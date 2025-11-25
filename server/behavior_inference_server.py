import os
import json
import time
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import cv2
import httpx
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlmodel import (
    SQLModel,
    Field,
    Session,
    create_engine,
    select,
)
from project_core.env import load_env, env_str, env_float, env_int, env_bool, env_path

from modeling.inference.pose_features import features_from_pose_seq
from modeling.inference.behavior_det import BehaviorDetector, BehaviorDetectorConfig
from modeling.inference.behavior_cls import BehaviorClassifier, BehaviorClassifierConfig
from modeling.tracking.engine import MultiCameraTracker, TrackResult


# ---------------------------------------------------------
# 로깅
# ---------------------------------------------------------

logger = logging.getLogger("god.behavior_server")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)

# ---------------------------------------------------------
# 환경 변수 / 경로 유틸
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
load_env(BASE_DIR)

# ---------------------------------------------------------
# DB 설정
# ---------------------------------------------------------

DATABASE_URL = env_str("DATABASE_URL")

engine = create_engine(DATABASE_URL, echo=False)


def get_session() -> Session:
    with Session(engine) as session:
        yield session


# ---------------------------------------------------------
# SQLModel 정의
# ---------------------------------------------------------


class Agent(SQLModel, table=True):
    __tablename__ = "agents"

    id: int | None = Field(default=None, primary_key=True)
    created_at: float = Field(default_factory=lambda: time.time())

    # 외부에서 사용하는 고유 코드 (예: 에이전트 장치 식별용)
    code: str = Field(index=True, unique=True)

    # 메타 정보 (예: 설치 위치, 담당자 등)
    name: str | None = None
    metadata_json: str | None = None


class BehaviorResult(SQLModel, table=True):
    __tablename__ = "behavior_result"

    id: int | None = Field(default=None, primary_key=True)
    created_at: float = Field(default_factory=lambda: time.time())

    agent_id: int = Field(foreign_key="agents.id", index=True)
    camera_id: str = Field(index=True)
    source_id: str | None = None

    window_index: int
    window_start_ts: float | None = None
    window_end_ts: float | None = None

    is_anomaly: bool
    stage1_normal: float
    stage1_anomaly: float

    stage2_labels_json: str
    stage2_probs_json: str
    stage2_top_label: str | None = None
    stage2_top_prob: float = 0.0


class KakaoAlarmLog(SQLModel, table=True):
    __tablename__ = "kakao_alarm_log"

    id: int | None = Field(default=None, primary_key=True)
    created_at: float = Field(default_factory=lambda: time.time())

    agent_id: int = Field(foreign_key="agents.id", index=True)
    camera_id: str = Field(index=True)
    source_id: str | None = None

    event_label: str
    start_window_index: int
    end_window_index: int
    duration_sec: float
    top_prob: float

    text_preview: str

    kakao_mode: str = "disabled"  # "disabled" | "real"
    kakao_ok: bool = False
    kakao_status_code: int | None = None
    kakao_error: str | None = None
    kakao_raw_response_json: str | None = None


class TrackingSnapshot(SQLModel, table=True):
    __tablename__ = "tracking_snapshot"

    id: int | None = Field(default=None, primary_key=True)
    created_at: float = Field(default_factory=lambda: time.time())

    agent_id: int = Field(foreign_key="agents.id", index=True)
    camera_id: str = Field(index=True)

    timestamp: float
    frame_index: int | None = None
    objects_json: str  # list[TrackingObject]-like dicts JSON


# ---------------------------------------------------------
# Pydantic 모델 (요청/응답)
# ---------------------------------------------------------


class PoseWindowRequest(BaseModel):
    agent_code: str
    camera_id: str
    source_id: str | None = None
    window_index: int
    keypoints: list[list[list[float]]]  # (T,33,4)


class BehaviorResultResponse(BaseModel):
    agent_code: str
    camera_id: str
    source_id: str | None = None

    window_index: int
    window_start_ts: float | None = None
    window_end_ts: float | None = None

    is_anomaly: bool
    det_prob: float
    top_label: str | None = None
    top_prob: float = 0.0
    prob: dict[str, float]


class BehaviorResultListResponse(BaseModel):
    results: list[BehaviorResultResponse]


class TrackingBBox(BaseModel):
    x: float
    y: float
    w: float
    h: float


class TrackingObjectResponse(BaseModel):
    global_id: str | None = None
    local_track_id: int
    label: str
    confidence: float
    bbox: TrackingBBox


class TrackingSnapshotResponse(BaseModel):
    agent_code: str
    camera_id: str
    timestamp: float
    frame_index: int | None
    objects: list[TrackingObjectResponse]


# ---------------------------------------------------------
# 에이전트 / 상태 관리
# ---------------------------------------------------------


def get_or_create_agent_by_code(session: Session, code: str) -> Agent:
    stmt = select(Agent).where(Agent.code == code)
    agent = session.exec(stmt).first()
    if agent is not None:
        return agent

    agent = Agent(code=code)
    session.add(agent)
    session.commit()
    session.refresh(agent)
    logger.info("Created new agent: id=%s code=%s", agent.id, agent.code)
    return agent


@dataclass
class BehaviorState:
    """카메라별 상태 관리용 구조체."""

    ema: float = 0.0
    last_window_index: int = -1

    # 현재 진행 중인 이벤트 여부
    in_event: bool = False
    event_label: str | None = None
    event_top_prob: float = 0.0
    event_start_ts: float | None = None
    event_start_window_index: int | None = None

    # 최근 윈도우들의 "이상 표" (True/False/None) 기록
    votes: deque[Any] = None  # type: ignore[assignment]
    last_update_ts: float = 0.0


_behavior_states: dict[tuple[int, str], BehaviorState] = {}


def _get_state(agent_id: int, camera_id: str) -> BehaviorState:
    key = (agent_id, camera_id)
    st = _behavior_states.get(key)
    if st is None:
        st = BehaviorState(
            ema=0.0,
            last_window_index=-1,
            in_event=False,
            event_label=None,
            event_top_prob=0.0,
            event_start_ts=None,
            event_start_window_index=None,
            votes=deque(maxlen=5),
            last_update_ts=0.0,
        )
        _behavior_states[key] = st
    return st


# ---------------------------------------------------------
# 모델 로딩 (Prep_v3 기반)
# ---------------------------------------------------------

DET_CKPT_PATH = env_path("DET_CKPT_PATH", BASE_DIR)
CLS_CKPT_PATH = env_path("CLS_CKPT_PATH", BASE_DIR)


_det_cfg: BehaviorDetectorConfig | None = None
_cls_cfg: BehaviorClassifierConfig | None = None
_detector: BehaviorDetector | None = None
_classifier: BehaviorClassifier | None = None


def get_detector() -> BehaviorDetector:
    global _det_cfg, _detector
    if _detector is None:
        det_cfg = BehaviorDetectorConfig(ckpt_path=DET_CKPT_PATH)
        _det_cfg = det_cfg
        _detector = BehaviorDetector(det_cfg)
        logger.info("Loaded BehaviorDetector from %s", DET_CKPT_PATH)
    assert _detector is not None
    return _detector


def get_classifier() -> BehaviorClassifier:
    global _cls_cfg, _classifier
    if _classifier is None:
        cls_cfg = BehaviorClassifierConfig(ckpt_path=CLS_CKPT_PATH)
        _cls_cfg = cls_cfg
        _classifier = BehaviorClassifier(cls_cfg)
        logger.info("Loaded BehaviorClassifier from %s", CLS_CKPT_PATH)
    assert _classifier is not None
    return _classifier


# ---------------------------------------------------------
# 이상행동 판정 파라미터
# ---------------------------------------------------------

EMA_ALPHA = env_float("EMA_ALPHA", 0.3)
DET_START_THR = env_float("DET_START_THR", 0.8)
DET_END_THR = env_float("DET_END_THR", 0.55)

DET_VOTE_WIN = env_int("DET_VOTE_WIN", 5)
DET_MIN_START_VOTES = env_int("DET_MIN_START_VOTES", 4)
DET_MIN_END_VOTES = env_int("DET_MIN_END_VOTES", 4)

DET_MIN_EVENT_SEC = env_float("DET_MIN_EVENT_SEC", 2.0)
DET_COOLDOWN_SEC = env_float("DET_COOLDOWN_SEC", 2.0)

logger.info(
    "Detection params: EMA_ALPHA=%.3f, START_THR=%.3f, END_THR=%.3f, "
    "VOTE_WIN=%d, MIN_START_VOTES=%d, MIN_END_VOTES=%d, "
    "MIN_EVENT_SEC=%.3f, COOLDOWN_SEC=%.3f",
    EMA_ALPHA,
    DET_START_THR,
    DET_END_THR,
    DET_VOTE_WIN,
    DET_MIN_START_VOTES,
    DET_MIN_END_VOTES,
    DET_MIN_EVENT_SEC,
    DET_COOLDOWN_SEC,
)


# ---------------------------------------------------------
# Kakao 알림 설정
# ---------------------------------------------------------

KAKAO_SEND_URL = os.getenv(
    "KAKAO_SEND_URL",
    "https://kapi.kakao.com/v2/api/talk/memo/default/send",
)


def _format_event_text(
    agent: Agent,
    camera_id: str,
    label: str,
    duration_sec: float,
    prob: float,
) -> str:
    return (
        f"[이상행동 감지]\n\n"
        f"- 에이전트: {agent.code}\n"
        f"- 카메라: {camera_id}\n"
        f"- 라벨: {label}\n"
        f"- 지속시간: {duration_sec:.1f}초\n"
        f"- 확률: {prob:.3f}\n"
    )


def _send_kakao_alarm(
    kakao_access_token: str,
    text: str,
) -> tuple[bool, int | None, str | None, str | None]:
    headers = {
        "Authorization": f"Bearer {kakao_access_token}",
        "Content-Type": "application/x-www-form-urlencoded;charset=utf-8",
    }
    payload = {
        "template_object": json.dumps(
            {
                "object_type": "text",
                "text": text,
                "link": {
                    "web_url": "https://example.com",
                    "mobile_web_url": "https://example.com",
                },
            },
            ensure_ascii=False,
        )
    }

    try:
        resp = httpx.post(KAKAO_SEND_URL, headers=headers, data=payload, timeout=5.0)
    except Exception as e:
        logger.exception("Kakao send failed: %s", e)
        return False, None, type(e).__name__, str(e)

    try:
        raw_json = resp.text
    except Exception:
        raw_json = None  # type: ignore[assignment]

    if resp.status_code == 200:
        return True, resp.status_code, None, raw_json
    return False, resp.status_code, f"HTTP {resp.status_code}", raw_json


# ---------------------------------------------------------
# FastAPI 앱
# ---------------------------------------------------------

app = FastAPI(title="Guardians of Dongne - Behavior Inference Server")

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[s.strip() for s in ALLOWED_ORIGINS.split(",") if s.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    SQLModel.metadata.create_all(engine)
    logger.info("Database tables ensured.")


# ---------------------------------------------------------
# 엔드포인트: 포즈 기반 이상행동 분석
# ---------------------------------------------------------


def run_detection_from_keypoints(kpt_seq: NDArray[np.floating]) -> float:
    """
    kpt_seq: (T,33,4) -> p_anom
    """
    feat = features_from_pose_seq(kpt_seq)  # (T,169)
    det = get_detector()
    return det(feat)


def run_classification_from_keypoints(
    kpt_seq: NDArray[np.floating],
) -> tuple[list[str], NDArray[np.float32]]:
    """
    kpt_seq: (T,33,4) -> (labels, probs)
    """
    feat = features_from_pose_seq(kpt_seq)  # (T,169)
    cls_model = get_classifier()
    labels, probs = cls_model.predict_proba(feat)
    return labels, probs


@app.post("/behavior/analyze_pose", response_model=BehaviorResultResponse)
def analyze_pose(
    payload: PoseWindowRequest,
    session: Session = Depends(get_session),
) -> BehaviorResultResponse:
    # 1) agent / state
    agent = get_or_create_agent_by_code(session, payload.agent_code)
    state = _get_state(agent.id, payload.camera_id)

    # 2) 포즈 텐서 변환
    kpt_seq = np.asarray(payload.keypoints, dtype=np.float32)
    if kpt_seq.ndim != 3 or kpt_seq.shape[1:] != (33, 4):
        raise HTTPException(
            status_code=400,
            detail=f"keypoints shape must be (T,33,4), got {kpt_seq.shape}",
        )

    # 3) 모델 추론
    p_anom = run_detection_from_keypoints(kpt_seq)
    events, probs = run_classification_from_keypoints(kpt_seq)

    # softmax 결과를 dict로
    prob_map = {label: float(p) for label, p in zip(events, probs)}
    top_label: str | None
    top_prob: float
    if prob_map:
        top_label = max(prob_map, key=prob_map.get)
        top_prob = prob_map[top_label]
    else:
        top_label = None
        top_prob = 0.0

    # 4) EMA / voting 기반 이벤트 상태 업데이트
    now_ts = time.time()
    state.last_update_ts = now_ts

    prev_ema = state.ema
    ema = EMA_ALPHA * p_anom + (1.0 - EMA_ALPHA) * prev_ema
    state.ema = ema

    is_anomaly_window = ema >= DET_START_THR

    # votes deque 업데이트
    if state.votes is None:
        state.votes = deque(maxlen=DET_VOTE_WIN)  # type: ignore[assignment]
    votes = state.votes
    votes.append(is_anomaly_window)

    start_votes = sum(1 for v in votes if v is True)
    end_votes = sum(1 for v in votes if v is False)

    event_ended = False
    event_label: str | None = None
    event_duration_sec: float = 0.0
    event_start_index_for_log: int | None = None

    # 이벤트 시작 조건
    if (
        not state.in_event
        and ema >= DET_START_THR
        and start_votes >= DET_MIN_START_VOTES
    ):
        state.in_event = True
        state.event_label = top_label
        state.event_top_prob = top_prob
        state.event_start_ts = now_ts
        state.event_start_window_index = payload.window_index

        logger.info(
            "[EVENT START] agent=%s cam=%s label=%s win_idx=%s ema=%.3f p_anom=%.3f",
            agent.code,
            payload.camera_id,
            top_label,
            payload.window_index,
            ema,
            p_anom,
        )

    # 이벤트 종료 조건
    elif (
        state.in_event
        and ema <= DET_END_THR
        and end_votes >= DET_MIN_END_VOTES
    ):
        # 로그용으로 시작 인덱스/라벨 보관
        event_label = state.event_label
        event_start_index_for_log = state.event_start_window_index

        if state.event_start_ts is not None:
            event_duration_sec = now_ts - state.event_start_ts
        else:
            event_duration_sec = 0.0

        if event_duration_sec >= DET_MIN_EVENT_SEC:
            event_ended = True

        logger.info(
            "[EVENT END-CANDIDATE] agent=%s cam=%s label=%s duration=%.2fs ema=%.3f p_anom=%.3f",
            agent.code,
            payload.camera_id,
            event_label,
            event_duration_sec,
            ema,
            p_anom,
        )

        # 상태 초기화 + 쿨다운
        state.in_event = False
        state.event_label = None
        state.event_top_prob = 0.0
        state.event_start_ts = None
        state.event_start_window_index = None

        if event_ended:
            state.last_update_ts = now_ts + DET_COOLDOWN_SEC

    # 5) DB 기록 (BehaviorResult)
    is_anomaly = ema >= DET_START_THR
    result = BehaviorResult(
        agent_id=agent.id,
        camera_id=payload.camera_id,
        source_id=payload.source_id,
        window_index=payload.window_index,
        window_start_ts=None,
        window_end_ts=None,
        is_anomaly=is_anomaly,
        stage1_normal=1.0 - p_anom,
        stage1_anomaly=p_anom,
        stage2_labels_json=json.dumps(events, ensure_ascii=False),
        stage2_probs_json=json.dumps(probs.tolist(), ensure_ascii=False),
        stage2_top_label=top_label,
        stage2_top_prob=top_prob,
    )
    session.add(result)
    session.commit()
    session.refresh(result)

    # 6) Kakao 알림 (이벤트 종료 시)
    if event_ended and event_label is not None:
        kakao_access_token = os.getenv("KAKAO_ACCESS_TOKEN")
        kakao_mode = "disabled"
        kakao_ok = False
        kakao_status_code: int | None = None
        kakao_error: str | None = None
        kakao_raw_json: str | None = None

        text = ""
        if kakao_access_token:
            kakao_mode = "real"
            text = _format_event_text(
                agent=agent,
                camera_id=payload.camera_id,
                label=event_label,
                duration_sec=event_duration_sec,
                prob=state.event_top_prob,
            )
            kakao_ok, kakao_status_code, kakao_error, kakao_raw_json = _send_kakao_alarm(
                kakao_access_token,
                text,
            )
            logger.info(
                "Kakao send: ok=%s status=%s error=%s",
                kakao_ok,
                kakao_status_code,
                kakao_error,
            )
        else:
            kakao_mode = "disabled"
            logger.info(
                "Kakao disabled for agent_code=%s (no kakao_access_token), event=%s",
                agent.code,
                event_label,
            )

        log = KakaoAlarmLog(
            agent_id=agent.id,
            camera_id=payload.camera_id,
            source_id=payload.source_id,
            event_label=event_label,
            start_window_index=event_start_index_for_log,
            end_window_index=payload.window_index,
            duration_sec=event_duration_sec,
            top_prob=state.event_top_prob,
            text_preview=text[:2000] if kakao_access_token else "",
            kakao_mode=kakao_mode,
            kakao_ok=kakao_ok,
            kakao_status_code=kakao_status_code,
            kakao_error=kakao_error,
            kakao_raw_response_json=kakao_raw_json,
        )
        session.add(log)
        session.commit()

        logger.info(
            "[EVENT END] agent=%s cam=%s label=%s duration=%.2fs kakao_mode=%s kakao_ok=%s",
            agent.code,
            payload.camera_id,
            event_label,
            event_duration_sec,
            kakao_mode,
            kakao_ok,
        )

    return BehaviorResultResponse(
        agent_code=agent.code,
        camera_id=payload.camera_id,
        source_id=payload.source_id,
        window_index=payload.window_index,
        window_start_ts=None,
        window_end_ts=None,
        is_anomaly=is_anomaly,
        det_prob=p_anom,
        top_label=top_label,
        top_prob=top_prob,
        prob=prob_map,
    )


@app.get(
    "/behavior/latest_by_camera",
    response_model=list[BehaviorResultResponse],
)
def behavior_latest_by_camera(
    agent_code: str,
    session: Session = Depends(get_session),
) -> list[BehaviorResultResponse]:
    agent = get_or_create_agent_by_code(session, agent_code)

    stmt = (
        select(BehaviorResult)
        .where(BehaviorResult.agent_id == agent.id)
        .order_by(BehaviorResult.id.desc())
    )
    rows = session.exec(stmt).all()

    latest_by_camera: dict[str, BehaviorResult] = {}
    for r in rows:
        if r.camera_id not in latest_by_camera:
            latest_by_camera[r.camera_id] = r

    results: list[BehaviorResultResponse] = []
    for r in latest_by_camera.values():
        events = json.loads(r.stage2_labels_json)
        probs = json.loads(r.stage2_probs_json)
        prob_map = {label: float(p) for label, p in zip(events, probs)}

        results.append(
            BehaviorResultResponse(
                agent_code=agent.code,
                camera_id=r.camera_id,
                source_id=r.source_id,
                window_index=r.window_index,
                window_start_ts=r.window_start_ts,
                window_end_ts=r.window_end_ts,
                is_anomaly=r.is_anomaly,
                det_prob=r.stage1_anomaly,
                top_label=r.stage2_top_label,
                top_prob=r.stage2_top_prob,
                prob=prob_map,
            )
        )

    return results


# ---------------------------------------------------------
# 트래킹: 설정 / 엔드포인트 (YOLO+DeepSORT + optional ReID/Global ID)
# ---------------------------------------------------------

TRACKING_ENABLED = env_bool("TRACKING_ENABLED", False)
TRACK_ENABLE_REID = env_bool("TRACK_ENABLE_REID", False)
TRACK_ENABLE_GLOBAL_ID = env_bool("TRACK_ENABLE_GLOBAL_ID", False)

_tracking_engine: MultiCameraTracker | None = None


def get_tracking_engine() -> MultiCameraTracker | None:
    """TRACKING_ENABLED 이고 설정이 정상일 때만 트래킹 엔진을 반환한다."""
    global _tracking_engine

    if not TRACKING_ENABLED:
        return None

    if _tracking_engine is not None:
        return _tracking_engine

    # .env 에서 트래킹 관련 파라미터 읽기
    yolo_weights_path = env_path("TRACK_YOLO_WEIGHTS", BASE_DIR)
    conf_thres = env_float("TRACK_CONF_THRESH", 0.5)
    iou_thres = env_float("TRACK_IOU_THRESH", 0.7)
    max_age = env_int("TRACK_MAX_AGE", 30)

    enable_reid = TRACK_ENABLE_REID
    enable_global_id = TRACK_ENABLE_GLOBAL_ID

    reid_dist_thres = env_float("TRACK_REID_DIST_THRES", 0.25)
    reid_model_name = env_str("TRACK_REID_MODEL_NAME", "osnet_x1_0")

    # 선택적인 ReID weight 경로 (없으면 None)
    try:
        reid_model_path = env_path("TRACK_REID_MODEL_PATH", BASE_DIR)
        reid_model_path_str: str | None = str(reid_model_path)
    except RuntimeError:
        reid_model_path_str = None

    _tracking_engine = MultiCameraTracker(
        yolo_weights=str(yolo_weights_path),
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        max_age=max_age,
        reid_cosine_thres=reid_dist_thres,
        reid_model_name=reid_model_name,
        reid_model_path=reid_model_path_str,
        enable_reid=enable_reid,
        enable_global_id=enable_global_id,
    )
    logger.info(
        "Tracking engine initialized: enabled=%s, reid=%s, global_id=%s, weights=%s",
        TRACKING_ENABLED,
        enable_reid,
        enable_global_id,
        yolo_weights_path,
    )
    return _tracking_engine


@app.post("/tracking")
async def tracking_endpoint(
    agent_code: str = Form(...),
    camera_id: str = Form(...),
    timestamp: float | None = Form(None),
    frame_index: int | None = Form(None),
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """트래킹용 프레임 업로드 엔드포인트."""
    agent = get_or_create_agent_by_code(session, agent_code)

    frame_bytes = await file.read()
    ts = timestamp if timestamp is not None else time.time()

    objects_list: list[dict[str, Any]] = []

    engine = get_tracking_engine()
    if engine is None:
        logger.info(
            "tracking disabled: TRACKING_ENABLED=False (camera_id=%s)",
            camera_id,
        )
    else:
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame_bgr is None:
            logger.warning(
                "tracking: JPEG frame decode failed (camera_id=%s)",
                camera_id,
            )
        else:
            tracks: list[TrackResult] = engine.process_frame(
                camera_id=camera_id,
                frame_bgr=frame_bgr,
                timestamp=ts,
            )
            for t in tracks:
                objects_list.append(
                    {
                        "global_id": t.global_id,
                        "local_track_id": t.local_track_id,
                        "label": t.label,
                        "confidence": float(t.confidence),
                        "bbox": {
                            "x": float(t.bbox.x),
                            "y": float(t.bbox.y),
                            "w": float(t.bbox.w),
                            "h": float(t.bbox.h),
                        },
                    }
                )

    snap = TrackingSnapshot(
        agent_id=agent.id,
        camera_id=camera_id,
        timestamp=ts,
        frame_index=frame_index,
        objects_json=json.dumps(objects_list, ensure_ascii=False),
    )
    session.add(snap)
    session.commit()
    session.refresh(snap)

    return {
        "ok": True,
        "snapshot_id": snap.id,
        "agent_id": agent.id,
        "camera_id": camera_id,
        "num_objects": len(objects_list),
    }


@app.get("/tracking/latest/{camera_id}", response_model=TrackingSnapshotResponse | None)
def tracking_latest_for_camera(
    camera_id: str,
    agent_code: str,
    session: Session = Depends(get_session),
) -> TrackingSnapshotResponse | None:
    agent = get_or_create_agent_by_code(session, agent_code)
    stmt = (
        select(TrackingSnapshot)
        .where(
            TrackingSnapshot.agent_id == agent.id,
            TrackingSnapshot.camera_id == camera_id,
        )
        .order_by(TrackingSnapshot.id.desc())
        .limit(1)
    )
    row = session.exec(stmt).first()
    if row is None:
        return None

    objects_raw = json.loads(row.objects_json)
    objects: list[TrackingObjectResponse] = []
    for obj in objects_raw:
        bbox_dict = obj.get("bbox", {})
        objects.append(
            TrackingObjectResponse(
                global_id=obj.get("global_id"),
                local_track_id=int(obj.get("local_track_id", 0)),
                label=str(obj.get("label", "")),
                confidence=float(obj.get("confidence", 0.0)),
                bbox=TrackingBBox(
                    x=float(bbox_dict.get("x", 0.0)),
                    y=float(bbox_dict.get("y", 0.0)),
                    w=float(bbox_dict.get("w", 0.0)),
                    h=float(bbox_dict.get("h", 0.0)),
                ),
            )
        )

    return TrackingSnapshotResponse(
        agent_code=agent.code,
        camera_id=row.camera_id,
        timestamp=row.timestamp,
        frame_index=row.frame_index,
        objects=objects,
    )


@app.get("/tracking/latest_all", response_model=list[TrackingSnapshotResponse])
def tracking_latest_all(
    agent_code: str,
    session: Session = Depends(get_session),
) -> list[TrackingSnapshotResponse]:
    agent = get_or_create_agent_by_code(session, agent_code)

    stmt = (
        select(TrackingSnapshot)
        .where(TrackingSnapshot.agent_id == agent.id)
        .order_by(TrackingSnapshot.id.desc())
    )
    rows = session.exec(stmt).all()

    latest_by_camera: dict[str, TrackingSnapshot] = {}
    for r in rows:
        if r.camera_id not in latest_by_camera:
            latest_by_camera[r.camera_id] = r

    results: list[TrackingSnapshotResponse] = []
    for r in latest_by_camera.values():
        objects_raw = json.loads(r.objects_json)
        objects: list[TrackingObjectResponse] = []
        for obj in objects_raw:
            bbox_dict = obj.get("bbox", {})
            objects.append(
                TrackingObjectResponse(
                    global_id=obj.get("global_id"),
                    local_track_id=int(obj.get("local_track_id", 0)),
                    label=str(obj.get("label", "")),
                    confidence=float(obj.get("confidence", 0.0)),
                    bbox=TrackingBBox(
                        x=float(bbox_dict.get("x", 0.0)),
                        y=float(bbox_dict.get("y", 0.0)),
                        w=float(bbox_dict.get("w", 0.0)),
                        h=float(bbox_dict.get("h", 0.0)),
                    ),
                )
            )

        results.append(
            TrackingSnapshotResponse(
                agent_code=agent.code,
                camera_id=r.camera_id,
                timestamp=r.timestamp,
                frame_index=r.frame_index,
                objects=objects,
            )
        )

    return results
