import json
import logging
import time
from collections import deque
from collections.abc import Generator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from numpy.typing import NDArray
from pydantic import BaseModel
from sqlmodel import (
    Field,
    Session,
    SQLModel,
    create_engine,
    select,
)

from modeling.inference.behavior_cls import (
    BehaviorClassifier,
    BehaviorClassifierConfig,
)
from modeling.inference.behavior_det import (
    BehaviorDetector,
    BehaviorDetectorConfig,
)
from modeling.inference.pose_features import features_from_pose_seq
from modeling.tracking.engine import MultiCameraTracker, TrackResult
from project_core.env import (
    env_bool,
    env_float,
    env_int,
    env_path,
    env_str,
    load_env,
)


# ---------------------------------------------------------
# 기본 설정 / 로깅 / ENV
# ---------------------------------------------------------

logger = logging.getLogger("god.behavior_server")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)

BASE_DIR = Path(__file__).resolve().parent
load_env(BASE_DIR)

DATABASE_URL = env_str("DATABASE_URL")

# Detection / event 파라미터 (EMA, threshold, voting 등)
EMA_ALPHA = env_float("DET_EMA_ALPHA", 0.3)
DET_START_THR = env_float("DET_START_THR", 0.8)
DET_END_THR = env_float("DET_END_THR", 0.55)
DET_VOTE_WIN = env_int("DET_VOTE_WIN", 5)
DET_MIN_START_VOTES = env_int("DET_MIN_START_VOTES", 4)
DET_MIN_END_VOTES = env_int("DET_MIN_END_VOTES", 4)
DET_MIN_EVENT_SEC = env_float("DET_MIN_EVENT_SEC", 2.0)
DET_COOLDOWN_SEC = env_float("DET_COOLDOWN_SEC", 2.0)

logger.info(
    "Detection params: "
    "EMA_ALPHA=%.3f, START_THR=%.3f, END_THR=%.3f, VOTE_WIN=%d, "
    "MIN_START_VOTES=%d, MIN_END_VOTES=%d, MIN_EVENT_SEC=%.3f, COOLDOWN_SEC=%.3f",
    EMA_ALPHA,
    DET_START_THR,
    DET_END_THR,
    DET_VOTE_WIN,
    DET_MIN_START_VOTES,
    DET_MIN_END_VOTES,
    DET_MIN_EVENT_SEC,
    DET_COOLDOWN_SEC,
)

# Tracking 관련 ENV
TRACKING_ENABLED = env_bool("TRACKING_ENABLED", False)
TRACK_YOLO_WEIGHTS = env_path("TRACK_YOLO_WEIGHTS", BASE_DIR / "yolo11n.pt")
TRACK_CONF_THRES = env_float("TRACK_CONF_THRES", 0.4)
TRACK_IOU_THRES = env_float("TRACK_IOU_THRES", 0.5)
TRACK_MAX_AGE = env_int("TRACK_MAX_AGE", 30)

TRACK_ENABLE_REID = env_bool("TRACK_ENABLE_REID", False)
TRACK_ENABLE_GLOBAL_ID = env_bool("TRACK_ENABLE_GLOBAL_ID", False)
TRACK_REID_MODEL_NAME = env_str("TRACK_REID_MODEL_NAME", "osnet_x1_0")
TRACK_REID_MODEL_PATH = env_path("TRACK_REID_MODEL_PATH", BASE_DIR / "osnet_x1_0_imagenet.pth")
TRACK_REID_COSINE_THRES = env_float("TRACK_REID_COSINE_THRES", 0.3)

# Behavior 모델 경로
DET_CKPT_PATH = env_path("DET_CKPT_PATH", BASE_DIR / "detector.ckpt")
CLS_CKPT_PATH = env_path("CLS_CKPT_PATH", BASE_DIR / "classifier.ckpt")

# ---------------------------------------------------------
# DB 초기화
# ---------------------------------------------------------

engine = create_engine(DATABASE_URL, echo=False)


# ---------------------------------------------------------
# SQLModel 모델
# ---------------------------------------------------------


class Agent(SQLModel, table=True):
    __tablename__ = "agents"

    id: int | None = Field(default=None, primary_key=True)
    created_at: float = Field(default_factory=lambda: time.time())

    # 사실상 논리적 primary key 역할
    code: str = Field(index=True, unique=True)

    # kakao access token (nullable)
    kakao_access_token: str | None = None

    # 기타 메모 / 설명
    note: str | None = None


class BehaviorResult(SQLModel, table=True):
    """
    기존 InferenceResult 를 BehaviorResult 로 리네이밍.
    테이블 이름도 behavior_result 로 변경.
    """

    __tablename__ = "behavior_result"

    id: int | None = Field(default=None, primary_key=True)
    created_at: float = Field(default_factory=lambda: time.time())

    agent_id: int = Field(foreign_key="agents.id", index=True)
    camera_id: str = Field(index=True)
    source_id: str | None = None

    window_index: int
    window_start_ts: float | None = None
    window_end_ts: float | None = None

    # stage1: 정상/이상 확률
    is_anomaly: bool
    stage1_normal: float
    stage1_anomaly: float

    # stage2: 라벨들 / 확률들 (JSON으로 저장)
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
    start_window_index: int | None = None
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
    """
    한 시점(프레임)에 대한 트래킹 스냅샷.
    objects_json 안에 global_id / local_track_id / label / confidence / bbox(xywh_normalized)
    를 담는 리스트가 저장된다.
    """

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


class HealthResponse(BaseModel):
    status: str = "ok"


class AgentRegisterRequest(BaseModel):
    agent_code: str
    kakao_access_token: str | None = None
    note: str | None = None


class KakaoTestResponse(BaseModel):
    ok: bool
    status_code: int | None
    error: str | None


class PoseWindowRequest(BaseModel):
    agent_code: str
    camera_id: str
    source_id: str | None = None
    window_index: int
    # (T, 33, 4) 형태를 가정한 포즈 시퀀스
    keypoints: list[list[list[float]]]


class BehaviorResultResponse(BaseModel):
    agent_code: str
    camera_id: str
    source_id: str | None
    window_index: int
    window_start_ts: float | None
    window_end_ts: float | None

    is_anomaly: bool
    det_prob: float  # stage1 anomaly prob

    top_label: str | None
    top_prob: float
    # 2차 분류 라벨 -> 확률 맵
    prob: dict[str, float]


class TrackingBBox(BaseModel):
    x: float
    y: float
    w: float
    h: float


class TrackingObjectResponse(BaseModel):
    global_id: str | None
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
# DB Session dependency
# ---------------------------------------------------------


def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session


# ---------------------------------------------------------
# Behavior 모델 / 상태 관리
# ---------------------------------------------------------


@dataclass
class BehaviorState:
    ema: float = 0.0
    in_event: bool = False
    event_label: str | None = None
    event_top_prob: float = 0.0
    event_start_ts: float | None = None
    event_start_window_index: int | None = None
    last_update_ts: float = 0.0
    votes: deque[bool] | None = None


# (agent_id, camera_id) → BehaviorState
_behavior_states: dict[tuple[int, str], BehaviorState] = {}

_detector: BehaviorDetector | None = None
_classifier: BehaviorClassifier | None = None


def _get_state(agent_id: int, camera_id: str) -> BehaviorState:
    key = (agent_id, camera_id)
    state = _behavior_states.get(key)
    if state is None:
        state = BehaviorState()
        _behavior_states[key] = state
    return state


def _load_behavior_models() -> None:
    global _detector, _classifier

    logger.info("Loading behavior models...")
    det_cfg = BehaviorDetectorConfig(
        ckpt_path=str(DET_CKPT_PATH),
    )
    cls_cfg = BehaviorClassifierConfig(
        ckpt_path=str(CLS_CKPT_PATH),
    )
    _detector = BehaviorDetector(det_cfg)
    _classifier = BehaviorClassifier(cls_cfg)
    logger.info("Behavior models loaded.")


def run_detection_from_keypoints(kpt_seq: NDArray[np.float32]) -> float:
    if _detector is None:
        raise RuntimeError("BehaviorDetector not initialized")

    feats = features_from_pose_seq(kpt_seq)  # (T, F) or (F,) depending on impl
    # BehaviorDetector 설계에 따라 다를 수 있지만,
    # 여기서는 "이상일 확률" 하나를 float로 돌려준다고 가정한다.
    p_anom = _detector.predict_anomaly_proba(feats)
    return float(p_anom)


def run_classification_from_keypoints(
    kpt_seq: NDArray[np.float32],
) -> tuple[list[str], NDArray[np.float32]]:
    if _classifier is None:
        raise RuntimeError("BehaviorClassifier not initialized")

    feats = features_from_pose_seq(kpt_seq)
    labels, probs = _classifier.predict_proba(feats)
    return list(labels), probs.astype(np.float32)


# ---------------------------------------------------------
# Tracking 엔진
# ---------------------------------------------------------

_tracking_engine: MultiCameraTracker | None = None


def _init_tracking_engine() -> None:
    global _tracking_engine

    if not TRACKING_ENABLED:
        logger.info("Tracking is disabled by env (TRACKING_ENABLED=false).")
        _tracking_engine = None
        return

    try:
        logger.info(
            "Initializing MultiCameraTracker: weights=%s, conf_thres=%.3f, iou_thres=%.3f, max_age=%d, "
            "enable_reid=%s, enable_global_id=%s",
            TRACK_YOLO_WEIGHTS,
            TRACK_CONF_THRES,
            TRACK_IOU_THRES,
            TRACK_MAX_AGE,
            TRACK_ENABLE_REID,
            TRACK_ENABLE_GLOBAL_ID,
        )

        _tracking_engine = MultiCameraTracker(
            yolo_weights=str(TRACK_YOLO_WEIGHTS),
            conf_thres=TRACK_CONF_THRES,
            iou_thres=TRACK_IOU_THRES,
            max_age=TRACK_MAX_AGE,
            reid_cosine_thres=TRACK_REID_COSINE_THRES,
            reid_model_name=TRACK_REID_MODEL_NAME,
            reid_model_path=str(TRACK_REID_MODEL_PATH),
            enable_reid=TRACK_ENABLE_REID,
            enable_global_id=TRACK_ENABLE_GLOBAL_ID,
        )
        logger.info("MultiCameraTracker initialized.")
    except Exception:
        logger.exception("Failed to initialize MultiCameraTracker. Disabling tracking.")
        _tracking_engine = None


# ---------------------------------------------------------
# Kakao helpers
# ---------------------------------------------------------


def _send_kakao_alarm(
    access_token: str,
    text: str,
) -> tuple[bool, int | None, str | None, str | None]:
    """
    Kakao 메시지를 실제로 보내는 helper.
    ACCESS TOKEN 은 항상 Agent.kakao_access_token 에서 가져온다.
    """
    url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/x-www-form-urlencoded;charset=utf-8",
    }
    payload = {
        "template_object": json.dumps(
            {
                "object_type": "text",
                "text": text,
                "link": {"web_url": "https://example.com"},
            },
            ensure_ascii=False,
        )
    }
    try:
        resp = httpx.post(url, headers=headers, data=payload, timeout=5.0)
        ok = resp.status_code == 200
        err = None if ok else resp.text
        return ok, resp.status_code, err, resp.text
    except Exception as exc:
        logger.exception("Kakao send failed")
        return False, None, str(exc), None


def _format_event_text(
    agent: Agent,
    camera_id: str,
    label: str | None,
    duration_sec: float,
    prob: float,
) -> str:
    lbl = label or "unknown"
    return (
        f"[이상행동 감지]\n"
        f"- 에이전트: {agent.code}\n"
        f"- 카메라: {camera_id}\n"
        f"- 라벨: {lbl}\n"
        f"- 지속 시간: {duration_sec:.1f}초\n"
        f"- 신뢰도(추정): {prob:.2f}"
    )


# ---------------------------------------------------------
# FastAPI 앱 / lifespan
# ---------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # DB 테이블 보장
    SQLModel.metadata.create_all(engine)

    # 모델 / 트래킹 엔진 로드
    _load_behavior_models()
    _init_tracking_engine()

    yield
    # 종료 시 별도 정리는 현재 없음


app = FastAPI(
    title="GoD Behavior Inference & Tracking Server",
    lifespan=lifespan,
)

# CORS (필요하다면 수정)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Helper: Agent
# ---------------------------------------------------------


def get_or_create_agent_by_code(session: Session, agent_code: str) -> Agent:
    stmt = select(Agent).where(Agent.code == agent_code)
    agent = session.exec(stmt).first()
    if agent is None:
        agent = Agent(code=agent_code)
        session.add(agent)
        session.commit()
        session.refresh(agent)
        logger.info("Created new Agent row for code=%s", agent_code)
    return agent


# ---------------------------------------------------------
# /health
# ---------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


# ---------------------------------------------------------
# /agent/register_kakao
# ---------------------------------------------------------


@app.post("/agent/register_kakao", response_model=AgentRegisterRequest)
def register_kakao(
    payload: AgentRegisterRequest,
    session: Session = Depends(get_session),
) -> AgentRegisterRequest:
    """
    에이전트 최초 실행 시, agent-code + kakao_access_token 등록/업데이트.
    """
    agent = get_or_create_agent_by_code(session, payload.agent_code)
    agent.kakao_access_token = payload.kakao_access_token
    agent.note = payload.note
    session.add(agent)
    session.commit()
    logger.info(
        "Registered/updated Kakao token for agent=%s has_token=%s",
        agent.code,
        bool(agent.kakao_access_token),
    )
    return payload


# ---------------------------------------------------------
# /kakao/test
# ---------------------------------------------------------


@app.get("/kakao/test", response_model=KakaoTestResponse)
def kakao_test(
    agent_code: str,
    text: str = "테스트 메시지입니다.",
    session: Session = Depends(get_session),
) -> KakaoTestResponse:
    """
    에이전트에 저장된 Kakao 토큰으로 테스트 메시지를 보내본다.
    """
    stmt = select(Agent).where(Agent.code == agent_code)
    agent = session.exec(stmt).first()
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")

    if not agent.kakao_access_token:
        raise HTTPException(status_code=400, detail="Agent has no Kakao token")

    ok, status_code, err, _raw = _send_kakao_alarm(agent.kakao_access_token, text)
    return KakaoTestResponse(ok=ok, status_code=status_code, error=err)


# ---------------------------------------------------------
# /behavior/analyze_pose
# ---------------------------------------------------------


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

    prob_map = {label: float(p) for label, p in zip(events, probs)}

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
    event_top_prob_for_log: float = 0.0

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
        event_label = state.event_label
        event_start_index_for_log = state.event_start_window_index
        event_top_prob_for_log = state.event_top_prob

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

    # 6) Kakao 알림 (이벤트 종료 시만)
    if event_ended and event_label is not None:
        kakao_mode = "disabled"
        kakao_ok = False
        kakao_status_code: int | None = None
        kakao_error: str | None = None
        kakao_raw_json: str | None = None

        text = ""
        if agent.kakao_access_token:
            kakao_mode = "real"
            text = _format_event_text(
                agent=agent,
                camera_id=payload.camera_id,
                label=event_label,
                duration_sec=event_duration_sec,
                prob=event_top_prob_for_log,
            )
            kakao_ok, kakao_status_code, kakao_error, kakao_raw_json = _send_kakao_alarm(
                agent.kakao_access_token,
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
            top_prob=event_top_prob_for_log,
            text_preview=text[:2000] if agent.kakao_access_token else "",
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


# ---------------------------------------------------------
# /behavior/latest* (legacy 유지)
# ---------------------------------------------------------


def _behavior_row_to_response(
    agent_code: str,
    row: BehaviorResult,
) -> BehaviorResultResponse:
    events = json.loads(row.stage2_labels_json)
    probs = json.loads(row.stage2_probs_json)
    prob_map = {label: float(p) for label, p in zip(events, probs)}

    return BehaviorResultResponse(
        agent_code=agent_code,
        camera_id=row.camera_id,
        source_id=row.source_id,
        window_index=row.window_index,
        window_start_ts=row.window_start_ts,
        window_end_ts=row.window_end_ts,
        is_anomaly=row.is_anomaly,
        det_prob=row.stage1_anomaly,
        top_label=row.stage2_top_label,
        top_prob=row.stage2_top_prob,
        prob=prob_map,
    )


@app.get(
    "/behavior/latest/{camera_id}",
    response_model=BehaviorResultResponse | None,
)
def behavior_latest_for_camera(
    camera_id: str,
    agent_code: str,
    session: Session = Depends(get_session),
) -> BehaviorResultResponse | None:
    """특정 에이전트 + 특정 카메라의 최신 1개 결과만 반환."""
    agent = get_or_create_agent_by_code(session, agent_code)

    stmt = (
        select(BehaviorResult)
        .where(
            BehaviorResult.agent_id == agent.id,
            BehaviorResult.camera_id == camera_id,
        )
        .order_by(BehaviorResult.id.desc())
        .limit(1)
    )
    row = session.exec(stmt).first()
    if row is None:
        return None

    return _behavior_row_to_response(agent.code, row)


@app.get(
    "/behavior/latest_all",
    response_model=list[BehaviorResultResponse],
)
def behavior_latest_all(
    agent_code: str,
    session: Session = Depends(get_session),
) -> list[BehaviorResultResponse]:
    """특정 에이전트에 대해, 각 카메라별 최신 1개 결과를 모두 반환."""
    agent = get_or_create_agent_by_code(session, agent_code)

    stmt = (
        select(BehaviorResult)
        .where(BehaviorResult.agent_id == agent.id)
        .order_by(BehaviorResult.id.desc())
    )
    rows = session.exec(stmt).all()

    latest_by_camera: dict[str, BehaviorResult] = {}
    for r in rows:
        # id desc 순으로 보고 있으므로, 처음 보는 camera_id 가 그 카메라의 최신 1개
        if r.camera_id not in latest_by_camera:
            latest_by_camera[r.camera_id] = r

    return [
        _behavior_row_to_response(agent.code, r)
        for r in latest_by_camera.values()
    ]


# ---------------------------------------------------------
# /tracking
# ---------------------------------------------------------


@app.post("/tracking")
async def tracking_upload(
    agent_code: str = Form(...),
    camera_id: str = Form(...),
    timestamp: float = Form(...),
    frame_index: int | None = Form(None),
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """
    에이전트가 프레임 이미지를 업로드하는 엔드포인트.
    - bbox 는 항상 0~1 정규화된 xywh 로 DB/응답에 저장한다.
    """
    agent = get_or_create_agent_by_code(session, agent_code)

    # 파일 → 이미지
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image data")
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Failed to decode image")

    h, w = frame.shape[:2]

    objects_list: list[dict[str, Any]] = []

    if _tracking_engine is not None and TRACKING_ENABLED:
        try:
            tracks: list[TrackResult] = _tracking_engine.process_frame(
                frame_bgr=frame,
                camera_id=camera_id,
                timestamp=timestamp,
            )
            for t in tracks:
                x_px, y_px, w_px, h_px = t.bbox  # pixel 단위
                # 0~1 정규화
                x_norm = float(x_px) / float(w) if w > 0 else 0.0
                y_norm = float(y_px) / float(h) if h > 0 else 0.0
                w_norm = float(w_px) / float(w) if w > 0 else 0.0
                h_norm = float(h_px) / float(h) if h > 0 else 0.0

                objects_list.append(
                    {
                        "global_id": t.global_id,
                        "local_track_id": int(t.local_track_id),
                        "label": t.label,
                        "confidence": float(t.confidence),
                        "bbox": {
                            "x": x_norm,
                            "y": y_norm,
                            "w": w_norm,
                            "h": h_norm,
                        },
                    }
                )
        except Exception:
            logger.exception("Tracking failed; storing empty objects.")
            objects_list = []

    snapshot = TrackingSnapshot(
        agent_id=agent.id,
        camera_id=camera_id,
        timestamp=timestamp,
        frame_index=frame_index,
        objects_json=json.dumps(objects_list, ensure_ascii=False),
    )
    session.add(snapshot)
    session.commit()

    return {"ok": True, "num_objects": len(objects_list)}


# ---------------------------------------------------------
# /tracking/latest* (legacy 유지, bbox 정규화된 값 사용)
# ---------------------------------------------------------


def _tracking_snapshot_to_response(
    agent_code: str,
    snap: TrackingSnapshot,
) -> TrackingSnapshotResponse:
    try:
        raw_objects = json.loads(snap.objects_json)
    except Exception:
        raw_objects = []

    objects: list[TrackingObjectResponse] = []
    for obj in raw_objects:
        bbox = obj.get("bbox") or {}
        objects.append(
            TrackingObjectResponse(
                global_id=obj.get("global_id"),
                local_track_id=int(obj.get("local_track_id", 0)),
                label=str(obj.get("label", "person")),
                confidence=float(obj.get("confidence", 0.0)),
                bbox=TrackingBBox(
                    x=float(bbox.get("x", 0.0)),
                    y=float(bbox.get("y", 0.0)),
                    w=float(bbox.get("w", 0.0)),
                    h=float(bbox.get("h", 0.0)),
                ),
            )
        )

    return TrackingSnapshotResponse(
        agent_code=agent_code,
        camera_id=snap.camera_id,
        timestamp=snap.timestamp,
        frame_index=snap.frame_index,
        objects=objects,
    )


@app.get(
    "/tracking/latest/{camera_id}",
    response_model=TrackingSnapshotResponse | None,
)
def tracking_latest_for_camera(
    camera_id: str,
    agent_code: str,
    session: Session = Depends(get_session),
) -> TrackingSnapshotResponse | None:
    """
    특定 에이전트 + 카메라의 최신 1개 트래킹 스냅샷.
    """
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
    snap = session.exec(stmt).first()
    if snap is None:
        return None

    return _tracking_snapshot_to_response(agent.code, snap)


@app.get(
    "/tracking/latest_all",
    response_model=list[TrackingSnapshotResponse],
)
def tracking_latest_all(
    agent_code: str,
    session: Session = Depends(get_session),
) -> list[TrackingSnapshotResponse]:
    """
    특定 에이전트에 대해, 각 카메라별 최신 1개 트래킹 스냅샷을 모두 반환.
    bbox 는 이미 0~1로 정규화된 xywh.
    """
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

    return [
        _tracking_snapshot_to_response(agent.code, r)
        for r in latest_by_camera.values()
    ]
