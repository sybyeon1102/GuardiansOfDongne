# server/behavior_inference_server.py

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
from dotenv import load_dotenv

from modeling.inference.pose_features import features_from_pose_seq
from modeling.inference.behavior_det import BehaviorDetector, BehaviorDetectorConfig
from modeling.inference.behavior_cls import BehaviorClassifier, BehaviorClassifierConfig


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
load_dotenv(BASE_DIR / ".env")


def _env_str(name: str, default: str | None = None) -> str:
    v = os.getenv(name)
    if v is None:
        if default is None:
            raise RuntimeError(f"missing environment variable: {name}")
        return default
    return v


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        logger.warning("ENV %s=%r is not a valid float, using default %s", name, v, default)
        return default


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError:
        logger.warning("ENV %s=%r is not a valid int, using default %s", name, v, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    v2 = v.strip().lower()
    if v2 in ("1", "true", "yes", "on"):
        return True
    if v2 in ("0", "false", "no", "off"):
        return False
    logger.warning("ENV %s=%r is not a valid bool, using default %s", name, v, default)
    return default


def _resolve_env_path(name: str) -> Path:
    raw = _env_str(name)
    expanded = os.path.expanduser(raw)
    p = Path(expanded)
    if not p.is_absolute():
        p = (BASE_DIR / p).resolve()
    return p


# ---------------------------------------------------------
# DB 설정
# ---------------------------------------------------------

DATABASE_URL = _env_str("DATABASE_URL")

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
    code: str = Field(index=True, unique=True)
    kakao_access_token: str | None = None
    note: str | None = None
    created_at: float = Field(default_factory=lambda: time.time())
    updated_at: float = Field(default_factory=lambda: time.time())


class InferenceResult(SQLModel, table=True):
    __tablename__ = "inference_results"

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

    event_label: str | None = None
    start_window_index: int | None = None
    end_window_index: int | None = None
    duration_sec: float | None = None
    top_prob: float | None = None

    text_preview: str | None = None

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
    window_start_ts: float | None = None
    window_end_ts: float | None = None

    # (T,33,4) [x,y,z,visibility]
    keypoints: list[list[list[float]]]


class BehaviorResultResponse(BaseModel):
    agent_code: str
    camera_id: str
    source_id: str | None

    window_index: int
    window_start_ts: float | None
    window_end_ts: float | None

    is_anomaly: bool
    det_prob: float
    top_label: str | None
    top_prob: float
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
# FastAPI 앱 / CORS
# ---------------------------------------------------------

app = FastAPI(title="GoD Behavior Inference Server", version="0.2.0")

_allowed_origins_raw = os.getenv("ALLOWED_ORIGINS")  # 예: "http://localhost:3000,http://127.0.0.1:3000"
if _allowed_origins_raw:
    ALLOWED_ORIGINS = [o.strip() for o in _allowed_origins_raw.split(",") if o.strip()]
else:
    # 개발 편의를 위해 기본은 모두 허용, 필요 시 .env 에서 제한
    ALLOWED_ORIGINS = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("CORS enabled for origins: %r", ALLOWED_ORIGINS)

# ---------------------------------------------------------
# DB 초기화
# ---------------------------------------------------------


@app.on_event("startup")
def on_startup() -> None:
    SQLModel.metadata.create_all(engine)
    load_models()
    logger.info("Startup complete: DB tables ensured, models loaded.")


# ---------------------------------------------------------
# 모델 로딩 (Prep_v3 기반)
# ---------------------------------------------------------

DET_CKPT_PATH = _resolve_env_path("DET_CKPT_PATH")
CLS_CKPT_PATH = _resolve_env_path("CLS_CKPT_PATH")

_detector: BehaviorDetector | None = None
_classifier: BehaviorClassifier | None = None


def get_detector() -> BehaviorDetector:
    if _detector is None:
        raise RuntimeError("BehaviorDetector is not initialized")
    return _detector


def get_classifier() -> BehaviorClassifier:
    if _classifier is None:
        raise RuntimeError("BehaviorClassifier is not initialized")
    return _classifier


def load_models() -> None:
    global _detector, _classifier

    det_cfg = BehaviorDetectorConfig(ckpt_path=DET_CKPT_PATH)
    cls_cfg = BehaviorClassifierConfig(ckpt_path=CLS_CKPT_PATH)

    _detector = BehaviorDetector(det_cfg)
    _classifier = BehaviorClassifier(cls_cfg)

    logger.info(
        "Loaded models:\n"
        "  DET_CKPT_PATH=%s (win=%s, feat_dim=%s)\n"
        "  CLS_CKPT_PATH=%s (win=%s, classes=%s)",
        DET_CKPT_PATH,
        _detector.win,
        _detector.feat_dim,
        CLS_CKPT_PATH,
        _classifier.win,
        _classifier.events,
    )


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
    kpt_seq: (T,33,4) -> (events, probs)
    """
    feat = features_from_pose_seq(kpt_seq)  # (T,169)
    clsf = get_classifier()
    prob_map = clsf(feat)  # dict[label->prob]

    events = list(clsf.events)
    probs = np.array([prob_map[label] for label in events], dtype=np.float32)
    return events, probs


# ---------------------------------------------------------
# 이상행동 판정 파라미터 (EMA / voting / cooldown)
# ---------------------------------------------------------

EMA_ALPHA = _env_float("EMA_ALPHA", 0.3)
DET_START_THR = _env_float("DET_START_THR", 0.8)
DET_END_THR = _env_float("DET_END_THR", 0.55)

DET_VOTE_WIN = _env_int("DET_VOTE_WIN", 5)
DET_MIN_START_VOTES = _env_int("DET_MIN_START_VOTES", 4)
DET_MIN_END_VOTES = _env_int("DET_MIN_END_VOTES", 4)

DET_MIN_EVENT_SEC = _env_float("DET_MIN_EVENT_SEC", 2.0)
DET_COOLDOWN_SEC = _env_float("DET_COOLDOWN_SEC", 2.0)

logger.info(
    "Detection params: EMA_ALPHA=%.3f, START_THR=%.3f, END_THR=%.3f, "
    "VOTE_WIN=%d, MIN_START_VOTES=%d, MIN_END_VOTES=%d, "
    "MIN_EVENT_SEC=%.1f, COOLDOWN_SEC=%.1f",
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
# 상태 머신 (에이전트+카메라 단위)
# ---------------------------------------------------------


@dataclass
class BehaviorState:
    ema_prob: float | None = None
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
            ema_prob=None,
            in_event=False,
            event_label=None,
            event_top_prob=0.0,
            event_start_ts=None,
            event_start_window_index=None,
            votes=deque(maxlen=DET_VOTE_WIN),
            last_update_ts=time.time(),
        )
        _behavior_states[key] = st
    return st


# ---------------------------------------------------------
# Kakao 연동
# ---------------------------------------------------------

KAKAO_SEND_URL = os.getenv("KAKAO_SEND_URL", "https://kapi.kakao.com/v2/api/talk/memo/default/send")


def _send_kakao_message(access_token: str, text: str) -> tuple[bool, int | None, str | None, str | None]:
    """
    실제 카카오 API 호출.
    """
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/x-www-form-urlencoded;charset=utf-8",
    }
    payload = {
        "template_object": json.dumps(
            {
                "object_type": "text",
                "text": text,
                "link": {"web_url": "https://example.com", "mobile_web_url": "https://example.com"},
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
# 에이전트 헬퍼
# ---------------------------------------------------------


def get_or_create_agent_by_code(session: Session, agent_code: str) -> Agent:
    stmt = select(Agent).where(Agent.code == agent_code)
    agent = session.exec(stmt).first()
    if agent is None:
        agent = Agent(code=agent_code)
        session.add(agent)
        session.commit()
        session.refresh(agent)
    return agent


# ---------------------------------------------------------
# 엔드포인트: 헬스 체크
# ---------------------------------------------------------


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "time": time.time()}


# ---------------------------------------------------------
# 엔드포인트: 에이전트 / 카카오 토큰 등록
# ---------------------------------------------------------


class AgentRegisterRequest(BaseModel):
    agent_code: str
    kakao_access_token: str
    note: str | None = None


@app.post("/agent/register_kakao")
def register_kakao(
    payload: AgentRegisterRequest,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    agent = get_or_create_agent_by_code(session, payload.agent_code)
    agent.kakao_access_token = payload.kakao_access_token
    agent.note = payload.note
    agent.updated_at = time.time()
    session.add(agent)
    session.commit()
    logger.info("Registered Kakao token for agent_code=%s (id=%s)", agent.code, agent.id)
    return {"ok": True, "agent_id": agent.id}


@app.get("/kakao/test")
def kakao_test(
    agent_code: str,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    stmt = select(Agent).where(Agent.code == agent_code)
    agent = session.exec(stmt).first()
    if agent is None:
        raise HTTPException(status_code=404, detail="agent not found")
    if not agent.kakao_access_token:
        raise HTTPException(status_code=400, detail="agent has no kakao_access_token")

    text = f"[테스트] 에이전트 {agent.code} Kakao 테스트 메시지입니다."
    ok, status_code, err, raw = _send_kakao_message(agent.kakao_access_token, text)

    log = KakaoAlarmLog(
        agent_id=agent.id,
        camera_id="__kakao_test__",
        source_id=None,
        event_label="__kakao_test__",
        start_window_index=None,
        end_window_index=None,
        duration_sec=None,
        top_prob=None,
        text_preview=text,
        kakao_mode="real",
        kakao_ok=ok,
        kakao_status_code=status_code,
        kakao_error=err,
        kakao_raw_response_json=raw,
    )
    session.add(log)
    session.commit()

    return {"ok": ok, "status_code": status_code, "error": err}


# ---------------------------------------------------------
# 엔드포인트: 포즈 기반 이상행동 분석
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

    # softmax 결과를 dict로
    prob_map = {label: float(p) for label, p in zip(events, probs)}
    top_label: str | None
    top_prob: float
    if prob_map:
        top_label, top_prob = max(prob_map.items(), key=lambda kv: kv[1])
    else:
        top_label, top_prob = None, 0.0

    # 4) EMA 업데이트
    prev_ema = state.ema_prob
    if prev_ema is None:
        ema = p_anom
    else:
        ema = EMA_ALPHA * p_anom + (1.0 - EMA_ALPHA) * prev_ema
    state.ema_prob = ema
    state.last_update_ts = time.time()

    # 5) voting
    vote: bool | None
    if ema >= DET_START_THR:
        vote = True
    elif ema <= DET_END_THR:
        vote = False
    else:
        vote = None
    state.votes.append(vote)

    # 6) START / END 판정
    is_anomaly_window = ema >= DET_START_THR
    now_ts = time.time()
    win_start_ts = payload.window_start_ts if payload.window_start_ts is not None else now_ts
    win_end_ts = payload.window_end_ts if payload.window_end_ts is not None else now_ts

    def _count_votes(vs: deque[Any], target: bool | None) -> int:
        return sum(1 for v in vs if v is target)

    # 이벤트 시작
    if not state.in_event:
        start_votes = _count_votes(state.votes, True)
        if ema >= DET_START_THR and start_votes >= DET_MIN_START_VOTES:
            state.in_event = True
            state.event_label = top_label
            state.event_top_prob = top_prob
            state.event_start_ts = win_start_ts
            state.event_start_window_index = payload.window_index
            logger.info(
                "[EVENT START] agent=%s cam=%s win=%s label=%s ema=%.3f p=%.3f",
                agent.code,
                payload.camera_id,
                payload.window_index,
                top_label,
                ema,
                p_anom,
            )
    else:
        # 이벤트 종료 후보
        end_votes = _count_votes(state.votes, False) + _count_votes(state.votes, None)
        duration_sec = (win_end_ts - (state.event_start_ts or win_start_ts)) if state.event_start_ts else 0.0

        if ema <= DET_END_THR and end_votes >= DET_MIN_END_VOTES:
            # 쿨다운/최소 길이 체크
            if duration_sec >= DET_MIN_EVENT_SEC:
                # END 확정 → 카카오/로그
                _handle_event_end(
                    session=session,
                    agent=agent,
                    payload=payload,
                    state=state,
                    duration_sec=duration_sec,
                )
            else:
                logger.info(
                    "[EVENT SHORT] agent=%s cam=%s win=%s duration=%.2fs (< %.2fs)",
                    agent.code,
                    payload.camera_id,
                    payload.window_index,
                    duration_sec,
                    DET_MIN_EVENT_SEC,
                )

            # 상태 리셋 + 쿨다운
            state.in_event = False
            state.event_label = None
            state.event_top_prob = 0.0
            state.event_start_ts = None
            state.event_start_window_index = None
            state.votes.clear()
            state.last_update_ts = now_ts

    # 7) inference_results 저장
    stage1_normal = float(1.0 - p_anom)
    stage1_anomaly = float(p_anom)

    res = InferenceResult(
        agent_id=agent.id,
        camera_id=payload.camera_id,
        source_id=payload.source_id,
        window_index=payload.window_index,
        window_start_ts=payload.window_start_ts,
        window_end_ts=payload.window_end_ts,
        is_anomaly=is_anomaly_window,
        stage1_normal=stage1_normal,
        stage1_anomaly=stage1_anomaly,
        stage2_labels_json=json.dumps(events, ensure_ascii=False),
        stage2_probs_json=json.dumps(probs.tolist(), ensure_ascii=False),
        stage2_top_label=top_label,
        stage2_top_prob=top_prob,
    )
    session.add(res)
    session.commit()

    return BehaviorResultResponse(
        agent_code=agent.code,
        camera_id=payload.camera_id,
        source_id=payload.source_id,
        window_index=payload.window_index,
        window_start_ts=payload.window_start_ts,
        window_end_ts=payload.window_end_ts,
        is_anomaly=is_anomaly_window,
        det_prob=float(ema),
        top_label=top_label,
        top_prob=top_prob,
        prob=prob_map,
    )


def _handle_event_end(
    session: Session,
    agent: Agent,
    payload: PoseWindowRequest,
    state: BehaviorState,
    duration_sec: float,
) -> None:
    """
    이벤트 END 시점에 Kakao + kakao_alarm_log 기록.
    """
    label = state.event_label or "__unknown__"
    text = (
        f"[이상행동 감지]\n"
        f"- 에이전트: {agent.code}\n"
        f"- 카메라: {payload.camera_id}\n"
        f"- 이벤트: {label}\n"
        f"- 지속시간: {duration_sec:.1f}초\n"
        f"- 윈도우: {state.event_start_window_index} ~ {payload.window_index}\n"
    )

    kakao_mode = "disabled"
    kakao_ok = False
    kakao_status_code: int | None = None
    kakao_error: str | None = None
    kakao_raw_json: str | None = None

    if agent.kakao_access_token:
        kakao_mode = "real"
        kakao_ok, kakao_status_code, kakao_error, kakao_raw_json = _send_kakao_message(
            agent.kakao_access_token,
            text,
        )
    else:
        kakao_mode = "disabled"
        logger.info(
            "Kakao disabled for agent_code=%s (no kakao_access_token), event=%s",
            agent.code,
            label,
        )

    log = KakaoAlarmLog(
        agent_id=agent.id,
        camera_id=payload.camera_id,
        source_id=payload.source_id,
        event_label=label,
        start_window_index=state.event_start_window_index,
        end_window_index=payload.window_index,
        duration_sec=duration_sec,
        top_prob=state.event_top_prob,
        text_preview=text[:2000],
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
        label,
        duration_sec,
        kakao_mode,
        kakao_ok,
    )


# ---------------------------------------------------------
# 엔드포인트: Behavior 최신 결과 조회 (UI용)
# ---------------------------------------------------------


@app.get("/behavior/latest/{camera_id}", response_model=BehaviorResultResponse | None)
def behavior_latest_for_camera(
    camera_id: str,
    agent_code: str,
    session: Session = Depends(get_session),
) -> BehaviorResultResponse | None:
    agent = get_or_create_agent_by_code(session, agent_code)
    stmt = (
        select(InferenceResult)
        .where(InferenceResult.agent_id == agent.id, InferenceResult.camera_id == camera_id)
        .order_by(InferenceResult.id.desc())
        .limit(1)
    )
    row = session.exec(stmt).first()
    if row is None:
        return None

    events = json.loads(row.stage2_labels_json)
    probs = json.loads(row.stage2_probs_json)
    prob_map = {label: float(p) for label, p in zip(events, probs)}

    return BehaviorResultResponse(
        agent_code=agent.code,
        camera_id=row.camera_id,
        source_id=row.source_id,
        window_index=row.window_index,
        window_start_ts=row.window_start_ts,
        window_end_ts=row.window_end_ts,
        is_anomaly=row.is_anomaly,
        det_prob=row.stage1_anomaly,  # EMA 대신 마지막 anomaly prob 저장 원하면 변경
        top_label=row.stage2_top_label,
        top_prob=row.stage2_top_prob,
        prob=prob_map,
    )


@app.get("/behavior/latest_all", response_model=list[BehaviorResultResponse])
def behavior_latest_all(
    agent_code: str,
    session: Session = Depends(get_session),
) -> list[BehaviorResultResponse]:
    agent = get_or_create_agent_by_code(session, agent_code)

    # camera_id 별 마지막 row
    stmt = (
        select(InferenceResult)
        .where(InferenceResult.agent_id == agent.id)
        .order_by(InferenceResult.camera_id, InferenceResult.id.desc())
    )
    rows = session.exec(stmt).all()

    latest_by_camera: dict[str, InferenceResult] = {}
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
# 트래킹: 설정 / 엔드포인트 (YOLO는 나중에 붙이고, 일단 no-op + DB 저장)
# ---------------------------------------------------------

TRACKING_ENABLED = _env_bool("TRACKING_ENABLED", False)
logger.info("Tracking enabled: %s", TRACKING_ENABLED)


@app.post("/tracking")
async def tracking_endpoint(
    agent_code: str = Form(...),
    camera_id: str = Form(...),
    timestamp: float | None = Form(None),
    frame_index: int | None = Form(None),
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """
    트래킹용 프레임 업로드 엔드포인트.

    현재는 YOLO/DeepSort 를 실제로 돌리지는 않고,
    TRACKING_ENABLED=False 여도 엔드포인트는 정상 응답을 돌려준다.

    - TrackingSnapshot 에는 objects_json="[]" 으로 비워서 저장해 둔다.
      (나중에 YOLO を 붙이면 이 부분을 실제 결과로 대체)
    """
    agent = get_or_create_agent_by_code(session, agent_code)

    # 프레임 바이트는 지금은 사용하지 않고 버린다.
    _ = await file.read()

    ts = timestamp if timestamp is not None else time.time()

    objects_list: list[dict[str, Any]] = []

    if TRACKING_ENABLED:
        # TODO: 나중에 modeling.tracking.yolo_deepsort.YoloDeepSortEngine 을 붙이면,
        #       여기서 JPEG -> BGR -> engine.track(...) 호출 후 결과를 objects_list 로 채운다.
        logger.warning(
            "TRACKING_ENABLED=True 이지만 YOLO 엔진이 아직 구현되지 않았습니다. "
            "objects_json 은 빈 배열로 저장됩니다."
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
        .where(TrackingSnapshot.agent_id == agent.id, TrackingSnapshot.camera_id == camera_id)
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
        .order_by(TrackingSnapshot.camera_id, TrackingSnapshot.id.desc())
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
