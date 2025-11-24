import os
import json
from pathlib import Path
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from collections import deque
from collections.abc import Generator
from typing import Any
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import requests
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlmodel import Field, Session, SQLModel, create_engine, select

# =========================================================
# .env 설정
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _resolve_path(raw: str | None, *, required: bool = False) -> str | None:
    """
    .env 에서 읽은 경로 문자열을 안전하게 절대 경로로 변환한다.
    """
    if not raw:
        if required:
            raise RuntimeError("필수 경로 환경변수가 비어 있습니다.")
        return None

    expanded = os.path.expandvars(os.path.expanduser(raw))
    p = Path(expanded)

    if not p.is_absolute():
        p = (BASE_DIR / p).resolve()

    return str(p)


DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL 환경 변수가 설정되어 있지 않습니다.")

# 두 개의 PT 경로 (필수)
DET_CKPT_PATH_RAW = os.getenv("DET_CKPT_PATH")
CLS_CKPT_PATH_RAW = os.getenv("CLS_CKPT_PATH")

DET_CKPT_PATH = _resolve_path(DET_CKPT_PATH_RAW, required=True)
CLS_CKPT_PATH = _resolve_path(CLS_CKPT_PATH_RAW, required=True)

DET_META_PATH_RAW = os.getenv("DET_META_PATH")   # 없으면 ckpt["meta"] 사용
CLS_META_PATH_RAW = os.getenv("CLS_META_PATH")   # 없으면 ckpt["meta"] 사용

DET_META_PATH = _resolve_path(DET_META_PATH_RAW, required=False)
CLS_META_PATH = _resolve_path(CLS_META_PATH_RAW, required=False)

# 이상 탐지용 voting / threshold 파라미터
EMA_ALPHA = float(os.getenv("EMA_ALPHA", "0.3"))
DET_START_THR = float(os.getenv("DET_START_THR", "0.80"))
DET_END_THR = float(os.getenv("DET_END_THR", "0.55"))
DET_VOTE_WIN = int(os.getenv("DET_VOTE_WIN", "5"))
DET_MIN_START_VOTES = int(os.getenv("DET_MIN_START_VOTES", "4"))
DET_MIN_END_VOTES = int(os.getenv("DET_MIN_END_VOTES", "4"))
DET_MIN_EVENT_SEC = float(os.getenv("DET_MIN_EVENT_SEC", "2.0"))
DET_COOLDOWN_SEC = float(os.getenv("DET_COOLDOWN_SEC", "2.0"))

# Kakao API URL (엔드포인트는 고정이지만, 필요하면 .env 에서 override 가능)
KAKAO_SEND_URL = os.getenv(
    "KAKAO_SEND_URL",
    "https://kapi.kakao.com/v2/api/talk/memo/default/send",
)


# =========================
# DB / SQLModel
# =========================

engine = create_engine(DATABASE_URL, echo=False)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Agent(SQLModel, table=True):
    """
    에이전트 마스터 테이블.

    - 외부에서는 agent_code (예: "agent-main-building-01")로 식별
    - 내부에서는 id(int) PK로 참조
    - kakao_access_token 은 에이전트/점포별 카카오 토큰 (없으면 알람 비활성)
    """
    __tablename__ = "agents"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=utcnow, nullable=False)

    # 사람이 읽기 좋은 에이전트 코드 (unique, non-null)
    code: str = Field(index=True, unique=True)

    # 이 에이전트(점포)의 전용 카카오 토큰 (없으면 알람 안 쏨)
    kakao_access_token: str | None = None

    # 옵션: 메모/설명
    note: str | None = None

    def touch(self) -> None:
        self.updated_at = utcnow()


class InferenceResult(SQLModel, table=True):
    __tablename__ = "inference_results"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=utcnow, nullable=False)

    # 내부 FK (에이전트)
    agent_id: int = Field(foreign_key="agents.id", index=True)

    camera_id: str
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
    created_at: datetime = Field(default_factory=utcnow, nullable=False)

    agent_id: int = Field(foreign_key="agents.id", index=True)

    camera_id: str
    source_id: str | None = None
    event_label: str | None = None

    start_window_index: int | None = None
    end_window_index: int | None = None
    duration_sec: float | None = None
    top_prob: float | None = None

    text_preview: str

    kakao_mode: str  # "disabled" | "real"
    kakao_ok: bool
    kakao_status_code: int | None = None
    kakao_error: str | None = None

    kakao_raw_response_json: str | None = None


class TrackingSnapshot(SQLModel, table=True):
    """
    트래킹 결과 스냅샷 테이블.

    한 row = 특정 시각에 특정 에이전트의 특정 카메라에서
    감지된 모든 객체(사람) 리스트.
    """
    __tablename__ = "tracking_snapshot"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=utcnow, nullable=False)

    agent_id: int = Field(foreign_key="agents.id", index=True)
    camera_id: str = Field(index=True)

    # 에이전트 기준 Unix timestamp (float)
    timestamp: float = Field(index=True)
    # 선택: 디버깅용 frame index
    frame_index: int | None = None

    # UI에 내려줄 objects 리스트 전체를 JSON 문자열로 직렬화해서 저장
    objects_json: str


def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session


def get_or_create_agent(
    session: Session,
    agent_code: str,
) -> Agent:
    """
    agent_code(예: "agent-main-building-01") 기준으로
    Agent row를 조회하거나, 없으면 생성한다.
    """
    stmt = select(Agent).where(Agent.code == agent_code)
    agent = session.exec(stmt).first()
    if agent is not None:
        agent.touch()
        session.add(agent)
        session.commit()
        session.refresh(agent)
        return agent

    agent = Agent(code=agent_code)
    session.add(agent)
    session.commit()
    session.refresh(agent)
    return agent


# =========================
# 모델 정의
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LSTMAnom(nn.Module):
    """
    단순 LSTM 기반 이상 탐지/분류 모델.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_out: int,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, num_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        out, _ = self.lstm(x)
        # 마지막 타임스텝만 사용
        last = out[:, -1, :]
        return self.fc(last)


def build_model_from_meta(meta: dict[str, Any]) -> LSTMAnom:
    input_dim = int(meta["input_dim"])
    hidden_dim = int(meta["hidden_dim"])
    num_layers = int(meta["num_layers"])
    num_out = int(meta["num_out"])
    return LSTMAnom(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_out=num_out,
    )


def features_from_buf(buf: list[np.ndarray]) -> np.ndarray:
    """
    keypoints 시퀀스를 입력으로 받아
    모델 입력용 피처 시퀀스 (T, D)를 만든다.
    """
    arr = np.asarray(buf, dtype=np.float32)  # (T,33,4)
    flat = arr.reshape(arr.shape[0], -1)     # (T, 33*4)
    diff = np.diff(flat, axis=0, prepend=flat[:1])
    feat = np.concatenate([flat, diff], axis=-1)
    return feat


def ema_filter(arr: np.ndarray, alpha: float) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.float32)
    acc = 0.0
    for i, v in enumerate(arr):
        acc = alpha * v + (1 - alpha) * acc
        out[i] = acc
    return out


def nan_forward_fill(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr2 = arr.reshape(-1, 1)
    else:
        arr2 = arr
    T, D = arr2.shape
    out = arr2.copy()
    has = np.zeros(D, dtype=bool)
    last = np.zeros(D, dtype=np.float32)
    for t in range(T):
        nz = ~np.isnan(out[t])
        last[nz] = out[t, nz]
        has |= nz
        miss = np.isnan(out[t]) & has
        out[t, miss] = last[miss]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class LSTMBundle:
    model: LSTMAnom
    norm_mean: np.ndarray
    norm_std: np.ndarray
    events: list[str] | None = None  # 분류 모델만 사용


_det_bundle: LSTMBundle | None = None
_cls_bundle: LSTMBundle | None = None


def _load_meta_from_ckpt_or_json(
    ck: dict[str, Any],
    meta_path: str | None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    if "meta" in ck and isinstance(ck["meta"], dict):
        meta = dict(ck["meta"])
    if meta_path:
        p = Path(meta_path)
        if p.is_file():
            with p.open(encoding="utf-8") as f:
                meta = json.load(f)
    return meta


def load_lstm_bundle(
    ckpt_path: str,
    meta_path: str | None = None,
) -> LSTMBundle:
    ck = torch.load(ckpt_path, map_location="cpu")
    meta = _load_meta_from_ckpt_or_json(ck, meta_path)

    model = build_model_from_meta(meta)
    model.load_state_dict(ck["model"])
    model.to(DEVICE)
    model.eval()

    norm_mean = np.asarray(meta["norm_mean"], dtype=np.float32)
    norm_std = np.asarray(meta["norm_std"], dtype=np.float32)

    events = None
    if "events" in meta and isinstance(meta["events"], list):
        events = list(map(str, meta["events"]))

    return LSTMBundle(
        model=model,
        norm_mean=norm_mean,
        norm_std=norm_std,
        events=events,
    )


def ensure_models_loaded() -> None:
    global _det_bundle, _cls_bundle
    if _det_bundle is None:
        _det_bundle = load_lstm_bundle(DET_CKPT_PATH, DET_META_PATH)
    if _cls_bundle is None:
        _cls_bundle = load_lstm_bundle(CLS_CKPT_PATH, CLS_META_PATH)


# =========================
# Kakao 알림
# =========================

def send_kakao_alarm(text: str, access_token: str | None) -> dict[str, Any]:
    """
    주어진 access_token으로 Kakao 메시지를 전송한다.
    access_token 이 없으면 'disabled' 로 동작.
    """
    if not access_token:
        return {
            "ok": False,
            "mode": "disabled",
            "status_code": None,
            "error": "Kakao disabled (no access token)",
            "response": None,
        }

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/x-www-form-urlencoded;charset=utf-8",
    }
    payload = {
        "object_type": "text",
        "text": text,
        "link": {"web_url": "https://example.com"},
    }
    data = {"template_object": json.dumps(payload, ensure_ascii=False)}

    try:
        res = requests.post(KAKAO_SEND_URL, headers=headers, data=data, timeout=5)
    except Exception as e:
        return {
            "ok": False,
            "mode": "real",
            "status_code": None,
            "error": repr(e),
            "response": None,
        }

    ok = (200 <= res.status_code < 300)
    try:
        body = res.json()
    except Exception:
        body = res.text

    return {
        "ok": ok,
        "mode": "real",
        "status_code": res.status_code,
        "error": None if ok else str(body),
        "response": body,
    }


# =========================
# FastAPI app & lifespan
# =========================

@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_models_loaded()
    SQLModel.metadata.create_all(engine)
    yield


app = FastAPI(
    title="Behavior Inference Server (agent table + per-agent Kakao token)",
    lifespan=lifespan,
)


# =========================================================
# 요청/응답 스키마
# =========================================================

class PoseWindowRequest(BaseModel):
    # 에이전트가 알고 있는 코드 (예: "agent-main-building-01")
    agent_code: str
    camera_id: str
    source_id: str | None = None
    window_index: int
    window_start_ts: float | None = None
    window_end_ts: float | None = None
    # (T,33,4): [[ [x,y,z,vis], ...33 ], ...T ]
    keypoints: list[list[list[float]]]


class BehaviorResultResponse(BaseModel):
    # 클라이언트에게도 code 기준으로 돌려준다.
    agent_code: str
    camera_id: str
    source_id: str | None = None
    window_index: int
    window_start_ts: float | None = None
    window_end_ts: float | None = None

    is_anomaly: bool
    det_prob: float
    top_label: str | None
    top_prob: float
    prob: dict[str, float]


class TrackingBBox(BaseModel):
    """
    0~1 정규화 좌표계 기준 바운딩 박스.
    (x, y)는 좌상단, (w, h)는 폭/높이.
    """
    x: float
    y: float
    w: float
    h: float


class TrackingObject(BaseModel):
    """
    한 사람(또는 객체)에 대한 트래킹 결과 한 개.
    """
    global_id: str
    local_track_id: int
    label: str
    confidence: float
    bbox: TrackingBBox


class TrackingSnapshotResponse(BaseModel):
    """
    특정 에이전트/카메라의 한 시점 기준 트래킹 스냅샷 응답.
    """
    agent_code: str
    camera_id: str
    timestamp: float
    frame_index: int | None = None
    objects: list[TrackingObject]


class AgentKakaoRegisterRequest(BaseModel):
    """
    에이전트가 처음 실행될 때, 자신의 Kakao 토큰을 등록할 때 사용하는 요청 본문.
    """
    agent_code: str
    kakao_access_token: str
    note: str | None = None


# ============================
# 카메라별 상태 추적용 dataclass
# ============================

@dataclass
class CameraState:
    ema_det: float = 0.0
    last_ts: float = 0.0
    votes: deque[bool | None] = field(
        default_factory=lambda: deque(maxlen=DET_VOTE_WIN)
    )
    in_anomaly: bool = False
    current_label: str | None = None
    start_ts: float = 0.0
    start_window_index: int | None = None
    last_end_ts: float = 0.0

    def update(self, det: float, ts: float) -> float:
        # EMA 업데이트
        if self.last_ts == 0.0:
            self.ema_det = det
        else:
            self.ema_det = EMA_ALPHA * det + (1 - EMA_ALPHA) * self.ema_det
        self.last_ts = ts

        # voting 버퍼 업데이트
        if det >= DET_START_THR:
            vote: bool | None = True
        elif det <= DET_END_THR:
            vote = False
        else:
            vote = None
        self.votes.append(vote)

        return self.ema_det


def build_kakao_text(
    agent_code: str,
    camera_id: str,
    source_id: str | None,
    event_label: str,
    duration_sec: float,
    top_prob: float,
) -> str:
    src = source_id or "unknown"
    return (
        f"[이상행동 감지]\n"
        f"- 에이전트: {agent_code}\n"
        f"- 카메라: {camera_id}\n"
        f"- 소스: {src}\n"
        f"- 이벤트: {event_label}\n"
        f"- 지속시간: {duration_sec:.1f}초\n"
        f"- 신뢰도: {top_prob:.2f}\n"
    )


# =========================================================
# 에이전트용 Kakao 토큰 등록 엔드포인트
# =========================================================

@app.post("/agent/register_kakao")
def register_agent_kakao(
    payload: AgentKakaoRegisterRequest,
    session: Session = Depends(get_session),
):
    """
    에이전트가 처음 실행될 때 자신의 Kakao 토큰을 등록/갱신하는 엔드포인트.

    - agent_code 가 없으면 새 Agent row 생성
    - 이미 있으면 kakao_access_token / note 갱신
    """
    agent = get_or_create_agent(session, payload.agent_code)
    agent.kakao_access_token = payload.kakao_access_token
    if payload.note is not None:
        agent.note = payload.note
    agent.touch()
    session.add(agent)
    session.commit()
    session.refresh(agent)

    return {
        "agent_code": agent.code,
        "has_token": bool(agent.kakao_access_token),
        "note": agent.note,
        "updated_at": agent.updated_at.isoformat(),
    }


# =========================================================
# 메인 엔드포인트
# =========================================================

@app.post("/behavior/analyze_pose", response_model=BehaviorResultResponse)
def analyze_pose(
    payload: PoseWindowRequest,
    session: Session = Depends(get_session),
):
    """
    포즈 윈도우 기반 이상행동 분석.

    - 외부에서는 agent_code 로 요청
    - 내부에서는 agent_id(FK)로 저장
    - 알림 보낼 때는 해당 agent의 kakao_access_token 사용
    """
    kpt_seq = np.asarray(payload.keypoints, dtype=np.float32)
    if kpt_seq.ndim != 3 or kpt_seq.shape[1:] != (33, 4):
        return JSONResponse(
            status_code=400,
            content={"detail": f"keypoints shape must be (T,33,4), got {kpt_seq.shape}"},
        )

    agent_code = payload.agent_code
    camera_id = payload.camera_id
    source_id = payload.source_id
    window_index = payload.window_index
    ts = payload.window_end_ts or payload.window_start_ts or float(window_index)

    ensure_models_loaded()

    # 에이전트 조회/생성
    agent = get_or_create_agent(session, agent_code)
    agent_id = agent.id
    assert agent_id is not None

    # (T,33,4) → (T,D) 피처 시퀀스
    feat = features_from_buf(list(kpt_seq))
    feat = np.clip(
        (feat - _det_bundle.norm_mean) / (_det_bundle.norm_std + 1e-6),
        -6,
        6,
    )
    x = torch.from_numpy(feat).unsqueeze(0).float().to(DEVICE)

    # 이상 탐지 모델
    with torch.no_grad():
        logits_det = _det_bundle.model(x)
        probs_det = torch.sigmoid(logits_det).detach().cpu().numpy()[0]

    p_anom = float(probs_det[0])

    # 이상 분류 모델 (softmax)
    feat2 = features_from_buf(list(kpt_seq))
    feat2 = np.clip(
        (feat2 - _cls_bundle.norm_mean) / (_cls_bundle.norm_std + 1e-6),
        -6,
        6,
    )
    x2 = torch.from_numpy(feat2).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        logits_cls = _cls_bundle.model(x2)
        probs_cls = torch.softmax(logits_cls, dim=-1).detach().cpu().numpy()[0]

    events = _cls_bundle.events or []
    prob_dict: dict[str, float] = {}
    top_label: str | None = None
    top_prob: float = 0.0
    if len(events) == probs_cls.shape[0]:
        for label, p in zip(events, probs_cls):
            prob_dict[label] = float(p)
        top_idx = int(np.argmax(probs_cls))
        top_label = events[top_idx]
        top_prob = float(probs_cls[top_idx])

    # ============================
    # (agent_id, camera_id) 단위 상태 추적
    # ============================
    if not hasattr(analyze_pose, "_states"):
        analyze_pose._states = {}  # type: ignore[attr-defined]
    states: dict[str, CameraState] = analyze_pose._states  # type: ignore[attr-defined]

    state_key = f"{agent_id}:{camera_id}"
    if state_key not in states:
        states[state_key] = CameraState()

    state = states[state_key]
    det_now = p_anom
    ema_det = state.update(det_now, ts)

    is_anomaly_now = ema_det >= DET_START_THR

    # DB 저장
    infer = InferenceResult(
        agent_id=agent_id,
        camera_id=camera_id,
        source_id=source_id,
        window_index=window_index,
        window_start_ts=payload.window_start_ts,
        window_end_ts=payload.window_end_ts,
        is_anomaly=is_anomaly_now,
        stage1_normal=float(1.0 - p_anom),
        stage1_anomaly=float(p_anom),
        stage2_labels_json=json.dumps(events, ensure_ascii=False),
        stage2_probs_json=json.dumps(probs_cls.tolist(), ensure_ascii=False),
        stage2_top_label=top_label,
        stage2_top_prob=top_prob,
    )
    session.add(infer)
    session.commit()

    # START / END 판정 + Kakao 로그
    if not state.in_anomaly:
        gap_ok = (ts - state.last_end_ts) >= DET_COOLDOWN_SEC
        if gap_ok and (state.votes.count(True) >= DET_MIN_START_VOTES):
            state.in_anomaly = True
            state.current_label = top_label
            state.start_ts = ts
            state.start_window_index = window_index
            state.votes.clear()
    else:
        end_votes = state.votes.count(None)
        long_enough = (ts - (state.start_ts or ts)) >= DET_MIN_EVENT_SEC
        end_ok = (
            ema_det <= DET_END_THR
            and end_votes >= DET_MIN_END_VOTES
            and long_enough
        )
        if end_ok:
            dur = ts - (state.start_ts or ts)
            label_for_log = state.current_label or top_label or "unknown"
            text = build_kakao_text(
                agent_code=agent_code,
                camera_id=camera_id,
                source_id=source_id,
                event_label=label_for_log,
                duration_sec=dur,
                top_prob=top_prob,
            )
            # 에이전트별 Kakao 토큰 사용 (없으면 disabled)
            kakao_res = send_kakao_alarm(text, agent.kakao_access_token)

            log = KakaoAlarmLog(
                agent_id=agent_id,
                camera_id=camera_id,
                source_id=source_id,
                event_label=label_for_log,
                start_window_index=state.start_window_index,
                end_window_index=window_index,
                duration_sec=dur,
                top_prob=top_prob,
                text_preview=text[:200],
                kakao_mode=kakao_res.get("mode", "unknown"),
                kakao_ok=bool(kakao_res.get("ok", False)),
                kakao_status_code=kakao_res.get("status_code"),
                kakao_error=kakao_res.get("error"),
                kakao_raw_response_json=json.dumps(
                    kakao_res.get("response"),
                    ensure_ascii=False,
                    default=str,
                ),
            )
            session.add(log)
            session.commit()

            state.in_anomaly = False
            state.current_label = None
            state.start_ts = 0.0
            state.start_window_index = None
            state.last_end_ts = ts
            state.votes.clear()

    return BehaviorResultResponse(
        agent_code=agent_code,
        camera_id=camera_id,
        source_id=source_id,
        window_index=window_index,
        window_start_ts=payload.window_start_ts,
        window_end_ts=payload.window_end_ts,
        is_anomaly=is_anomaly_now,
        det_prob=ema_det,
        top_label=top_label,
        top_prob=top_prob,
        prob=prob_dict,
    )


# =========================================================
# kakao test
# =========================================================

@app.get("/kakao/test")
def kakao_test(
    agent_code: str = Query(..., description="테스트할 에이전트 코드"),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """
    Kakao 연동 테스트.

    - 반드시 agent_code 를 지정해야 한다.
    - 해당 에이전트의 kakao_access_token 으로 메시지를 보낸다.
    """
    agent = get_or_create_agent(session, agent_code)
    token = agent.kakao_access_token

    text = "[테스트] 이상행동 알림 시스템 카카오 연동 테스트 메시지입니다."
    res = send_kakao_alarm(text, token)

    return {
        "agent_code": agent_code,
        "has_token": bool(token),
        "send_ok": bool(res.get("ok", False)),
        "mode": res.get("mode"),
        "status_code": res.get("status_code"),
        "error": res.get("error"),
        "raw_response": res.get("response"),
    }


# =========================================================
# Behavior / Tracking 최신 결과 조회
# =========================================================

@app.get(
    "/behavior/latest/{camera_id}",
    response_model=BehaviorResultResponse | None,
)
def get_latest_behavior(
    camera_id: str,
    agent_code: str = Query(..., description="에이전트 코드"),
    session: Session = Depends(get_session),
):
    """
    주어진 agent_code, camera_id 에 대해 DB에 저장된
    가장 최신 InferenceResult 한 건을 반환한다.
    없으면 null.
    """
    agent = get_or_create_agent(session, agent_code)
    agent_id = agent.id
    assert agent_id is not None

    stmt = (
        select(InferenceResult)
        .where(
            (InferenceResult.agent_id == agent_id)
            & (InferenceResult.camera_id == camera_id)
        )
        .order_by(InferenceResult.created_at.desc())
        .limit(1)
    )
    row = session.exec(stmt).first()
    if row is None:
        return None

    try:
        labels = json.loads(row.stage2_labels_json or "[]")
    except Exception:
        labels = []

    try:
        probs_raw = json.loads(row.stage2_probs_json or "[]")
        probs_list = [float(p) for p in probs_raw]
    except Exception:
        probs_list = []

    prob_dict: dict[str, float] = {
        label: float(p) for label, p in zip(labels, probs_list)
    }

    if row.stage2_top_label is not None:
        top_label = row.stage2_top_label
        top_prob = float(row.stage2_top_prob or 0.0)
    elif labels and probs_list:
        top_idx = int(np.argmax(probs_list))
        top_label = labels[top_idx]
        top_prob = float(probs_list[top_idx])
    else:
        top_label = None
        top_prob = 0.0

    det_prob = float(row.stage1_anomaly)

    return BehaviorResultResponse(
        agent_code=agent_code,
        camera_id=row.camera_id,
        source_id=row.source_id,
        window_index=row.window_index,
        window_start_ts=row.window_start_ts,
        window_end_ts=row.window_end_ts,
        is_anomaly=row.is_anomaly,
        det_prob=det_prob,
        top_label=top_label,
        top_prob=top_prob,
        prob=prob_dict,
    )


@app.get(
    "/tracking/latest/{camera_id}",
    response_model=TrackingSnapshotResponse | None,
)
def get_latest_tracking(
    camera_id: str,
    agent_code: str = Query(..., description="에이전트 코드"),
    session: Session = Depends(get_session),
):
    """
    주어진 agent_code, camera_id 에 대해 DB에 저장된
    가장 최신 TrackingSnapshot 한 건을 반환한다.
    없으면 null.
    """
    agent = get_or_create_agent(session, agent_code)
    agent_id = agent.id
    assert agent_id is not None

    stmt = (
        select(TrackingSnapshot)
        .where(
            (TrackingSnapshot.agent_id == agent_id)
            & (TrackingSnapshot.camera_id == camera_id)
        )
        .order_by(TrackingSnapshot.timestamp.desc())
        .limit(1)
    )
    row = session.exec(stmt).first()
    if row is None:
        return None

    try:
        raw_list = json.loads(row.objects_json or "[]")
    except Exception:
        raw_list = []

    objects: list[TrackingObject] = []
    for obj in raw_list:
        bbox_raw = obj.get("bbox") or {}
        try:
            bbox = TrackingBBox(
                x=float(bbox_raw.get("x", 0.0)),
                y=float(bbox_raw.get("y", 0.0)),
                w=float(bbox_raw.get("w", 0.0)),
                h=float(bbox_raw.get("h", 0.0)),
            )
        except Exception:
            continue

        try:
            tracking_obj = TrackingObject(
                global_id=str(obj.get("global_id", "")),
                local_track_id=int(obj.get("local_track_id", 0)),
                label=str(obj.get("label", "person")),
                confidence=float(obj.get("confidence", 0.0)),
                bbox=bbox,
            )
        except Exception:
            continue

        objects.append(tracking_obj)

    return TrackingSnapshotResponse(
        agent_code=agent_code,
        camera_id=row.camera_id,
        timestamp=row.timestamp,
        frame_index=row.frame_index,
        objects=objects,
    )
