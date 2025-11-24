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

    - raw 가 None/빈 문자열:
        * required=True 면 RuntimeError
        * required=False 면 None 반환
    - ~ (홈 디렉터리) 확장
    - 환경변수($HOME 등) 확장
    - 상대 경로면 BASE_DIR(server 디렉터리) 기준 절대 경로로 변환
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
if not DET_CKPT_PATH_RAW or not CLS_CKPT_PATH_RAW:
    raise RuntimeError("DET_CKPT_PATH, CLS_CKPT_PATH 둘 다 .env 에 설정되어야 합니다.")

DET_CKPT_PATH = _resolve_path(DET_CKPT_PATH_RAW, required=True)
CLS_CKPT_PATH = _resolve_path(CLS_CKPT_PATH_RAW, required=True)

# meta.json 경로 (선택)
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
DET_COOLDOWN_SEC = float(os.getenv("DET_COOLDOWN_SEC", "3.0"))

# Kakao API URL (토큰은 Agent 별로 관리)
KAKAO_SEND_URL = os.getenv(
    "KAKAO_SEND_URL",
    "https://kapi.kakao.com/v2/api/talk/memo/default/send",
)

# torch 디바이스
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] using {DEVICE}")

# =========================================================
# DB / SQLModel
# =========================================================

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
        agent.updated_at = utcnow()
        session.add(agent)
        session.commit()
        session.refresh(agent)
        return agent

    agent = Agent(code=agent_code, created_at=utcnow(), updated_at=utcnow())
    session.add(agent)
    session.commit()
    session.refresh(agent)
    return agent


# =========================================================
# LSTM 모델 / feature 전처리 (legacy_2 그대로)
# =========================================================

class AttPool(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.w = nn.Linear(d, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B,T,D)
        a = self.w(h).squeeze(-1)  # (B,T)
        w = torch.softmax(a, dim=1).unsqueeze(-1)
        return (h * w).sum(1)  # (B,D)


class LSTMAnom(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        hidden: int = 128,
        layers: int = 2,
        num_out: int = 1,
        bidir: bool = True,
    ) -> None:
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv1d(feat_dim, feat_dim, 3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            feat_dim,
            hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidir,
            dropout=0.1,
        )
        d = hidden * (2 if bidir else 1)
        self.pool = AttPool(d)
        self.head = nn.Linear(d, num_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,F)
        # conv1d 를 쓰기 위해 (B,F,T) 로 transpose
        h = x.transpose(1, 2)
        h = self.pre(h)
        h = h.transpose(1, 2)  # 다시 (B,T,F)

        h, _ = self.lstm(h)    # (B,T,D)
        h = self.pool(h)       # (B,D)
        return self.head(h)    # (B,num_out)


def _ffill_bfill(arr: np.ndarray) -> np.ndarray:
    T, D = arr.shape
    out = arr.copy()
    last = np.zeros(D, np.float32)
    has = np.zeros(D, bool)
    for t in range(T):
        nz = ~np.isnan(out[t])
        last[nz] = out[t, nz]
        has |= nz
        miss = np.isnan(out[t]) & has
        out[t, miss] = last[miss]
    last[:] = 0
    has[:] = False
    # 뒤쪽 방향으로도 한 번 채우고 싶으면 여기에 bfill 추가할 수도 있지만,
    # 현재 구현은 forward-fill 만 사용.
    return out


def features_from_buf(buf: list[np.ndarray]) -> np.ndarray:
    """
    legacy_2 에서 사용하던 pose -> feature 변환 그대로.
    k: (T,33,4) [x,y,z,visibility]
    """
    k = np.stack(buf, 0)                        # (T,33,4)
    T = k.shape[0]
    xy = k[:, :, :2].reshape(T, -1)             # (T,66)
    vis = k[:, :, 3:4].reshape(T, -1)           # (T,33)
    xy = _ffill_bfill(xy).reshape(T, 33, 2)
    vis = _ffill_bfill(vis).reshape(T, 33, 1)

    hip = np.mean(xy[:, [23, 24], :], axis=1)
    sh = np.mean(xy[:, [11, 12], :], axis=1)
    sc = np.linalg.norm(sh - hip, axis=1, keepdims=True)
    sc[sc < 1e-3] = 1.0

    xy_n = (xy - hip[:, None, :]) / sc[:, None, :]
    vel = np.diff(xy_n, axis=0, prepend=xy_n[:1])

    def ang(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        ba, bc = a - b, c - b
        na = np.linalg.norm(ba, axis=-1, keepdims=True)
        nc = np.linalg.norm(bc, axis=-1, keepdims=True)
        na[na < 1e-6] = 1.0
        nc[nc < 1e-6] = 1.0
        cos = np.sum(ba * bc, axis=-1, keepdims=True) / (na * nc)
        cos = np.clip(cos, -1, 1)
        return np.arccos(cos) / np.pi

    def pick(i: int) -> np.ndarray:
        return xy_n[:, i, :]

    angs = np.stack(
        [
            ang(pick(11), pick(13), pick(15)),
            ang(pick(12), pick(14), pick(16)),
            ang(pick(23), pick(25), pick(27)),
            ang(pick(24), pick(26), pick(28)),
        ],
        axis=1,
    )  # (T,4)

    feat = np.concatenate(
        [
            xy_n.reshape(T, -1),       # 66
            vel.reshape(T, -1),        # 66
            angs,                      # 4
            vis.reshape(T, -1),        # 33
        ],
        axis=1,
    )  # (T,169)
    feat = feat.astype(np.float32)
    return np.clip(feat, -10, 10)


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


def load_models_and_meta() -> None:
    global _det_bundle, _cls_bundle

    # --- 이상 탐지 모델 ---
    ck_det = torch.load(DET_CKPT_PATH, map_location=DEVICE)
    meta_det = _load_meta_from_ckpt_or_json(ck_det, DET_META_PATH)

    feat_dim_det = meta_det.get("feat_dim") or ck_det.get("feat_dim")
    if feat_dim_det is None:
        raise RuntimeError("검출 모델 feat_dim 정보를 찾을 수 없습니다.")
    feat_dim_det = int(feat_dim_det)

    num_out_det = meta_det.get("num_out") or ck_det.get("num_out") or 1
    num_out_det = int(num_out_det)

    norm_mean_det = np.asarray(meta_det.get("norm_mean", 0.0), np.float32)
    norm_std_det = np.asarray(meta_det.get("norm_std", 1.0), np.float32)
    if norm_mean_det.ndim == 0:
        norm_mean_det = np.full([feat_dim_det], float(norm_mean_det), np.float32)
    if norm_std_det.ndim == 0:
        norm_std_det = np.full([feat_dim_det], float(norm_std_det), np.float32)

    det_model = LSTMAnom(feat_dim=feat_dim_det, num_out=num_out_det)
    det_model.load_state_dict(ck_det["model"])
    det_model.to(DEVICE)
    det_model.eval()

    _det_bundle = LSTMBundle(
        model=det_model,
        norm_mean=norm_mean_det,
        norm_std=norm_std_det,
        events=None,
    )

    # --- 이상 분류 모델 ---
    ck_cls = torch.load(CLS_CKPT_PATH, map_location=DEVICE)
    meta_cls = _load_meta_from_ckpt_or_json(ck_cls, CLS_META_PATH)

    feat_dim_cls = meta_cls.get("feat_dim") or ck_cls.get("feat_dim")
    if feat_dim_cls is None:
        raise RuntimeError("분류 모델 feat_dim 정보를 찾을 수 없습니다.")
    feat_dim_cls = int(feat_dim_cls)

    num_out_cls = meta_cls.get("num_out") or ck_cls.get("num_out")
    events_cls = meta_cls.get("events")
    if events_cls is None:
        raise RuntimeError("분류 모델 meta['events'] 가 필요합니다.")
    if num_out_cls is None:
        num_out_cls = len(events_cls)
    num_out_cls = int(num_out_cls)
    if num_out_cls != len(events_cls):
        raise RuntimeError("분류 모델 num_out 과 events 길이가 일치하지 않습니다.")

    norm_mean_cls = np.asarray(meta_cls.get("norm_mean", 0.0), np.float32)
    norm_std_cls = np.asarray(meta_cls.get("norm_std", 1.0), np.float32)
    if norm_mean_cls.ndim == 0:
        norm_mean_cls = np.full([feat_dim_cls], float(norm_mean_cls), np.float32)
    if norm_std_cls.ndim == 0:
        norm_std_cls = np.full([feat_dim_cls], float(norm_std_cls), np.float32)

    cls_model = LSTMAnom(feat_dim=feat_dim_cls, num_out=num_out_cls)
    cls_model.load_state_dict(ck_cls["model"])
    cls_model.to(DEVICE)
    cls_model.eval()

    _cls_bundle = LSTMBundle(
        model=cls_model,
        norm_mean=norm_mean_cls,
        norm_std=norm_std_cls,
        events=list(events_cls),
    )

    print(f"[MODEL] det ckpt={DET_CKPT_PATH}, feat_dim={feat_dim_det}, out={num_out_det}")
    print(
        f"[MODEL] cls ckpt={CLS_CKPT_PATH}, feat_dim={feat_dim_cls}, "
        f"out={num_out_cls}, events={events_cls}"
    )


def run_detection_from_keypoints(kpt_seq: np.ndarray) -> float:
    """
    이상 탐지 모델: p_anom (0~1) 반환.
    """
    if _det_bundle is None:
        raise RuntimeError("검출 모델이 로드되지 않았습니다.")

    feat = features_from_buf(list(kpt_seq))  # (T,F)
    feat = np.clip(
        (feat - _det_bundle.norm_mean) / (_det_bundle.norm_std + 1e-6),
        -6,
        6,
    )
    x = torch.from_numpy(feat).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        logits = _det_bundle.model(x)
        probs = torch.sigmoid(logits).detach().cpu().numpy()[0]

    # num_out>1 이어도 "이상 확률"은 첫 채널로 본다고 가정
    p_anom = float(probs[0])
    return p_anom


def run_classification_from_keypoints(
    kpt_seq: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    """
    이상 분류 모델: (events, probs) 반환.
    """
    if _cls_bundle is None or _cls_bundle.events is None:
        raise RuntimeError("분류 모델이 로드되지 않았습니다.")

    feat = features_from_buf(list(kpt_seq))
    feat = np.clip(
        (feat - _cls_bundle.norm_mean) / (_cls_bundle.norm_std + 1e-6),
        -6,
        6,
    )
    x = torch.from_numpy(feat).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        logits = _cls_bundle.model(x)
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]

    return _cls_bundle.events, probs


# =========================================================
# 카메라별 상태 (EMA + voting + START/END)
#  - key: (agent_id, camera_id) 조합
# =========================================================

@dataclass
class CameraState:
    in_anomaly: bool = False
    start_window_index: int | None = None
    start_ts: float | None = None
    last_end_ts: float = -1e9

    ema_det: float | None = None                # 이상 탐지 EMA
    ema_cls: np.ndarray | None = None           # 분류 softmax EMA
    votes: deque[bool | None] = field(
        default_factory=lambda: deque(maxlen=DET_VOTE_WIN)
    )

    current_label: str | None = None


_camera_states: dict[str, CameraState] = {}


# =========================================================
# Kakao 유틸 (에이전트별 토큰 기반)
# =========================================================

def build_kakao_text(
    *,
    agent_code: str,
    camera_id: str,
    source_id: str | None,
    event_label: str,
    duration_sec: float | None,
    top_prob: float,
) -> str:
    src = source_id or "-"
    dur_str = f"{duration_sec:.1f}s" if duration_sec is not None else "unknown"
    prob_pct = int(top_prob * 100)
    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return (
        "[이상행동 감지]\n"
        f"- 에이전트: {agent_code}\n"
        f"- 카메라: {camera_id}\n"
        f"- 소스: {src}\n"
        f"- 유형: {event_label}\n"
        f"- 신뢰도: {prob_pct}%\n"
        f"- 지속시간: {dur_str}\n"
        f"- 시각: {ts_str}"
    )


def send_kakao_alarm(text: str, access_token: str | None) -> dict[str, Any]:
    """
    에이전트별 kakao_access_token 을 사용하여 메시지를 전송한다.
    access_token 이 없으면 'disabled' 로 동작.
    """
    if not access_token:
        print("[KAKAO] disabled (no access token for agent)")
        return {"ok": False, "mode": "disabled", "reason": "no_token"}

    headers = {
        "Authorization": f"Bearer {access_token}",
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
        resp = requests.post(KAKAO_SEND_URL, headers=headers, data=payload, timeout=3.0)
        try:
            data = resp.json()
        except Exception:
            data = resp.text
        ok = 200 <= resp.status_code < 300
        if not ok:
            print(f"[KAKAO] send failed: status={resp.status_code}, body={data}")
        return {
            "ok": ok,
            "mode": "real",
            "status_code": resp.status_code,
            "response": data,
        }
    except Exception as e:
        print(f"[KAKAO] send exception: {e}")
        return {"ok": False, "mode": "real", "error": str(e)}


# =========================================================
# FastAPI 앱 / lifespan
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    SQLModel.metadata.create_all(engine)
    load_models_and_meta()
    print("[STARTUP] behavior inference server ready.")
    yield
    # 필요하면 종료 처리


app = FastAPI(
    title="Behavior Inference Server (agent-aware, multi-cam, Kakao per agent)",
    lifespan=lifespan,
)


# =========================================================
# 요청/응답 스키마
# =========================================================

class PoseWindowRequest(BaseModel):
    # 에이전트 코드 (점포/설치 위치 식별용)
    agent_code: str

    camera_id: str
    source_id: str | None = None
    window_index: int
    window_start_ts: float | None = None
    window_end_ts: float | None = None
    # (T,33,4): [[ [x,y,z,vis], ...33 ], ...T ]
    keypoints: list[list[list[float]]]


class BehaviorResultResponse(BaseModel):
    # 응답에도 agent_code 포함 (UI/로깅 편의를 위해)
    agent_code: str
    camera_id: str
    source_id: str | None = None
    window_index: int
    window_start_ts: float | None = None
    window_end_ts: float | None = None

    is_anomaly: bool
    det_prob: float            # 현재 EMA 기준 이상 확률
    top_label: str | None
    top_prob: float
    prob: dict[str, float]     # 이상 분류 (EMA 기준 softmax)


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
    에이전트가 처음 실행될 때, 자신의 Kakao 토큰을 등록/갱신하는 요청 본문.
    """
    agent_code: str
    kakao_access_token: str
    note: str | None = None


# =========================================================
# 에이전트 Kakao 토큰 등록 엔드포인트
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
    agent.updated_at = utcnow()
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
# 메인 엔드포인트 (포즈 -> 이상행동 추론)
# =========================================================

@app.post("/behavior/analyze_pose", response_model=BehaviorResultResponse)
def analyze_pose(
    payload: PoseWindowRequest,
    session: Session = Depends(get_session),
):
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

    # 에이전트 조회/생성
    agent = get_or_create_agent(session, agent_code)
    agent_id = agent.id
    assert agent_id is not None

    # (agent, camera) 단위 상태
    state_key = f"{agent_id}:{camera_id}"
    state = _camera_states.get(state_key)
    if state is None:
        state = CameraState()
        _camera_states[state_key] = state

    try:
        # 1) 이상 탐지 (Model_1)
        p_anom = run_detection_from_keypoints(kpt_seq)
        if state.ema_det is None:
            state.ema_det = p_anom
        else:
            state.ema_det = EMA_ALPHA * p_anom + (1.0 - EMA_ALPHA) * state.ema_det

        # 2) 이상 분류 (Model_2)
        events, probs_cls = run_classification_from_keypoints(kpt_seq)
        if state.ema_cls is None:
            state.ema_cls = probs_cls.copy()
        else:
            state.ema_cls = EMA_ALPHA * probs_cls + (1.0 - EMA_ALPHA) * state.ema_cls

    except NotImplementedError as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})
    except Exception as e:
        print(f"[ERR] inference failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"inference failed: {e}"},
        )

    ema_det = float(state.ema_det if state.ema_det is not None else p_anom)
    ema_cls = state.ema_cls if state.ema_cls is not None else probs_cls

    top_idx = int(np.argmax(ema_cls))
    top_label = events[top_idx] if 0 <= top_idx < len(events) else None
    top_prob = float(ema_cls[top_idx])

    # voting 업데이트 (이상 여부만)
    is_high = ema_det >= DET_START_THR
    state.votes.append(True if is_high else None)

    # per-window is_anomaly (UI/DB용): 단순 threshold + EMA 기준
    is_anomaly_now = ema_det >= DET_START_THR

    # DB: inference_results 저장 (매 윈도우)
    row = InferenceResult(
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
        stage2_probs_json=json.dumps([float(x) for x in probs_cls]),
        stage2_top_label=top_label,
        stage2_top_prob=top_prob,
    )
    session.add(row)
    session.commit()

    # START / END 판정 + Kakao 로그 (에이전트별 토큰 사용)
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
            kakao_res = send_kakao_alarm(text, agent.kakao_access_token)
            raw_resp = kakao_res.get("response")
            raw_json = (
                json.dumps(raw_resp, ensure_ascii=False)
                if isinstance(raw_resp, dict)
                else None
            )

            log_row = KakaoAlarmLog(
                agent_id=agent_id,
                camera_id=camera_id,
                source_id=source_id,
                event_label=label_for_log,
                start_window_index=state.start_window_index,
                end_window_index=window_index,
                duration_sec=dur,
                top_prob=top_prob,
                text_preview=text,
                kakao_mode=str(kakao_res.get("mode", "disabled")),
                kakao_ok=bool(kakao_res.get("ok", False)),
                kakao_status_code=kakao_res.get("status_code"),
                kakao_error=kakao_res.get("error"),
                kakao_raw_response_json=raw_json,
            )
            session.add(log_row)
            session.commit()

            state.in_anomaly = False
            state.current_label = None
            state.start_ts = None
            state.start_window_index = None
            state.last_end_ts = ts
            state.votes.clear()

    # UI 응답용 확률 맵
    prob_dict: dict[str, float] = {
        label: float(p) for label, p in zip(events, ema_cls)
    }

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
# kakao test (에이전트별)
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
# Behavior 최신 결과 조회 (단일 캠)
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


# =========================================================
# Behavior 최신 결과 조회 (에이전트 단위 멀티캠)
# =========================================================

@app.get(
    "/behavior/latest_all",
    response_model=list[BehaviorResultResponse],
)
def get_latest_behavior_all(
    agent_code: str = Query(..., description="에이전트 코드"),
    session: Session = Depends(get_session),
):
    """
    한 에이전트의 모든 카메라에 대해,
    각 카메라별 최신 InferenceResult 를 한 번에 반환.

    - UI 는 이걸 호출해서 "카메라가 몇 개 있는지" 파악하고
      그때 레이아웃/그리드를 유연하게 잡으면 된다.
    """
    agent = get_or_create_agent(session, agent_code)
    agent_id = agent.id
    assert agent_id is not None

    stmt = (
        select(InferenceResult)
        .where(InferenceResult.agent_id == agent_id)
        .order_by(InferenceResult.camera_id, InferenceResult.created_at.desc())
    )
    rows = session.exec(stmt).all()

    latest_by_camera: dict[str, InferenceResult] = {}
    for row in rows:
        if row.camera_id not in latest_by_camera:
            latest_by_camera[row.camera_id] = row

    results: list[BehaviorResultResponse] = []

    for cam_id in sorted(latest_by_camera.keys()):
        row = latest_by_camera[cam_id]
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

        results.append(
            BehaviorResultResponse(
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
        )

    return results


# =========================================================
# Tracking 최신 결과 조회 (단일 캠)
# =========================================================

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


# =========================================================
# Tracking 최신 결과 조회 (에이전트 단위 멀티캠)
# =========================================================

@app.get(
    "/tracking/latest_all",
    response_model=list[TrackingSnapshotResponse],
)
def get_latest_tracking_all(
    agent_code: str = Query(..., description="에이전트 코드"),
    session: Session = Depends(get_session),
):
    """
    한 에이전트의 모든 카메라에 대해,
    각 카메라별 최신 TrackingSnapshot 를 한 번에 반환.
    """
    agent = get_or_create_agent(session, agent_code)
    agent_id = agent.id
    assert agent_id is not None

    stmt = (
        select(TrackingSnapshot)
        .where(TrackingSnapshot.agent_id == agent_id)
        .order_by(TrackingSnapshot.camera_id, TrackingSnapshot.timestamp.desc())
    )
    rows = session.exec(stmt).all()

    latest_by_camera: dict[str, TrackingSnapshot] = {}
    for row in rows:
        if row.camera_id not in latest_by_camera:
            latest_by_camera[row.camera_id] = row

    results: list[TrackingSnapshotResponse] = []

    for cam_id in sorted(latest_by_camera.keys()):
        row = latest_by_camera[cam_id]
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

        results.append(
            TrackingSnapshotResponse(
                agent_code=agent_code,
                camera_id=row.camera_id,
                timestamp=row.timestamp,
                frame_index=row.frame_index,
                objects=objects,
            )
        )

    return results
