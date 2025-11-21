from collections.abc import Generator
from contextlib import asynccontextmanager
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import requests
import torch
import torch.nn as nn
from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlmodel import Field, Session, SQLModel, create_engine


# =========================================================
# 환경 변수 / 설정
# =========================================================

CKPT_PATH = os.getenv("CKPT_PATH", "lstm_multilabel_05.pt")
META_PATH = os.getenv("META_PATH")  # 없으면 ckpt 안의 meta 사용

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 예: postgresql+psycopg2://user:password@localhost:5432/dbname
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://god_user:secret@localhost:5432/god",
)

# 1차 정상/이상 threshold (pt 하나에서 파생)
ANOMALY_THRESHOLD = float(os.getenv("ANOMALY_THRESHOLD", "0.5"))

# Kakao
KAKAO_ACCESS_TOKEN = (os.getenv("KAKAO_ACCESS_TOKEN") or "").strip()
KAKAO_ENABLED = bool(KAKAO_ACCESS_TOKEN)


# =========================================================
# SQLModel ORM 정의
# =========================================================

engine = create_engine(DATABASE_URL, echo=False)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class InferenceResult(SQLModel, table=True):
    __tablename__ = "inference_results"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=utcnow, nullable=False)

    camera_id: str
    source_id: str | None = None
    window_index: int
    window_start_ts: float | None = None
    window_end_ts: float | None = None

    is_anomaly: bool
    stage1_normal: float
    stage1_anomaly: float

    # JSON 문자열로 저장 (labels, probs)
    stage2_labels_json: str
    stage2_probs_json: str
    stage2_top_label: str | None = None
    stage2_top_prob: float = 0.0


class KakaoAlarmLog(SQLModel, table=True):
    __tablename__ = "kakao_alarm_log"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=utcnow, nullable=False)

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

    kakao_raw_response_json: str | None = None  # 카카오 응답 JSON 문자열


def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session


# =========================================================
# LSTM 모델 / 전처리 (레거시 기반)
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
        z = self.pre(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.lstm(z)
        z = self.pool(h)
        return self.head(z)


def features_from_buf(buf: list[np.ndarray]) -> np.ndarray:
    k = np.stack(buf,0)                        # (T,33,4)
    T = k.shape[0]
    xy  = k[:,:,:2].reshape(T,-1)              # (T,66)
    vis = k[:,:,3:4].reshape(T,-1)             # (T,33)
    xy  = _ffill_bfill(xy).reshape(T,33,2)
    vis = _ffill_bfill(vis).reshape(T,33,1)

    hip = np.mean(xy[:,[23,24],:],axis=1)
    sh  = np.mean(xy[:,[11,12],:],axis=1)
    sc  = np.linalg.norm(sh-hip,axis=1,keepdims=True)
    sc[sc<1e-3] = 1.0

    xy_n = (xy-hip[:,None,:])/sc[:,None,:]
    vel  = np.diff(xy_n,axis=0,prepend=xy_n[:1])

    def ang(a,b,c):
        v1 = a - b
        v2 = c - b
        n1 = np.linalg.norm(v1,axis=-1)
        n2 = np.linalg.norm(v2,axis=-1)
        n1[n1==0] = 1e-6
        n2[n2==0] = 1e-6
        cos=(v1*v2).sum(-1)/(n1*n2)
        return np.arccos(np.clip(cos,-1,1))
    def pick(i): return xy_n[:,i,:]
    angs = np.stack([
        ang(pick(11),pick(13),pick(15)),
        ang(pick(12),pick(14),pick(16)),
        ang(pick(23),pick(25),pick(27)),
        ang(pick(24),pick(26),pick(28)),
    ],axis=1)                                  # (T,4)

    feat = np.concatenate([xy_n.reshape(T,-1),   # 66
                         vel.reshape(T,-1),    # 66
                         angs,                 # 4
                         vis.reshape(T,-1)],1) # 33
    feat = feat.astype(np.float32)               # (T,169)
    return np.clip(feat,-10,10)


def _ffill_bfill(arr):
    T,D = arr.shape
    out = arr.copy()
    last = np.zeros(D,np.float32)
    has = np.zeros(D,bool)
    for t in range(T):
        nz = ~np.isnan(out[t])
        last[nz]=out[t,nz]
        has |= nz
        miss = np.isnan(out[t])&has
        out[t,miss] = last[miss]
    last[:] = 0
    has[:] = False
    for t in range(T-1,-1,-1):
        nz = ~np.isnan(out[t])
        last[nz] = out[t,nz]
        has |=nz
        miss = np.isnan(out[t])&has
        out[t,miss] = last[miss]
    return np.nan_to_num(out,nan=0.0,posinf=0.0,neginf=0.0)


# --- 전역 상태 ---
_model: LSTMAnom | None = None
_events: list[str] = []
_norm_mean: np.ndarray | None = None
_norm_std: np.ndarray | None = None


def load_model_and_meta() -> None:
    global _model, _events, _norm_mean, _norm_std

    ck = torch.load(CKPT_PATH, map_location=DEVICE)
    meta: dict[str, Any] = ck.get("meta", {})

    if not meta and META_PATH and os.path.exists(META_PATH):
        with open(META_PATH, encoding="utf-8") as f:
            meta = json.load(f)

    events = meta.get("events")
    feat_dim: int | None = ck.get("feat_dim")
    num_out: int | None = ck.get("num_out", len(events) if events else None)

    assert events is not None, "meta['events'] 가 없습니다."
    assert feat_dim is not None, "ckpt['feat_dim'] 이 없습니다."
    assert num_out == len(events), "num_out 과 events 길이가 맞지 않습니다."

    norm_mean = np.asarray(meta.get("norm_mean", 0.0), np.float32)
    norm_std = np.asarray(meta.get("norm_std", 1.0), np.float32)
    if norm_mean.ndim == 0:
        norm_mean = np.full([feat_dim], float(norm_mean), np.float32)
    if norm_std.ndim == 0:
        norm_std = np.full([feat_dim], float(norm_std), np.float32)

    model = LSTMAnom(feat_dim=feat_dim, num_out=num_out)
    model.load_state_dict(ck["model"])
    model.to(DEVICE)
    model.eval()

    _model = model
    _events = list(events)
    _norm_mean = norm_mean
    _norm_std = norm_std

    print(f"[MODEL] loaded ckpt={CKPT_PATH}, events={_events}")


def run_inference_from_keypoints_sequence(kpt_seq: np.ndarray) -> tuple[
    list[str],
    np.ndarray,
    float,
    float,
]:
    """
    kpt_seq: (T, 33, 4) float32

    반환:
      events: 레이블 리스트
      probs:  (C,) 각 레이블별 확률
      normal_score, anomaly_score: 1차용 점수 (max_prob 기반)
    """
    if _model is None or not _events or _norm_mean is None or _norm_std is None:
        raise RuntimeError("모델/메타가 로드되지 않았습니다. load_model_and_meta() 먼저 호출 필요.")

    # (T,33,4) 버퍼 → (T,169) feature
    feat = features_from_buf(list(kpt_seq))  # (T,169)

    feat = np.clip((feat - _norm_mean) / (_norm_std + 1e-6), -6, 6)
    x = torch.from_numpy(feat).unsqueeze(0).float().to(DEVICE)  # (1,T,F)

    with torch.no_grad():
        logits = _model(x)  # type: ignore[arg-type]
        probs = torch.sigmoid(logits).cpu().numpy()[0]  # (C,)

    max_prob = float(probs.max()) if probs.size > 0 else 0.0
    anomaly_score = max_prob
    normal_score = 1.0 - anomaly_score

    return _events, probs, normal_score, anomaly_score


# =========================================================
# 카메라별 anomaly 상태 관리 (START/END 추적)
# =========================================================

@dataclass
class CameraState:
    in_anomaly: bool = False
    current_label: str | None = None
    start_window_index: int | None = None
    start_ts: float | None = None


_camera_states: dict[str, CameraState] = {}


def handle_anomaly_transition(
    session: Session,
    *,
    camera_id: str,
    source_id: str | None,
    window_index: int,
    window_start_ts: float | None,
    window_end_ts: float | None,
    is_anomaly: bool,
    top_label: str | None,
    top_prob: float,
) -> None:
    """
    카메라별로 이전 상태를 보고 START/END 이벤트를 판정하고,
    END 시점에 Kakao 알람 + KakaoAlarmLog에 기록.
    """
    state = _camera_states.get(camera_id, CameraState())

    end_ts = window_end_ts if window_end_ts is not None else window_start_ts
    now_ts = end_ts if end_ts is not None else float(window_index)

    if not state.in_anomaly and is_anomaly:
        # START
        state.in_anomaly = True
        state.current_label = top_label
        state.start_window_index = window_index
        state.start_ts = window_start_ts if window_start_ts is not None else now_ts

    elif state.in_anomaly and not is_anomaly:
        # END
        duration = None
        if state.start_ts is not None and now_ts is not None:
            duration = max(0.0, float(now_ts - state.start_ts))

        label = state.current_label or top_label or "unknown"
        text = build_kakao_text(
            camera_id=camera_id,
            source_id=source_id,
            event_label=label,
            duration_sec=duration,
            top_prob=top_prob,
        )
        kakao_res = send_kakao_alarm(text)

        raw_resp = kakao_res.get("response")
        raw_json = json.dumps(raw_resp, ensure_ascii=False) if isinstance(raw_resp, dict) else None

        log = KakaoAlarmLog(
            camera_id=camera_id,
            source_id=source_id,
            event_label=label,
            start_window_index=state.start_window_index,
            end_window_index=window_index,
            duration_sec=duration,
            top_prob=top_prob,
            text_preview=text,
            kakao_mode=str(kakao_res.get("mode", "disabled")),
            kakao_ok=bool(kakao_res.get("ok", False)),
            kakao_status_code=kakao_res.get("status_code"),
            kakao_error=kakao_res.get("error"),
            kakao_raw_response_json=raw_json,
        )
        session.add(log)
        session.commit()

        state = CameraState()

    elif state.in_anomaly and is_anomaly:
        # 계속 이상 상태 유지.
        # 라벨이 바뀌면 이전 END + 새 START로 쪼개고 싶다면 여기서 처리 가능.
        pass

    _camera_states[camera_id] = state


# =========================================================
# Kakao 알림 유틸
# =========================================================

def build_kakao_text(
    *,
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
        f"- 카메라: {camera_id}\n"
        f"- 소스: {src}\n"
        f"- 유형: {event_label}\n"
        f"- 신뢰도: {prob_pct}%\n"
        f"- 지속시간: {dur_str}\n"
        f"- 시각: {ts_str}"
    )


def send_kakao_alarm(text: str) -> dict[str, Any]:
    """
    .env에 KAKAO_ACCESS_TOKEN 이 없으면 아무 것도 보내지 않고 disabled 모드로 동작.
    있으면 카카오톡 "나에게" 텍스트 보내기 API를 사용.
    """
    if not KAKAO_ENABLED:
        print("[KAKAO] disabled (no access token)")
        return {"ok": False, "mode": "disabled", "reason": "no_token"}

    headers = {
        "Authorization": f"Bearer {KAKAO_ACCESS_TOKEN}",
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

    url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
    try:
        resp = requests.post(url, headers=headers, data=payload, timeout=3.0)
        data = resp.json()
        ok = resp.status_code == 200
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
# FastAPI 스키마 / 엔드포인트
# =========================================================


class PoseWindowRequest(BaseModel):
    camera_id: str
    source_id: str | None = None
    window_index: int
    window_start_ts: float | None = None
    window_end_ts: float | None = None

    # (T,33,4) : [[ [x,y,z,vis], ...33 ], ...T ]
    keypoints: list[list[list[float]]]


class BehaviorResultResponse(BaseModel):
    camera_id: str
    source_id: str | None = None
    window_index: int
    window_start_ts: float | None = None
    window_end_ts: float | None = None

    is_anomaly: bool
    top_label: str | None
    top_prob: float
    prob: dict[str, float]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    SQLModel.metadata.create_all(engine)
    load_model_and_meta()
    print("[STARTUP] behavior inference server ready.")

    yield

    # shutdown 시 필요하면 여기서 정리
    # 예: 엔진 디스포즈, 리소스 정리 등
    # engine.dispose()
    # print("[SHUTDOWN] behavior inference server stopped.")


app = FastAPI(
    title="Behavior Inference Server (single-pt, SQLModel, Kakao)",
    lifespan=lifespan,
)


@app.post("/behavior/analyze_pose", response_model=BehaviorResultResponse)
def analyze_pose(
    payload: PoseWindowRequest,
    session: Session = Depends(get_session),
):
    """
    에이전트가 보낸 포즈 시퀀스(T,33,4)에 대해
    - LSTM pt로 추론
    - inference_results 테이블에 저장
    - START/END 기반 Kakao 알람 + 로그 처리
    - UI가 사용할 JSON 응답 반환
    """
    kpt_seq = np.asarray(payload.keypoints, dtype=np.float32)
    if kpt_seq.ndim != 3 or kpt_seq.shape[1:] != (33, 4):
        return JSONResponse(
            status_code=400,
            content={
                "detail": f"keypoints shape must be (T,33,4), got {kpt_seq.shape}"
            },
        )

    try:
        labels, probs, normal_score, anomaly_score = run_inference_from_keypoints_sequence(
            kpt_seq
        )
    except NotImplementedError as e:
        # features_from_buf 연결 전까지는 여기서 에러가 날 수 있음
        return JSONResponse(status_code=500, content={"detail": str(e)})
    except Exception as e:
        print(f"[ERR] inference failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"inference failed: {e}"},
        )

    if probs.size > 0:
        max_idx = int(probs.argmax())
        max_prob = float(probs[max_idx])
        top_label = labels[max_idx] if 0 <= max_idx < len(labels) else None
    else:
        max_prob = 0.0
        top_label = None

    is_anomaly = max_prob >= ANOMALY_THRESHOLD
    prob_dict: dict[str, float] = {ev: float(p) for ev, p in zip(labels, probs)}

    # DB: inference_results 저장
    row = InferenceResult(
        camera_id=payload.camera_id,
        source_id=payload.source_id,
        window_index=payload.window_index,
        window_start_ts=payload.window_start_ts,
        window_end_ts=payload.window_end_ts,
        is_anomaly=is_anomaly,
        stage1_normal=float(normal_score),
        stage1_anomaly=float(anomaly_score),
        stage2_labels_json=json.dumps(labels, ensure_ascii=False),
        stage2_probs_json=json.dumps([float(x) for x in probs]),
        stage2_top_label=top_label,
        stage2_top_prob=max_prob,
    )
    session.add(row)
    session.commit()

    # Kakao START/END 처리 (END에서만 알람 발송)
    handle_anomaly_transition(
        session,
        camera_id=payload.camera_id,
        source_id=payload.source_id,
        window_index=payload.window_index,
        window_start_ts=payload.window_start_ts,
        window_end_ts=payload.window_end_ts,
        is_anomaly=is_anomaly,
        top_label=top_label,
        top_prob=max_prob,
    )

    return BehaviorResultResponse(
        camera_id=payload.camera_id,
        source_id=payload.source_id,
        window_index=payload.window_index,
        window_start_ts=payload.window_start_ts,
        window_end_ts=payload.window_end_ts,
        is_anomaly=is_anomaly,
        top_label=top_label,
        top_prob=max_prob,
        prob=prob_dict,
    )


# ---- kakao_test.py 대체용 엔드포인트 ----

@app.get("/kakao/test")
def kakao_test() -> dict[str, Any]:
    """
    Kakao 토큰 상태를 확인하고, 필요하면 테스트 메시지를 보낸다.

    - 토큰이 없으면: {"enabled": False, ...}
    - 토큰이 있으면: 실제로 "테스트 메시지"를 보내고 응답 요약을 반환
    """
    if not KAKAO_ENABLED:
        return {
            "enabled": False,
            "message": "KAKAO_ACCESS_TOKEN not set",
        }

    text = "[테스트] 이상행동 알림 시스템 카카오 연동 테스트 메시지입니다."
    res = send_kakao_alarm(text)

    return {
        "enabled": True,
        "send_ok": bool(res.get("ok", False)),
        "mode": res.get("mode"),
        "status_code": res.get("status_code"),
        "error": res.get("error"),
        "raw_response": res.get("response"),
    }
