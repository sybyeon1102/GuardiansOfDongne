from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# 원래 c_server.py에서 사용하던 이벤트 타입 리터럴.
# Event type literal used in the original c_server.py implementation.
EventType = Literal["START", "END", "PROBS", "HEARTBEAT"]


class EventPayload(BaseModel):
    """
    실시간 클라이언트가 전송하는 단일 이벤트 페이로드.
    Single event payload sent from the realtime client.

    원래 b_inference_server/c_server.py 의 EventPayload 모델을 옮겨온 것이다.
    This model mirrors the EventPayload definition from b_inference_server/c_server.py.
    """

    # 공통 필드 / Common fields
    type: EventType
    ts: float = Field(
        default_factory=lambda: datetime.utcnow().timestamp(),
        description="이벤트 발생 시각(UNIX epoch, seconds) / Event timestamp.",
    )
    frame_idx: int | None = Field(
        default=None,
        description="영상 프레임 인덱스(있다면) / Video frame index, if any.",
    )

    # START 전송 예: {"event":"fire","prob":0.83}
    # Example for START: {"event": "fire", "prob": 0.83}
    event: str | None = Field(
        default=None,
        description="감지된 이벤트 이름 / Detected event name.",
    )
    prob: float | None = Field(
        default=None,
        description="이벤트에 대한 신뢰도(probability) / Probability for the event.",
    )

    # END 전송 예: {"event":"fire","duration_sec":5.45}
    # Example for END: {"event": "fire", "duration_sec": 5.45}
    duration_sec: float | None = Field(
        default=None,
        description="이벤트 지속 시간(초) / Event duration in seconds.",
    )

    # HEARTBEAT/PROBS 보조 정보
    # Auxiliary fields for HEARTBEAT/PROBS
    state: str | None = Field(
        default=None,
        description="클라이언트/세션 상태 문자열 / Client or session state string.",
    )
    top_event: str | None = Field(
        default=None,
        description="가장 높은 확률의 이벤트 이름 / Top-1 event name.",
    )
    top_prob: float | None = Field(
        default=None,
        description="가장 높은 확률 값 / Top-1 probability value.",
    )
    probs: dict[str, float] | None = Field(
        default=None,
        description=(
            "이벤트별 확률 맵 (event → prob). "
            "Per-event probability map (event → probability)."
        ),
    )


class EventLog(BaseModel):
    """
    단일 이벤트 로그 레코드.
    Represents a single persisted event log record.

    - timestamp: 서버가 기록한 시각 (UTC 기준 권장)
    - event_type: START/END/PROBS/HEARTBEAT 등 이벤트 타입 문자열
    - session_id, user_id: 세션/사용자 식별자(선택)
    - data: 원본 페이로드(예: EventPayload.model_dump()) 또는 추가 메타데이터
    """

    timestamp: datetime
    event_type: str
    session_id: str | None = None
    user_id: str | None = None
    data: dict[str, Any] | None = None


class EventLogRequest(BaseModel):
    """
    이벤트 로그를 기록하기 위한 요청 본문.
    Request body for recording an event log.

    기본 형태는 {"event": {...}} 이며,
    event 필드 안에는 EventLog 인스턴스가 직렬화되어 들어간다.
    The basic shape is {"event": {...}} where `event` holds a serialized EventLog.
    """

    event: EventLog


class EventLogResponse(BaseModel):
    """
    이벤트 로그 기록 결과 응답.
    Response for event log recording result.
    """

    success: bool
    event_id: str | None = None
