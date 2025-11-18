from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Final
from uuid import uuid4
import json
import os

from project_core import Err, Ok, Result

from app.internal.kakao_notification import (
    AnomalyNotificationPayload,
    build_kakao_text,
    send_kakao_memo,
)
from app.models.model_io_event import (
    EventLog,
    EventLogRequest,
    EventLogResponse,
    EventPayload,
)


# JSONL 로그 파일 경로 기본값 / Default path for JSONL log file
_LOG_JSONL_DEFAULT: Final[str] = "events_log.jsonl"

# 메모리에 보관할 최근 이벤트 개수 기본값 / Default number of recent events kept in memory
_KEEP_RECENT_DEFAULT: Final[int] = 200

# 환경 변수에서 설정값을 읽어온다. 없으면 기본값을 사용한다.
# Load configuration from environment variables, falling back to defaults.
_LOG_JSONL_PATH: Final[str] = os.getenv(
    "BACKEND_EVENT_LOG_JSONL",
    _LOG_JSONL_DEFAULT,
)
_KEEP_RECENT: Final[int] = int(
    os.getenv("BACKEND_EVENT_KEEP_RECENT", str(_KEEP_RECENT_DEFAULT)),
)

# 최근 수신 이벤트를 메모리에 유지하는 버퍼.
# In-memory buffer to keep the most recent events.
_RECENT_EVENTS: Deque[dict[str, Any]] = deque(maxlen=_KEEP_RECENT)


def _event_to_dict(event: EventLog) -> dict[str, Any]:
    """EventLog 모델을 JSON 직렬화 가능한 dict 로 변환한다.
    Convert an EventLog model into a JSON-serializable dict.
    """
    return {
        # timezone-aware datetime 을 ISO 문자열로 저장 (예: 2025-11-18T12:34:56+00:00)
        # Store timezone-aware datetime as ISO string.
        "timestamp": event.timestamp.isoformat(),
        "event_type": event.event_type,
        "session_id": event.session_id,
        "user_id": event.user_id,
        "data": event.data,
    }


def _append_and_maybe_dump(event_dict: dict[str, Any]) -> None:
    """이벤트를 메모리 버퍼와 JSONL 파일에 기록한다.
    Append the event to the in-memory buffer and optionally to a JSONL file.
    """
    _RECENT_EVENTS.append(event_dict)

    # LOG_JSONL 경로가 비어 있으면 파일로 기록하지 않는다.
    # If LOG_JSONL path is empty, skip file persistence.
    if not _LOG_JSONL_PATH:
        return

    path = Path(_LOG_JSONL_PATH)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            json.dump(event_dict, file, ensure_ascii=False)
            file.write("\n")
    except OSError as exc:
        # 파일 기록 실패는 치명적 오류로 취급하지 않고 경고만 남긴다.
        # Failing to write the log file is treated as a warning, not a fatal error.
        print(f"[WARN] Failed to write event log JSONL: {exc}")


def record_event_log(request: EventLogRequest) -> Result[EventLogResponse, str]:
    """EventLogRequest 기반으로 이벤트 로그를 기록한다.
    Record an event log based on EventLogRequest.
    """
    try:
        event_dict = _event_to_dict(request.event)
        event_id = uuid4().hex
        event_dict["event_id"] = event_id

        _append_and_maybe_dump(event_dict)

        response = EventLogResponse(
            success=True,
            event_id=event_id,
        )
        return Ok(response)
    except Exception as exc:  # noqa: BLE001
        # 간단히 문자열 에러로 감싼다 (추후 EventError 타입으로 확장 가능).
        # Wrap error as a simple string (can be replaced with EventError later).
        return Err(f"Failed to record event log: {exc}")


def _maybe_send_kakao_for_end(payload: EventPayload) -> None:
    """EventPayload가 END 타입인 경우 카카오 알림을 시도한다.
    Try sending a Kakao memo when the payload type is END.

    이 함수는 실패해도 예외를 전파하지 않고 경고만 출력한다.
    This function does not raise on failure; it only prints a warning.
    """
    if payload.type != "END":
        return

    try:
        notif = AnomalyNotificationPayload(
            event=payload.event,
            duration_sec=payload.duration_sec,
            # ts 를 영상 상 시간으로 간주 / Treat ts as video timestamp.
            video_ts=payload.ts,
            frame_idx=payload.frame_idx,
        )
        text = build_kakao_text(notif)
        result = send_kakao_memo(text)

        if not result.get("ok", False):
            print(f"[WARN] Failed to send Kakao memo: {result}")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Exception while sending Kakao memo: {exc}")


def record_event_payload(
    payload: EventPayload,
    *,
    session_id: str | None = None,
    user_id: str | None = None,
) -> Result[EventLogResponse, str]:
    """legacy EventPayload 기반으로 이벤트 로그를 기록한다.
    Record an event log based on legacy EventPayload.

    - payload.ts 는 UNIX epoch(sec) 로 들어오며, 이를 timezone-aware datetime 으로 변환하여 기록한다.
    - payload 전체를 data 필드에 그대로 저장한다 (추후 스키마를 좁힐 수 있음).
    - type == "END" 인 경우 카카오톡 "나에게 보내기" 알림을 시도한다.
    """
    try:
        # ts 가 없으면 현재 UTC 시각으로 대체 / Fallback to current UTC time if ts is missing
        if payload.ts is None:
            timestamp = datetime.now(timezone.utc)
        else:
            timestamp = datetime.fromtimestamp(payload.ts, tz=timezone.utc)

        event_log = EventLog(
            timestamp=timestamp,
            event_type=payload.type,
            session_id=session_id,
            user_id=user_id,
            data=payload.model_dump(),
        )

        # END 이벤트에 대해서는 카카오 알림을 비동기적으로(느슨하게) 시도.
        # For END events, loosely try to send a Kakao memo (optional side-effect).
        _maybe_send_kakao_for_end(payload)

        request = EventLogRequest(event=event_log)
        return record_event_log(request)
    except Exception as exc:  # noqa: BLE001
        return Err(f"Failed to record event payload: {exc}")


def get_recent_events(limit: int) -> Result[list[dict[str, Any]], str]:
    """요청된 개수만큼 최근 이벤트 목록을 반환한다.
    Return a list of most recent events up to the requested limit.
    """
    try:
        if limit <= 0:
            items: list[dict[str, Any]] = []
        else:
            snapshot = list(_RECENT_EVENTS)
            if limit >= len(snapshot):
                items = snapshot
            else:
                items = snapshot[-limit:]

        return Ok(items)
    except Exception as exc:  # noqa: BLE001
        return Err(f"Failed to get recent events: {exc}")
