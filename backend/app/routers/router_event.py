from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from project_core import Err, Ok

from app.models.model_io_event import (
    EventLogRequest,
    EventLogResponse,
    EventPayload,
)
from app.services.service_event import (
    get_recent_events as get_recent_events_service,
    record_event_log as record_event_log_service,
    record_event_payload as record_event_payload_service,
)
from app.internal.kakao_notification import (
    check_kakao_token,
    send_kakao_memo,
)

router = APIRouter()


@router.post(
    "/",
    summary="이벤트 로그 기록 / Record event log",
    response_model=EventLogResponse,
)
async def record_event_log_endpoint(
    request: EventLogRequest,
) -> EventLogResponse:
    """
    정규 EventLogRequest 기반으로 이벤트 로그를 기록한다.
    Record an event log based on EventLogRequest.
    """
    result = record_event_log_service(request)

    match result:
        case Ok(value=response):
            return response
        case Err(error=message):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=message,
            )
        case _:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unexpected result type.",
            )


@router.get(
    "/recent",
    summary="최근 이벤트 조회 / Get recent events",
)
async def get_recent_events_endpoint(
    limit: int = 50,
) -> list[dict[str, Any]]:
    """
    최근 이벤트 로그를 최대 limit 개수만큼 조회한다.
    Get up to `limit` recent event logs.
    """
    result = get_recent_events_service(limit=limit)

    match result:
        case Ok(value=events):
            return events
        case Err(error=message):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=message,
            )
        case _:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unexpected result type.",
            )


@router.post(
    "/payload",
    summary="EventPayload 기반 이벤트 기록 / Record event via EventPayload",
    response_model=EventLogResponse,
)
async def record_event_payload_endpoint(
    payload: EventPayload,
) -> EventLogResponse:
    """
    legacy EventPayload 형식으로 들어오는 이벤트를 기록한다.
    Record an event log based on legacy EventPayload.

    - 원래 c_server.py 의 `/events` 에 해당하는 기능을,
      새 구조의 `/event/payload` 라는 경로로 흡수한다.
    """
    result = record_event_payload_service(payload)

    match result:
        case Ok(value=response):
            return response
        case Err(error=message):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=message,
            )
        case _:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unexpected result type.",
            )


class KakaoTestRequest(BaseModel):
    """카카오 테스트 메모 발송용 요청 바디.

    Request body for sending a Kakao test memo.
    """

    text: str | None = None


@router.get(
    "/debug/kakao/token-check",
    summary="Kakao 액세스 토큰 유효성 점검",
    tags=["debug", "kakao"],
)
def kakao_token_check() -> dict[str, Any]:
    """KAKAO_ACCESS_TOKEN이 유효한지 /v2/user/me로 점검한다.

    Check whether KAKAO_ACCESS_TOKEN is valid by calling /v2/user/me.
    """
    return check_kakao_token()


@router.post(
    "/debug/kakao/send-test",
    summary="Kakao 테스트 메모 발송",
    tags=["debug", "kakao"],
)
def kakao_send_test(request: KakaoTestRequest) -> dict[str, Any]:
    """임의의 텍스트로 카카오 '나에게 보내기' 테스트 메모를 보낸다.

    Send a test Kakao memo with the given text.
    """
    text = request.text or "Kakao memo test from backend /debug/kakao/send-test"
    return send_kakao_memo(text)
