"""
카카오톡 "나에게 보내기" 메모 발송 유틸리티.

This module provides small utilities to send "send to me" KakaoTalk
memo messages using Kakao's default memo API.
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Final
from urllib import parse, request, error as urlerror


KAKAO_MEMO_ENDPOINT: Final[str] = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
KAKAO_ME_ENDPOINT: Final[str] = "https://kapi.kakao.com/v2/user/me"  # 토큰 점검용

ENV_ACCESS_TOKEN: Final[str] = "KAKAO_ACCESS_TOKEN"
ENV_ALERT_LINK_URL: Final[str] = "ALERT_LINK_URL"
ENV_BUTTON_TITLE: Final[str] = "ALERT_BUTTON_TITLE"


@dataclass
class AnomalyNotificationPayload:
    """
    이상 행동 종료(END) 시점에 카카오 알림으로 보낼 핵심 정보.

    Core information to be sent via Kakao memo when an anomaly END event occurs.
    """

    event: str | None = None
    duration_sec: float | None = None
    video_ts: float | None = None
    frame_idx: int | None = None


def format_duration_hhmmss(duration_sec: float | None) -> str:
    """
    초 단위 duration_sec 을 "HH:MM:SS" 문자열로 변환한다.
    Convert duration in seconds to "HH:MM:SS" string.
    """
    if duration_sec is None or duration_sec < 0:
        return "-"

    total = int(duration_sec)
    hours = total // 3600
    minutes = (total % 3600) // 60
    seconds = total % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def build_kakao_text(payload: AnomalyNotificationPayload) -> str:
    """
    이상 행동 알림용 카카오 메모 텍스트를 생성한다.
    Build Kakao memo text for anomaly notification.
    """
    lines: list[str] = []

    lines.append("[이상행동탐지] 최종 이벤트 알림")
    lines.append(f"- 최종 이벤트 : {payload.event or '-'}")

    if payload.duration_sec is not None:
        hhmmss = format_duration_hhmmss(payload.duration_sec)
        lines.append(
            f"- 지속시간 : {hhmmss} ({payload.duration_sec:.1f}s)",
        )

    if payload.video_ts is not None:
        lines.append(f"- 영상 시간 : {payload.video_ts:.2f}s")

    if payload.frame_idx is not None:
        lines.append(f"- 프레임 : {payload.frame_idx}")

    # 수신 시각(로컬 시간) / Received time (local time)
    lines.append(time.strftime("- 수신시각 : %Y-%m-%d / %H:%M:%S", time.localtime()))

    return "\n".join(lines)


def _get_env_access_token() -> str | None:
    """
    KAKAO_ACCESS_TOKEN 환경 변수를 읽어온다.
    Read KAKAO_ACCESS_TOKEN from environment variables.
    """
    token = os.getenv(ENV_ACCESS_TOKEN)
    return token or None


def _get_env_link_url() -> str | None:
    """
    ALERT_LINK_URL 환경 변수를 읽어온다.
    Read ALERT_LINK_URL from environment variables.
    """
    url = os.getenv(ENV_ALERT_LINK_URL)
    return url or None


def _get_env_button_title() -> str:
    """
    ALERT_BUTTON_TITLE 환경 변수를 읽어온다 (기본값 제공).
    Read ALERT_BUTTON_TITLE from environment variables with a default.
    """
    return os.getenv(ENV_BUTTON_TITLE, "확인하러 가기")


def send_kakao_memo(text: str) -> dict[str, Any]:
    """
    카카오 "나에게 보내기" 메모 API를 호출한다.
    Call Kakao "send to me" memo API.

    반환값은 {"ok": bool, ...} 형태의 간단한 dict 이다.
    The return value is a simple dict like {"ok": bool, ...}.
    """
    access_token = _get_env_access_token()
    if not access_token:
        return {"ok": False, "err": f"{ENV_ACCESS_TOKEN} not set"}

    alert_link_url = _get_env_link_url()
    button_title = _get_env_button_title()

    template_object: dict[str, Any] = {
        "object_type": "text",
        "text": text,
        "link": {},
        "button_title": button_title,
    }

    if alert_link_url:
        template_object["link"] = {
            "web_url": alert_link_url,
            "mobile_web_url": alert_link_url,
        }

    # form-urlencoded 로 template_object 를 전송해야 한다.
    # The API expects template_object as URL-encoded form data.
    payload_str = json.dumps(template_object, ensure_ascii=False)
    form_data = parse.urlencode({"template_object": payload_str}).encode("utf-8")

    req = request.Request(
        KAKAO_MEMO_ENDPOINT,
        data=form_data,
        method="POST",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
        },
    )

    try:
        with request.urlopen(req, timeout=5.0) as resp:
            resp_body = resp.read().decode("utf-8", errors="ignore")
    except urlerror.URLError as exc:  # noqa: PERF203
        return {"ok": False, "err": f"request failed: {exc}"}

    try:
        data = json.loads(resp_body)
    except json.JSONDecodeError:
        # 카카오 API는 {"result_code":0, ...} 형식을 사용한다.
        # Kakao API usually responds with {"result_code":0, ...}.
        return {
            "ok": False,
            "err": "invalid JSON response from Kakao",
            "raw": resp_body,
        }

    # result_code == 0 이면 성공으로 본다.
    ok = data.get("result_code") == 0
    return {"ok": ok, "response": data}


def check_kakao_token() -> dict[str, Any]:
    """
    KAKAO_ACCESS_TOKEN 이 유효한지 /v2/user/me 호출로 점검한다.
    Check KAKAO_ACCESS_TOKEN validity by calling /v2/user/me.
    """
    access_token = _get_env_access_token()
    if not access_token:
        return {"ok": False, "err": f"{ENV_ACCESS_TOKEN} not set"}

    req = request.Request(
        KAKAO_ME_ENDPOINT,
        method="GET",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
        },
    )

    try:
        with request.urlopen(req, timeout=5.0) as resp:
            resp_body = resp.read().decode("utf-8", errors="ignore")
    except urlerror.URLError as exc:  # noqa: PERF203
        return {"ok": False, "err": f"request failed: {exc}"}

    try:
        data = json.loads(resp_body)
    except json.JSONDecodeError:
        return {
            "ok": False,
            "err": "invalid JSON response from Kakao /v2/user/me",
            "raw": resp_body,
        }

    return {"ok": True, "response": data}


def _parse_args() -> argparse.Namespace:
    """
    간단한 CLI용 인자를 파싱한다.
    Parse arguments for simple CLI usage.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Send Kakao 'send to me' memo for anomaly END events.\n"
            "이상 행동 종료(END) 시점을 알리는 카카오톡 메모를 전송합니다."
        ),
    )

    parser.add_argument(
        "--event",
        type=str,
        default=None,
        help="최종 이벤트 이름 (예: 'loitering') / Final event name.",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=None,
        help="지속 시간(초) / Duration in seconds.",
    )
    parser.add_argument(
        "--video-ts",
        type=float,
        default=None,
        help="영상 상의 시간(초) / Video timestamp in seconds.",
    )
    parser.add_argument(
        "--frame-idx",
        type=int,
        default=None,
        help="프레임 인덱스 / Frame index.",
    )

    return parser.parse_args()


def main() -> None:
    """
    CLI 엔트리 포인트.

    CLI entry point for manual testing:
    python -m app.internal.kakao_notification --event ... --duration-sec ...
    """
    args = _parse_args()

    payload = AnomalyNotificationPayload(
        event=args.event,
        duration_sec=args.duration_sec,
        video_ts=args.video_ts,
        frame_idx=args.frame_idx,
    )

    text = build_kakao_text(payload)
    print("[INFO] Kakao memo text:")
    print(text)
    print("-" * 40)

    result = send_kakao_memo(text)
    print("[INFO] send_kakao_memo result:")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
