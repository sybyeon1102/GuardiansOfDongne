"""
이 모듈은 JSONL 이벤트 로그 파일에서 최근 이벤트를 읽어와
터미널에 간단한 테이블 형태로 출력하는 CLI 유틸입니다.

This module provides a simple CLI utility that reads recent events
from a JSONL event log file and prints them as a table on the terminal.
"""

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final


# backend/app/services/service_event.py 와 동일한 기본값을 사용한다.
# Use the same defaults as backend/app/services/service_event.py.
_LOG_JSONL_DEFAULT: Final[str] = "events_log.jsonl"


@dataclass
class EventRow:
    """단일 이벤트 행을 표현하는 내부 모델.
    Internal model representing a single event row.
    """

    timestamp: str
    event_type: str
    session_id: str
    user_id: str
    summary: str


def _parse_timestamp(value: Any) -> str:
    """JSON에서 읽은 timestamp 값을 사람이 읽기 쉬운 문자열로 변환한다.
    Convert a timestamp value from JSON into a human-readable string.
    """
    if isinstance(value, str):
        # ISO 8601 형식을 그대로 사용하되, 사람이 읽기 쉽게 약간만 다듬는다.
        # Keep ISO 8601-ish string; optionally trim microseconds.
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return value
    return ""


def _summarize_data(data: Any) -> str:
    """data 필드를 짧은 문자열로 요약한다.
    Summarize the 'data' field into a short string.
    """
    if data is None:
        return ""

    # EventPayload 스타일이면 top_event / top_prob 를 우선 보여준다.
    # For EventPayload-like data, prefer top_event/top_prob if present.
    if isinstance(data, dict):
        top_event = data.get("top_event")
        top_prob = data.get("top_prob")
        if top_event is not None and top_prob is not None:
            return f"{top_event} ({top_prob:.2f})"

        # 그 외에는 JSON을 한 줄로 직렬화해서 보여준다.
        # Otherwise, serialize dict as a single-line JSON string.
        try:
            return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        except TypeError:
            return str(data)

    # dict 가 아니면 str() 로 fallback.
    # Non-dict values fall back to str().
    return str(data)


def _read_last_lines(path: Path, limit: int) -> list[str]:
    """JSONL 파일에서 마지막 limit 줄을 읽어온다.
    Read the last `limit` lines from a JSONL file.
    """
    if not path.exists():
        return []

    # 간단히 전체를 읽어서 끝부분만 잘라낸다.
    # For moderate-sized logs, read all and slice the tail.
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = [line for line in text.splitlines() if line.strip()]
    if limit <= 0 or limit >= len(lines):
        return lines
    return lines[-limit:]


def load_recent_events(log_path: Path, limit: int) -> list[EventRow]:
    """JSONL 로그 파일에서 최근 이벤트를 EventRow 리스트로 변환한다.
    Load recent events from the JSONL log file and convert them into EventRow list.
    """
    lines = _read_last_lines(log_path, limit)
    rows: list[EventRow] = []

    for line in lines:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        timestamp = _parse_timestamp(obj.get("timestamp"))
        event_type = str(obj.get("event_type", "") or "")
        session_id = str(obj.get("session_id", "") or "")
        user_id = str(obj.get("user_id", "") or "")
        summary = _summarize_data(obj.get("data"))

        rows.append(
            EventRow(
                timestamp=timestamp,
                event_type=event_type,
                session_id=session_id,
                user_id=user_id,
                summary=summary,
            ),
        )

    return rows


def print_table(rows: list[EventRow]) -> None:
    """EventRow 리스트를 터미널 테이블로 출력한다.
    Print a list of EventRow objects as a simple table in the terminal.
    """
    if not rows:
        print("No events found.")
        return

    # 각 컬럼별 최대 너비 계산 / Compute max width for each column
    headers = ["#", "Timestamp", "Type", "Session", "User", "Summary"]

    index_width = max(len(headers[0]), len(str(len(rows))))
    ts_width = max(len(headers[1]), *(len(r.timestamp) for r in rows))
    type_width = max(len(headers[2]), *(len(r.event_type) for r in rows))
    sess_width = max(len(headers[3]), *(len(r.session_id) for r in rows))
    user_width = max(len(headers[4]), *(len(r.user_id) for r in rows))

    # Summary는 너무 넓어지지 않도록 적당히 자른다. / Limit summary width.
    summary_width_max = 80
    summary_width = min(
        max(len(headers[5]), *(len(r.summary) for r in rows)),
        summary_width_max,
    )

    header_fmt = (
        f"{{:>{index_width}}}  "
        f"{{:<{ts_width}}}  "
        f"{{:<{type_width}}}  "
        f"{{:<{sess_width}}}  "
        f"{{:<{user_width}}}  "
        f"{{:<{summary_width}}}"
    )
    row_fmt = header_fmt

    print(
        header_fmt.format(
            headers[0],
            headers[1],
            headers[2],
            headers[3],
            headers[4],
            headers[5],
        ),
    )
    print("-" * (index_width + ts_width + type_width + sess_width + user_width + summary_width + 10))

    for idx, row in enumerate(rows, start=1):
        summary = row.summary
        if len(summary) > summary_width:
            summary = summary[: summary_width - 3] + "..."
        print(
            row_fmt.format(
                idx,
                row.timestamp,
                row.event_type,
                row.session_id,
                row.user_id,
                summary,
            ),
        )


def parse_args() -> argparse.Namespace:
    """명령행 인자를 파싱한다.
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "JSONL 이벤트 로그에서 최근 이벤트를 읽어와 테이블로 출력합니다.\n"
            "Read recent events from a JSONL log file and print them as a table."
        ),
    )

    parser.add_argument(
        "--path",
        type=str,
        default=os.getenv("BACKEND_EVENT_LOG_JSONL", _LOG_JSONL_DEFAULT),
        help=(
            "이벤트 로그 JSONL 파일 경로. "
            "기본값은 BACKEND_EVENT_LOG_JSONL 또는 'events_log.jsonl' 입니다.\n"
            "Path to the event log JSONL file "
            "(default: BACKEND_EVENT_LOG_JSONL or 'events_log.jsonl')."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help=(
            "가져올 최대 이벤트 개수 (기본: 50).\n"
            "Maximum number of events to load (default: 50)."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """CLI 엔트리 포인트.
    CLI entry point.
    """
    args = parse_args()
    log_path = Path(args.path)
    limit = max(1, args.limit)

    print(f"[INFO] Reading last {limit} events from: {log_path}")
    rows = load_recent_events(log_path, limit=limit)
    print_table(rows)


if __name__ == "__main__":
    main()
