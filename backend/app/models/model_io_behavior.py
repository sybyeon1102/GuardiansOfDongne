"""
행동(포즈 기반) 분석 API의 입출력 모델을 정의합니다.

Defines input/output models for pose-based behavior analysis API.
"""

from typing import Annotated

from pydantic import BaseModel, Field


class PoseFrame(BaseModel):
    """
    단일 시점의 포즈/특징 벡터를 표현합니다.

    Represents a single frame of pose/features.
    """

    index: Annotated[int | None, Field(
        default=None,
        description=(
            "프레임 인덱스(선택). 0부터 시작하는 순서를 나타낼 수 있습니다.\n"
            "Optional frame index starting from 0."
        ),
    )]

    features: Annotated[list[float], Field(
        description=(
            "LSTM 입력용 특징 벡터 (예: 포즈 좌표/각도 등 전처리된 값들).\n"
            "Feature vector for LSTM input (e.g., preprocessed pose coordinates/angles)."
        ),
    )]


class BehaviorAnalyzeRequest(BaseModel):
    """
    포즈 프레임 시퀀스를 기반으로 정상/이상 행동을 분석해 달라는 요청입니다.

    Request to analyze normal/anomalous behavior from a sequence of pose frames.
    """

    frames: Annotated[list[PoseFrame], Field(
        min_length=1,
        description=(
            "시간 순서대로 정렬된 포즈 프레임 리스트입니다.\n"
            "List of pose frames ordered in time."
        ),
    )]


class BehaviorAnalyzeResponse(BaseModel):
    """
    LSTM 기반 정상/이상 행동 분석 결과입니다.

    LSTM-based normal/anomaly behavior analysis result.
    """

    is_anomaly: Annotated[bool, Field(
        description=(
            "이상 행동으로 판단되면 True, 정상 행동으로 판단되면 False입니다.\n"
            "True if the sequence is considered anomalous, False otherwise."
        ),
    )]

    normal_score: Annotated[float, Field(
        ge=0.0,
        le=1.0,
        description=(
            "정상일 가능성을 나타내는 점수(대략 0~1 범위 가이드)입니다.\n"
            "Score (roughly 0–1) indicating how likely the sequence is normal."
        ),
    )]

    anomaly_score: Annotated[float, Field(
        ge=0.0,
        le=1.0,
        description=(
            "이상일 가능성을 나타내는 점수(대략 0~1 범위 가이드)입니다.\n"
            "Score (roughly 0–1) indicating how likely the sequence is anomalous."
        ),
    )]

    # 선택 필드들: 필요 시 프론트엔드/디버깅 용도로 사용할 수 있습니다.
    # Optional detailed fields for frontend/debugging if needed.

    events: Annotated[list[str] | None, Field(
        default=None,
        description=(
            "클래스(이벤트) 이름 리스트입니다. scores/thresholds와 인덱스로 정렬됩니다.\n"
            "Optional list of class/event names aligned with scores/thresholds by index."
        ),
    )]

    scores: Annotated[list[float] | None, Field(
        default=None,
        description=(
            "각 이벤트(클래스)별 sigmoid 점수 리스트입니다.\n"
            "Optional list of per-event sigmoid scores."
        ),
    )]

    thresholds: Annotated[list[float] | None, Field(
        default=None,
        description=(
            "각 이벤트(클래스)별 의사결정 임계값 리스트입니다.\n"
            "Optional list of per-event decision thresholds."
        ),
    )]
