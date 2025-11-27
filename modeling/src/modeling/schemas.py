from typing import Any

from pydantic import BaseModel


class PoseWindowRequest(BaseModel):
    """
    에이전트 → 서버로 전달되는 윈도우 요청 공통 스키마.

    """

    # 어떤 에이전트/카메라에서 온 윈도우인지 식별
    agent_code: str
    camera_id: str
    source_id: str | None = None

    # 윈도우 인덱스 & 시간 정보
    window_index: int
    window_start_ts: float | None = None
    window_end_ts: float | None = None

    # (선택) 전처리된 feature 시퀀스 (T, feat_dim). 예: feat_dim=169
    features: list[list[float]]

    # 이 윈도우에 '의미 있는 포즈'가 하나라도 있는지 여부
    # - False: 0-프레임-윈도우 (추론 스킵용)
    # - True : 적어도 한 프레임 이상 포즈 있음
    has_pose: bool = True
