from functools import lru_cache
from pathlib import Path
from typing import Final

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


APP_TITLE_DEFAULT: Final[str] = "Behavior Inference API"
APP_VERSION_DEFAULT: Final[str] = "0.1.0"
APP_DESCRIPTION_DEFAULT: Final[str] = "Behavior analysis backend (FastAPI)"


class Settings(BaseSettings):
    """
    애플리케이션 전역 설정.
    Global application settings.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="BACKEND_",
        extra="ignore",
    )

    app_title: str = APP_TITLE_DEFAULT
    app_version: str = APP_VERSION_DEFAULT
    app_description: str = APP_DESCRIPTION_DEFAULT

    model_checkpoint_path: Path = Field(
        ...,
        description=(
            "LSTM 모델 체크포인트(.pt) 파일 경로.\n"
            "Path to the LSTM model checkpoint (.pt) file."
        ),
    )

    environment: str = Field(
        default="local",
        description=(
            "실행 환경(local/dev/prod 등) / "
            "Runtime environment (local/dev/prod, etc.)."
        ),
    )
    debug: bool = Field(
        default=False,
        description="디버그 모드 활성화 여부 / Whether to enable debug mode.",
    )


@lru_cache
def get_settings() -> Settings:
    """
    환경 변수 및 .env 파일에서 설정을 로드한다.
    Load settings from environment variables and .env file (cached).
    """
    return Settings()
