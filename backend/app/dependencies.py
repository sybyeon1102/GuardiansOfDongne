from typing import Annotated

from fastapi import Depends

from app.config import Settings, get_settings


def get_app_settings() -> Settings:
    """
    FastAPI 의존성으로 사용할 설정 객체를 반환한다.
    Return application settings for FastAPI dependency injection.
    """
    return get_settings()


type SettingsDep = Annotated[Settings, Depends(get_app_settings)]
