from pathlib import Path
from typing import Any
import logging
import os

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def load_env(base_dir: Path, filename: str = ".env") -> None:
    """해당 디렉터리 기준 .env 파일을 로드한다."""
    load_dotenv(base_dir / filename)


def env_str(name: str, default: str | None = None) -> str:
    v = os.getenv(name)
    if v is None:
        if default is None:
            raise RuntimeError(f"missing environment variable: {name}")
        return default
    return v


def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        logger.warning("ENV %s=%r is not a valid float, using default %s", name, v, default)
        return default


def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError:
        logger.warning("ENV %s=%r is not a valid int, using default %s", name, v, default)
        return default


def env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    v2 = v.strip().lower()
    if v2 in ("1", "true", "yes", "on"):
        return True
    if v2 in ("0", "false", "no", "off"):
        return False
    logger.warning("ENV %s=%r is not a valid bool, using default %s", name, v, default)
    return default


def env_path(name: str, base_dir: Path) -> Path:
    """~ / 상대 경로를 base_dir 기준으로 안전하게 resolve한다."""
    raw = env_str(name)
    expanded = os.path.expanduser(raw)
    p = Path(expanded)
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p
