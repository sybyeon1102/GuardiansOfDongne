"""
project_core 패키지.

프로젝트 전반에서 공유되는 코어 유틸리티와 에러 표현(Result 타입 등)을 제공합니다.

The `project_core` package.

Provides core utilities and shared error representations (such as the Result type)
used across the project.
"""

from .result import Ok, Err, Result, is_ok, is_err

__all__ = [
    "Ok",
    "Err",
    "Result",
    "is_ok",
    "is_err",
]
