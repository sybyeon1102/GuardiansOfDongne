from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class Ok[T]:
    """
    성공 결과 값을 담는 래퍼입니다.

    Wrapper type that represents the successful branch of a Result.
    """

    # match Ok(value) 구문에서 위치 인자로 매칭될 필드 정의
    # Define which field is used positionally in `match Ok(value)`
    __match_args__ = ("value",)

    value: T


@dataclass(slots=True, frozen=True)
class Err[E]:
    """
    실패(에러) 정보를 담는 래퍼입니다.

    Wrapper type that represents the error branch of a Result.
    """
    # match Err(error) 구문에서 위치 인자로 매칭될 필드 정의
    # Define which field is used positionally in `match Err(error)`
    __match_args__ = ("error",)

    error: E


type Result[T, E] = Ok[T] | Err[E]
"""
도메인/서비스 계층에서 사용하는 공용 Result 타입입니다.

Generic Result type used as the shared error/value representation
across domain and service boundaries.

- T: 성공 시 반환되는 값의 타입 (success type)
- E: 실패(에러) 시 반환되는 정보의 타입 (error type)
"""


def is_ok[T, E](result: Result[T, E]) -> bool:
    """
    Result가 Ok 인지 여부를 반환합니다.

    Return True if the given Result is an Ok value.
    """
    return isinstance(result, Ok)


def is_err[T, E](result: Result[T, E]) -> bool:
    """
    Result가 Err 인지 여부를 반환합니다.

    Return True if the given Result is an Err value.
    """
    return isinstance(result, Err)


__all__ = [
    "Ok",
    "Err",
    "Result",
    "is_ok",
    "is_err",
]
