from dataclasses import dataclass
from enum import Enum

from fastapi import HTTPException, status

from project_core import Result


class InferenceErrorCode(str, Enum):
    """
    모델 추론/행동 분석 과정에서 발생하는 에러 코드.
    Error codes for model inference / behavior analysis.
    """

    INVALID_INPUT = "invalid_input"
    MODEL_NOT_READY = "model_not_ready"
    INFERENCE_FAILED = "inference_failed"
    INTERNAL_ERROR = "internal_error"


@dataclass(slots=True, frozen=True)
class InferenceError:
    """
    모델 추론/행동 분석 도메인 에러 표현.
    Domain error representation for model inference / behavior analysis.
    """

    code: InferenceErrorCode
    message: str


type InferenceResult[T] = Result[T, InferenceError]


def map_inference_error_to_http_exception(error: InferenceError) -> HTTPException:
    """
    InferenceError 를 HTTPException 으로 변환한다.
    Map an InferenceError into an HTTPException.
    """
    match error.code:
        case InferenceErrorCode.INVALID_INPUT:
            return HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error.message,
            )
        case InferenceErrorCode.MODEL_NOT_READY:
            return HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=error.message,
            )
        case InferenceErrorCode.INFERENCE_FAILED:
            return HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error.message,
            )
        case _:
            # INTERNAL_ERROR 또는 알 수 없는 코드
            # INTERNAL_ERROR or unknown error code
            return HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error.message,
            )
