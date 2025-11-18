from functools import lru_cache

from project_core import Err, Ok

from modeling.inference import (
    LstmInferenceConfig,
    LstmInferenceContext,
    LstmInferenceError,
    LstmInferenceErrorCode,
    LstmInferenceResult,
    load_model_lstm,
    run_inference_lstm_single,
)

from app.config import get_settings
from app.errors import InferenceError, InferenceErrorCode, InferenceResult
from app.models.model_io_behavior import (
    BehaviorAnalyzeRequest,
    BehaviorAnalyzeResponse,
)


@lru_cache
def _get_lstm_context() -> LstmInferenceResult[LstmInferenceContext]:
    """
    설정에서 체크포인트 경로를 읽어 LSTM 추론 컨텍스트를 로드한다(캐시됨).
    Load LSTM inference context from settings (cached).
    """
    settings = get_settings()

    # TODO: Settings 클래스에 model_checkpoint_path 필드를 정의해야 한다.
    # TODO: You must define `model_checkpoint_path` field in Settings.
    checkpoint_path = settings.model_checkpoint_path  # type: ignore[attr-defined]

    config = LstmInferenceConfig(
        checkpoint_path=checkpoint_path,
        device="auto",
    )

    return load_model_lstm(config)


def _map_lstm_error_to_inference_error(
    error: LstmInferenceError,
) -> InferenceError:
    """
    modeling 레이어의 LSTM 에러를 backend InferenceError 로 변환한다.
    Map LstmInferenceError from modeling layer to backend InferenceError.
    """
    match error.code:
        case LstmInferenceErrorCode.INVALID_INPUT:
            code = InferenceErrorCode.INVALID_INPUT
        case (
            LstmInferenceErrorCode.CHECKPOINT_NOT_FOUND
            | LstmInferenceErrorCode.CHECKPOINT_LOAD_FAILED
            | LstmInferenceErrorCode.MODEL_BUILD_FAILED
        ):
            code = InferenceErrorCode.MODEL_NOT_READY
        case LstmInferenceErrorCode.INFERENCE_FAILED:
            code = InferenceErrorCode.INFERENCE_FAILED
        case _:
            code = InferenceErrorCode.INTERNAL_ERROR

    return InferenceError(
        code=code,
        message=error.message,
    )


def _frames_from_request(
    request: BehaviorAnalyzeRequest,
) -> list[list[float]]:
    """
    BehaviorAnalyzeRequest에서 LSTM 입력용 feature 시퀀스를 추출한다.
    Extract feature sequence for LSTM input from BehaviorAnalyzeRequest.
    """
    return [frame.features for frame in request.frames]


def analyze_behavior(
    request: BehaviorAnalyzeRequest,
) -> InferenceResult[BehaviorAnalyzeResponse]:
    """
    LSTM/모델링 레이어를 호출해 이상 행동 여부를 계산한다.
    Call the modeling/inference layer to compute anomaly scores.
    """
    # 1) 모델 컨텍스트 로드 / Load model context
    try:
        result_ctx = _get_lstm_context()
    except Exception as exc:  # noqa: BLE001
        return Err(
            InferenceError(
                code=InferenceErrorCode.INTERNAL_ERROR,
                message=f"Failed to initialize LSTM context: {exc}",
            )
        )

    match result_ctx:
        case Ok(value=context):
            pass
        case Err(error=lstm_error):
            return Err(_map_lstm_error_to_inference_error(lstm_error))
        case _:
            return Err(
                InferenceError(
                    code=InferenceErrorCode.INTERNAL_ERROR,
                    message="Unexpected result type from load_model_lstm.",
                )
            )

    # 2) 요청에서 feature 시퀀스 추출 / Extract feature sequence from request
    frames = _frames_from_request(request)
    if not frames:
        return Err(
            InferenceError(
                code=InferenceErrorCode.INVALID_INPUT,
                message="frames must not be empty.",
            )
        )

    # 3) LSTM 추론 수행 / Run LSTM inference
    try:
        result_inf = run_inference_lstm_single(context, frames)
    except Exception as exc:  # noqa: BLE001
        return Err(
            InferenceError(
                code=InferenceErrorCode.INTERNAL_ERROR,
                message=f"Unexpected exception during inference: {exc}",
            )
        )

    match result_inf:
        case Ok(value=output):
            pass
        case Err(error=lstm_error):
            return Err(_map_lstm_error_to_inference_error(lstm_error))
        case _:
            return Err(
                InferenceError(
                    code=InferenceErrorCode.INTERNAL_ERROR,
                    message="Unexpected result type from run_inference_lstm_single.",
                )
            )

    # 4) 도메인 응답 모델로 매핑 / Map LSTM output to API response model
    response = BehaviorAnalyzeResponse(
        is_anomaly=output.is_anomaly,
        normal_score=output.normal_score,
        anomaly_score=output.anomaly_score,
        # 아래 필드들은 modeling 레이어의 LstmInferenceOutput에서 그대로 전달합니다.
        # These fields are passed through from LstmInferenceOutput in the modeling layer.
        events=output.events,
        scores=output.scores,
        thresholds=output.thresholds,
    )

    return Ok(response)
