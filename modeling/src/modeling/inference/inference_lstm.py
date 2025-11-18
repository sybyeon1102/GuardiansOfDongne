"""
LSTM 기반 이상/정상 행동 추론(inference) 레이어를 제공합니다.

Provides LSTM-based inference utilities for normal/anomaly behavior
classification, using checkpoints produced by the training module.
"""

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, Sequence
import json
import sys

import numpy as np
import torch

from project_core import Err, Ok, Result
from modeling.training.trainer_lstm import LstmAnomalyModel


class LstmInferenceErrorCode(str, Enum):
    """
    LSTM 추론 레이어 전용 에러 코드입니다.

    Error codes specific to the LSTM inference layer.
    """

    INVALID_INPUT = "invalid_input"
    CHECKPOINT_NOT_FOUND = "checkpoint_not_found"
    CHECKPOINT_LOAD_FAILED = "checkpoint_load_failed"
    MODEL_BUILD_FAILED = "model_build_failed"
    INFERENCE_FAILED = "inference_failed"


@dataclass(slots=True)
class LstmInferenceError:
    """
    LSTM 추론 레이어 도메인 에러 표현입니다.

    Domain error representation for the LSTM inference layer.
    """

    code: LstmInferenceErrorCode
    message: str


type LstmInferenceResult[T] = Result[T, LstmInferenceError]


@dataclass(slots=True)
class LstmInferenceConfig:
    """
    LSTM 추론에 필요한 설정값입니다.

    Configuration for LSTM inference.
    """

    checkpoint_path: Path
    """
    학습이 완료된 LSTM 체크포인트 파일(.pt) 경로입니다.

    Path to the trained LSTM checkpoint file (.pt).
    """

    device: Literal["auto", "cpu", "cuda"] = "auto"
    """
    추론에 사용할 디바이스 선호도입니다.

    Device preference for inference.
    """


@dataclass(slots=True)
class LstmInferenceContext:
    """
    로드된 LSTM 모델과 전처리 메타 정보를 담는 컨텍스트입니다.

    Context holding the loaded LSTM model and preprocessing metadata.
    """

    model: torch.nn.Module
    device: torch.device
    events: list[str]
    thresholds: list[float]
    norm_mean: np.ndarray  # shape: (1, 1, F)
    norm_std: np.ndarray   # shape: (1, 1, F)
    window_size: int


@dataclass(slots=True)
class LstmInferenceOutput:
    """
    LSTM 기반 이상행동 추론 결과입니다.

    LSTM-based behavior anomaly inference result.
    """

    is_anomaly: bool
    """
    이상 행동으로 판단되면 True, 아니면 False입니다.

    True if the sequence is considered anomalous, otherwise False.
    """

    normal_score: float
    """
    정상일 가능성을 나타내는 점수(0~1 범위 가이드)입니다.

    A score (roughly 0~1) indicating how likely the sequence is normal.
    """

    anomaly_score: float
    """
    이상일 가능성을 나타내는 점수(0~1 범위 가이드)입니다.

    A score (roughly 0~1) indicating how likely the sequence is anomalous.
    """

    scores: list[float]
    """
    각 이벤트(클래스) 별 sigmoid 확률 값입니다.

    Per-event sigmoid scores for each class.
    """

    thresholds: list[float]
    """
    각 이벤트(클래스) 별 의사결정 임계값입니다.

    Per-event decision thresholds.
    """

    events: list[str]
    """
    클래스(이벤트) 이름 리스트입니다.

    Names of the classes (events), aligned with `scores` and `thresholds`.
    """


def _select_inference_device(preference: Literal["auto", "cpu", "cuda"]) -> str:
    """
    선호 옵션에 따라 추론에 사용할 디바이스 문자열을 선택합니다.

    Select device string ("cpu" or "cuda") from the preference.
    """
    if preference == "cpu":
        return "cpu"
    if preference == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    # auto
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model_lstm(config: LstmInferenceConfig) -> LstmInferenceResult[LstmInferenceContext]:
    """
    체크포인트에서 LSTM 모델과 메타 정보를 로드합니다.

    Load the LSTM model and metadata from a checkpoint file.
    """
    ckpt_path = config.checkpoint_path

    if not ckpt_path.is_file():
        return Err(
            LstmInferenceError(
                code=LstmInferenceErrorCode.CHECKPOINT_NOT_FOUND,
                message=f"checkpoint file not found: {ckpt_path}",
            )
        )

    device_str = _select_inference_device(config.device)
    device = torch.device(device_str)

    try:
        # map_location 에서 바로 디바이스로 로딩
        # Load checkpoint to the selected device.
        checkpoint = torch.load(ckpt_path, map_location=device)
    except Exception as exc:  # noqa: BLE001 - 도메인 에러로 변환
        return Err(
            LstmInferenceError(
                code=LstmInferenceErrorCode.CHECKPOINT_LOAD_FAILED,
                message=f"failed to load checkpoint: {exc}",
            )
        )

    if "state_dict" not in checkpoint or "meta" not in checkpoint:
        return Err(
            LstmInferenceError(
                code=LstmInferenceErrorCode.MODEL_BUILD_FAILED,
                message="invalid checkpoint format: 'state_dict' or 'meta' missing",
            )
        )

    state_dict = checkpoint["state_dict"]
    meta = checkpoint["meta"]
    thresholds_raw = checkpoint.get("thresholds")

    # meta.json 구조는 dataset_common.py 의 build_dataset 에서 생성한 형식을 따른다.
    # The meta structure follows build_dataset in dataset_common.py.
    try:
        events = list(meta.get("events", []))
        window_size = int(meta.get("win", 0))

        norm_mean = np.asarray(meta["norm_mean"], dtype=np.float32).reshape(1, 1, -1)
        norm_std = np.asarray(meta["norm_std"], dtype=np.float32).reshape(1, 1, -1)
    except Exception as exc:  # noqa: BLE001
        return Err(
            LstmInferenceError(
                code=LstmInferenceErrorCode.MODEL_BUILD_FAILED,
                message=f"invalid meta structure in checkpoint: {exc}",
            )
        )

    feat_dim = int(norm_mean.shape[-1])

    # 클래스 수는 events 길이를 우선으로 하고, 없으면 thresholds 길이를 사용
    # Prefer length of events; fallback to length of thresholds.
    if events:
        num_classes = len(events)
    elif thresholds_raw is not None:
        num_classes = len(thresholds_raw)
        events = [f"class_{i}" for i in range(num_classes)]
    else:
        # state_dict 기반의 안전한 fallback (head.weight: (C, D))
        # Safe fallback from state_dict (head.weight: (C, D)).
        head_weight = state_dict.get("head.weight")
        if head_weight is None:
            return Err(
                LstmInferenceError(
                    code=LstmInferenceErrorCode.MODEL_BUILD_FAILED,
                    message="cannot infer number of classes (no events, thresholds, or head.weight)",
                )
            )
        num_classes = int(head_weight.shape[0])
        events = [f"class_{i}" for i in range(num_classes)]

    if thresholds_raw is None:
        thresholds = [0.5] * num_classes
    else:
        thresholds = [float(t) for t in thresholds_raw]
        if len(thresholds) != num_classes:
            # 길이가 맞지 않는 경우에는 잘라내거나 패딩해서 맞춘다.
            # If lengths mismatch, trim or pad with 0.5 to match.
            if len(thresholds) > num_classes:
                thresholds = thresholds[:num_classes]
            else:
                thresholds = thresholds + [0.5] * (num_classes - len(thresholds))

    try:
        model = LstmAnomalyModel(
            feat_dim=feat_dim,
            num_out=num_classes,
        )
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
    except Exception as exc:  # noqa: BLE001
        return Err(
            LstmInferenceError(
                code=LstmInferenceErrorCode.MODEL_BUILD_FAILED,
                message=f"failed to build LSTM model from checkpoint: {exc}",
            )
        )

    context = LstmInferenceContext(
        model=model,
        device=device,
        events=events,
        thresholds=thresholds,
        norm_mean=norm_mean,
        norm_std=norm_std,
        window_size=window_size,
    )
    return Ok(context)


def run_inference_lstm_single(
    context: LstmInferenceContext,
    frames: Sequence[Sequence[float]],
) -> LstmInferenceResult[LstmInferenceOutput]:
    """
    단일 시퀀스(윈도우)에 대해 LSTM 추론을 수행합니다.

    Run LSTM inference on a single sequence (one window).
    """
    # 입력을 (T, F) numpy 배열로 정규화
    # Normalize input to a (T, F) numpy array.
    try:
        x = np.asarray(frames, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        return Err(
            LstmInferenceError(
                code=LstmInferenceErrorCode.INVALID_INPUT,
                message=f"failed to convert input frames to numpy array: {exc}",
            )
        )

    if x.ndim != 2:
        return Err(
            LstmInferenceError(
                code=LstmInferenceErrorCode.INVALID_INPUT,
                message=f"expected 2D array (T, F), got shape {x.shape}",
            )
        )

    t_len, feat_dim = x.shape

    if context.window_size > 0 and t_len != context.window_size:
        return Err(
            LstmInferenceError(
                code=LstmInferenceErrorCode.INVALID_INPUT,
                message=f"expected window_size={context.window_size}, got T={t_len}",
            )
        )

    expected_feat_dim = int(context.norm_mean.shape[-1])
    if feat_dim != expected_feat_dim:
        return Err(
            LstmInferenceError(
                code=LstmInferenceErrorCode.INVALID_INPUT,
                message=f"expected feature dimension {expected_feat_dim}, got {feat_dim}",
            )
        )

    # meta에서 저장된 정규화 파라미터를 이용해 정규화
    # Normalize using mean/std stored in meta.
    mean = context.norm_mean.reshape(1, 1, expected_feat_dim)
    std = context.norm_std.reshape(1, 1, expected_feat_dim)

    xn = (x.reshape(1, t_len, feat_dim) - mean) / std
    xn = np.clip(xn, -6.0, 6.0)

    try:
        with torch.no_grad():
            tensor = torch.from_numpy(xn).to(context.device)
            logits = context.model(tensor)  # (1, C)
            probs = torch.sigmoid(logits).cpu().numpy()[0]  # (C,)
    except Exception as exc:  # noqa: BLE001
        return Err(
            LstmInferenceError(
                code=LstmInferenceErrorCode.INFERENCE_FAILED,
                message=f"failed to run LSTM inference: {exc}",
            )
        )

    scores = probs.tolist()

    thr = np.asarray(context.thresholds, dtype=np.float32)
    if thr.shape[0] != probs.shape[0]:
        # 길이가 다르면 맞춰준다 (최소 길이 기준).
        # If lengths differ, align to the minimum length.
        n = min(thr.shape[0], probs.shape[0])
        thr = thr[:n]
        probs = probs[:n]
        scores = scores[:n]
        events = context.events[:n]
    else:
        events = list(context.events)

    # 임계값을 넘는 클래스가 하나라도 있으면 이상으로 간주
    # Consider the sequence anomalous if any score is above its threshold.
    above = probs >= thr
    is_anomaly = bool(above.any())

    anomaly_score = float(probs.max())
    normal_score = float(1.0 - anomaly_score)

    output = LstmInferenceOutput(
        is_anomaly=is_anomaly,
        normal_score=normal_score,
        anomaly_score=anomaly_score,
        scores=scores,
        thresholds=thr.tolist(),
        events=events,
    )
    return Ok(output)


# ---------------------------------------------------------------------------
# CLI helpers / 커맨드 라인 헬퍼
# ---------------------------------------------------------------------------


def _build_arg_parser() -> ArgumentParser:
    """
    LSTM 추론용 커맨드 라인 파서 생성.
    Build argument parser for LSTM inference CLI.
    """
    parser = ArgumentParser(
        description=(
            "LSTM 기반 정상/이상 행동 추론을 단일 시퀀스에 대해 수행합니다.\n"
            "Run LSTM-based normal/anomaly inference on a single sequence."
        )
    )

    parser.add_argument(
        "-c",
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "체크포인트(.pt) 파일 경로.\n"
            "Path to the checkpoint (.pt) file."
        ),
    )

    parser.add_argument(
        "-i",
        "--input-json",
        type=Path,
        required=True,
        help=(
            "입력 프레임이 담긴 JSON 파일 경로. "
            "최상위가 [[...], ...] 또는 {'frames': [[...], ...]} 형식을 가정합니다.\n"
            "Path to input JSON file. Expects [[...], ...] or "
            "{'frames': [[...], ...]} at top level."
        ),
    )

    parser.add_argument(
        "-d",
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="추론 디바이스 선택 (auto/cpu/cuda). / Device for inference.",
    )

    parser.add_argument(
        "--pretty",
        action="store_true",
        help="결과를 보기 좋게 들여쓰기 해서 출력합니다. / Pretty-print JSON output.",
    )

    return parser


def _load_frames_from_json(path: Path) -> LstmInferenceResult[list[list[float]]]:
    """
    JSON 파일에서 프레임 시퀀스를 로드합니다.

    Load frame sequence from JSON file.
    """
    if not path.is_file():
        return Err(
            LstmInferenceError(
                code=LstmInferenceErrorCode.INVALID_INPUT,
                message=f"input JSON file not found: {path}",
            )
        )

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:  # noqa: BLE001
        return Err(
            LstmInferenceError(
                code=LstmInferenceErrorCode.INVALID_INPUT,
                message=f"failed to parse JSON from {path}: {exc}",
            )
        )

    # 두 가지 패턴 지원:
    # 1) 최상위가 바로 [[...], ...]
    # 2) {"frames": [[...], ...]}
    if isinstance(data, dict):
        frames = data.get("frames")
    else:
        frames = data

    if not isinstance(frames, list) or not frames:
        return Err(
            LstmInferenceError(
                code=LstmInferenceErrorCode.INVALID_INPUT,
                message="input JSON must contain a non-empty list of frames",
            )
        )

    # 최소한의 형태 검사: 각 프레임이 리스트이고, 그 안이 수(float/int)인지만 확인.
    # Minimal shape check: each frame is a list of numbers.
    for idx, frame in enumerate(frames):
        if not isinstance(frame, list):
            return Err(
                LstmInferenceError(
                    code=LstmInferenceErrorCode.INVALID_INPUT,
                    message=f"frame[{idx}] is not a list",
                )
            )
        for jdx, value in enumerate(frame):
            if not isinstance(value, (int, float)):
                return Err(
                    LstmInferenceError(
                        code=LstmInferenceErrorCode.INVALID_INPUT,
                        message=f"frame[{idx}][{jdx}] is not a number",
                    )
                )

    return Ok(frames)


def _print_error_and_exit(error: LstmInferenceError, *, prefix: str) -> "None":
    """
    에러를 stderr로 출력하고 프로세스를 실패 코드로 종료합니다.

    Print error to stderr and terminate the process with non-zero exit code.
    """
    sys.stderr.write(f"[{prefix}] {error.code}: {error.message}\n")
    sys.exit(1)


def main() -> None:
    """
    LSTM 추론용 커맨드 라인 엔트리 포인트입니다.

    Command-line entry point for LSTM inference.
    """
    parser = _build_arg_parser()
    args: Namespace = parser.parse_args()

    config = LstmInferenceConfig(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    # 1) 모델 로드 / Load model
    result_ctx = load_model_lstm(config)

    match result_ctx:
        case Ok(value=context):
            pass
        case Err(error=error):
            _print_error_and_exit(error, prefix="LOAD")

        case _:
            sys.stderr.write("[LOAD] unexpected result type from load_model_lstm\n")
            sys.exit(1)

    # 2) 입력 로드 / Load input frames
    result_frames = _load_frames_from_json(args.input_json)

    match result_frames:
        case Ok(value=frames):
            pass
        case Err(error=error):
            _print_error_and_exit(error, prefix="INPUT")

        case _:
            sys.stderr.write("[INPUT] unexpected result type from _load_frames_from_json\n")
            sys.exit(1)

    # 3) 추론 수행 / Run inference
    result_inf = run_inference_lstm_single(context, frames)

    match result_inf:
        case Ok(value=output):
            pass
        case Err(error=error):
            _print_error_and_exit(error, prefix="INFER")

        case _:
            sys.stderr.write("[INFER] unexpected result type from run_inference_lstm_single\n")
            sys.exit(1)

    # 4) 결과를 JSON으로 출력 / Print result as JSON
    result_payload = {
        "is_anomaly": output.is_anomaly,
        "normal_score": output.normal_score,
        "anomaly_score": output.anomaly_score,
        "events": output.events,
        "scores": output.scores,
        "thresholds": output.thresholds,
    }

    if args.pretty:
        text = json.dumps(result_payload, ensure_ascii=False, indent=2)
    else:
        text = json.dumps(result_payload, ensure_ascii=False)

    sys.stdout.write(f"{text}\n")


__all__ = [
    "LstmInferenceErrorCode",
    "LstmInferenceError",
    "LstmInferenceResult",
    "LstmInferenceConfig",
    "LstmInferenceContext",
    "LstmInferenceOutput",
    "load_model_lstm",
    "run_inference_lstm_single",
    "main",
]


if __name__ == "__main__":
    main()
