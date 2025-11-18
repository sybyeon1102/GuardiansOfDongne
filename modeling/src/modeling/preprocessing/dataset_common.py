"""
공통 전처리 유틸과 데이터셋 구축 파이프라인을 제공합니다.

Provides shared preprocessing utilities and dataset building pipeline
for both anomaly and normal behavior datasets.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import os
import argparse
import json
import time

import cv2
import mediapipe as mp
import numpy as np

from project_core import Err, Ok, Result


# --------------------------
# 타입 정의 / 설정
# --------------------------


type Intervals = dict[str, list[tuple[int, int]]]
type BuildFn = Callable[[DatasetConfig], Result[DatasetArrays, str]]
"""
이벤트별 (start_frame, end_frame) 구간들의 모음입니다.

Per-event list of (start_frame, end_frame) intervals.
"""


@dataclass(slots=True)
class DatasetConfig:
    """
    데이터셋 구축에 필요한 공통 설정값입니다.

    Common configuration values used to build a dataset.
    """

    video_root: Path
    xml_root: Path
    out_dir: Path
    events: list[str]

    window_size: int = 16
    stride: int = 4
    overlap: float = 0.25

    resize_w: int = 640
    model_complexity: int = 0


type DatasetArrays = tuple[np.ndarray, np.ndarray, dict[str, object]]
"""
(X, Y, meta) 배열 묶음을 나타냅니다.

Tuple of (X, Y, meta) produced by dataset builders.
"""

type ParseIntervalsFunc = Callable[[Path], tuple[int, int, Intervals]]
"""
XML에서 (start_frame, end_frame, intervals)를 추출하는 함수 타입입니다.

Function type that parses a CVAT XML file into
(start_frame, end_frame, intervals).
"""


# --------------------------
# 포즈 추출 / 피처 구성
# --------------------------


def extract_pose_video(
    video_path: Path,
    *,
    resize_w: int,
    model_complexity: int,
) -> np.ndarray:
    """
    영상에서 MediaPipe Pose를 이용해 프레임별 포즈 키포인트를 추출합니다.

    Extract per-frame pose keypoints from a video using MediaPipe Pose.

    반환:
    - shape: (T, 33, 4)  (x, y, z, visibility)
    """
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] cannot open video: {video_path}")
        pose.close()
        return np.zeros((0, 33, 4), np.float32)

    seq: list[list[list[float]]] = []
    n = 0

    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break

            n += 1

            if resize_w:
                scale = resize_w / bgr.shape[1]
                bgr = cv2.resize(bgr, (resize_w, int(bgr.shape[0] * scale)))

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if res and res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                seq.append([[p.x, p.y, p.z, p.visibility] for p in lm])
            else:
                # 포즈 감지 실패 시 NaN으로 채워둔다.
                seq.append([[np.nan] * 4] * 33)

            if n % 100 == 0:
                print(f"   [pose] {n} frames")
    finally:
        cap.release()
        pose.close()

    return np.asarray(seq, dtype=np.float32)


def ffill_bfill(arr: np.ndarray) -> np.ndarray:
    """
    NaN을 앞/뒤 방향으로 채운 뒤, 남은 NaN/inf는 0으로 채웁니다.

    Forward/backward fill NaNs and replace remaining NaN/inf with zeros.
    """
    T, D = arr.shape
    out = arr.copy()

    last = np.zeros(D, np.float32)
    has = np.zeros(D, bool)

    # forward fill
    for t in range(T):
        nz = ~np.isnan(out[t])
        last[nz] = out[t, nz]
        has |= nz
        miss = np.isnan(out[t]) & has
        out[t, miss] = last[miss]

    # backward fill
    last[:] = 0
    has[:] = False

    for t in range(T - 1, -1, -1):
        nz = ~np.isnan(out[t])
        last[nz] = out[t, nz]
        has |= nz
        miss = np.isnan(out[t]) & has
        out[t, miss] = last[miss]

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def build_features(kpts: np.ndarray) -> np.ndarray:
    """
    (T, 33, 4) 형태의 키포인트 배열에서 LSTM 입력용 피처를 구성합니다.

    Build LSTM input features from pose keypoints (T, 33, 4).
    """
    if kpts.size == 0:
        return np.zeros((0, 169), np.float32)

    T = kpts.shape[0]

    xy = kpts[:, :, :2].reshape(T, -1)
    vis = kpts[:, :, 3:4].reshape(T, -1)

    xy = ffill_bfill(xy).reshape(T, 33, 2)
    vis = ffill_bfill(vis).reshape(T, 33, 1)

    hip = np.mean(xy[:, [23, 24], :], axis=1)
    sh = np.mean(xy[:, [11, 12], :], axis=1)
    sc = np.linalg.norm(sh - hip, axis=1, keepdims=True)
    sc[sc < 1e-3] = 1.0

    xy_n = (xy - hip[:, None, :]) / sc[:, None, :]
    vel = np.diff(xy_n, axis=0, prepend=xy_n[:1])

    def ang(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        v1 = a - b
        v2 = c - b
        n1 = np.linalg.norm(v1, axis=-1)
        n2 = np.linalg.norm(v2, axis=-1)
        n1[n1 == 0] = 1e-6
        n2[n2 == 0] = 1e-6
        cos = (v1 * v2).sum(-1) / (n1 * n2)
        return np.arccos(np.clip(cos, -1.0, 1.0))

    def pick(i: int) -> np.ndarray:
        return xy_n[:, i, :]

    angs = np.stack(
        [
            ang(pick(11), pick(13), pick(15)),
            ang(pick(12), pick(14), pick(16)),
            ang(pick(23), pick(25), pick(27)),
            ang(pick(24), pick(26), pick(28)),
        ],
        axis=1,
    )

    feat = np.concatenate(
        [xy_n.reshape(T, -1), vel.reshape(T, -1), angs, vis.reshape(T, -1)],
        axis=1,
    )

    return np.clip(feat.astype(np.float32), -10.0, 10.0)


def slice_windows_multilabel(
    feat: np.ndarray,
    *,
    start_f: int,
    stop_f: int,
    intervals: Intervals,
    events: list[str],
    window_size: int,
    stride: int,
    overlap: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    프레임 단위 피처를 슬라이딩 윈도우로 잘라 멀티라벨 Y를 생성합니다.

    Slice frame-wise features into sliding windows and build multilabel Y.
    """
    num_frames, feat_dim = feat.shape
    num_classes = len(events)

    if num_frames == 0:
        return (
            np.zeros((0, window_size, feat_dim), np.float32),
            np.zeros((0, num_classes), np.float32),
        )

    def ov(a0: int, a1: int, b0: int, b1: int) -> int:
        return max(0, min(a1, b1) - max(a0, b0) + 1)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for s in range(start_f, stop_f - window_size + 2, stride):
        e = s + window_size - 1

        y = np.zeros(num_classes, np.float32)

        for i, ev in enumerate(events):
            for (a, b) in intervals.get(ev, []):
                if ov(s, e, a, b) / window_size >= overlap:
                    y[i] = 1.0
                    break

        xs.append(feat[s : e + 1])
        ys.append(y)

    if not xs:
        return (
            np.zeros((0, window_size, feat_dim), np.float32),
            np.zeros((0, num_classes), np.float32),
        )

    return np.stack(xs, axis=0), np.stack(ys, axis=0)


# --------------------------
# 데이터셋 통합 / 정규화 / 저장
# --------------------------


def build_dataset_multilabel(
    config: DatasetConfig,
    *,
    parse_intervals: ParseIntervalsFunc,
) -> Result[DatasetArrays, str]:
    """
    공통 파이프라인으로 멀티라벨 LSTM용 데이터셋을 구축합니다.

    Build a multilabel dataset for LSTM using the common preprocessing pipeline.
    """
    try:
        video_root = config.video_root
        xml_root = config.xml_root

        if not video_root.is_dir():
            return Err(f"video_root not found: {video_root}")
        if not xml_root.is_dir():
            return Err(f"xml_root not found: {xml_root}")

        all_X: list[np.ndarray] = []
        all_Y: list[np.ndarray] = []

        for sub in sorted(os.listdir(video_root)):
            vdir = video_root / sub
            if not vdir.is_dir():
                continue

            v_list = sorted(os.listdir(vdir))

            for fname in v_list:
                if not fname.lower().endswith(".mp4"):
                    continue

                vpath = vdir / fname
                xml_path = xml_root / sub / fname.replace(".mp4", ".xml")

                if not xml_path.is_file():
                    print(f"[SKIP] {sub} :: {fname} → xml 없음")
                    continue

                print(f"[PROC] {sub} :: {fname}")

                s, e, intervals = parse_intervals(xml_path)

                kpts = extract_pose_video(
                    vpath,
                    resize_w=config.resize_w,
                    model_complexity=config.model_complexity,
                )
                feat = build_features(kpts)

                X, Y = slice_windows_multilabel(
                    feat,
                    start_f=s,
                    stop_f=e,
                    intervals=intervals,
                    events=config.events,
                    window_size=config.window_size,
                    stride=config.stride,
                    overlap=config.overlap,
                )

                if X.shape[0] == 0:
                    continue

                all_X.append(X)
                all_Y.append(Y)

        if not all_X:
            return Err("no windows generated; please check video/xml paths and config")

        X = np.concatenate(all_X, axis=0)
        Y = np.concatenate(all_Y, axis=0)

        print(f"[INFO] 통합 데이터: X={X.shape}, Y={Y.shape}")

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        ds_mean = X.mean(axis=(0, 1), keepdims=True)
        ds_std = X.std(axis=(0, 1), keepdims=True) + 1e-6
        Xn = np.clip((X - ds_mean) / ds_std, -6.0, 6.0)

        meta: dict[str, object] = {
            "events": config.events,
            "win": config.window_size,
            "stride": config.stride,
            "overlap": config.overlap,
            "norm_mean": ds_mean.squeeze().tolist(),
            "norm_std": ds_std.squeeze().tolist(),
            "resize_w": config.resize_w,
            "model_complexity": config.model_complexity,
        }

        return Ok((Xn.astype(np.float32), Y.astype(np.float32), meta))
    except Exception as exc:
        return Err(f"failed to build dataset: {exc}")


def run_dataset_cli(
    *,
    description: str,
    default_events: list[str],
    build_fn: BuildFn,
    dataset_name: str,
) -> None:
    """
    DatasetConfig + build_fn을 이용해 CLI에서 데이터셋을 생성하는 헬퍼입니다.

    Helper to build a dataset from CLI using a DatasetConfig and build_fn.
    """
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--video-root",
        type=Path,
        required=True,
        help="영상(.mp4)들이 들어 있는 루트 디렉터리 "
             "(root directory containing input videos)",
    )
    parser.add_argument(
        "--xml-root",
        type=Path,
        required=True,
        help="CVAT XML 어노테이션 파일들이 들어 있는 루트 디렉터리 "
             "(root directory containing CVAT XML annotations)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="생성된 X.npy / Y.npy / meta.json을 저장할 디렉터리 "
             "(output directory for X.npy / Y.npy / meta.json)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=16,
        help="슬라이딩 윈도우 길이 (프레임 수, 기본값: 16) "
             "(sliding window length in frames, default: 16)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help="슬라이딩 윈도우 이동 간격 (프레임 수, 기본값: 4) "
             "(sliding window stride in frames, default: 4)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.25,
        help="윈도우와 이벤트 구간의 최소 겹침 비율 (0.0~1.0, 기본값: 0.25) "
             "(minimum overlap ratio between window and event interval, default: 0.25)",
    )
    parser.add_argument(
        "--resize-w",
        type=int,
        default=640,
        help="영상 가로 리사이즈 크기 (0이면 리사이즈하지 않음, 기본값: 640) "
             "(target video width, 0 means no resize, default: 640)",
    )
    parser.add_argument(
        "--model-complexity",
        type=int,
        default=0,
        help="MediaPipe Pose model_complexity 값 (0, 1, 2 중 하나, 기본값: 0) "
             "(MediaPipe Pose model_complexity: 0, 1 or 2, default: 0)",
    )

    args = parser.parse_args()
    start = time.time()

    config = DatasetConfig(
        video_root=args.video_root,
        xml_root=args.xml_root,
        out_dir=args.out_dir,
        events=default_events,
        window_size=args.window_size,
        stride=args.stride,
        overlap=args.overlap,
        resize_w=args.resize_w,
        model_complexity=args.model_complexity,
    )

    print(
        f"[INFO] building {dataset_name} dataset\n"
        f"       video_root={config.video_root}\n"
        f"       xml_root  ={config.xml_root}\n"
        f"       out_dir   ={config.out_dir}\n"
        f"       events    ={default_events}\n"
        f"       win/stride/overlap = "
        f"{config.window_size}/{config.stride}/{config.overlap}"
    )

    result = build_fn(config)

    if isinstance(result, Err):
        print(f"[ERROR] {result.error}")
        raise SystemExit(1)

    X, Y, meta = result.value

    args.out_dir.mkdir(parents=True, exist_ok=True)

    np.save(args.out_dir / "X.npy", X)
    np.save(args.out_dir / "Y.npy", Y)
    (args.out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    elapsed = time.time() - start
    print(f"[OK] {dataset_name} dataset saved to {args.out_dir}")
    print(f"     X: {X.shape}, Y: {Y.shape}")
    print(f"[TIME] {elapsed:.2f} seconds")
