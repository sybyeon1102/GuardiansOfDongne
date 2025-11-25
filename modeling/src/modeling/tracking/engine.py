"""Multi-camera tracking engine using YOLO + DeepSORT (+ optional ReID/global ID)."""

from dataclasses import dataclass
import time
from typing import Any

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

from modeling.device import choose_torch_device, is_cuda_device


@dataclass
class BBoxNorm:
    """0~1 스케일의 tlwh 바운딩 박스."""

    x: float
    y: float
    w: float
    h: float


@dataclass
class TrackResult:
    """트래킹 결과 1개.

    - camera_id: 어느 카메라에서 나온 트랙인지
    - global_id: 멀티캠 공통 ID (옵션)
    - local_track_id: 해당 카메라 내 DeepSORT track_id
    """

    camera_id: str
    global_id: str
    local_track_id: int
    label: str
    confidence: float
    bbox: BBoxNorm


@dataclass
class _GlobalTrack:
    global_id: str
    embedding: np.ndarray  # L2 정규화된 feature (D,)
    last_seen_ts: float


class ReIDEmbedder:
    """Torchreid 기반 Re-ID 임베더.

    - enable_reid=True 인 경우에만 실제로 사용된다.
    - 이 클래스가 생성될 때에만 torchreid가 import 된다(지연 import).
    """

    def __init__(
        self,
        device: str,
        model_name: str = "osnet_x1_0",
        model_path: str | None = None,
    ) -> None:
        # 지연 import: enable_reid=True 일 때만 torchreid 의존
        from torchreid.utils import FeatureExtractor  # type: ignore[import]

        kwargs: dict[str, Any] = {
            "model_name": model_name,
            "device": device,
        }
        if model_path is not None:
            kwargs["model_path"] = model_path

        self._extractor = FeatureExtractor(**kwargs)

    def __call__(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        if not crops_bgr:
            return np.zeros((0, 512), dtype=np.float32)

        # FeatureExtractor 는 RGB 이미지를 받으므로 변환
        rgb_images = [crop[:, :, ::-1] for crop in crops_bgr]  # BGR -> RGB
        feats = self._extractor(rgb_images)

        if isinstance(feats, np.ndarray):
            arr = feats.astype("float32")
        else:
            # torch.Tensor 인 경우
            arr = feats.detach().cpu().numpy().astype("float32")

        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        arr = arr / norms
        return arr


class MultiCameraTracker:
    """YOLO + DeepSORT 기반 멀티 카메라 트래커.

    - 기본: YOLO11n + DeepSORT 기본 embedder 로 per-camera tracking
    - enable_reid=True: Torchreid 로 외부 임베딩 생성
    - enable_global_id=True: ReID 임베딩 기반 global_id 계산
      (단, enable_reid=False 이면 global_id 는 camera_id:local_track_id 형태의 단순 값)
    """

    def __init__(
        self,
        yolo_weights: str = "yolo11n.pt",
        conf_thres: float = 0.4,
        iou_thres: float = 0.7,
        max_age: int = 30,
        reid_cosine_thres: float = 0.25,
        reid_model_name: str = "osnet_x1_0",
        reid_model_path: str | None = None,
        enable_reid: bool = False,
        enable_global_id: bool = False,
    ) -> None:
        self.device = choose_torch_device()
        self.enable_reid = enable_reid
        self.enable_global_id = enable_global_id

        # YOLO detector
        self._yolo = YOLO(yolo_weights)
        self._yolo.to(self.device)

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_age = max_age
        self.reid_cosine_thres = reid_cosine_thres

        # 카메라별 DeepSort 인스턴스
        self._trackers: dict[str, DeepSort] = {}

        # ReID 임베더
        if self.enable_reid:
            self._embedder: ReIDEmbedder | None = ReIDEmbedder(
                device=str(self.device),
                model_name=reid_model_name,
                model_path=reid_model_path,
            )
        else:
            self._embedder = None

        # 글로벌 ID 관리
        self._global_tracks: dict[str, _GlobalTrack] = {}
        self._local_to_global: dict[tuple[str, int], str] = {}
        self._next_global_int = 1

    # ------------------------------------------------------------------
    # 내부 유틸
    # ------------------------------------------------------------------

    def _get_tracker(self, camera_id: str) -> DeepSort:
        tracker = self._trackers.get(camera_id)
        if tracker is not None:
            return tracker

        if self.enable_reid:
            # 외부 임베더(ReIDEmbedder)를 사용하므로 embedder=None
            tracker = DeepSort(
                max_age=self.max_age,
                max_iou_distance=self.iou_thres,
                n_init=3,
                nms_max_overlap=1.0,
                embedder=None,
                half=is_cuda_device(self.device),
            )
        else:
            # DeepSORT 기본 embedder 사용 (mobilenet 등)
            tracker = DeepSort(
                max_age=self.max_age,
                max_iou_distance=self.iou_thres,
                n_init=3,
                nms_max_overlap=1.0,
                # embedder 인자 생략 → 기본 embedder 사용
            )

        self._trackers[camera_id] = tracker
        return tracker

    def _assign_global_id(
        self,
        camera_id: str,
        local_track_id: int,
        embedding: np.ndarray | None,
        timestamp: float,
    ) -> str:
        """(camera_id, local_track_id) → global_id 매핑 + 멀티캠 re-id."""
        key = (camera_id, local_track_id)
        existing = self._local_to_global.get(key)

        if existing is not None:
            g = self._global_tracks.get(existing)
            if g is not None and embedding is not None:
                # EMA 업데이트
                alpha = 0.7
                g.embedding = alpha * g.embedding + (1.0 - alpha) * embedding
                g.embedding /= np.linalg.norm(g.embedding) + 1e-12
                g.last_seen_ts = timestamp
            return existing

        if embedding is None or not self._global_tracks:
            global_id = f"G{self._next_global_int}"
            self._next_global_int += 1
            if embedding is None:
                embedding = np.zeros(1, dtype=np.float32)
            self._global_tracks[global_id] = _GlobalTrack(
                global_id=global_id,
                embedding=embedding,
                last_seen_ts=timestamp,
            )
            self._local_to_global[key] = global_id
            return global_id

        # 기존 global track들과 cosine similarity 비교
        best_gid: str | None = None
        best_sim = -1.0
        for gid, gtrack in self._global_tracks.items():
            sim = float(np.dot(gtrack.embedding, embedding))
            if sim > best_sim:
                best_sim = sim
                best_gid = gid

        # (1 - sim) <= threshold → 같은 사람으로 본다.
        if best_gid is not None and (1.0 - best_sim) <= self.reid_cosine_thres:
            gtrack = self._global_tracks[best_gid]
            alpha = 0.7
            gtrack.embedding = alpha * gtrack.embedding + (1.0 - alpha) * embedding
            gtrack.embedding /= np.linalg.norm(gtrack.embedding) + 1e-12
            gtrack.last_seen_ts = timestamp
            self._local_to_global[key] = best_gid
            return best_gid

        # 유사한 global track이 없으면 새 global id 부여
        global_id = f"G{self._next_global_int}"
        self._next_global_int += 1
        self._global_tracks[global_id] = _GlobalTrack(
            global_id=global_id,
            embedding=embedding,
            last_seen_ts=timestamp,
        )
        self._local_to_global[key] = global_id
        return global_id

    # ------------------------------------------------------------------
    # 메인 엔트리 포인트
    # ------------------------------------------------------------------

    def process_frame(
        self,
        camera_id: str,
        frame_bgr: np.ndarray,
        timestamp: float | None = None,
    ) -> list[TrackResult]:
        """한 프레임에 대해 YOLO detection + DeepSORT + (옵션) 멀티캠 ReID 수행."""
        if timestamp is None:
            timestamp = time.time()

        h, w, _ = frame_bgr.shape

        # 1) YOLO로 사람만 검출
        yolo_out = self._yolo(
            frame_bgr,
            conf=self.conf_thres,
            verbose=False,
            device=self.device,
        )
        boxes = yolo_out[0].boxes
        raw_dets: list[tuple[list[float], float, int]] = []

        if boxes is not None and boxes.xywh is not None:
            xywh = boxes.xywh.cpu().numpy()  # (N, 4)
            conf = boxes.conf.cpu().numpy()  # (N,)
            cls = boxes.cls.cpu().numpy().astype(int)  # (N,)

            for (cx, cy, bw, bh), c, cls_id in zip(xywh, conf, cls):
                if c < self.conf_thres:
                    continue
                # COCO 기준 사람 class id == 0
                if cls_id != 0:
                    continue

                x = float(cx - bw / 2.0)
                y = float(cy - bh / 2.0)
                raw_dets.append(
                    ([x, y, float(bw), float(bh)], float(c), int(cls_id)),
                )

        tracker = self._get_tracker(camera_id)

        if not raw_dets:
            tracker.update_tracks([])
            return []

        # 2) DeepSORT 업데이트 (ReID on/off 에 따라 분기)
        embeds: np.ndarray | None = None

        if self.enable_reid and self._embedder is not None:
            # crop + 외부 ReID 임베딩 사용
            crops: list[np.ndarray] = []
            for (x, y, bw, bh), _, _ in raw_dets:
                x0 = max(int(x), 0)
                y0 = max(int(y), 0)
                x1 = min(int(x + bw), w)
                y1 = min(int(y + bh), h)
                if x1 <= x0 or y1 <= y0:
                    crops.append(np.zeros((10, 10, 3), dtype=frame_bgr.dtype))
                else:
                    crops.append(frame_bgr[y0:y1, x0:x1].copy())

            embeds = self._embedder(crops)  # (N, D)
            others = list(range(len(raw_dets)))  # detection index 전달
            tracks = tracker.update_tracks(
                raw_detections=raw_dets,
                embeds=embeds,
                others=others,
            )
        else:
            # DeepSORT 기본 embedder에 frame만 전달
            tracks = tracker.update_tracks(
                raw_detections=raw_dets,
                frame=frame_bgr,
            )

        results: list[TrackResult] = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            local_id = int(track.track_id)

            # 글로벌 ID 결정
            if self.enable_global_id and self.enable_reid and embeds is not None:
                det_index = track.get_det_supplementary()
                if det_index is not None and 0 <= det_index < embeds.shape[0]:
                    emb_vec = embeds[det_index]
                    global_id = self._assign_global_id(
                        camera_id=camera_id,
                        local_track_id=local_id,
                        embedding=emb_vec,
                        timestamp=timestamp,
                    )
                else:
                    # detection index 를 못 찾으면 fallback
                    global_id = f"{camera_id}:{local_id}"
            else:
                # ReID/글로벌 ID 비활성화 → per-camera 기반 단순 ID
                global_id = f"{camera_id}:{local_id}"

            # 바운딩 박스 (가능하면 원본 detection 기준)
            tlwh = track.to_tlwh(orig=True, orig_strict=False)
            if tlwh is None:
                tlwh = track.to_tlwh()
            x, y, bw, bh = tlwh

            bbox_norm = BBoxNorm(
                x=float(x / w),
                y=float(y / h),
                w=float(bw / w),
                h=float(bh / h),
            )

            conf = float(getattr(track, "det_confid", 1.0))

            results.append(
                TrackResult(
                    camera_id=camera_id,
                    global_id=global_id,
                    local_track_id=local_id,
                    label="person",
                    confidence=conf,
                    bbox=bbox_norm,
                ),
            )

        return results
