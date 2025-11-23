# pose_extractor.py
"""
Mediapipe Pose 로부터 (33,4) 키포인트를 추출하는 모듈.
Module to extract (33,4) keypoints from Mediapipe Pose results.
"""

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class PoseExtractorConfig:
    """Mediapipe Pose 설정 / Mediapipe Pose configuration."""
    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


class PoseExtractor:
    """
    BGR 프레임을 받아서 Mediapipe Pose 결과와 keypoints 배열을 반환하는 헬퍼.
    Helper that takes BGR frames and returns Mediapipe Pose results and keypoints.
    """

    def __init__(self, config: PoseExtractorConfig | None = None) -> None:
        if config is None:
            config = PoseExtractorConfig()

        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            model_complexity=config.model_complexity,
            enable_segmentation=False,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )

    @staticmethod
    def extract_keypoints(
        results: "mp.solutions.pose.PoseLandmark | None",
        num_landmarks: int = 33,
    ) -> np.ndarray:
        """
        Mediapipe Pose 결과에서 (33,4) 키포인트 배열을 만든다.
        Build (33,4) keypoints array from Mediapipe Pose result.

        각 행은 [x, y, z, visibility].
        Each row is [x, y, z, visibility].
        """
        if results is None or results.pose_landmarks is None:
            return np.zeros((num_landmarks, 4), dtype=np.float32)

        kpts = np.zeros((num_landmarks, 4), dtype=np.float32)

        for i, lm in enumerate(results.pose_landmarks.landmark):
            if i >= num_landmarks:
                break
            kpts[i, 0] = lm.x
            kpts[i, 1] = lm.y
            kpts[i, 2] = lm.z
            kpts[i, 3] = lm.visibility

        return kpts

    def process_frame(self, frame_bgr: "cv2.Mat") -> tuple[np.ndarray, "cv2.Mat"]:
        """
        한 프레임을 처리해 keypoints 와 랜드마크가 그려진 이미지를 반환한다.
        Process one frame and return keypoints and annotated image.

        Returns
        -------
        keypoints : np.ndarray
            (33,4) float32 배열.
        annotated : cv2.Mat
            관절 랜드마크가 그려진 BGR 이미지.
        """
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._pose.process(image_rgb)

        keypoints = self.extract_keypoints(results)

        annotated = frame_bgr.copy()
        if results is not None and results.pose_landmarks is not None:
            mp_drawing = mp.solutions.drawing_utils
            mp_styles = mp.solutions.drawing_styles
            mp_drawing.draw_landmarks(
                annotated,
                results.pose_landmarks,
                self._mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
            )

        return keypoints, annotated

    def close(self) -> None:
        """Mediapipe 리소스를 정리한다 / Close Mediapipe resources."""
        self._pose.close()
