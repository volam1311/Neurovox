from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from mediapipe.tasks.python.components.containers import landmark as landmark_lib
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.vision import face_landmarker as fl_module
from mediapipe.tasks.python.vision.core import image as mp_image_module
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module

from stroke_eye_monitor.config import MonitorConfig
from stroke_eye_monitor.core.assets import ensure_face_landmarker_model

NormalizedLandmark = landmark_lib.NormalizedLandmark


@dataclass
class FrameResult:
    landmarks: Optional[List[NormalizedLandmark]]
    process_ms: float
    image_shape: tuple[int, int]
    # 4x4 row-major: canonical face → runtime face (when enabled in FaceLandmarker).
    face_matrix: Optional[np.ndarray] = None


class FaceMeshEyeDetector:
    """Face mesh + iris landmarks via MediaPipe Tasks FaceLandmarker (video mode)."""

    def __init__(self, config: MonitorConfig, model_path: str | None = None) -> None:
        if model_path:
            p = Path(model_path)
            if not p.is_file():
                raise FileNotFoundError(f"Face landmarker model not found: {p}")
        else:
            p = ensure_face_landmarker_model()

        opts = fl_module.FaceLandmarkerOptions(
            base_options=base_options_module.BaseOptions(model_asset_path=str(p.resolve())),
            running_mode=running_mode_module.VisionTaskRunningMode.VIDEO,
            num_faces=config.max_num_faces,
            min_face_detection_confidence=config.min_detection_confidence,
            min_face_presence_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = fl_module.FaceLandmarker.create_from_options(opts)
        self._t0 = time.perf_counter()
        self._last_ts_ms = 0

    def close(self) -> None:
        self._landmarker.close()

    def process_bgr(self, bgr) -> FrameResult:
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        mp_image = mp_image_module.Image(mp_image_module.ImageFormat.SRGB, rgb)

        ts = int((time.perf_counter() - self._t0) * 1000)
        if ts <= self._last_ts_ms:
            ts = self._last_ts_ms + 1
        self._last_ts_ms = ts

        t0 = time.perf_counter()
        result = self._landmarker.detect_for_video(mp_image, ts)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        lm: Optional[List[NormalizedLandmark]] = None
        fm: Optional[np.ndarray] = None
        if result.face_landmarks:
            lm = result.face_landmarks[0]
            if result.facial_transformation_matrixes:
                raw = result.facial_transformation_matrixes[0]
                if raw is not None and np.asarray(raw).size >= 16:
                    fm = np.asarray(raw, dtype=np.float64).reshape(4, 4)

        return FrameResult(
            landmarks=lm, process_ms=dt_ms, image_shape=(h, w), face_matrix=fm
        )
