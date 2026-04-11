"""Per-frame outputs shared by all eye / gaze backends (no MediaPipe import here)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FrameResult:
    landmarks: list[Any] | None
    process_ms: float
    image_shape: tuple[int, int]
    # 4x4 row-major: canonical face → runtime face (MediaPipe FaceLandmarker).
    face_matrix: np.ndarray | None = None
