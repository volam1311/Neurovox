from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np


def _lm_xy(
    landmarks: Sequence, h: int, w: int, idx: int
) -> tuple[float, float]:
    lm = landmarks[idx]
    x = float(lm.x if lm.x is not None else 0.0)
    y = float(lm.y if lm.y is not None else 0.0)
    return x * w, y * h


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


# Six-point EAR-style ratio for MediaPipe Face Mesh (478 pts with iris refine).
# Pairs: vertical / vertical / horizontal — same layout as common OpenFace/Mediapipe recipes.
_LEFT_EAR_IDX = (362, 385, 387, 263, 373, 380)
_RIGHT_EAR_IDX = (33, 160, 158, 133, 153, 144)


def eye_aspect_ratio(landmarks: Sequence, h: int, w: int, indices: tuple[int, ...]) -> float:
    p = [_lm_xy(landmarks, h, w, i) for i in indices]
    v1 = _dist(p[1], p[5])
    v2 = _dist(p[2], p[4])
    hspan = _dist(p[0], p[3])
    if hspan < 1e-6:
        return 0.0
    return (v1 + v2) / (2.0 * hspan)


def left_ear(landmarks: Sequence, h: int, w: int) -> float:
    return eye_aspect_ratio(landmarks, h, w, _LEFT_EAR_IDX)


def right_ear(landmarks: Sequence, h: int, w: int) -> float:
    return eye_aspect_ratio(landmarks, h, w, _RIGHT_EAR_IDX)


# Iris landmark ranges with refine_landmarks=True (478 landmarks total).
_LEFT_IRIS_IDX = range(468, 473)
_RIGHT_IRIS_IDX = range(473, 478)


def _iris_center(
    landmarks: Sequence, h: int, w: int, indices: Iterable[int]
) -> tuple[float, float] | None:
    pts = [_lm_xy(landmarks, h, w, i) for i in indices]
    if not pts:
        return None
    ax = sum(p[0] for p in pts) / len(pts)
    ay = sum(p[1] for p in pts) / len(pts)
    return ax, ay


def iris_offset_normalized(
    landmarks: Sequence,
    h: int,
    w: int,
    *,
    left_corner: int,
    right_corner: int,
    iris_indices: Iterable[int],
) -> tuple[float, float] | None:
    """Horizontal/vertical offset of iris center within eye span, roughly in [-1, 1]."""
    ic = _iris_center(landmarks, h, w, iris_indices)
    if ic is None:
        return None
    outer = _lm_xy(landmarks, h, w, left_corner)
    inner = _lm_xy(landmarks, h, w, right_corner)
    eye_w = _dist(outer, inner)
    if eye_w < 1e-6:
        return None
    mid_x = (outer[0] + inner[0]) / 2.0
    mid_y = (outer[1] + inner[1]) / 2.0
    nx = (ic[0] - mid_x) / (eye_w / 2.0)
    ny = (ic[1] - mid_y) / (eye_w / 2.0)
    return float(nx), float(ny)


@dataclass(frozen=True)
class EyeMetrics:
    left_ear: float
    right_ear: float
    ear_asymmetry: float
    left_iris_offset: tuple[float, float] | None
    right_iris_offset: tuple[float, float] | None


def compute_eye_metrics(landmarks: Sequence, h: int, w: int) -> EyeMetrics:
    le = left_ear(landmarks, h, w)
    re = right_ear(landmarks, h, w)
    asym = abs(le - re)
    li = iris_offset_normalized(
        landmarks,
        h,
        w,
        left_corner=263,
        right_corner=362,
        iris_indices=_LEFT_IRIS_IDX,
    )
    ri = iris_offset_normalized(
        landmarks,
        h,
        w,
        left_corner=33,
        right_corner=133,
        iris_indices=_RIGHT_IRIS_IDX,
    )
    return EyeMetrics(
        left_ear=le,
        right_ear=re,
        ear_asymmetry=asym,
        left_iris_offset=li,
        right_iris_offset=ri,
    )


def head_pose_rvec(face_matrix: np.ndarray | None) -> np.ndarray:
    """3-vector from the rotation block of MediaPipe's 4×4 face transform (axis–angle via Rodrigues).

    Zeros when the matrix is missing or ill-conditioned.
    """
    if face_matrix is None:
        return np.zeros(3, dtype=np.float64)
    M = np.asarray(face_matrix, dtype=np.float64).reshape(4, 4)
    R = M[:3, :3]
    if not np.all(np.isfinite(R)):
        return np.zeros(3, dtype=np.float64)
    rvec, _ = cv2.Rodrigues(R)
    v = rvec.reshape(3).astype(np.float64)
    if not np.all(np.isfinite(v)):
        return np.zeros(3, dtype=np.float64)
    return v


def gaze_feature_vector(
    m: EyeMetrics, face_matrix: np.ndarray | None = None
) -> np.ndarray | None:
    """
    Feature row for calibrated gaze regression (8-D):
    [Lnx, Lny, Rnx, Rny, r0, r1, r2, 1]

    ``(r0,r1,r2)`` is the Rodrigues vector of the 3×3 rotation from MediaPipe's
    facial transformation matrix (canonical face → runtime face).
    """
    if m.left_iris_offset is None or m.right_iris_offset is None:
        return None
    lx, ly = m.left_iris_offset
    rx, ry = m.right_iris_offset
    h = head_pose_rvec(face_matrix)
    return np.array(
        [lx, ly, rx, ry, float(h[0]), float(h[1]), float(h[2]), 1.0],
        dtype=np.float64,
    )


class BlinkDetector:
    """Detect blinks by tracking EAR state transitions.

    Returns True from feed() once per blink, on the rising edge (eye reopens
    after being closed for at least ``min_closed_frames``).
    """

    def __init__(
        self,
        ear_threshold: float = 0.18,
        min_closed_frames: int = 2,
        cooldown_frames: int = 8,
    ) -> None:
        self._threshold = ear_threshold
        self._min_closed = min_closed_frames
        self._cooldown = cooldown_frames
        self._closed_count = 0
        self._cooldown_remaining = 0

    def feed(self, avg_ear: float) -> bool:
        """Feed the average EAR of both eyes. Returns True once per blink."""
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            if avg_ear >= self._threshold:
                self._closed_count = 0
            return False

        if avg_ear < self._threshold:
            self._closed_count += 1
            return False

        # Eye just reopened
        if self._closed_count >= self._min_closed:
            self._closed_count = 0
            self._cooldown_remaining = self._cooldown
            return True

        self._closed_count = 0
        return False


def smooth_exponential(prev: float, new: float, alpha: float) -> float:
    if math.isnan(new):
        return prev
    return alpha * new + (1.0 - alpha) * prev


def smooth_vec2(
    prev: tuple[float, float] | None,
    new: tuple[float, float] | None,
    alpha: float,
) -> tuple[float, float] | None:
    if new is None:
        return prev
    if prev is None:
        return new
    return (
        alpha * new[0] + (1.0 - alpha) * prev[0],
        alpha * new[1] + (1.0 - alpha) * prev[1],
    )
