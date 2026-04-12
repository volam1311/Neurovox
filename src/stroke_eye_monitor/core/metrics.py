from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


def _lm_xy(landmarks: Sequence, h: int, w: int, idx: int) -> tuple[float, float]:
    lm = landmarks[idx]
    x = float(lm.x if lm.x is not None else 0.0)
    y = float(lm.y if lm.y is not None else 0.0)
    return x * w, y * h


def _dist(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


_LEFT_EAR_IDX = (362, 385, 387, 263, 373, 380)
_RIGHT_EAR_IDX = (33, 160, 158, 133, 153, 144)


def eye_aspect_ratio(
    landmarks: Sequence, h: int, w: int, indices: tuple[int, ...]
) -> float:
    p = [_lm_xy(landmarks, h, w, i) for i in indices]
    v1 = _dist(p[1], p[5])
    v2 = _dist(p[2], p[4])
    horiz = _dist(p[0], p[3])
    if horiz < 1e-6:
        return 0.0
    return (v1 + v2) / (2.0 * horiz)


def iris_offset(
    landmarks: Sequence,
    h: int,
    w: int,
    center_idx: int,
    corner_indices: tuple[int, int],
) -> tuple[float, float]:
    """Calculate iris (x,y) offset normalized by eye width."""
    c = _lm_xy(landmarks, h, w, center_idx)
    p0 = _lm_xy(landmarks, h, w, corner_indices[0])
    p1 = _lm_xy(landmarks, h, w, corner_indices[1])

    eye_w = _dist(p0, p1)
    if eye_w < 1e-6:
        return 0.0, 0.0

    # Center of the two corners
    mid_x = (p0[0] + p1[0]) / 2.0
    mid_y = (p0[1] + p1[1]) / 2.0

    # Offset from center normalized by width
    dx = (c[0] - mid_x) / eye_w
    dy = (c[1] - mid_y) / eye_w
    return dx, dy


@dataclass
class EyeMetrics:
    left_ear: float
    right_ear: float
    ear_asymmetry: float
    left_iris_offset: tuple[float, float]
    right_iris_offset: tuple[float, float]


def compute_eye_metrics(landmarks: Sequence, h: int, w: int) -> EyeMetrics:
    l_ear = eye_aspect_ratio(landmarks, h, w, _LEFT_EAR_IDX)
    r_ear = eye_aspect_ratio(landmarks, h, w, _RIGHT_EAR_IDX)

    # MediaPipe iris landmarks: Left=468, Right=473
    l_iris = iris_offset(landmarks, h, w, 468, (362, 263))
    r_iris = iris_offset(landmarks, h, w, 473, (33, 133))

    return EyeMetrics(
        left_ear=l_ear,
        right_ear=r_ear,
        ear_asymmetry=abs(l_ear - r_ear),
        left_iris_offset=l_iris,
        right_iris_offset=r_iris,
    )


def smooth_exponential(current: float, target: float, alpha: float) -> float:
    return alpha * target + (1.0 - alpha) * current


def smooth_vec2(
    current: tuple[float, float] | None, target: tuple[float, float], alpha: float
) -> tuple[float, float]:
    if current is None:
        return target
    return (
        alpha * target[0] + (1.0 - alpha) * current[0],
        alpha * target[1] + (1.0 - alpha) * current[1],
    )


def gaze_feature_vector(
    m: EyeMetrics,
    face_matrix: np.ndarray | None,
) -> np.ndarray:
    """17D vector: [l_iris_x, l_iris_y, r_iris_x, r_iris_y, head_rot_x, head_rot_y, head_rot_z, sin(pitch), cos(pitch), sin(yaw), cos(yaw), sin(roll), cos(roll), head_tx, head_ty, head_tz, 1.0].

    If MediaPipe does not return a face transform matrix, head transformation is filled with
    zeros so gaze regression still runs (quality is reduced until the matrix is back).
    """
    if face_matrix is None:
        rv = np.zeros(3, dtype=np.float64)
        tv = np.zeros(3, dtype=np.float64)
    else:
        rmat = face_matrix[:3, :3]
        rvec, _ = cv2.Rodrigues(rmat)
        rv = rvec.reshape(3)
        tv = face_matrix[:3, 3]

    return np.array(
        [
            m.left_iris_offset[0],
            m.left_iris_offset[1],
            m.right_iris_offset[0],
            m.right_iris_offset[1],
            float(rv[0]),
            float(rv[1]),
            float(rv[2]),
            float(math.sin(float(rv[0]))),
            float(math.cos(float(rv[0]))),
            float(math.sin(float(rv[1]))),
            float(math.cos(float(rv[1]))),
            float(math.sin(float(rv[2]))),
            float(math.cos(float(rv[2]))),
            float(tv[0]),
            float(tv[1]),
            float(tv[2]),
            1.0,
        ],
        dtype=np.float64,
    )


class BlinkDetector:
    """Detect blinks by tracking EAR state transitions with hysteresis (Schmitt Trigger).

    Returns True from feed() once per blink, on the rising edge when the eye reopens
    above open_threshold after dropping below close_threshold. This allows robust
    detection of rapid "flutters" where the eye doesn't fully open between blinks.
    """

    def __init__(
        self,
        close_threshold: float = 0.12,
        open_threshold: float = 0.16,
        min_closed_frames: int = 1,
        cooldown_frames: int = 4,
    ) -> None:
        self._close_thresh = close_threshold
        self._open_thresh = open_threshold
        self._min_closed = min_closed_frames
        self._cooldown = cooldown_frames

        self._closed_count = 0
        self._cooldown_remaining = 0
        self._is_closed = False

    def feed(self, avg_ear: float) -> bool:
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        if self._is_closed:
            # Eye is currently considered closed. Wait for it to open.
            if avg_ear >= self._open_thresh:
                self._is_closed = False

                blink_triggered = False
                if (
                    self._closed_count >= self._min_closed
                    and self._cooldown_remaining == 0
                ):
                    blink_triggered = True
                    self._cooldown_remaining = self._cooldown
                    print(">>> BLINK DETECTED! <<<")

                self._closed_count = 0
                return blink_triggered
            else:
                self._closed_count += 1
        else:
            # Eye is currently considered open. Wait for it to close.
            if avg_ear <= self._close_thresh:
                self._is_closed = True
                self._closed_count = 1

        return False
