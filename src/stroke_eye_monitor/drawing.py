from __future__ import annotations

from enum import Enum
from typing import Any, List, Sequence

import cv2
from mediapipe.tasks.python.components.containers import landmark as landmark_lib
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections

Connection = FaceLandmarksConnections.Connection


class DrawStyle(Enum):
    FULL = "full"
    EYES_ONLY = "eyes_only"


def _norm_pt(
    landmarks: Sequence[landmark_lib.NormalizedLandmark], w: int, h: int, idx: int
) -> tuple[int, int]:
    lm = landmarks[idx]
    x = float(lm.x if lm.x is not None else 0.0)
    y = float(lm.y if lm.y is not None else 0.0)
    return int(round(x * w)), int(round(y * h))


def _draw_connections(
    frame: Any,
    landmarks: Sequence[landmark_lib.NormalizedLandmark],
    connections: List[Connection],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    h, w = frame.shape[:2]
    for c in connections:
        a = _norm_pt(landmarks, w, h, c.start)
        b = _norm_pt(landmarks, w, h, c.end)
        cv2.line(frame, a, b, color, thickness, cv2.LINE_AA)


def draw_face_mesh_eyes(
    frame: Any,
    landmarks: Sequence[landmark_lib.NormalizedLandmark],
    *,
    style: DrawStyle = DrawStyle.EYES_ONLY,
) -> None:
    """Draw landmarks using Tasks API topology (no legacy `mp.solutions`)."""
    if style is DrawStyle.FULL:
        _draw_connections(
            frame,
            landmarks,
            FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            (192, 192, 192),
            1,
        )
    # Eyes-focused overlay (lower cost than full tessellation)
    _draw_connections(
        frame, landmarks, FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE, (48, 48, 255), 1
    )
    _draw_connections(
        frame, landmarks, FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE, (48, 255, 48), 1
    )
    _draw_connections(
        frame, landmarks, FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS, (0, 215, 255), 2
    )
    _draw_connections(
        frame, landmarks, FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS, (0, 215, 255), 2
    )


def draw_hud(
    frame: Any,
    *,
    fps: float,
    process_ms: float,
    face_ok: bool,
    lines: list[str],
) -> None:
    y = 24
    fh, _ = cv2.getTextSize("A", cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
    gap = fh + 6
    status = "face locked" if face_ok else "searching face"
    cv2.putText(
        frame,
        f"FPS {fps:.1f}  infer {process_ms:.1f} ms  {status}",
        (8, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (30, 240, 30) if face_ok else (40, 40, 255),
        1,
        cv2.LINE_AA,
    )
    y += gap
    for line in lines:
        cv2.putText(
            frame,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        y += gap
