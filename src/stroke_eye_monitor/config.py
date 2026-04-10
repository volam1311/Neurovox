from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MonitorConfig:
    """Tuned defaults for live webcam use (latency over max resolution)."""

    camera_index: int = 0
    # Downscale width for inference; height preserves aspect ratio. Smaller = faster.
    process_width: int = 640
    max_num_faces: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    refine_landmarks: bool = True
    # Mirror display like a typical selfie preview
    mirror_display: bool = True
    window_name: str = "Webcam eye / gaze (research demo)"
