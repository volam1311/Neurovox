from __future__ import annotations

import subprocess
from dataclasses import dataclass


def detect_screen_resolution() -> tuple[int, int] | None:
    """Auto-detect the primary display resolution on macOS."""
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"],
            text=True,
            timeout=5,
        )
        for line in out.splitlines():
            if "Resolution" in line and "Retina" in line:
                parts = line.split(":")[-1].strip().split()
                return int(parts[0]), int(parts[2])
        for line in out.splitlines():
            if "Resolution" in line:
                parts = line.split(":")[-1].strip().split()
                return int(parts[0]), int(parts[2])
    except Exception:
        pass
    return None


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
