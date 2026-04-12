from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass


def _detect_windows() -> tuple[int, int] | None:
    try:
        import ctypes
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        user32.SetProcessDPIAware()
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except Exception:
        return None


def _detect_macos() -> tuple[int, int] | None:
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


def detect_screen_resolution() -> tuple[int, int] | None:
    """Auto-detect primary display resolution (Windows + macOS)."""
    if os.name == "nt":
        return _detect_windows()
    return _detect_macos()


@dataclass(frozen=True)
class MonitorConfig:
    """Tuned defaults for live webcam use (latency over max resolution)."""

    camera_index: int = 2
    # Downscale width for inference; height preserves aspect ratio. Smaller = faster.
    process_width: int = 640
    max_num_faces: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    refine_landmarks: bool = True
    # Mirror display like a typical selfie preview
    mirror_display: bool = True
    window_name: str = "Webcam eye / gaze (research demo)"
