from __future__ import annotations

from typing import Any

import cv2


def letterbox_to_width(frame: Any, target_w: int) -> Any:
    """Downscale so width is at most ``target_w`` (keeps aspect)."""
    h, w = frame.shape[:2]
    if w <= target_w:
        return frame
    scale = target_w / float(w)
    nh = int(round(h * scale))
    return cv2.resize(frame, (target_w, nh), interpolation=cv2.INTER_AREA)
