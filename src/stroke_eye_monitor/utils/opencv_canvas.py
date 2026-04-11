from __future__ import annotations

import cv2
import numpy as np


def sync_opencv_window_canvas(window_name: str, width: int, height: int) -> tuple[int, int]:
    """Poll OpenCV until the window reports a plausible client size (Retina / scaling safe).

    Returns ``(width, height)`` possibly updated from the actual window rectangle.
    """
    board = np.zeros((max(height, 1), max(width, 1), 3), dtype=np.uint8)
    cv2.imshow(window_name, board)
    best_w, best_h = width, height
    for _ in range(60):
        cv2.waitKey(16)
        r = cv2.getWindowImageRect(window_name)
        cw, ch = int(r[2]), int(r[3])
        if cw >= 320 and ch >= 240:
            best_w, best_h = cw, ch
            break
    return best_w, best_h
