from __future__ import annotations

import threading
import time
from typing import Any

import cv2


class ThreadedVideoCapture:
    """Background thread reads ``VideoCapture``; main thread consumes frames.

    Uses a thread-safe variable so if inference/UI is slower than the camera, 
    stale frames are dropped and you keep the most recent image (lower latency).
    """

    def __init__(self, cap: cv2.VideoCapture) -> None:
        self._cap = cap

        self._lock = threading.Lock()
        self._latest_item: tuple[bool, Any] | None = None
        self._new_frame_event = threading.Event()

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="video-capture", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()

            if frame is not None:
                # Force a robust copy so memory isn't shared with OpenCV C++
                frame = frame.copy()

            with self._lock:
                self._latest_item = (ok, frame)

            self._new_frame_event.set()  # Signal that data is ready

            if not ok:
                # If camera died, stop the thread.
                # Main thread will read ok=False and handle the crash.
                break

    def read(self, timeout: float = 0.5) -> tuple[bool, Any]:
        """Block up to ``timeout`` seconds for the next frame."""
        # Wait until the background thread signals a frame is ready
        if self._new_frame_event.wait(timeout=timeout):
            self._new_frame_event.clear()  # Reset for the next frame
            
            with self._lock:
                # Add fallback to satisfy type checker
                if self._latest_item is not None:
                    return self._latest_item
                return False, None
        else:
            return False, None  # Timed out waiting for camera

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None