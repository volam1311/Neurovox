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
        self._cond = threading.Condition(self._lock)
        self._frame_id = 0
        self._latest_item: tuple[bool, Any] | None = None

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

            with self._cond:
                self._latest_item = (ok, frame)
                self._frame_id += 1
                self._cond.notify_all()

            if not ok:
                # Transient camera read failures can occur on some backends.
                # The failed read has been published to the consumer above.
                # Now sleep briefly and retry unless the capture has been explicitly stopped.
                time.sleep(0.05)
                continue

    def read(self, timeout: float = 0.5) -> tuple[bool, Any]:
        """Block up to ``timeout`` seconds for the next frame."""
        with self._cond:
            current_id = self._frame_id

            # wait_for blocks until the lambda returns True or the timeout is reached
            new_frame_arrived = self._cond.wait_for(
                lambda: self._frame_id > current_id or self._stop.is_set(),
                timeout=timeout,
            )

            if new_frame_arrived and self._latest_item is not None:
                return self._latest_item

            return False, None  # Timed out waiting for camera

    def stop(self) -> None:
        self._stop.set()

        # Wake up any thread waiting in read() so it can exit gracefully
        with self._cond:
            self._cond.notify_all()

        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
