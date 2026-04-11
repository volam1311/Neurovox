from __future__ import annotations

import queue
import threading
import time
from typing import Any

import cv2


class ThreadedVideoCapture:
    """Background thread reads ``VideoCapture``; main thread consumes frames.

    Uses a queue of depth 1 so if inference/UI is slower than the camera, stale
    frames are dropped and you keep the most recent image (lower latency).
    """

    def __init__(self, cap: cv2.VideoCapture) -> None:
        self._cap = cap
        self._q: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="video-capture", daemon=True)
        self._thread.start()

    def _offer(self, item: tuple[bool, Any]) -> None:
        """Replace queue contents with ``item`` (latest frame wins)."""
        try:
            self._q.put_nowait(item)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(item)
            except queue.Full:
                pass

    def _run(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.002)
                continue
            # Copy so the processing thread never shares a buffer with OpenCV's capture path.
            self._offer((True, frame.copy()))

    def read(self, timeout: float = 0.5) -> tuple[bool, Any]:
        """Block up to ``timeout`` seconds for the next frame (newest in queue)."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return False, None

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
