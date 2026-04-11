from __future__ import annotations

import collections
import time


class FpsMeter:
    """Rolling mean FPS from frame-to-frame deltas (call ``tick()`` once per frame)."""

    def __init__(self, maxlen: int = 30) -> None:
        self._buf: collections.deque[float] = collections.deque(maxlen=maxlen)
        self._last_tick = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        self._buf.append(1.0 / max(now - self._last_tick, 1e-6))
        self._last_tick = now
        return sum(self._buf) / max(len(self._buf), 1)
