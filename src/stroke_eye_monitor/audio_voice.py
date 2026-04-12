"""Microphone speech-to-text (Whisper) + OpenAI text-to-speech for the gaze keyboard demo."""

from __future__ import annotations

import io
import os
import threading
import time
import wave
from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from LLM.audio_platform import (
    check_recording_available,
    describe_audio_stack,
    record_mono_float32,
)

if TYPE_CHECKING:
    from LLM.openai_backend import OpenAICompletion


class SpokenContextBuffer:
    """Thread-safe rolling buffer of Whisper transcripts.

    Shown in the UI while STT is active (typing mode). ``get_for_llm`` consumes
    the buffer for the next inference; ``clear`` drops it when returning to idle.
    """

    def __init__(self, *, max_snippets: int = 12, max_chars: int = 6000) -> None:
        self._lock = threading.Lock()
        self._recent: deque[str] = deque(maxlen=max_snippets)
        self._max_chars = max_chars
        self.last_snippet: str = ""

    def push(self, text: str) -> None:
        t = text.strip()
        if not t:
            return
        with self._lock:
            self.last_snippet = t[:500]
            self._recent.append(t)

    def clear(self) -> None:
        with self._lock:
            self._recent.clear()
            self.last_snippet = ""

    def snapshot_lines_for_ui(self, max_lines: int = 4) -> list[str]:
        with self._lock:
            return list(self._recent)[-max(1, max_lines) :]

    def get_for_llm(self) -> str | None:
        """Return buffered speech for one LLM call and clear it (consume)."""
        with self._lock:
            if not self._recent:
                return None
            joined = " ".join(self._recent)
            if len(joined) > self._max_chars:
                joined = joined[-self._max_chars :]
            self._recent.clear()
            return joined


class SttListener:
    """Background loop: record short chunks, transcribe with Whisper when not silent."""

    def __init__(
        self,
        llm: OpenAICompletion,
        buffer: SpokenContextBuffer,
        *,
        chunk_seconds: float = 4.0,
        sample_rate: int = 16000,
        rms_threshold: float | None = None,
        peak_threshold: float | None = None,
    ) -> None:
        self._llm = llm
        self._buffer = buffer
        self._chunk_seconds = chunk_seconds
        self._sr = sample_rate
        try:
            self._rms_threshold = float(
                rms_threshold
                if rms_threshold is not None
                else os.environ.get("NEUROVOX_STT_RMS_THRESHOLD", "0.018")
            )
        except ValueError:
            self._rms_threshold = 0.018
        peak_env = os.environ.get("NEUROVOX_STT_PEAK_THRESHOLD", "0")
        try:
            self._peak_threshold = float(
                peak_threshold if peak_threshold is not None else peak_env
            )
        except ValueError:
            self._peak_threshold = 0.0
        self._running = False
        self._thread: threading.Thread | None = None
        self._paused = False
        self._pause_lock = threading.Lock()

    def pause(self) -> None:
        """Skip recording/transcription while assistant audio is playing (avoid echo)."""
        with self._pause_lock:
            self._paused = True

    def resume(self) -> None:
        with self._pause_lock:
            self._paused = False

    def resume_delayed(self, delay_s: float) -> None:
        """Resume STT after ``delay_s`` (reduces speaker bleed into the mic after TTS)."""

        def run() -> None:
            time.sleep(max(0.0, delay_s))
            self.resume()

        threading.Thread(target=run, daemon=True, name="stt-resume-delay").start()

    def _is_paused(self) -> bool:
        with self._pause_lock:
            return self._paused

    def start(self) -> None:
        if self._running:
            return
        if not check_recording_available():
            raise ImportError(
                "No microphone backend available: install sounddevice (needs PortAudio) "
                "or `pip install soundcard` and set NEUROVOX_AUDIO_RECORD_BACKEND=soundcard. "
                + describe_audio_stack()
            )
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="stt-listener", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def _loop(self) -> None:
        while self._running:
            try:
                if self._is_paused():
                    time.sleep(0.05)
                    continue
                n = int(self._chunk_seconds * self._sr)
                audio = record_mono_float32(n, self._sr)
                if not self._running:
                    break
                a = np.asarray(audio, dtype=np.float32).reshape(-1)
                a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=-1.0)
                np.clip(a, -1.0, 1.0, out=a)
                rms = float(np.sqrt(np.mean(np.square(a))))
                peak = float(np.max(np.abs(a)))
                if rms < self._rms_threshold:
                    continue
                if self._peak_threshold > 0.0 and peak < self._peak_threshold:
                    continue
                pcm = (a * 32767.0).astype(np.int16)
                wav_bytes = _pcm_mono_to_wav_bytes(pcm, self._sr)
                text = self._llm.transcribe_wav(wav_bytes).strip()
                if text:
                    self._buffer.push(text)
                    print(f"[mic] {text}", flush=True)
            except Exception as exc:
                print(f"STT error: {exc}", flush=True)
                time.sleep(0.5)


def _pcm_mono_to_wav_bytes(pcm16: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()
