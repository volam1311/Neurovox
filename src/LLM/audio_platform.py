"""Cross-platform audio I/O with graceful fallbacks (no single point of failure).

Playback: ``sounddevice`` when available, else OS-native tools (Windows winmm,
macOS ``afplay``, Linux ``paplay`` / ``aplay`` / ``ffplay``).

Recording: ``sounddevice`` first; optional ``soundcard`` if installed (useful when
PortAudio is unavailable, e.g. some Windows setups).

Environment (optional):
  NEUROVOX_AUDIO_PLAY_BACKEND   auto | sounddevice | system
  NEUROVOX_AUDIO_RECORD_BACKEND auto | sounddevice | soundcard
  NEUROVOX_TTS_PAD_MS           leading/trailing silence padding (helps DAC / Bluetooth)
"""

from __future__ import annotations

import io
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import wave
from pathlib import Path
from typing import Literal

import numpy as np

_play_lock = threading.Lock()

PlayBackend = Literal["auto", "sounddevice", "system"]
RecordBackend = Literal["auto", "sounddevice", "soundcard"]


def _env_play_backend() -> PlayBackend:
    v = (os.environ.get("NEUROVOX_AUDIO_PLAY_BACKEND") or "auto").strip().lower()
    if v in ("auto", "sounddevice", "system"):
        return v  # type: ignore[return-value]
    return "auto"


def _env_record_backend() -> RecordBackend:
    v = (os.environ.get("NEUROVOX_AUDIO_RECORD_BACKEND") or "auto").strip().lower()
    if v in ("auto", "sounddevice", "soundcard"):
        return v  # type: ignore[return-value]
    return "auto"


def play_wav_bytes(wav_bytes: bytes) -> bool:
    """Play mono or stereo PCM WAV bytes. Thread-safe; serializes overlapping calls.

    Decodes to float32 (including 24-bit PCM), pads start/end for cleaner playback,
    then plays via sounddevice or OS fallback.
    """
    if not wav_bytes:
        return False
    try:
        audio, sr = _wav_bytes_to_mono_float32(wav_bytes)
        audio = _pad_tts_edges(audio, sr)
    except Exception:
        # Unknown WAV layout (e.g. float); let the OS / player decode the original bytes.
        with _play_lock:
            return _play_wav_system(wav_bytes)
    with _play_lock:
        backend = _env_play_backend()
        if backend == "system":
            return _play_wav_system(_pcm_mono_float32_to_wav_bytes(audio, sr))
        if backend == "sounddevice":
            return _play_float_sounddevice(audio, sr)
        if _play_float_sounddevice(audio, sr):
            return True
        return _play_wav_system(_pcm_mono_float32_to_wav_bytes(audio, sr))


def _wav_bytes_to_mono_float32(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode WAV to mono float32 [-1, 1] and sample rate."""
    bio = io.BytesIO(wav_bytes)
    with wave.open(bio, "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        n = wf.getnframes()
        raw = wf.readframes(n)
    if sw == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0) - 1.0
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        x = (
            b[:, 0].astype(np.int32)
            | (b[:, 1].astype(np.int32) << 8)
            | (b[:, 2].astype(np.int32) << 16)
        )
        x = np.where(x >= 0x800000, x - 0x1000000, x)
        samples = x.astype(np.float32) / 8388608.0
    elif sw == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if nch > 1:
        samples = samples.reshape(-1, nch).mean(axis=1)
    return samples, sr


def _pad_tts_edges(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    try:
        pad_ms = float(os.environ.get("NEUROVOX_TTS_PAD_MS", "90") or 90)
    except ValueError:
        pad_ms = 90.0
    if pad_ms <= 0:
        return audio
    n = int(sample_rate * pad_ms / 1000.0)
    if n <= 0:
        return audio
    z = np.zeros(n, dtype=np.float32)
    a = audio.astype(np.float32, copy=False).reshape(-1)
    return np.concatenate([z, a, z])


def _pcm_mono_float32_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    a = np.clip(audio.reshape(-1), -1.0, 1.0)
    pcm16 = (a * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


def _play_float_sounddevice(audio: np.ndarray, sr: int) -> bool:
    try:
        import sounddevice as sd
    except ImportError:
        return False
    try:
        a = np.ascontiguousarray(audio.reshape(-1), dtype=np.float32)
        sd.play(a, int(sr))
        sd.wait()
        return True
    except Exception:
        return False


def _play_wav_system(wav_bytes: bytes) -> bool:
    try:
        fd, path_str = tempfile.mkstemp(suffix=".wav")
        path = Path(path_str)
        os.close(fd)
        path.write_bytes(wav_bytes)
        try:
            return _play_wav_path_system(path)
        finally:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
    except OSError:
        return False


def _play_wav_path_system(path: Path) -> bool:
    system = platform.system()
    if system == "Windows":
        return _play_windows(path)
    if system == "Darwin":
        r = subprocess.run(
            ["afplay", str(path)],
            capture_output=True,
            timeout=600,
        )
        return r.returncode == 0
    # Linux and other Unix
    return _play_linux(path)


def _play_windows(path: Path) -> bool:
    p = str(path.resolve())
    try:
        import ctypes

        winmm = ctypes.windll.winmm  # type: ignore[attr-defined]
        SND_FILENAME = 0x20000
        SND_SYNC = 0x0000
        ok = bool(winmm.PlaySoundW(p, None, SND_FILENAME | SND_SYNC))
        if ok:
            return True
    except Exception:
        pass
    try:
        import winsound

        winsound.PlaySound(p, winsound.SND_FILENAME)
        return True
    except Exception:
        pass
    ps = p.replace("'", "''")
    r = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            f"(New-Object Media.SoundPlayer '{ps}').PlaySync()",
        ],
        capture_output=True,
        timeout=600,
    )
    return r.returncode == 0


def _record_soundcard(num_samples: int, sample_rate: int) -> np.ndarray:
    """Record via the ``soundcard`` package (optional dependency)."""
    import soundcard as sc

    mic = sc.default_microphone()
    if mic is None:
        raise RuntimeError("soundcard: no default microphone")
    try:
        return mic.record(samplerate=int(sample_rate), numframes=int(num_samples))
    except TypeError:
        return mic.record(numframes=int(num_samples), samplerate=int(sample_rate))


def _play_linux(path: Path) -> bool:
    p = str(path)
    for cmd in (
        ["paplay", p],
        ["aplay", "-q", p],
    ):
        exe = cmd[0]
        if shutil.which(exe):
            r = subprocess.run(cmd, capture_output=True, timeout=600)
            if r.returncode == 0:
                return True
    ffplay = shutil.which("ffplay")
    if ffplay:
        r = subprocess.run(
            [
                ffplay,
                "-nodisp",
                "-autoexit",
                "-loglevel",
                "quiet",
                p,
            ],
            capture_output=True,
            timeout=600,
        )
        return r.returncode == 0
    return False


def record_mono_float32(
    num_samples: int,
    sample_rate: int,
) -> np.ndarray:
    """Record ``num_samples`` mono samples at ``sample_rate`` (float32, shape Nx1)."""
    if num_samples <= 0 or sample_rate <= 0:
        raise ValueError("num_samples and sample_rate must be positive")
    backend = _env_record_backend()
    errors: list[str] = []

    if backend in ("sounddevice", "auto"):
        try:
            import sounddevice as sd

            audio = sd.rec(
                num_samples,
                sample_rate,
                channels=1,
                dtype=np.float32,
            )
            sd.wait()
            return audio
        except Exception as exc:
            errors.append(f"sounddevice: {exc}")
            if backend == "sounddevice":
                raise RuntimeError(
                    "Recording failed with NEUROVOX_AUDIO_RECORD_BACKEND=sounddevice. "
                    + "; ".join(errors)
                ) from exc

    if backend in ("soundcard", "auto"):
        try:
            data = _record_soundcard(num_samples, sample_rate)
            if data is None or len(data) == 0:
                raise RuntimeError("soundcard returned empty audio")
            if data.ndim == 1:
                return data.reshape(-1, 1).astype(np.float32, copy=False)
            return data[:, :1].astype(np.float32, copy=False)
        except ImportError as exc:
            errors.append(
                "soundcard: package not installed (pip install soundcard for this backend)"
            )
            if backend == "soundcard":
                raise RuntimeError("; ".join(errors)) from exc
        except Exception as exc:
            errors.append(f"soundcard: {exc}")
            if backend == "soundcard":
                raise RuntimeError("; ".join(errors)) from exc

    raise RuntimeError(
        "No working microphone backend. Install PortAudio + sounddevice, or "
        "`pip install soundcard` and set NEUROVOX_AUDIO_RECORD_BACKEND=soundcard. "
        f"Details: {'; '.join(errors)}"
    )


def check_playback_available() -> bool:
    """Whether at least one playback path is likely to work (lightweight probe)."""
    if _env_play_backend() == "sounddevice":
        try:
            import sounddevice  # noqa: F401

            return True
        except ImportError:
            return False
    if _env_play_backend() == "system":
        return True
    try:
        import sounddevice  # noqa: F401

        return True
    except ImportError:
        return platform.system() in ("Windows", "Darwin") or bool(
            shutil.which("paplay") or shutil.which("aplay") or shutil.which("ffplay")
        )


def check_recording_available() -> bool:
    """Whether recording is likely to work (imports + optional soundcard)."""
    backend = _env_record_backend()
    if backend == "sounddevice":
        try:
            import sounddevice  # noqa: F401

            return True
        except ImportError:
            return False
    if backend == "soundcard":
        try:
            import soundcard  # noqa: F401

            return True
        except ImportError:
            return False
    try:
        import sounddevice  # noqa: F401

        return True
    except ImportError:
        try:
            import soundcard  # noqa: F401

            return True
        except ImportError:
            return False


def describe_audio_stack() -> str:
    """One-line summary for logs (OS, backends, env)."""
    parts = [
        f"os={platform.system()}",
        f"python={sys.version.split()[0]}",
        f"play={_env_play_backend()}",
        f"record={_env_record_backend()}",
    ]
    try:
        import sounddevice as sd

        parts.append(f"sounddevice={getattr(sd, '__version__', '?')}")
    except ImportError:
        parts.append("sounddevice=no")
    try:
        import soundcard as sc

        parts.append(f"soundcard={getattr(sc, '__version__', '?')}")
    except ImportError:
        parts.append("soundcard=no")
    return " | ".join(parts)
