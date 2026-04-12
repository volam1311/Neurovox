"""Whisper post-filters driven by environment (no hardcoded phrase lists).

Hyperparameters (optional; defaults favor normal capture — tighten via env in noisy rooms):

**Substring blocklist (optional)**
  ``NEUROVOX_WHISPER_BLOCKED_SUBSTRINGS`` — comma-separated substrings; if any normalized
  substring appears in the transcript, it is dropped. Empty = disabled.

**Verbose API metrics** (only when ``NEUROVOX_WHISPER_PROB_FILTER=1`` in ``transcribe_wav``)
  ``NEUROVOX_WHISPER_MAX_COMPRESSION_RATIO`` — reject if above (default 2.45).
  ``NEUROVOX_WHISPER_MAX_NO_SPEECH`` — reject if any segment's ``no_speech_prob`` ≥ this (default 0.72).
  ``NEUROVOX_WHISPER_AVG_NO_SPEECH`` — reject if mean ``no_speech_prob`` ≥ this (default 0.54).

**Repetition heuristic** (pathological stutter / loop artifacts)
  ``NEUROVOX_WHISPER_REPETITION_FILTER`` — ``1`` (default) or ``0`` to disable.
  ``NEUROVOX_GARBAGE_MIN_CHARS`` — min transcript length to evaluate (default 18).
  ``NEUROVOX_GARBAGE_MIN_UNIT`` / ``NEUROVOX_GARBAGE_MAX_UNIT`` — repeat-unit width for regex (2–4).
  ``NEUROVOX_GARBAGE_MIN_REPEATS`` — min extra repeats of that unit (regex tail, default 4).
  ``NEUROVOX_GARBAGE_CHAR_DOMINANCE_MIN_LEN`` — min length for dominance check (default 24).
  ``NEUROVOX_GARBAGE_CHAR_DOMINANCE_RATIO`` — reject if one char fraction ≥ this (default 0.42).

Mic gating (not this module): ``NEUROVOX_STT_RMS_THRESHOLD`` in ``audio_voice.SttListener``.
Language hint: ``OPENAI_WHISPER_LANGUAGE`` (e.g. ``en``) in ``openai_backend.transcribe_wav``.
"""

from __future__ import annotations

import os
import re
import unicodedata
from collections import Counter
from typing import Any


def normalize_for_match(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold().strip()


def _env_blocked_substrings_normalized() -> list[str]:
    raw = (os.environ.get("NEUROVOX_WHISPER_BLOCKED_SUBSTRINGS") or "").strip()
    if not raw:
        return []
    return [normalize_for_match(p) for p in raw.split(",") if p.strip()]


def is_hallucination_phrase(text: str) -> bool:
    """True if ``text`` matches optional ``NEUROVOX_WHISPER_BLOCKED_SUBSTRINGS`` (comma-separated)."""
    if not text or not str(text).strip():
        return False
    blocked = _env_blocked_substrings_normalized()
    if not blocked:
        return False
    n = normalize_for_match(text)
    if len(n) < 4:
        return False
    for hn in blocked:
        if not hn:
            continue
        if hn in n or (len(n) <= len(hn) + 8 and n in hn):
            return True
    return False


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, "") or default)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, "") or default)
    except ValueError:
        return default


def is_garbage_repetition(text: str) -> bool:
    """Detect pathological repetition using ``NEUROVOX_GARBAGE_*`` / ``NEUROVOX_WHISPER_REPETITION_FILTER``."""
    off = os.environ.get("NEUROVOX_WHISPER_REPETITION_FILTER", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    )
    if off:
        return False

    t = (text or "").strip()
    min_chars = _env_int("NEUROVOX_GARBAGE_MIN_CHARS", 18)
    if len(t) < min_chars:
        return False
    compact = "".join(t.split())
    if len(compact) < min_chars:
        return False

    gmin = _env_int("NEUROVOX_GARBAGE_MIN_UNIT", 2)
    gmax = _env_int("NEUROVOX_GARBAGE_MAX_UNIT", 4)
    tail = _env_int("NEUROVOX_GARBAGE_MIN_REPEATS", 4)
    if gmin < 1 or gmax < gmin or tail < 1:
        return False
    try:
        pat = re.compile(rf"(.{{{gmin},{gmax}}})\1{{{tail},}}")
    except re.error:
        return False
    if pat.search(compact):
        return True

    dom_min = _env_int("NEUROVOX_GARBAGE_CHAR_DOMINANCE_MIN_LEN", 24)
    dom_ratio = _env_float("NEUROVOX_GARBAGE_CHAR_DOMINANCE_RATIO", 0.42)
    if len(compact) >= dom_min:
        top = Counter(compact).most_common(1)[0][1]
        if top / len(compact) >= dom_ratio:
            return True
    return False


def _no_speech_probs(segments: Any) -> list[float]:
    out: list[float] = []
    if not segments:
        return out
    for s in segments:
        if isinstance(s, dict):
            v = s.get("no_speech_prob")
        else:
            v = getattr(s, "no_speech_prob", None)
        if v is not None:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                pass
    return out


def should_reject_whisper_verbose(
    *,
    text: str,
    segments: Any,
    compression_ratio: float | None,
) -> bool:
    """Return True if this transcript should be discarded (env-tuned thresholds)."""
    if is_hallucination_phrase(text) or is_garbage_repetition(text):
        return True
    cr_max = _env_float("NEUROVOX_WHISPER_MAX_COMPRESSION_RATIO", 2.45)
    if compression_ratio is not None:
        try:
            if float(compression_ratio) > cr_max:
                return True
        except (TypeError, ValueError):
            pass
    probs = _no_speech_probs(segments)
    if not probs:
        return False
    max_thr = _env_float("NEUROVOX_WHISPER_MAX_NO_SPEECH", 0.72)
    avg_thr = _env_float("NEUROVOX_WHISPER_AVG_NO_SPEECH", 0.54)
    mx = max(probs)
    av = sum(probs) / len(probs)
    if mx >= max_thr or av >= avg_thr:
        return True
    return False
