"""Reduce Whisper silence/noise hallucinations (e.g. \"thank you for watching\")."""

from __future__ import annotations

import os
import unicodedata
from typing import Any

# Common YouTube-style / silence hallucinations; keep specific phrases to avoid blocking real "thank you".
_HALLUCINATION_SUBSTRINGS = (
    "thank you for watching",
    "thanks for watching",
    "thank you so much for watching",
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございます",
    "ご視聴ありがとうござい",
    "見てくれてありがとう",
    "like and subscribe",
    "subscribe to my channel",
    "smash that like",
)


def normalize_for_match(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold().strip()


def is_hallucination_phrase(text: str) -> bool:
    if not text or not str(text).strip():
        return False
    n = normalize_for_match(text)
    if len(n) < 4:
        return False
    for h in _HALLUCINATION_SUBSTRINGS:
        hn = normalize_for_match(h)
        if hn in n or (len(n) <= len(hn) + 8 and n in hn):
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
    """Return True if this transcript should be discarded."""
    if is_hallucination_phrase(text):
        return True
    try:
        cr_max = float(os.environ.get("NEUROVOX_WHISPER_MAX_COMPRESSION_RATIO", "2.35") or 2.35)
    except ValueError:
        cr_max = 2.35
    if compression_ratio is not None:
        try:
            if float(compression_ratio) > cr_max:
                return True
        except (TypeError, ValueError):
            pass
    probs = _no_speech_probs(segments)
    if not probs:
        return False
    try:
        max_thr = float(os.environ.get("NEUROVOX_WHISPER_MAX_NO_SPEECH", "0.68") or 0.68)
        avg_thr = float(os.environ.get("NEUROVOX_WHISPER_AVG_NO_SPEECH", "0.48") or 0.48)
    except ValueError:
        max_thr, avg_thr = 0.68, 0.48
    mx = max(probs)
    av = sum(probs) / len(probs)
    if mx >= max_thr or av >= avg_thr:
        return True
    return False
