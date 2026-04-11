from __future__ import annotations

import io
import json
import os
import re
import textwrap
from typing import Any

from .audio_platform import play_wav_bytes
from .completion import RankedSuggestion
from .env import load_llm_env
from .stt_whisper import is_hallucination_phrase, should_reject_whisper_verbose

_DEFAULT_RANKED_SYSTEM = textwrap.dedent(
    """\
    You are an AAC (Augmentative and Alternative Communication) abbreviation expander,
    inspired by the Google SpeakFaster methodology.

    The user types with an eye-gaze keyboard. They enter **highly abbreviated text**:
    word-initial letters, fragments, common acronyms, or short phonetic approximations
    — often ALL CAPS with no spaces.

    **Primary signal:** The **current fragment** (the typed letters) is the main
    instruction. Your expansions must **decode that abbreviation** into full sentences.
    Do **not** replace the fragment with an unrelated sentence that merely fits the
    transcript or spoken audio.

    **Secondary context:** You may receive a **session transcript** (Turn 1, Turn 2, …)
    and/or **spoken voice** from a microphone. Use these **only** to:
    - resolve ambiguity (which homonym fits),
    - keep pronouns/topic continuity,
    - match tone.

    If transcript or voice **conflicts** with what the letters could reasonably expand
    to, **trust the letters** and pick the best expansion of the fragment; do not
    invent a different message.

    Guidelines:
    - Rank 1 = most likely expansion of **this fragment**.
    - Each suggestion must be a **complete, natural sentence** ready to be spoken aloud.
    - Keep it concise — one sentence per suggestion unless the abbreviation clearly
      implies more.
    - If there is no transcript or voice, use general knowledge and the fragment alone.

    Output only valid JSON per the schema in the first system block.
    """
).strip()


def _format_session_transcript(
    prior_turns: list[str],
    *,
    max_turns: int,
    max_chars: int,
) -> str:
    """Oldest-first numbered turns; sliding window: drop oldest until under ``max_chars``."""
    cleaned = [p.strip() for p in prior_turns if p and str(p).strip()]
    if not cleaned:
        return ""
    lo = max(0, len(cleaned) - max_turns)
    while lo < len(cleaned):
        lines = [f"Turn {i + 1}: {cleaned[i]}" for i in range(lo, len(cleaned))]
        text = "\n".join(lines)
        if len(text) <= max_chars or lo >= len(cleaned) - 1:
            return text
        lo += 1
    return f"Turn {len(cleaned)}: {cleaned[-1]}"


def _parse_ranked_json(raw: str, k: int) -> list[RankedSuggestion] | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        data: Any = json.loads(raw)
    except json.JSONDecodeError:
        return None
    items = data.get("suggestions")
    if not isinstance(items, list):
        return None
    out: list[RankedSuggestion] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            rank = int(item["rank"])
        except (KeyError, TypeError, ValueError):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        out.append(RankedSuggestion(rank=rank, text=text))
    out.sort(key=lambda s: s.rank)
    seen: set[int] = set()
    deduped: list[RankedSuggestion] = []
    for s in out:
        if s.rank in seen or s.rank < 1 or s.rank > k:
            continue
        seen.add(s.rank)
        deduped.append(s)
    return deduped if deduped else None


def _parse_ranked_lines(raw: str, k: int) -> list[RankedSuggestion]:
    """Fallback: lines like ``1. foo`` or ``2) bar``."""
    out: list[RankedSuggestion] = []
    for line in raw.splitlines():
        m = re.match(r"^\s*(\d+)\s*[\.)]\s*(.+)\s*$", line)
        if not m:
            continue
        rank = int(m.group(1))
        text = m.group(2).strip()
        if text and 1 <= rank <= k:
            out.append(RankedSuggestion(rank=rank, text=text))
    out.sort(key=lambda s: s.rank)
    seen: set[int] = set()
    deduped: list[RankedSuggestion] = []
    for s in out:
        if s.rank in seen:
            continue
        seen.add(s.rank)
        deduped.append(s)
    return deduped[:k]


class OpenAICompletion:
    """Chat completion via the OpenAI API (``OPENAI_API_KEY`` in environment or ``.env``)."""

    def __init__(
        self,
        *,
        model: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.3,
    ) -> None:
        load_llm_env()
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key or not str(api_key).strip():
            raise ValueError(
                "OPENAI_API_KEY is missing. Set it in the environment or in a .env file at the repo root."
            )
        self._model = (model or os.environ.get("OPENAI_MODEL", "gpt-4o")).strip()
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._context_max_turns = int(
            os.environ.get("OPENAI_CONTEXT_MAX_TURNS", "24") or 24
        )
        self._context_max_chars = int(
            os.environ.get("OPENAI_CONTEXT_MAX_CHARS", "12000") or 12000
        )

        kwargs: dict = {"api_key": api_key.strip()}
        base_url = os.environ.get("OPENAI_BASE_URL")
        if base_url and str(base_url).strip():
            kwargs["base_url"] = str(base_url).strip()
        self._client = OpenAI(**kwargs)
        self._tts_model = (os.environ.get("OPENAI_TTS_MODEL", "tts-1")).strip()
        self._tts_voice = (os.environ.get("OPENAI_TTS_VOICE", "alloy")).strip()

    def transcribe_wav(self, wav_bytes: bytes) -> str:
        """Transcribe mono WAV bytes with Whisper (drops known silence hallucinations).

        By default only a phrase blocklist is applied. Set ``NEUROVOX_WHISPER_PROB_FILTER=1``
        to also use verbose_json segment probabilities (stricter; can block real speech).
        """
        bio = io.BytesIO(wav_bytes)
        bio.name = "chunk.wav"
        lang = (os.environ.get("OPENAI_WHISPER_LANGUAGE") or "").strip() or None
        use_prob = os.environ.get("NEUROVOX_WHISPER_PROB_FILTER", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        kwargs: dict[str, Any] = {
            "model": "whisper-1",
            "file": bio,
            "temperature": 0.0,
        }
        if lang:
            kwargs["language"] = lang
        if use_prob:
            kwargs["response_format"] = "verbose_json"
        r = self._client.audio.transcriptions.create(**kwargs)
        text = (r.text or "").strip()
        if use_prob:
            segments = getattr(r, "segments", None) or []
            cr_raw = getattr(r, "compression_ratio", None)
            try:
                cr_f = float(cr_raw) if cr_raw is not None else None
            except (TypeError, ValueError):
                cr_f = None
            if should_reject_whisper_verbose(
                text=text, segments=segments, compression_ratio=cr_f
            ):
                return ""
            return text
        if is_hallucination_phrase(text):
            return ""
        return text

    def speak(self, text: str) -> None:
        """Speak text with OpenAI TTS (WAV) and play via the default audio output."""
        text = text.strip()
        if not text:
            return
        try:
            raw_sp = os.environ.get("OPENAI_TTS_SPEED")
            sp = float(raw_sp) if raw_sp is not None and str(raw_sp).strip() else 1.0
        except ValueError:
            sp = 1.0
        sp = max(0.25, min(4.0, sp))
        speech_kw: dict[str, Any] = {
            "model": self._tts_model,
            "voice": self._tts_voice,
            "input": text[:4096],
            "response_format": "wav",
        }
        if sp != 1.0:
            speech_kw["speed"] = sp
        try:
            resp = self._client.audio.speech.create(**speech_kw)
        except Exception as exc:
            if "speed" in speech_kw:
                speech_kw.pop("speed", None)
                try:
                    resp = self._client.audio.speech.create(**speech_kw)
                except Exception as exc2:
                    print(f"TTS API error: {exc2}", flush=True)
                    return
            else:
                print(f"TTS API error: {exc}", flush=True)
                return
        if not play_wav_bytes(resp.content):
            print(
                "TTS: playback failed (try NEUROVOX_AUDIO_PLAY_BACKEND=system or install "
                "sounddevice / OS audio tools).",
                flush=True,
            )

    def complete_ranked(
        self,
        *,
        abbreviated: str,
        history: list[str] | None = None,
        spoken_context: str | None = None,
        k: int = 3,
    ) -> list[RankedSuggestion]:
        k = max(1, min(10, int(k)))
        json_rules = f"""You output ONLY valid JSON (no markdown, no code fences).

Schema:
{{
  "suggestions": [
    {{"rank": 1, "text": "most likely full sentence the user intended"}},
    {{"rank": 2, "text": "second most likely expansion"}},
    ...
  ]
}}

Rules:
- Provide up to {k} objects in "suggestions" (ranks 1..{k}).
- "rank" must be integers 1..{k} with no duplicates.
- "text" = one natural, speakable sentence expanding the abbreviated input.
- No extra keys. No commentary outside the JSON."""

        system = "\n\n".join([json_rules, _DEFAULT_RANKED_SYSTEM])

        prior = [p.strip() for p in (history or []) if p and str(p).strip()]
        transcript = _format_session_transcript(
            prior,
            max_turns=max(1, self._context_max_turns),
            max_chars=max(500, self._context_max_chars),
        )

        spoken = (spoken_context or "").strip()

        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        if spoken:
            messages.append(
                {
                    "role": "user",
                    "content": textwrap.dedent(
                        f"""\
                        ## Spoken voice (secondary — do not override the typed fragment)
                        Optional background from the room. Use only if it helps interpret
                        the abbreviation below; ignore if irrelevant or conflicting.

                        \"\"\"{spoken}\"\"\"
                        """
                    ).strip(),
                }
            )
        if transcript:
            messages.append(
                {
                    "role": "user",
                    "content": textwrap.dedent(
                        f"""\
                        ## Session transcript (secondary context only)
                        Prior confirmed phrases. Use for continuity; the next message must
                        still be a direct expansion of the current typed fragment.

                        {transcript}
                        """
                    ).strip(),
                }
            )
        messages.append(
            {
                "role": "user",
                "content": textwrap.dedent(
                    f"""\
                    ## Typed fragment — expand THIS (primary)
                    \"\"\"{abbreviated.strip()}\"\"\"

                    Expand into up to {k} ranked full sentences (rank 1 = most likely).
                    The fragment is the anchor; transcript/voice above are hints only.
                    """
                ).strip(),
            }
        )

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        parsed = _parse_ranked_json(raw, k)
        if parsed is None:
            parsed = _parse_ranked_lines(raw, k)
        if not parsed:
            return [RankedSuggestion(rank=1, text=raw[:500] if raw else "(no parseable response)")]
        return parsed[:k]

    def complete(
        self,
        *,
        abbreviated: str,
        history: list[str] | None = None,
        spoken_context: str | None = None,
    ) -> str:
        ranked = self.complete_ranked(
            abbreviated=abbreviated,
            history=history,
            spoken_context=spoken_context,
            k=3,
        )
        return "\n".join(f"{s.rank}. {s.text}" for s in ranked)
