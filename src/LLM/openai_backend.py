from __future__ import annotations

import json
import os
import re
import textwrap
from typing import Any

from .completion import RankedSuggestion
from .env import load_llm_env

_DEFAULT_RANKED_SYSTEM = textwrap.dedent(
    """\
    You always see **two** inputs together: (1) the **chatbot's question**, and (2) the
    **person's short reply** from a gaze or AAC keyboard — acronyms (WDYM, IMFTK, …),
    fragments, typos, often ALL CAPS.

    Use **both** the question and the short reply as one context window. Your job is to
    list the **top ranked possible full replies from the person** — complete things they
    might be trying to say **back to the chatbot** in that situation. Rank 1 = the reply
    they most likely intended; ranks 2–5 are plausible alternatives.

    Each suggestion should read as a **natural message from the person** (not a meta
    explanation like "they mean X"). Plain, supportive tone. One short sentence or phrase
    each unless a slightly longer answer is clearly right.

    Output only valid JSON per the schema in the first system block.
    """
).strip()


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
        max_tokens: int = 512,
        temperature: float = 0.5,
    ) -> None:
        load_llm_env()
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key or not str(api_key).strip():
            raise ValueError(
                "OPENAI_API_KEY is missing. Set it in the environment or in a .env file at the repo root."
            )
        self._model = (model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")).strip()
        self._max_tokens = max_tokens
        self._temperature = temperature

        kwargs: dict = {"api_key": api_key.strip()}
        base_url = os.environ.get("OPENAI_BASE_URL")
        if base_url and str(base_url).strip():
            kwargs["base_url"] = str(base_url).strip()
        self._client = OpenAI(**kwargs)

    def complete_ranked(
        self,
        *,
        question: str,
        reply: str,
        context: str | None = None,
        k: int = 5,
    ) -> list[RankedSuggestion]:
        k = max(1, min(10, int(k)))
        json_rules = f"""You output ONLY valid JSON (no markdown, no code fences).

Schema:
{{
  "suggestions": [
    {{"rank": 1, "text": "most likely full reply the person could be sending to the chatbot"}},
    {{"rank": 2, "text": "second most likely full reply from the person"}},
    ...
  ]
}}

Rules:
- Provide up to {k} objects in "suggestions" (ranks 1..{k}); prefer {k} unless input is empty or nonsensical.
- "rank" must be integers 1..{k} with no duplicates.
- "text" = one **candidate reply from the person** (worded as they might say it), using **both** the given question and their short fragment together — not a dictionary gloss of the acronym alone.
- No extra keys. No commentary outside the JSON."""
        system_parts = [json_rules]
        if context:
            system_parts.append(context.strip())
        else:
            system_parts.append(_DEFAULT_RANKED_SYSTEM)
        system = "\n\n".join(system_parts)

        user = textwrap.dedent(
            f"""\
            Chatbot question:
            \"\"\"{question.strip()}\"\"\"

            Person's short typed reply (same context — use both):
            \"\"\"{reply.strip()}\"\"\"

            Return JSON with up to {k} ranked **possible full replies from the person**
            (rank 1 = most likely). Each "text" is one candidate message they might intend
            to send, grounded in **both** the question and the fragment above.
            """
        ).strip()

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
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
        self, *, question: str, reply: str, context: str | None = None
    ) -> str:
        ranked = self.complete_ranked(question=question, reply=reply, context=context, k=5)
        return "\n".join(f"{s.rank}. {s.text}" for s in ranked)
