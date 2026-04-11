from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class RankedSuggestion:
    """One ranked **candidate full reply** the person might intend (not a third-party gloss)."""

    rank: int
    text: str


@runtime_checkable
class CompletionBackend(Protocol):
    """Expand abbreviated gaze-typed text into ranked full sentences."""

    def complete(
        self, *, abbreviated: str, history: list[str] | None = None
    ) -> str:
        """Human-readable ranked lines (``1. …`` through ``k``)."""

    def complete_ranked(
        self,
        *,
        abbreviated: str,
        history: list[str] | None = None,
        spoken_context: str | None = None,
        k: int = 3,
    ) -> list[RankedSuggestion]:
        """Return up to ``k`` expansions.

        ``history`` = prior **confirmed** utterances in this session (oldest first),
        **excluding** the current abbreviated fragment.
        ``spoken_context`` = optional microphone transcript (questions / room context).
        """


class EchoBackend:
    """Stub: echoes the raw input as rank 1 (for wiring without an API key)."""

    def complete(
        self, *, abbreviated: str, history: list[str] | None = None
    ) -> str:
        ranked = self.complete_ranked(abbreviated=abbreviated, history=history, k=1)
        return "\n".join(f"{s.rank}. {s.text}" for s in ranked)

    def complete_ranked(
        self,
        *,
        abbreviated: str,
        history: list[str] | None = None,
        spoken_context: str | None = None,
        k: int = 3,
    ) -> list[RankedSuggestion]:
        _ = k
        _ = history
        _ = spoken_context
        return [RankedSuggestion(rank=1, text=abbreviated.strip())]
