from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class RankedSuggestion:
    """One ranked interpretation of what the user's short reply likely means."""

    rank: int
    text: str


@runtime_checkable
class CompletionBackend(Protocol):
    """Rank what a short AAC-style reply most likely means given a chatbot question."""

    def complete(
        self, *, question: str, reply: str, context: str | None = None
    ) -> str:
        """Human-readable ranked lines (``1. …`` through ``k``)."""

    def complete_ranked(
        self,
        *,
        question: str,
        reply: str,
        context: str | None = None,
        k: int = 5,
    ) -> list[RankedSuggestion]:
        """Return up to ``k`` interpretations sorted by ``rank`` (1 = most likely)."""


class EchoBackend:
    """Stub: echoes the raw reply as rank 1 (for wiring without an API key)."""

    def complete(
        self, *, question: str, reply: str, context: str | None = None
    ) -> str:
        ranked = self.complete_ranked(question=question, reply=reply, context=context, k=1)
        return "\n".join(f"{s.rank}. {s.text}" for s in ranked)

    def complete_ranked(
        self,
        *,
        question: str,
        reply: str,
        context: str | None = None,
        k: int = 5,
    ) -> list[RankedSuggestion]:
        _ = context
        _ = k
        _ = question
        return [RankedSuggestion(rank=1, text=reply.strip())]
