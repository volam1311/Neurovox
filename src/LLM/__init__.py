"""Text completion / LLM layer (OpenAI-backed completion available)."""

from .completion import CompletionBackend, EchoBackend, RankedSuggestion
from .env import load_llm_env
from .openai_backend import OpenAICompletion

__all__ = [
    "CompletionBackend",
    "EchoBackend",
    "OpenAICompletion",
    "RankedSuggestion",
    "load_llm_env",
]
