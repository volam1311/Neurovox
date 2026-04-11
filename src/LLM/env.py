from __future__ import annotations

from pathlib import Path

_loaded = False


def load_llm_env() -> None:
    """Load ``.env`` from the repository root once (optional ``python-dotenv``)."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    # .../AIML/src/LLM/env.py -> repo root is parents[2]
    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env")
