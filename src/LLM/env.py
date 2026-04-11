from __future__ import annotations

from pathlib import Path

_loaded = False


def load_llm_env() -> None:
    """Load ``.env`` from ``src/`` or repo root (whichever exists as a file)."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    # src/LLM/env.py -> parents[1] = src/, parents[2] = repo root
    src_dir = Path(__file__).resolve().parents[1]
    repo_root = src_dir.parent
    for candidate in [src_dir / ".env", repo_root / ".env"]:
        if candidate.is_file():
            load_dotenv(candidate)
            return
