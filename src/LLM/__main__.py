"""Smoke-test: abbreviated gaze-typed text -> ranked sentence expansions.

Examples:
  PYTHONPATH=src python -m LLM "HAU"
  PYTHONPATH=src python -m LLM "TYSM"
"""

from __future__ import annotations

import sys

from . import OpenAICompletion


def main() -> None:
    argv = [a.strip() for a in sys.argv[1:] if a.strip()]
    if argv:
        abbreviated = " ".join(argv)
    else:
        abbreviated = input("Abbreviated text (e.g. HAU, WDYW): ").strip()
    if not abbreviated:
        print("Need abbreviated text to expand.", file=sys.stderr)
        raise SystemExit(1)
    backend = OpenAICompletion()
    for s in backend.complete_ranked(abbreviated=abbreviated, k=3):
        print(f"{s.rank}. {s.text}")


if __name__ == "__main__":
    main()
