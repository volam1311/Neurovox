"""Smoke-test: chatbot ``question`` + user ``reply`` → top 5 ranked interpretations.

Examples:
  PYTHONPATH=src python -m LLM "How are you feeling today?" "IMFTK"
  PYTHONPATH=src python -m LLM "What did you mean by that?" "WDYM"
"""

from __future__ import annotations

import sys

from . import OpenAICompletion


def main() -> None:
    argv = [a.strip() for a in sys.argv[1:] if a.strip()]
    if len(argv) >= 2:
        question, reply = argv[0], " ".join(argv[1:])
    else:
        question = input("Chatbot question: ").strip()
        reply = input("User short reply (e.g. WDYM, IMFTK): ").strip()
    if not question or not reply:
        print("Need both a question and a reply.", file=sys.stderr)
        raise SystemExit(1)
    backend = OpenAICompletion()
    for s in backend.complete_ranked(question=question, reply=reply, k=5):
        print(f"{s.rank}. {s.text}")


if __name__ == "__main__":
    main()
