"""Run a set of common questions through the tutor and save the answers.

    python faq_check.py            # all questions -> faq_answers.md
    python faq_check.py 2          # only the first 2

Needs LLM_API_KEY in the environment (see rag/ask.py).
"""
import os
import sys
import time
from pathlib import Path

from rag.ask import answer
from rag.retriever import Retriever

QUESTIONS = [
    "What is the convoy effect?",
    "Explain SRTF with an example",
    # a numerical that is not in the slides: solved by rag/solver.py, explained by the LLM
    "Solve using Round Robin with time quantum 3. P1: AT 0, BT 5. P2: AT 1, BT 4. P3: AT 3, BT 2. "
    "P4: AT 6, BT 3. Give the Gantt chart, CT, TAT, WT and the averages.",
    "Difference between preemptive and non-preemptive scheduling",
    "How does round robin work and how to choose the time quantum?",
    "What is starvation and how does aging solve it?",
    "What are the scheduling criteria?",
    "What is a process control block?",
    "Explain multilevel feedback queue scheduling",
    "What is a context switch?",
    "What is the difference between paging and segmentation?",
    "Explain the Banker's algorithm with an example",
    "How can the dining philosophers problem cause deadlock?",
    "How does TCP congestion control work?",  # not in the knowledge base (networking chapter not included)
]
OUT = Path("faq_answers.md")
PAUSE = 15  # seconds between questions, to stay inside free-tier rate limits


def main():
    if not os.environ.get("LLM_API_KEY"):
        sys.exit("LLM_API_KEY is not set in this terminal. In PowerShell: $env:LLM_API_KEY = 'your key'")
    questions = QUESTIONS[: int(sys.argv[1])] if len(sys.argv) > 1 else QUESTIONS
    r = Retriever()
    parts = [f"# OSmosis - FAQ answers\n\nRetrieval mode: {r.mode}\n"]
    for n, q in enumerate(questions, 1):
        print(f"[{n}/{len(questions)}] {q}")
        reply = answer(r, q)
        if reply.error:
            print(f"  LLM unavailable: {reply.error}")
        hits, figs = reply.hits, reply.figs
        parts.append(f"## {n}. {q}\n\n{reply.text}\n")
        if hits:
            parts.append("**Sources**\n")
        parts += [f"- [S{i}] {h.cite}" for i, h in enumerate(hits, 1)]
        if figs:
            parts.append("\n**Figures**\n")
            parts += [f"- {f['label']} ({f['source_short']})" for f in figs]
        parts.append("")
        OUT.write_text("\n".join(parts), encoding="utf-8")
        if n < len(questions):
            time.sleep(PAUSE)
    print(f"\nDone. Answers saved to {OUT}")


if __name__ == "__main__":
    main()
