"""Ask the tutor a question from the command line.

    python -m rag.ask "Explain SRTF with an example"
    python -m rag.ask "What is the convoy effect?" --html answer.html   # open in a browser to see figures
    python -m rag.ask "Round robin time quantum" --no-llm                # only show what was retrieved

LLM settings come from environment variables (any OpenAI-compatible API):
    LLM_BASE_URL   default https://api.groq.com/openai/v1   (Ollama: http://localhost:11434/v1)
    LLM_API_KEY    your key (Ollama: any text, e.g. "ollama")
    LLM_MODEL      default openai/gpt-oss-120b              (Ollama: e.g. qwen2.5:3b)
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path

from . import intent
from .config import CONVENTIONS, DATA_DIR, LLM_BASE_URL_DEFAULT, LLM_MODEL_DEFAULT, TOP_K
from .retriever import Retriever
from .examples import Example, find_example
from .numericals import ProblemError, follow_up, narrow, parse_problem, render, solve, to_text

SYSTEM_PROMPT = f"""You are OSmosis, a patient teaching assistant for the Operating Systems course
VIT BCSE303L (processes and threads, CPU scheduling, synchronization, deadlocks, memory
management, storage and file systems, Linux commands and shell programming).

Rules:
1. Answer ONLY from the numbered SOURCES. If they do not contain the answer, say so and stop;
   do not fill the gap from your own knowledge.
2. Cite every factual sentence with its source tag, e.g. [S2]. Prefer the teacher's slides
   when they and the textbook differ, and point out the difference.
   - The only valid tags are [S1], [S2], ... exactly as numbered in SOURCES, in plain square
     brackets. Never invent other tags (no "Rule 3", no section or slide numbers as tags).
   - Cite a source only for what that passage itself states. If a definition comes from [S3],
     cite [S3], not the source of the example that follows it.
3. For CPU-scheduling numericals: show the Gantt chart as a timeline (e.g. | P1 0-3 | P2 3-7 |),
   then a table with CT, TAT, WT (and RT if asked), then the averages, step by step.
   For other numericals (Banker's algorithm, page replacement, disk scheduling, address
   translation): follow the method in the SOURCES one step at a time, show each step's working
   in a table or list, and check every arithmetic step before stating the result.
   - If a source already works the problem, reproduce its steps and values exactly.
   - Treat every arrival, completion and return from I/O as a decision point. At each one, state
     the remaining time (or priority) of every ready process before saying who runs, and make
     sure the explanation agrees with the Gantt chart.
   - A preemption is only a switch that takes the CPU from a process that still has burst time
     left. A switch after a process completes or leaves for I/O is a context switch, not a
     preemption; count the two separately.
   - Every row of a table must have the same number of cells as its header.
   - If a COMPUTED SOLUTION follows the SOURCES, it comes from the course's exact solver (CPU
     scheduling, Banker's algorithm, page replacement or disk scheduling). The student already
     sees it in full above your answer, so do not repeat its tables and do not recompute
     anything. Walk through its "Decisions" or "Steps" lines in order to explain how the result
     arises, quote its numbers unchanged, and cite the SOURCES for the method and the formulas.
     State any assumption it lists. Worked examples in the SOURCES are different problems: never
     use their numbers.
4. If a FIGURE is listed that helps, tell the student to look at it by its label
   (e.g. "see Figure 5.5" or "see Slide 79"). Ignore listed figures that are off-topic.
5. For CPU scheduling, use these course conventions unless the question states otherwise. They
   are course rules, not sources, so they take no [S#] tag. Write "(course convention)" only
   after a sentence that states one of the five conventions below, and nowhere else.
{CONVENTIONS}
6. Stay on the algorithm or topic asked about. Do not bring in mechanisms of other algorithms
   (for example a time quantum when explaining FCFS, SJF, SRTF or priority scheduling, or FCFS
   CPU scheduling when the question is about FCFS disk scheduling) unless the SOURCES do.
7. Keep it exam-oriented: short paragraphs, key terms in bold, end with one common mistake to
   avoid. The mistake must concern the topic asked about and follow from the SOURCES or the
   course conventions; cite it like any other sentence. Never repeat the wording of these rules.
8. Earlier turns of the conversation may come before the SOURCES. Use them only to work out
   what the student is referring to ("that step", "it", "explain again"). The answer itself
   must still come from the SOURCES and any COMPUTED SOLUTION.
"""

HISTORY_TURNS = 3      # earlier exchanges sent along with a question
HISTORY_CHARS = 1500   # each earlier answer is cut to this length

_CITE = re.compile(r"\[S\d+")


def clean_answer(text: str, n_sources: int) -> str:
    """Normalise citation brackets (models often write 【S1】), drop zero-width characters and
    remove tags that point at no source (e.g. [S74] written for "slide 74")."""
    text = text.replace("【", "[").replace("】", "]")
    text = text.translate({cp: None for cp in (0x200B, 0x200C, 0x200D, 0xFEFF)})
    text = re.sub(r"\[\s*\**\s*S(\d+)\s*\**\s*\]", r"[S\1]", text)  # [**S3**], [ S3 ] -> [S3]
    return re.sub(r"\[S(\d+)\]", lambda m: m.group(0) if 1 <= int(m.group(1)) <= n_sources else "", text)


def is_grounded(answer: str) -> bool:
    """False when the answer cites no source, i.e. the tutor said the material does not cover it."""
    return bool(_CITE.search(answer))


def build_context(hits, figs) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        parts.append(f"[S{i}] ({h.cite})\n{h.item['text']}")
    if figs:
        parts.append("FIGURES available to show the student:")
        for f in figs:
            parts.append(f"- {f['label']} ({f['source_short']}): {f['caption']}")
    return "\n\n".join(parts)


def call_llm(question: str, context: str, history=()) -> str:
    """The tutor's answer. `history` is earlier (question, answer) pairs of the conversation."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for q, a in history:
        messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a[:HISTORY_CHARS]}]
    messages.append({"role": "user", "content": f"SOURCES:\n{context}\n\nQUESTION: {question}"})
    return llm_chat(messages)


def llm_chat(messages: list[dict], temperature: float = 0) -> str:
    """One call to the configured OpenAI-compatible chat endpoint -> the reply text."""
    base = os.environ.get("LLM_BASE_URL", LLM_BASE_URL_DEFAULT).rstrip("/")
    key = os.environ.get("LLM_API_KEY", "")
    model = os.environ.get("LLM_MODEL", LLM_MODEL_DEFAULT)
    if not key:
        raise RuntimeError("LLM_API_KEY is not set (see the top of rag/ask.py).")
    # temperature 0: same question -> same answer, as far as the provider allows
    body = {"model": model, "temperature": temperature, "messages": messages}
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            # Groq sits behind Cloudflare, which rejects urllib's default agent (403, code 1010)
            "User-Agent": "osmosis-tutor/1.0",
        },
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                text = json.loads(r.read())["choices"][0]["message"]["content"]
            # reasoning models (Qwen 3, DeepSeek-R1) may put their scratch work in <think> tags
            return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 3:  # free-tier rate limit
                print("[rate limited, retrying in 30 s]", file=sys.stderr)
                time.sleep(30)
                continue
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}") from e


@dataclass
class Reply:
    body: str                        # the LLM's answer; the retrieved passages when no LLM was used
    hits: list                       # passages behind the answer ([S1] is hits[0])
    figs: list
    solution: object = None          # set when a numerical was solved for this answer (any solver)
    llm_used: bool = False
    error: str = ""                  # why the LLM was not used, if it was asked for
    example: Example | None = None   # set when `solution` is one of the course's worked examples

    @property
    def text(self) -> str:
        if self.solution and self.llm_used:
            head = f"Worked example: {self.example.name} ({self.example.where})\n{self.example.question}\n\n" \
                if self.example else ""
            return f"{head}{render(self.solution)}\n\nExplanation\n-----------\n{self.body}"
        return self.body


is_follow_up = intent.is_follow_up   # a short question leaning on the previous turn


def answer(r: Retriever, question: str, k: int = TOP_K, use_llm: bool = True,
           history=(), previous=None) -> Reply:
    """Answer one question.

    A numerical (CPU scheduling, Banker's algorithm, page replacement, disk scheduling) is
    solved exactly by rag.numericals first; the LLM then only explains the computed result. The
    same happens when the student asks for an example of an algorithm: the course's worked
    example is solved in code. Raises ProblemError if a numerical cannot be read.

    `history` holds the earlier (question, answer) pairs of the conversation and `previous`
    the last numerical solved in it, so follow-ups ("explain that step again", "now with
    quantum 2") work.
    """
    problem = parse_problem(question) or follow_up(question, previous)
    leans_back = bool(history) and is_follow_up(question)
    hits, figs = r.retrieve(f"{history[-1][0]} {question}" if leans_back else question, k=k)
    example = None if problem else find_example(question, hits)
    if example:
        whole = parse_problem(example.question)
        problem = narrow(whole, question)
        if problem != whole:   # only the algorithm asked about ("Explain LRU with an example")
            example = replace(example, question=to_text(problem))
    solution = solve(problem) if problem else None
    context = build_context(hits, figs)
    if example:
        context += (f"\n\nCOMPUTED SOLUTION (exact) of the worked example \"{example.name}\" "
                    f"({example.where}): {example.question}\n" + render(solution))
    elif solution:
        context += "\n\nCOMPUTED SOLUTION (exact):\n" + render(solution)
    elif previous and leans_back:
        context += "\n\nCOMPUTED SOLUTION (exact) from the previous question:\n" + render(solve(previous))
    reply = Reply(context, hits, figs, solution, example=example)
    if use_llm:
        try:
            reply.body = clean_answer(call_llm(question, context, history[-HISTORY_TURNS:]), len(hits))
            reply.llm_used = True
        except Exception as e:
            reply.error = str(e)
    if not is_grounded(reply.body):
        reply.hits, reply.figs = [], []  # nothing was used, so do not list unrelated sources / figures
    return reply


def explain_mistake(r: Retriever, problem, where: str, hint: str, explanation: str) -> Reply:
    """An LLM explanation of a mistake the Reasoning Debugger found, grounded in retrieved course
    passages. The solver's exact solution and its description of the mistake are given to the LLM
    as fixed facts; nothing is parsed from free text, and the LLM is asked only to explain."""
    from .numericals import title
    hits, figs = r.retrieve(f"{title(problem)}. {hint}")
    context = (build_context(hits, figs) + "\n\nCOMPUTED SOLUTION (exact):\n" + render(solve(problem))
               + f"\n\nTHE STUDENT'S MISTAKE (found by comparing their working with the exact solution):\n"
                 f"{where}: {explanation}")
    question = ("Explain to the student why this step is wrong and which rule from the course applies, citing the "
                "SOURCES. Use only the numbers in the COMPUTED SOLUTION and the mistake description.")
    reply = Reply(context, hits, figs)
    try:
        reply.body = clean_answer(call_llm(question, context), len(hits))
        reply.llm_used = True
    except Exception as e:
        reply.error = str(e)
    if not is_grounded(reply.body):
        reply.hits, reply.figs = [], []
    return reply


def write_html(path: Path, question: str, answer: str, hits, figs):
    try:
        rel = os.path.relpath(DATA_DIR, path.parent.resolve()).replace("\\", "/")
    except ValueError:  # Windows: the page is being written to a different drive
        rel = DATA_DIR.as_uri()
    fig_html = "".join(
        f'<figure><img src="{rel}/{f["image"]}"><figcaption><b>{html.escape(f["label"])}</b> '
        f'{html.escape(f["caption"])}<br><small>{html.escape(f["source_short"])}</small></figcaption></figure>'
        for f in figs
    )
    src_html = "".join(f"<li><b>[S{i}]</b> {html.escape(h.cite)}</li>" for i, h in enumerate(hits, 1))
    path.write_text(
        f"""<!doctype html><meta charset="utf-8"><title>OSmosis</title>
<style>body{{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;line-height:1.55}}
pre{{white-space:pre-wrap;background:#f6f7f9;padding:1rem;border-radius:8px}}
figure{{margin:1rem 0;border:1px solid #ddd;border-radius:8px;padding:.5rem}} img{{max-width:100%}}</style>
<h2>{html.escape(question)}</h2><pre>{html.escape(answer)}</pre>
{f'<h3>Figures</h3>{fig_html}' if figs else ''}{f'<h3>Sources</h3><ol>{src_html}</ol>' if hits else ''}""",
        encoding="utf-8",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("-k", type=int, default=TOP_K)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--html", type=Path)
    a = ap.parse_args()

    r = Retriever()
    print(f"[retrieval: {r.mode}]")
    try:
        reply = answer(r, a.question, k=a.k, use_llm=not a.no_llm)
    except ProblemError as e:
        sys.exit(f"This looks like a numerical, but it cannot be solved as written: {e}\n"
                 "Write it like:  Round Robin, time quantum 3. P1: AT 0, BT 5. P2: AT 1, BT 4.\n"
                 "  or  Banker's algorithm. Allocation: P0 0 1 0, P1 2 0 0. Max: P0 7 5 3, P1 3 2 2. Available: 3 3 2.\n"
                 "  or  LRU, 3 frames. Reference string: 7, 0, 1, 2, 0, 3, 0, 4\n"
                 "  or  SSTF. Head at 53. Requests: 98, 183, 37, 122, 14, 124, 65, 67")
    if reply.error:
        print(f"[LLM unavailable: {reply.error}] showing retrieved sources instead.\n", file=sys.stderr)
    text, hits, figs = reply.text, reply.hits, reply.figs

    print(text)
    if hits:
        print("\nSources:")
    for i, h in enumerate(hits, 1):
        print(f"  [S{i}] {h.cite}")
    if figs:
        print("Figures:")
        for f in figs:
            print(f"  {f['label']:<24} data/{f['image']}")
    if a.html:
        write_html(a.html, a.question, text, hits, figs)
        print(f"\nOpen {a.html} in a browser to see the figures.")


if __name__ == "__main__":
    main()
