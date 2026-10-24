"""Exact page-replacement simulator: FIFO, LRU and Optimal.

    python -m rag.paging "LRU, 3 frames. Reference string: 7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 0, 3, 2"

Conventions (teacher's slides, Module 5 part 2, and Silberschatz 10.4):
  - frames start empty; a fault with a free frame fills the first free frame
  - a replaced page's frame is reused in place, so each frame row of the table is one frame
  - FIFO replaces the page loaded earliest; LRU the page used least recently; Optimal the page
    whose next use is farthest away (among pages never used again, the one loaded earliest -
    the number of faults is the same whichever of them is chosen)
  - with no algorithm named, all three are solved and compared; with several frame counts
    ("3 and 4 frames") each is solved, which shows Belady's anomaly for FIFO
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass

from .numtext import longest_run, read_vector
from .solver import ProblemError

ALGOS = {"fifo": "FIFO (first-in, first-out)", "lru": "LRU (least recently used)",
         "optimal": "Optimal (OPT)"}
SHORT = {"fifo": "FIFO", "lru": "LRU", "optimal": "Optimal"}


@dataclass
class Problem:
    refs: list[int]
    frames: list[int]          # one or more frame counts
    algos: list[str]           # keys of ALGOS


@dataclass
class Run:
    algo: str
    frames: int
    table: list[list[int | None]]   # frame contents after each reference
    faults: list[bool]
    victims: list[int | None]       # page replaced at each reference (None: no replacement)
    steps: list[str]

    @property
    def n_faults(self) -> int:
        return sum(self.faults)


@dataclass
class Solution:
    problem: Problem
    runs: list[Run]


def _simulate(refs: list[int], n: int, algo: str) -> Run:
    frames: list[int | None] = [None] * n
    loaded, used = {}, {}            # page -> time loaded / time last used
    table, faults, victims, steps = [], [], [], []
    for t, page in enumerate(refs):
        if page in frames:
            used[page] = t
            table.append(list(frames))
            faults.append(False)
            victims.append(None)
            steps.append(f"Ref {t + 1}: page {page} is already in a frame -> hit.")
            continue
        if None in frames:
            slot, victim = frames.index(None), None
            why = "a frame is free"
        else:
            resident = [p for p in frames if p is not None]
            if algo == "fifo":
                victim = min(resident, key=lambda p: loaded[p])
                why = f"FIFO replaces {victim}, loaded first (at ref {loaded[victim] + 1})"
            elif algo == "lru":
                victim = min(resident, key=lambda p: used[p])
                why = f"LRU replaces {victim}, least recently used (at ref {used[victim] + 1})"
            else:
                def next_use(p):
                    return next((k for k in range(t + 1, len(refs)) if refs[k] == p), None)
                never = [p for p in resident if next_use(p) is None]
                if never:
                    victim = min(never, key=lambda p: loaded[p])
                    why = (f"Optimal replaces {victim}, never used again" if len(never) == 1 else
                           f"Optimal replaces {victim}: {', '.join(map(str, never))} are never used again, so the "
                           "one loaded first goes")
                else:
                    victim = max(resident, key=next_use)
                    why = f"Optimal replaces {victim}, whose next use (ref {next_use(victim) + 1}) is farthest away"
            slot = frames.index(victim)
            del loaded[victim]
        frames[slot] = page
        loaded[page] = used[page] = t
        table.append(list(frames))
        faults.append(True)
        victims.append(victim)
        steps.append(f"Ref {t + 1}: page {page} is not in memory -> fault; {why}. Frames: "
                     + " ".join("-" if p is None else str(p) for p in frames))
    return Run(algo, n, table, faults, victims, steps)


def solve(p: Problem) -> Solution:
    if len(p.refs) < 1:
        raise ProblemError("give the reference string, e.g. 'reference string 7, 0, 1, 2, 0, 3'.")
    if not p.frames or min(p.frames) < 1:
        raise ProblemError("say how many frames there are, e.g. 'with 3 frames'.")
    if max(p.frames) > 12:
        raise ProblemError("use at most 12 frames.")
    unknown = [a for a in p.algos if a not in ALGOS]
    if unknown or not p.algos:
        raise ProblemError("name the algorithm: FIFO, LRU or Optimal.")
    return Solution(p, [_simulate(p.refs, n, a) for a in p.algos for n in p.frames])


# ---------------------------------------------------------------------------------------
# Text output
# ---------------------------------------------------------------------------------------
def _pct(a: int, b: int) -> str:
    return f"{100 * a / b:.2f}".rstrip("0").rstrip(".") + "%"


def title(p: Problem) -> str:
    frames = " and ".join(map(str, p.frames))
    return f"Page replacement - {', '.join(SHORT[a] for a in p.algos)}, {frames} frame{'s' * (p.frames != [1])}"


def summary(s: Solution) -> str:
    """'FIFO 15 faults, LRU 12 faults, Optimal 9 faults' (+ Belady's anomaly if it shows)."""
    n = len(s.problem.refs)
    parts = [f"{SHORT[r.algo]}{f' ({r.frames} frames)' if len(s.problem.frames) > 1 else ''}: "
             f"{r.n_faults} faults, {n - r.n_faults} hits" for r in s.runs]
    text = "; ".join(parts) + "."
    fifo = sorted((r.frames, r.n_faults) for r in s.runs if r.algo == "fifo")
    for (f1, n1), (f2, n2) in zip(fifo, fifo[1:]):
        if n2 > n1:
            text += (f" Belady's anomaly: FIFO has more faults with {f2} frames ({n2}) than with {f1} "
                     f"frames ({n1}).")
    return text


def render(s: Solution) -> str:
    p = s.problem
    out = [f"Algorithm: {title(p)}", f"Reference string ({len(p.refs)} references): "
           + ", ".join(map(str, p.refs)), ""]
    for r in s.runs:
        out.append(f"{ALGOS[r.algo]} with {r.frames} frames:")
        head = ["Reference"] + [str(x) for x in p.refs]
        out += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for k in range(r.frames):
            out.append("| " + " | ".join([f"Frame {k + 1}"] + ["" if col[k] is None else str(col[k])
                                                                 for col in r.table]) + " |")
        out.append("| " + " | ".join(["Fault"] + ["F" if f else "" for f in r.faults]) + " |")
        n = len(p.refs)
        out.append(f"Page faults = {r.n_faults}, hits = {n - r.n_faults}; fault rate = {r.n_faults}/{n} = "
                   f"{_pct(r.n_faults, n)}, hit ratio = {n - r.n_faults}/{n} = {_pct(n - r.n_faults, n)}")
        out += ["Steps:"] + [f"  {line}" for line in r.steps] + [""]
    if len(s.runs) > 1:
        out.append("Comparison: " + summary(s))
    return "\n".join(out).rstrip()


def to_text(p: Problem) -> str:
    """The problem as a one-line question that parse_problem reads back unchanged."""
    return (f"Page replacement with {', '.join(SHORT[a] for a in p.algos)}. Reference string: "
            f"{', '.join(map(str, p.refs))}. Number of frames: {' and '.join(map(str, p.frames))}.")


# ---------------------------------------------------------------------------------------
# Reading a problem out of a question
# ---------------------------------------------------------------------------------------
_TOPIC = re.compile(r"page[\s-]*replacement|reference\s+string|page\s+(?:faults?|references?|hits?)|\blru\b|"
                    r"belady|\bframes?\b", re.I)
_ALGO_WORDS = {
    "fifo": re.compile(r"\bfifo\b|first[\s-]*in[\s,-]*first[\s-]*out|belady'?s\s+anomaly", re.I),
    "lru": re.compile(r"\blru\b|least[\s-]*recently[\s-]*used", re.I),
    "optimal": re.compile(r"\bopt(?:imal)?\b|\bmin\b(?=\s*(?:algorithm|page))", re.I),
}
# "3 frames", "3 and 4 frames", "frames = 3" (no commas: "..., 4, 5, 3 frames" must not take the string's tail)
_FRAMES = [
    re.compile(r"(\d+(?:\s*,?\s*(?:and|&|or|vs\.?)\s*\d+)?)\s*(?:empty\s+|page\s+|physical\s+|memory\s+|"
               r"available\s+)?frames?\b", re.I),
    re.compile(r"\b(?:number\s+of\s+)?(?:page\s+)?frames?\s*(?:=|:|is|are|of)?\s*(\d+(?:\s*(?:and|&|or)\s*\d+)?)",
               re.I),
]
_STRING = re.compile(r"\b(?:reference\s+string|page\s+references?|references?|string|sequence)\b"
                     r"(?:\s+(?:is|of|as\s+follows))?\s*[:=]?", re.I)


def mentions(text: str) -> bool:
    return bool(re.search(r"page[\s-]*replacement|page\s+faults?|reference\s+string|\blru\b|belady|"
                          r"(?:fifo|optimal)\b[^.?]{0,40}\bpages?\b|\bpages?\b[^.?]{0,40}(?:fifo|optimal)", text, re.I))


def algos_in(text: str) -> list[str]:
    return [a for a, rx in _ALGO_WORDS.items() if rx.search(text)]


def parse_problem(text: str) -> Problem | None:
    """Problem if `text` states a page-replacement numerical, None if it is an ordinary question.

    Raises ProblemError when it clearly is one but something needed is missing.
    """
    if not _TOPIC.search(text) or not re.search(r"\bpages?\b|\bframes?\b|reference|\blru\b|belady", text, re.I):
        return None
    if re.search(r"\b(burst|arrival|quantum|cylinders?|disk|alloc\w*)\b", text, re.I):
        return None
    body, frames = text, []
    for rx in _FRAMES:
        m = rx.search(body)
        if m:
            frames = [int(x) for x in re.findall(r"\d+", m.group(1))]
            body = body[:m.start()] + " . " + body[m.end():]
            break
    refs = []
    for m in _STRING.finditer(body):
        v = read_vector(body[m.end():], skip_words=2)
        if len(v) >= 3:
            refs = v
            break
    refs = refs or longest_run(body, at_least=5)
    if not refs:
        return None
    if not frames:
        if mentions(text):
            raise ProblemError("say how many frames there are, e.g. 'with 3 frames'.")
        return None
    return Problem(refs, sorted(set(frames)), algos_in(text) or list(ALGOS))


_AGAIN = re.compile(r"\b(now|instead|again|redo|re-?solve|repeat|what if|what about|how about|change|do it|"
                    r"try (it )?with|same (string|problem|question|data|numbers))\b", re.I)


def follow_up(text: str, previous: Problem | None) -> Problem | None:
    """'Now with 4 frames' or 'what about LRU?' after a page-replacement problem -> the changed problem."""
    if previous is None or not _AGAIN.search(text):
        return None
    frames = []
    for rx in _FRAMES:
        m = rx.search(text)
        if m:
            frames = [int(x) for x in re.findall(r"\d+", m.group(1))]
            break
    algos = algos_in(text)
    if not frames and not algos:
        return None
    return Problem(previous.refs, sorted(set(frames)) or previous.frames, algos or previous.algos)


def narrow(p: Problem, text: str) -> Problem:
    """A worked example solved only with the algorithm(s) the question names."""
    algos = algos_in(text)
    return Problem(p.refs, p.frames, algos) if algos else p


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    try:
        prob = parse_problem(" ".join(sys.argv[1:]))
        if prob is None:
            sys.exit("No reference string found. Write it like:  LRU, 3 frames. Reference string: 7, 0, 1, 2, 0, 3")
        print(render(solve(prob)))
    except ProblemError as e:
        sys.exit(f"Cannot solve this: {e}")


if __name__ == "__main__":
    main()
