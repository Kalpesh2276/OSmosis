"""Exact disk-scheduling simulator: FCFS, SSTF, SCAN, C-SCAN, LOOK and C-LOOK.

    python -m rag.disk "SSTF. Head at 53. Requests: 98, 183, 37, 122, 14, 124, 65, 67"

Conventions (Silberschatz 10e, 11.2; the teacher's slides have no disk-scheduling examples):
  - total head movement = sum of |next - current| over every move, in cylinders
  - SCAN goes all the way to the last cylinder (or 0) before reversing, when requests remain on
    the other side; LOOK reverses at the last request in that direction
  - C-SCAN goes to the end, jumps to the other end and carries on in the same direction;
    C-LOOK jumps from the last request to the first one on the other side. The return seek is
    counted in the total (the total without it is shown as well) unless the question says not to
  - direction: as stated ("towards 0", "towards larger cylinders", or from the previous
    request); if not stated, towards larger cylinder numbers, and the answer says so
  - disk size: as stated ("200 cylinders", "0 to 199"); if not stated, 200 cylinders (0-199)
  - SSTF: ties between two equally near requests go to the one in the current direction
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field

from .numtext import longest_run
from .solver import ProblemError

ALGOS = {"fcfs": "FCFS (first-come, first-served)", "sstf": "SSTF (shortest seek time first)",
         "scan": "SCAN (elevator)", "cscan": "C-SCAN (circular SCAN)", "look": "LOOK", "clook": "C-LOOK"}
SHORT = {"fcfs": "FCFS", "sstf": "SSTF", "scan": "SCAN", "cscan": "C-SCAN", "look": "LOOK", "clook": "C-LOOK"}
DEFAULT_CYLINDERS = 200


@dataclass
class Problem:
    requests: list[int]
    head: int
    algos: list[str]
    up: bool = True                       # moving towards larger cylinder numbers
    cylinders: int = DEFAULT_CYLINDERS    # cylinders 0 .. cylinders - 1
    count_return: bool = True             # C-SCAN / C-LOOK: count the return seek
    assumed: list[str] = field(default_factory=list, compare=False)   # what the question did not say


@dataclass
class Run:
    algo: str
    path: list[int]                 # head positions, starting with the initial head
    jumps: set[int]                 # indices i where path[i] -> path[i+1] is a return seek
    served: list[int]               # requests in the order they are served
    steps: list[str]

    @property
    def moves(self) -> list[int]:
        return [abs(b - a) for a, b in zip(self.path, self.path[1:])]

    @property
    def return_seek(self) -> int:
        return sum(self.moves[i] for i in self.jumps)


@dataclass
class Solution:
    problem: Problem
    runs: list[Run]

    def total(self, run: Run) -> int:
        moves = sum(run.moves)
        return moves if self.problem.count_return else moves - run.return_seek


def _simulate(p: Problem, algo: str) -> Run:
    pending, head, path, jumps, steps, served = list(p.requests), p.head, [p.head], set(), [], []
    top, up = p.cylinders - 1, p.up
    way = "towards larger cylinders" if up else "towards 0"

    def go(to, why, jump=False):
        nonlocal head
        if jump:
            jumps.add(len(path) - 1)
        steps.append(f"{head} -> {to}: {abs(to - head)} cylinders ({why}).")
        head = to
        path.append(to)

    def serve(c, why="serve the request", jump=False):
        pending.remove(c)
        served.append(c)
        go(c, why, jump)

    if algo == "fcfs":
        for c in list(p.requests):
            serve(c, "next request in arrival order")
    elif algo == "sstf":
        while pending:
            best = min(pending, key=lambda c: (abs(c - head), (c < head) if up else (c > head)))
            up = best >= head if best != head else up
            serve(best, f"nearest pending request, {abs(best - head)} away")
    else:
        ahead = sorted(c for c in pending if c >= head) if up else sorted((c for c in pending if c <= head),
                                                                          reverse=True)
        behind = [c for c in pending if c not in ahead]
        steps.append(f"Head at {head}, moving {way}. Requests ahead: {', '.join(map(str, ahead)) or 'none'}; "
                     f"behind: {', '.join(map(str, sorted(behind))) or 'none'}.")
        for c in ahead:
            serve(c)
        end = top if up else 0
        if algo in ("scan", "cscan") and behind and head != end:
            go(end, f"{'SCAN' if algo == 'scan' else 'C-SCAN'} continues to the end of the disk")
        if algo in ("scan", "look"):
            for c in sorted(behind, reverse=up):
                serve(c)
        elif behind:
            if algo == "cscan":
                start = 0 if up else top
                if head != start:
                    go(start, "return seek to the other end, nothing is served on the way", jump=True)
                order = sorted(behind, reverse=not up)
            else:
                order = sorted(behind, reverse=not up)
                serve(order[0], "C-LOOK jumps back to the first request on the other side and serves it", jump=True)
                order = order[1:]
            for c in order:
                serve(c)
    return Run(algo, path, jumps, served, steps)


def solve(p: Problem) -> Solution:
    if len(p.requests) < 1:
        raise ProblemError("give the request queue, e.g. 'requests: 98, 183, 37, 122'.")
    if p.cylinders < 2:
        raise ProblemError("the disk needs at least two cylinders.")
    bad = [c for c in p.requests + [p.head] if not 0 <= c < p.cylinders]
    if bad:
        raise ProblemError(f"cylinder {bad[0]} is outside the disk (0 to {p.cylinders - 1}); "
                           "give the disk size, e.g. '5000 cylinders (0-4999)'.")
    unknown = [a for a in p.algos if a not in ALGOS]
    if unknown or not p.algos:
        raise ProblemError("name the algorithm: FCFS, SSTF, SCAN, C-SCAN, LOOK or C-LOOK.")
    return Solution(p, [_simulate(p, a) for a in p.algos])


# ---------------------------------------------------------------------------------------
# Text output
# ---------------------------------------------------------------------------------------
def _num(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".")


def title(p: Problem) -> str:
    return (f"Disk scheduling - {', '.join(SHORT[a] for a in p.algos)}; head at {p.head}, moving "
            f"{'towards larger cylinders' if p.up else 'towards 0'}, cylinders 0-{p.cylinders - 1}")


def summary(s: Solution) -> str:
    return "; ".join(f"{SHORT[r.algo]}: {s.total(r)} cylinders" for r in s.runs) + "."


def render(s: Solution) -> str:
    p = s.problem
    out = [f"Algorithm: {title(p)}", f"Request queue (in arrival order): {', '.join(map(str, p.requests))}"]
    if p.assumed:
        out.append("Assumed, since the question does not say: " + "; ".join(p.assumed) + ".")
    out.append("")
    for r in s.runs:
        out.append(f"{ALGOS[r.algo]}:")
        out.append("Head path: " + " -> ".join(map(str, r.path)))
        out.append("Order served: " + ", ".join(map(str, r.served)))
        terms = " + ".join(f"|{b} - {a}|" for a, b in zip(r.path, r.path[1:]))
        out.append(f"Total head movement = {terms}")
        out.append(f"  = {' + '.join(map(str, r.moves))} = {sum(r.moves)} cylinders")
        if r.jumps:
            out.append(f"  of which the return seek is {r.return_seek}; without it: {sum(r.moves) - r.return_seek} "
                       f"cylinders. {'Counted' if p.count_return else 'Not counted'} in the answer.")
        out.append(f"Answer: {s.total(r)} cylinders; average seek = {s.total(r)}/{len(p.requests)} = "
                   f"{_num(s.total(r) / len(p.requests))} cylinders per request")
        out += ["Steps:"] + [f"  {line}" for line in r.steps] + [""]
    if len(s.runs) > 1:
        out.append("Comparison: " + summary(s))
    return "\n".join(out).rstrip()


def to_text(p: Problem) -> str:
    """The problem as a one-line question that parse_problem reads back unchanged."""
    text = (f"Disk scheduling with {', '.join(SHORT[a] for a in p.algos)}. The disk has {p.cylinders} cylinders "
            f"(0-{p.cylinders - 1}). Head at {p.head}, moving towards {'larger cylinders' if p.up else '0'}. "
            f"Request queue: {', '.join(map(str, p.requests))}.")
    if not p.count_return and any(a in ("cscan", "clook") for a in p.algos):
        text += " Do not count the return seek."
    return text


# ---------------------------------------------------------------------------------------
# Reading a problem out of a question
# ---------------------------------------------------------------------------------------
_TOPIC = re.compile(r"\bdisk|\bcylinders?\b|\btracks?\b|\bseek|\bsstf\b|\bc-?scan\b|\bc-?look\b|\bscan\b|\blook\b|"
                    r"elevator|r/w\s+head|read[\s/-]*write\s+head|head\s+(?:movement|position)", re.I)
_ALGO_WORDS = {
    "fcfs": re.compile(r"\bfcfs\b|\bfifo\b|first[\s-]*come", re.I),
    "sstf": re.compile(r"\bsstf\b|shortest[\s-]*seek", re.I),
    "scan": re.compile(r"(?<![-\w])scan\b|elevator", re.I),
    "cscan": re.compile(r"\bc[\s-]?scan\b|circular[\s-]*scan", re.I),
    "look": re.compile(r"(?<![-\w])look\b(?!\s+(?:at|for|up|into|like))", re.I),
    "clook": re.compile(r"\bc[\s-]?look\b|circular[\s-]*look", re.I),
}
_HEAD = [   # most explicit first: "head at 53", "currently serving a request at cylinder 2150", "head ... 53"
    re.compile(r"\b(?:head|arm)\b(?!\s+movement)[^\d\n.]{0,25}?\b(?:at|on|is|=|:|position(?:ed)?(?:\s+(?:at|is))?)"
               r"\s*(?:cylinder\s+|track\s+)?(\d+)", re.I),
    re.compile(r"\b(?:currently|initially|presently|starts?|starting)\s+(?:at|from|on|serving)(?:\s+a\s+request\s+at)?"
               r"\s+(?:cylinder\s+|track\s+)?(\d+)", re.I),
    re.compile(r"\b(?:head|arm)\b(?!\s+movement)[^\d\n.]{0,40}?(\d+)", re.I),
]
_PREV = re.compile(r"\bprevious(?:ly)?\b[^\d\n.]{0,40}?(\d+)", re.I)
_SIZE = [re.compile(r"\b(\d+)\s*(?:cylinders|tracks)\b", re.I),
         re.compile(r"\b0\s*(?:-|to|–|\.\.)\s*(\d+)\b", re.I)]
_UP = re.compile(r"towards?\s+(?:the\s+)?(?:larger|higher|greater|bigger|increasing|outer|outside|high|end\b|max)|"
                 r"\b(?:moving|direction|goes|going|moves)\s+(?:is\s+)?(?:up|right|outwards?)\b|"
                 r"\bincreasing\s+(?:order|direction)|moving\s+from\s+0\s+(?:to|towards)\s+\d+", re.I)
_DOWN = re.compile(r"towards?\s+(?:the\s+)?(?:smaller|lower|decreasing|inner|inside|low|0\b|zero|spindle|start|"
                   r"beginning)|\b(?:moving|direction|goes|going|moves)\s+(?:is\s+)?(?:down|left|inwards?)\b|"
                   r"\bdecreasing\s+(?:order|direction)", re.I)
_NO_RETURN = re.compile(r"\b(?:not|without|exclud\w*|ignor\w*|don'?t)\s+(?:count\w*\s+|includ\w*\s+)?(?:the\s+)?"
                        r"(?:return|jump|wrap)", re.I)
_QUEUE = re.compile(r"\b(?:request(?:s|ed)?|queue|order|sequence|cylinders?|tracks?)\b[^\d\n]{0,40}", re.I)


def mentions(text: str) -> bool:
    return bool(re.search(r"\bdisk|\bcylinders?\b|\bseek|\bsstf\b|\bc-?scan\b|\bc-?look\b|elevator|"
                          r"(?<![-\w])(?:scan|look)\b[^.?]{0,40}\bhead|head\s+movement", text, re.I))


def algos_in(text: str) -> list[str]:
    """Algorithms named in `text`, in ALGOS order ("C SCAN" is C-SCAN only, not SCAN as well)."""
    plain = re.sub(r"\bc[\s-]?(?:scan|look)\b|circular[\s-]*(?:scan|look)", " ", text, flags=re.I)
    return [a for a, rx in _ALGO_WORDS.items() if rx.search(plain if a in ("scan", "look") else text)]


def parse_problem(text: str) -> Problem | None:
    """Problem if `text` states a disk-scheduling numerical, None if it is an ordinary question.

    Raises ProblemError when it clearly is one but something needed is missing.
    """
    if not _TOPIC.search(text) or not re.search(r"\bdisk|\bcylinder|\btrack|\bseek|\bsstf|\bc-?scan|\bc-?look|"
                                                r"\bscan\b|\blook\b|elevator|\bhead\b", text, re.I):
        return None
    if re.search(r"\b(burst|arrival|quantum|frames?|alloc\w*)\b", text, re.I):
        return None
    body, assumed = text, []
    cylinders = None
    for rx in _SIZE:                                # "200 cylinders" and/or "0-199"
        m = rx.search(body)
        if m:
            cylinders = cylinders or int(m.group(1)) + (rx is _SIZE[1])
            body = body[:m.start()] + " . " + body[m.end():]
    head_m = next((m for m in (rx.search(body) for rx in _HEAD) if m), None)
    if not head_m:
        if mentions(text) and longest_run(text, 3):
            raise ProblemError("give the starting head position, e.g. 'head at 53'.")
        return None
    head = int(head_m.group(1))
    body = body[:head_m.start()] + " . " + body[head_m.end():]
    up = None
    prev = _PREV.search(body)
    if prev:
        up = head > int(prev.group(1))
        body = body[:prev.start()] + " . " + body[prev.end():]
    if up is None:
        if _DOWN.search(text):
            up = False
        elif _UP.search(text):
            up = True
    requests = []
    for m in _QUEUE.finditer(body):
        run = longest_run(body[m.end():m.end() + 400], at_least=2)
        if len(run) > len(requests):
            requests = run
    requests = requests or longest_run(body, at_least=3)
    if not requests:
        if mentions(text):
            raise ProblemError("give the request queue, e.g. 'requests: 98, 183, 37, 122'.")
        return None
    algos = algos_in(text) or list(ALGOS)
    sweeping = any(a in ("scan", "cscan", "look", "clook", "sstf") for a in algos)
    if up is None:
        up = True
        if sweeping:
            assumed.append("the head moves towards larger cylinder numbers")
    if cylinders is None:
        cylinders = max(DEFAULT_CYLINDERS, max(requests + [head]) + 1)
        if any(a in ("scan", "cscan") for a in algos):
            assumed.append(f"the disk has {cylinders} cylinders (0-{cylinders - 1})")
    return Problem(requests, head, algos, up, cylinders, not _NO_RETURN.search(text), assumed)


_AGAIN = re.compile(r"\b(now|instead|again|redo|re-?solve|repeat|what if|what about|how about|change|do it|"
                    r"try (it )?with|same (queue|requests|problem|question|data|numbers))\b", re.I)


def follow_up(text: str, previous: Problem | None) -> Problem | None:
    """'What about C-LOOK?' or 'now with the head moving towards 0' after a disk problem."""
    if previous is None or not _AGAIN.search(text):
        return None
    algos = algos_in(text)
    up = False if _DOWN.search(text) else True if _UP.search(text) else None
    if not algos and up is None:
        return None
    assumed = [a for a in previous.assumed if not (up is not None and "moves towards" in a)]
    return Problem(previous.requests, previous.head, algos or previous.algos,
                   previous.up if up is None else up, previous.cylinders, previous.count_return, assumed)


def narrow(p: Problem, text: str) -> Problem:
    """A worked example solved only with the algorithm(s) the question names."""
    algos = algos_in(text)
    return Problem(p.requests, p.head, algos, p.up, p.cylinders, p.count_return, p.assumed) if algos else p


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    try:
        prob = parse_problem(" ".join(sys.argv[1:]))
        if prob is None:
            sys.exit("No disk queue found. Write it like:  SSTF. Head at 53. Requests: 98, 183, 37, 122, 14")
        print(render(solve(prob)))
    except ProblemError as e:
        sys.exit(f"Cannot solve this: {e}")


if __name__ == "__main__":
    main()
