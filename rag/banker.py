"""Exact Banker's algorithm: Need matrix, safety algorithm, resource-request algorithm and
deadlock detection with several instances per resource.

    python -m rag.banker "Allocation: P0 0 1 0, P1 2 0 0, P2 3 0 2, P3 2 1 1, P4 0 0 2. Max: P0 7 5 3, P1 3 2 2, P2 9 0 2, P3 2 2 2, P4 4 3 3. Available: 3 3 2. P1 requests (1, 0, 2)."

It follows the teacher's slides (Module 3, deadlocks):
  - Need = Max - Allocation; total instances = column sums of Allocation + Available
  - the safety algorithm checks P0, P1, ... in order, pass after pass, and lets each process
    whose Need fits in Work finish as soon as its turn comes (Work = Work + Allocation). This
    gives the safe sequences of Example 1 and of P1's request in Example 2. Other safe sequences
    may exist: any order in which each Need fits the Work at its turn is correct.
  - a request is granted only if Request <= Need, Request <= Available and the state after
    pretending to allocate it is safe
  - detection: a process that holds nothing counts as finished from the start
Accepted layouts: matrices one after the other ("Allocation: P0 0 1 0, P1 2 0 0. Max: ..."),
one row per process with the matrices side by side under a header ("Process Allocation Max
Available"), or one process at a time ("P0: Allocation 0 1 0, Max 7 5 3").
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from dataclasses import dataclass, field

from .numtext import read_vector, vec
from .solver import ProblemError


@dataclass
class Problem:
    pids: list[int]                                  # process numbers, in table order
    alloc: list[list[int]]
    maxm: list[list[int]] | None                     # None in deadlock detection
    available: list[int]
    request: tuple[int, list[int]] | None = None     # (process number, request vector)
    requests: list[list[int]] | None = None          # deadlock detection: the Request matrix
    names: list[str] = field(default_factory=list)   # resource names; A, B, C, ... when empty
    prefix: str = "P"

    @property
    def detect(self) -> bool:
        return self.requests is not None

    @property
    def resources(self) -> list[str]:
        return self.names or [chr(65 + j) for j in range(len(self.available))]


@dataclass
class Check:
    """One run of the safety (or detection) algorithm."""
    order: list[int]          # row indices, in the order the processes finish
    safe: bool
    steps: list[str]
    works: list[list[int]]    # Work after each finish
    checks: list[dict] = field(default_factory=list)   # each comparison: pass, i, need, work, ok (as in steps)


@dataclass
class Solution:
    problem: Problem
    need: list[list[int]] | None
    total: list[int]
    check: Check
    request: dict | None = None   # verdict ("granted" | "unsafe" | "wait" | "error"), steps, check, new state


# ---------------------------------------------------------------------------------------
# Solving
# ---------------------------------------------------------------------------------------
def _name(p: Problem, i: int) -> str:
    return f"{p.prefix}{p.pids[i]}"


def _validate(p: Problem):
    m, n = len(p.available), len(p.pids)
    if n < 1 or m < 1:
        raise ProblemError("give at least one process and one resource type.")
    if len(set(p.pids)) != n:
        raise ProblemError("a process is listed twice.")
    mats = {"Allocation": p.alloc, "Max": p.maxm, "Request": p.requests}
    for label, mat in mats.items():
        if mat is None:
            continue
        if len(mat) != n:
            raise ProblemError(f"the {label} matrix needs one row for each of the {n} processes.")
        for i, row in enumerate(mat):
            if len(row) != m:
                raise ProblemError(f"{_name(p, i)}: its {label} row has {len(row)} values, but there are "
                                   f"{m} resource types (Available has {m}).")
            if min(row) < 0:
                raise ProblemError(f"{_name(p, i)}: {label} values cannot be negative.")
    if min(p.available) < 0:
        raise ProblemError("Available cannot be negative (is the total smaller than what is allocated?).")
    if p.maxm:
        for i, (mx, a) in enumerate(zip(p.maxm, p.alloc)):
            j = next((j for j in range(m) if a[j] > mx[j]), None)
            if j is not None:
                raise ProblemError(f"{_name(p, i)} holds more {p.resources[j]} ({a[j]}) than its Max ({mx[j]}).")
    if p.request:
        pid, req = p.request
        if pid not in p.pids:
            raise ProblemError(f"the request is from {p.prefix}{pid}, which is not in the table.")
        if len(req) != m:
            raise ProblemError(f"the request has {len(req)} values but there are {m} resource types.")


def _run(p: Problem, alloc, need, work, label: str, done=None) -> Check:
    """Safety algorithm: pass after pass over the processes in table order."""
    n, m = len(alloc), len(work)
    finish = list(done) if done else [False] * n
    work = list(work)
    order, steps, works, checks = [], [], [], []
    for i in range(n):
        if finish[i]:
            steps.append(f"{_name(p, i)} holds no resources, so it counts as finished (Finish = true).")
    pass_no = 0
    while not all(finish):
        pass_no += 1
        progressed = False
        for i in range(n):
            if finish[i]:
                continue
            nm = _name(p, i)
            short = next((j for j in range(m) if need[i][j] > work[j]), None)
            record = {"pass": pass_no, "i": i, "need": list(need[i]), "work": list(work), "ok": short is None,
                      "short": short, "step": len(steps)}
            if short is not None:
                steps.append(f"Pass {pass_no}: {nm}: {label} {vec(need[i])} > Work {vec(work)} "
                             f"({p.resources[short]}: {need[i][short]} > {work[short]}), so {nm} waits.")
                checks.append(record)
                continue
            new = [w + a for w, a in zip(work, alloc[i])]
            steps.append(f"Pass {pass_no}: {nm}: {label} {vec(need[i])} <= Work {vec(work)}, so {nm} finishes "
                         f"and releases its Allocation: Work = {vec(work)} + {vec(alloc[i])} = {vec(new)}.")
            checks.append({**record, "work_after": new})
            work, finish[i], progressed = new, True, True
            order.append(i)
            works.append(new)
        if not progressed:
            steps.append(f"Pass {pass_no}: no waiting process can finish, so the algorithm stops.")
            break
    return Check(order, all(finish), steps, works, checks)


def _request(p: Problem, need) -> dict:
    pid, req = p.request
    i = p.pids.index(pid)
    nm, steps = _name(p, i), []
    fits_need = all(r <= x for r, x in zip(req, need[i]))
    steps.append(f"Step 1: Request {vec(req)} <= Need({nm}) {vec(need[i])}? {'Yes' if fits_need else 'No'}.")
    if not fits_need:
        return {"verdict": "error", "steps": steps, "check": None,
                "reason": f"{nm} asks for more than its maximum claim, so the request is refused (error)."}
    fits_avail = all(r <= x for r, x in zip(req, p.available))
    steps.append(f"Step 2: Request {vec(req)} <= Available {vec(p.available)}? {'Yes' if fits_avail else 'No'}.")
    if not fits_avail:
        return {"verdict": "wait", "steps": steps, "check": None,
                "reason": f"The resources are not available, so {nm} must wait."}
    avail = [a - r for a, r in zip(p.available, req)]
    alloc = [list(row) for row in p.alloc]
    alloc[i] = [a + r for a, r in zip(alloc[i], req)]
    new_need = [list(row) for row in need]
    new_need[i] = [x - r for x, r in zip(need[i], req)]
    steps.append(f"Step 3: pretend to allocate: Available = {vec(p.available)} - {vec(req)} = {vec(avail)}; "
                 f"Allocation({nm}) = {vec(p.alloc[i])} + {vec(req)} = {vec(alloc[i])}; "
                 f"Need({nm}) = {vec(need[i])} - {vec(req)} = {vec(new_need[i])}. Run the safety algorithm on "
                 "this state:")
    check = _run(p, alloc, new_need, avail, "Need")
    if check.safe:
        seq = ", ".join(_name(p, k) for k in check.order)
        verdict, reason = "granted", f"The new state is safe (<{seq}>), so the request is GRANTED immediately."
    else:
        verdict, reason = "unsafe", (f"The new state is unsafe, so the request is NOT granted: {nm} must wait and "
                                     "the old state is restored (deadlock avoided).")
    return {"verdict": verdict, "steps": steps, "check": check, "reason": reason,
            "available": avail, "alloc": alloc, "need": new_need}


def solve(p: Problem) -> Solution:
    _validate(p)
    m = len(p.available)
    total = [p.available[j] + sum(row[j] for row in p.alloc) for j in range(m)]
    if p.detect:
        check = _run(p, p.alloc, p.requests, p.available, "Request", [not any(row) for row in p.alloc])
        return Solution(p, None, total, check)
    need = [[mx - a for mx, a in zip(mrow, arow)] for mrow, arow in zip(p.maxm, p.alloc)]
    check = _run(p, p.alloc, need, p.available, "Need")
    return Solution(p, need, total, check, _request(p, need) if p.request else None)


def sequence(p: Problem, check: Check) -> str:
    return "<" + ", ".join(_name(p, i) for i in check.order) + ">"


def state(p: Problem, c: Check) -> str:
    stuck = ", ".join(_name(p, i) for i in range(len(p.pids)) if i not in c.order
                      and (not p.detect or any(p.alloc[i])))
    if p.detect:
        return (f"No deadlock: the order {sequence(p, c)} lets every process finish." if c.safe else
                f"DEADLOCK: {stuck} cannot finish, so they are deadlocked.")
    return (f"The system is in a SAFE state; safe sequence {sequence(p, c)}." if c.safe else
            f"The system is NOT in a safe state: no safe sequence exists ({stuck} cannot finish).")


def verdict(s: Solution) -> str:
    """The result in one or two sentences."""
    return state(s.problem, s.check) + (" " + s.request["reason"] if s.request else "")


# ---------------------------------------------------------------------------------------
# Text output
# ---------------------------------------------------------------------------------------
def title(p: Problem) -> str:
    if p.detect:
        return "Deadlock detection (several instances per resource type)"
    return "Banker's algorithm - safety check" + (" and resource request" if p.request else "")


def _table(p: Problem, cols: list[tuple[str, list[list[int]]]]) -> list[str]:
    head = ["Process"] + [name for name, _ in cols]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for i in range(len(p.pids)):
        out.append("| " + " | ".join([_name(p, i)] + [" ".join(map(str, mat[i])) for _, mat in cols]) + " |")
    return out


def render(s: Solution) -> str:
    p = s.problem
    first, last = _name(p, 0), _name(p, len(p.pids) - 1)
    out = [f"Algorithm: {title(p)}", f"Resources: {' '.join(p.resources)}", ""]
    if p.detect:
        out += _table(p, [("Allocation", p.alloc), ("Request", p.requests)])
    else:
        out += _table(p, [("Allocation", p.alloc), ("Max", p.maxm), ("Need = Max - Allocation", s.need)])
    out += ["", f"Available = {vec(p.available)}",
            f"Total instances = column sums of Allocation + Available = {vec(s.total)}", ""]
    name = "Detection algorithm" if p.detect else "Safety algorithm"
    out.append(f"Steps ({name}: Work starts as Available; each pass checks {first} to {last} in order):")
    out += [f"  {line}" for line in s.check.steps]
    out.append("Result: " + state(p, s.check))
    if not p.detect:
        out.append("  " + ("Other safe sequences may exist; any order in which each process's Need fits in Work "
                           "at its turn is correct." if s.check.safe else "An unsafe state may lead to deadlock."))
    if s.request:
        r = s.request
        pid, req = p.request
        out += ["", f"Resource request: {p.prefix}{pid} requests {vec(req)} (resource-request algorithm)"]
        out += [f"  {line}" for line in r["steps"]]
        if r["check"]:
            out += [f"    {line}" for line in r["check"].steps]
        out.append(f"  Result: {r['reason']}")
    return "\n".join(out)


def to_text(p: Problem) -> str:
    """The problem as a one-line question that parse_problem reads back unchanged."""
    def rows(mat):
        return ", ".join(f"{p.prefix}{pid} " + " ".join(map(str, r)) for pid, r in zip(p.pids, mat))
    avail = " ".join(map(str, p.available))
    if p.detect:
        return (f"Deadlock detection (Banker's algorithm). Allocation: {rows(p.alloc)}. "
                f"Request: {rows(p.requests)}. Available: {avail}.")
    text = f"Banker's algorithm. Allocation: {rows(p.alloc)}. Max: {rows(p.maxm)}. Available: {avail}."
    if p.request:
        text += f" Request from {p.prefix}{p.request[0]}: ({' '.join(map(str, p.request[1]))})."
    return text


# ---------------------------------------------------------------------------------------
# Reading a problem out of a question
# ---------------------------------------------------------------------------------------
_TOPIC = re.compile(r"\bbanker|\bsafe(?:ty)?\b|\bunsafe\b|\bdeadlock|\bmax(?:imum)?\b|\bneed\b", re.I)
_ALLOC = re.compile(r"\balloc", re.I)
_DETECT = re.compile(r"\bdetect", re.I)
_PROC = re.compile(r"\b([PT])\s?(\d{1,2})\b")
_LABEL = re.compile(
    r"\b(allocat\w*|alloc|max(?:imum)?(?:\s+(?:need|demand|claim|matrix))?|need(?:\s+matrix)?|available|avail|"
    r"total(?:\s+(?:no\.?|number)\s+of)?(?:\s+(?:instances|resources))?(?:\s+of\s+each\s+(?:type|resource))?|"
    r"request(?:\s+matrix)?)\b", re.I)
_NUMS = r"(?P<nums>\d+(?:[ \t]*[, \t][ \t]*\d+)*)"
_PID = r"[PT]\s?(?P<pid>\d+)"
_REQUEST = [   # "P1 requests (1, 0, 2)", "P1's request (1,0,2)", "request from P1 arrives for (1,1,0,0)", ...
    re.compile(r"\b" + _PID + r"(?:'s|’s)?\s+(?:makes\s+a\s+request|requests|request|asks)\s*(?:for\s*)?(?:an?\s+)?"
               r"(?:additional\s+)?(?:resources?\s*|instances?\s*)?(?:of\s*)?[:=]?\s*[(\[<]?\s*" + _NUMS, re.I),
    re.compile(r"\brequests?\s*(?:vector\s*)?(?:from|by|of)\s+(?:process\s+)?" + _PID + r"\b[^\d\n]{0,30}?"
               r"[(\[<]?\s*" + _NUMS, re.I),
    re.compile(r"\brequest\s*(?:vector\s*)?[(\[]\s*" + _PID + r"\s*[)\]]\s*[:=]?\s*[(\[<]?\s*" + _NUMS, re.I),
    re.compile(r"\brequest\s*(?:vector\s*)?(?:of\s*)?[(\[<]\s*" + _NUMS + r"\s*[)\]>]\s*(?:from|by)\s+(?:process\s+)?"
               + _PID, re.I),
]
_MATRICES = ("alloc", "max", "need", "req")


def _kind(word: str) -> str:
    w = word.lower()
    for prefix, kind in (("alloc", "alloc"), ("max", "max"), ("need", "need"), ("avail", "avail"),
                         ("total", "total"), ("request", "req")):
        if w.startswith(prefix):
            return kind
    raise AssertionError(word)


def _take_request(text: str):
    for rx in _REQUEST:
        m = rx.search(text)
        if m:
            req = (int(m.group("pid")), [int(x) for x in re.findall(r"\d+", m.group("nums"))])
            return req, text[:m.start()] + " " + text[m.end():]
    return None, text


def mentions(text: str) -> bool:
    """True if the text is about the Banker's algorithm or deadlock detection."""
    return bool(re.search(r"\bbanker|safe(?:ty)?\s+(?:state|sequence|algorithm)|resource[\s-]*request\s+algorithm|"
                          r"deadlock\s+(?:avoidance|detection)|detect\w*\s+(?:a\s+)?deadlock", text, re.I))


def parse_problem(text: str) -> Problem | None:
    """Problem if `text` states a Banker's-algorithm numerical, None if it is an ordinary question.

    Raises ProblemError when it clearly is one but something needed is missing.
    """
    if not (_TOPIC.search(text) and _ALLOC.search(text)):
        return None
    t = re.sub(r"\b[Rr]\d+\b", " ", text)          # resource names R1, R2, ... would read as numbers
    request, t = _take_request(t)
    marks = list(_PROC.finditer(t))
    if len({int(m.group(2)) for m in marks}) < 2:
        return None
    labels = [(m.start(), m.end(), _kind(m.group(1))) for m in _LABEL.finditer(t)]
    mats: dict[str, dict[int, list[int]]] = {k: {} for k in _MATRICES}
    loose = []                                      # (section, pid, numbers) of rows written "P0 0 1 0"
    for i, m in enumerate(marks):
        pid, end = int(m.group(2)), marks[i + 1].start() if i + 1 < len(marks) else len(t)
        seg = t[m.end():end]
        lead = read_vector(seg, skip_words=0)
        if lead:
            before = [k for a, _, k in labels if a < m.start()]
            loose.append((before[-1] if before and before[-1] in _MATRICES else None, pid, lead, m.start()))
            continue
        inside = [(a - m.end(), b - m.end(), k) for a, b, k in labels if m.end() <= a < end]
        for j, (a, b, k) in enumerate(inside):     # "P0: Allocation 0 1 0, Max 7 5 3"
            stop = inside[j + 1][0] if j + 1 < len(inside) else len(seg)
            v = read_vector(seg[b:stop], skip_words=1)
            if v and k in mats:
                mats[k][pid] = v
    vectors: dict[str, list[int]] = {}

    if loose:
        # header labels just before the first row ("Process Allocation Max Available")
        first = loose[0][3]
        header, nxt = [], first
        for a, b, k in reversed([lab for lab in labels if lab[0] < first]):
            if re.search(r"\d", t[b:nxt]):
                break
            header.insert(0, k)
            nxt = a
        kinds = list(dict.fromkeys(header))
        side = [k for k in kinds if k in _MATRICES]
        pids = [pid for _, pid, _, _ in loose]
        if len(side) >= 2 and len(set(pids)) == len(pids):
            lengths = [len(v) for _, _, v, _ in loose]
            width = Counter(lengths[1:] or lengths).most_common(1)[0][0]
            if width % len(side):
                raise ProblemError(f"each row should have {len(side)} groups of numbers ({', '.join(side)}) of "
                                   "equal size.")
            m_res = width // len(side)
            for _, pid, v, _ in loose:
                for j, k in enumerate(side):
                    mats[k][pid] = v[j * m_res:(j + 1) * m_res]
                extra = v[len(side) * m_res:]
                for k in [k for k in kinds if k in ("avail", "total")]:
                    if len(extra) >= m_res and k not in vectors:
                        vectors[k], extra = extra[:m_res], extra[m_res:]
        else:
            for section, pid, v, _ in loose:
                if section is None:
                    raise ProblemError("label the matrices, e.g. 'Allocation: P0 0 1 0, P1 2 0 0. "
                                       "Max: P0 7 5 3, P1 3 2 2. Available: 3 3 2.'")
                mats[section][pid] = v

    for a, b, k in labels:
        if k in ("avail", "total") and k not in vectors:
            v = read_vector(t[b:], skip_words=4)
            if v:
                vectors[k] = v
    if not mats["alloc"]:
        raise ProblemError("give the Allocation matrix, e.g. 'Allocation: P0 0 1 0, P1 2 0 0'.")
    pids = sorted(mats["alloc"])
    m_res = Counter(len(v) for v in mats["alloc"].values()).most_common(1)[0][0]
    # a matrix copied without process names ("MAX A B C 0 0 1 2 1 7 5 0 ...")
    for j, (a, b, k) in enumerate(labels):
        if k in _MATRICES and not mats[k]:
            stop = min([x for x, _, _ in labels[j + 1:]] + [mk.start() for mk in marks if mk.start() > b] + [len(t)])
            nums = read_vector(t[b:stop], skip_words=2)
            if len(nums) >= len(pids) * m_res:
                mats[k] = {pid: nums[i * m_res:(i + 1) * m_res] for i, pid in enumerate(pids)}

    def matrix(kind, label):
        rows = mats[kind]
        missing = [p for p in pids if p not in rows]
        if missing:
            raise ProblemError(f"the {label} matrix has no row for {', '.join(f'P{x}' for x in missing)}.")
        return [rows[p] for p in pids]

    alloc = matrix("alloc", "Allocation")
    detect = bool(mats["req"]) and (not (mats["max"] or mats["need"]) or bool(_DETECT.search(text)))
    if "avail" in vectors:
        avail = vectors["avail"]
    elif "total" in vectors:
        tot = vectors["total"][:m_res]
        avail = [tot[j] - sum(r[j] for r in alloc) for j in range(min(m_res, len(tot)))]
    else:
        raise ProblemError("give Available (or the total number of instances of each resource), "
                           "e.g. 'Available: 3 3 2'.")
    if len(avail) > m_res:                          # "Available 3 3 2 1. Calculate ..." -> drop the "1."
        avail = avail[:m_res]
    prefix = marks[0].group(1).upper()
    if detect:
        return Problem(pids, alloc, None, avail, None, matrix("req", "Request"), prefix=prefix)
    if mats["max"]:
        maxm = matrix("max", "Max")
    elif mats["need"]:
        maxm = [[a + n for a, n in zip(ar, nr)] for ar, nr in zip(alloc, matrix("need", "Need"))]
    else:
        raise ProblemError("give the Max (or the Need) matrix, e.g. 'Max: P0 7 5 3, P1 3 2 2'.")
    return Problem(pids, alloc, maxm, avail, request, prefix=prefix)


def follow_up(text: str, previous: Problem | None) -> Problem | None:
    """'What if P4 requests (0, 0, 2, 0)?' after a Banker's problem -> the same state with that request."""
    if previous is None or previous.detect:
        return None
    request, _ = _take_request(text)
    if request is None:
        return None
    return Problem(previous.pids, previous.alloc, previous.maxm, previous.available, request,
                   names=previous.names, prefix=previous.prefix)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    try:
        prob = parse_problem(" ".join(sys.argv[1:]))
        if prob is None:
            sys.exit("No Banker's-algorithm table found. Write it like:  Allocation: P0 0 1 0, P1 2 0 0. "
                     "Max: P0 7 5 3, P1 3 2 2. Available: 3 3 2.")
        print(render(solve(prob)))
    except ProblemError as e:
        sys.exit(f"Cannot solve this: {e}")


if __name__ == "__main__":
    main()
