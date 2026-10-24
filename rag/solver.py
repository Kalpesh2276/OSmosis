"""Exact CPU-scheduling solver: FCFS, SJF, SRTF, priority (both kinds), Round Robin.

    python -m rag.solver "Round Robin, time quantum 3. P1: AT 0, BT 5. P2: AT 1, BT 4."

The tutor uses it for numerical questions so that the Gantt chart and the CT / TAT / WT
table are computed, not guessed by the LLM. It follows the course conventions in
config.CONVENTIONS:
  - priority: smallest integer = highest priority (unless the question says otherwise)
  - ties: earlier arrival first, then lower process number
  - Round Robin: a process arriving at the instant a quantum expires joins the ready
    queue BEFORE the preempted process
  - TAT = CT - AT, WT = TAT - BT, RT = first time on CPU - AT
Times must be whole numbers. A process may have I/O between CPU bursts
("P2: AT 3, BT 2, I/O 3, BT 2"); as in the teacher's slides, WT = TAT - total CPU burst.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field

ALGOS = {
    "fcfs": "First-Come, First-Served (FCFS)",
    "sjf": "Shortest Job First (non-preemptive)",
    "srtf": "Shortest Remaining Time First (preemptive SJF)",
    "priority": "Priority scheduling (non-preemptive)",
    "priority_p": "Priority scheduling (preemptive)",
    "rr": "Round Robin",
}
PREEMPTIVE = {"srtf", "priority_p"}
MAX_TIME = 100_000


class ProblemError(ValueError):
    """The question looks like a numerical but cannot be solved as written."""


@dataclass
class Proc:
    pid: int
    at: int
    cpu: list[int]                      # CPU bursts, in order
    io: list[int] = field(default_factory=list)   # I/O time after cpu[i]
    prio: int | None = None


@dataclass
class Problem:
    algo: str
    procs: list[Proc]
    quantum: int | None = None
    high_is_better: bool = False        # False: smallest integer = highest priority


@dataclass
class Solution:
    problem: Problem
    gantt: list[tuple[str, int, int]]   # (label, start, end); label "Idle" when the CPU is free
    rows: list[dict]                    # pid, at, bt, ct, tat, wt, rt
    trace: list[str]                    # one line per scheduling decision
    end: int
    steps: list[dict] = field(default_factory=list)   # the same decisions, structured (see solve)


# ---------------------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------------------
def solve(p: Problem) -> Solution:
    """Simulate one time unit at a time.

    Besides the text trace, Solution.steps holds one dict per decision point (an arrival, a
    completion, a return from I/O, an expired quantum, a preemption, or a dispatch):
      t, arrived, returned, completed, to_io (process ids), expired (pid whose quantum ran out),
      ready (pids waiting, in queue order, before the choice), remaining ({pid: CPU time left in
      the current burst, for every unfinished process}), before / after (pid on the CPU before
      and after the decision, None = idle), action ("dispatch" | "preempt" | "continue" | "idle"),
      preempted (pid or None).
    The Reasoning Debugger and Predict-Run-Explain read these; they do not change the schedule."""
    if p.algo not in ALGOS:
        raise ProblemError(f"unknown algorithm {p.algo!r}")
    if p.algo == "rr" and not p.quantum:
        raise ProblemError("Round Robin needs a time quantum (e.g. 'time quantum 3').")
    if p.algo.startswith("priority") and any(q.prio is None for q in p.procs):
        raise ProblemError("priority scheduling needs a priority for every process.")
    if len({q.pid for q in p.procs}) != len(p.procs):
        raise ProblemError("a process is listed twice.")
    for q in p.procs:
        if q.at < 0 or not q.cpu or min(q.cpu) <= 0 or len(q.io) != len(q.cpu) - 1:
            raise ProblemError(f"P{q.pid} has an invalid arrival / burst / I/O time.")

    procs = {q.pid: q for q in p.procs}
    burst = {pid: 0 for pid in procs}           # index of the current CPU burst
    rem = {pid: q.cpu[0] for pid, q in procs.items()}
    first, done = {}, {}
    back = {}                                   # pid -> time it returns from I/O
    ready: list[int] = []                       # in order of joining (FCFS / RR use this order)
    running, used = None, 0
    timeline: list[int | None] = []
    trace: list[str] = []
    steps: list[dict] = []

    def primary(pid):                           # what the algorithm compares
        if p.algo in ("sjf", "srtf"):
            return rem[pid]
        if p.algo.startswith("priority"):
            return -procs[pid].prio if p.high_is_better else procs[pid].prio
        return 0

    def tie(pid):
        return (procs[pid].at, pid)

    def show(pid):
        if p.algo in ("sjf", "srtf"):
            return f"P{pid} ({rem[pid]} left)"
        if p.algo.startswith("priority"):
            return f"P{pid} (priority {procs[pid].prio})"
        return f"P{pid}"

    t = 0
    while True:
        events, expired = [], None
        info = {"t": t, "arrived": [], "returned": [], "completed": [], "to_io": [], "expired": None,
                "before": running, "preempted": None}
        # 1. the running process finishes its burst, or its quantum runs out
        if running is not None and rem[running] == 0:
            q = procs[running]
            if burst[running] + 1 < len(q.cpu):
                back[running] = t + q.io[burst[running]]
                burst[running] += 1
                rem[running] = q.cpu[burst[running]]
                events.append(f"P{running} finishes a CPU burst and starts I/O until t={back[running]}")
                info["to_io"].append(running)
            else:
                done[running] = t
                events.append(f"P{running} completes")
                info["completed"].append(running)
            running = None
        elif running is not None and p.algo == "rr" and used == p.quantum:
            expired, running = running, None
            info["expired"] = expired
        if len(done) == len(procs):
            trace.append(f"t={t}: " + "; ".join(events) + ". All processes have finished.")
            steps.append({**info, "ready": [], "remaining": {}, "after": None, "action": "end"})
            break
        # 2. arrivals and I/O returns join the ready queue ...
        returning = [pid for pid, when in back.items() if when == t]
        for pid in sorted([pid for pid, q in procs.items() if q.at == t] + returning, key=tie):
            events.append(f"P{pid} returns from I/O" if pid in returning else f"P{pid} arrives")
            info["returned" if pid in returning else "arrived"].append(pid)
            back.pop(pid, None)
            ready.append(pid)
        # ... and only then the process whose quantum expired (course convention)
        if expired is not None:
            events.append(f"P{expired}'s quantum expires ({rem[expired]} left), it goes to the tail")
            ready.append(expired)
        # 3. preemption
        waiting = ", ".join(show(x) for x in ready) or "empty"
        info["ready"] = list(ready)
        info["remaining"] = {pid: rem[pid] for pid in procs if pid not in done}
        if running is not None and p.algo in PREEMPTIVE and ready:
            best = min(ready, key=lambda x: (primary(x), tie(x)))
            if primary(best) < primary(running):
                events.append(f"{show(best)} preempts {show(running)}")
                info["preempted"] = running
                ready.append(running)
                running = None
        # 4. dispatch
        if running is None and ready:
            if p.algo in ("fcfs", "rr"):
                running = ready[0]
            else:
                running = min(ready, key=lambda x: (primary(x), tie(x)))
            ready.remove(running)
            used = 0
            first.setdefault(running, t)
            what = "Ready queue (head first)" if p.algo in ("fcfs", "rr") else "Ready"
            trace.append(f"t={t}: " + "; ".join(events or ["CPU is free"])
                         + f". {what}: {waiting}. -> P{running} runs.")
            steps.append({**info, "after": running, "action": "preempt" if info["preempted"] else "dispatch"})
        elif events and running is not None:
            trace.append(f"t={t}: " + "; ".join(events)
                         + f". Running: {show(running)}. Ready: {waiting}. -> P{running} continues.")
            steps.append({**info, "after": running, "action": "continue"})
        elif events:
            trace.append(f"t={t}: " + "; ".join(events) + ". Ready: empty. -> CPU is idle.")
            steps.append({**info, "after": None, "action": "idle"})
        # 5. run one time unit
        timeline.append(running)
        if running is not None:
            rem[running] -= 1
            used += 1
        t += 1
        if t > MAX_TIME:
            raise ProblemError("the schedule did not finish; check the numbers.")

    gantt: list[tuple[str, int, int]] = []
    for i, pid in enumerate(timeline):
        label = "Idle" if pid is None else f"P{pid}"
        if gantt and gantt[-1][0] == label:
            gantt[-1] = (label, gantt[-1][1], i + 1)
        else:
            gantt.append((label, i, i + 1))
    rows = []
    for pid in sorted(procs):
        q = procs[pid]
        bt = sum(q.cpu)
        tat = done[pid] - q.at
        rows.append({"pid": pid, "at": q.at, "bt": bt, "prio": q.prio, "ct": done[pid],
                     "tat": tat, "wt": tat - bt, "rt": first[pid] - q.at})
    return Solution(p, gantt, rows, trace, t, steps)


# ---------------------------------------------------------------------------------------
# Text output
# ---------------------------------------------------------------------------------------
def _num(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".")


def title(p: Problem) -> str:
    text = ALGOS[p.algo] + (f", time quantum = {p.quantum}" if p.algo == "rr" else "")
    if p.algo.startswith("priority"):
        text += " - " + ("largest" if p.high_is_better else "smallest") + " number = highest priority"
    return text


def to_text(p: Problem) -> str:
    """The problem as a question that parse_problem reads back unchanged."""
    name = {"fcfs": "FCFS", "sjf": "SJF (non-preemptive)", "srtf": "SRTF", "rr": "Round Robin",
            "priority": "Priority scheduling (non-preemptive)",
            "priority_p": "Priority scheduling (preemptive)"}[p.algo]
    if p.algo == "rr":
        name += f", time quantum {p.quantum}"
    if p.algo.startswith("priority") and p.high_is_better:
        name += ", a higher number means a higher priority"
    procs = []
    for q in p.procs:
        bursts = [f"BT {q.cpu[0]}"] + [f"I/O {io}, BT {cpu}" for io, cpu in zip(q.io, q.cpu[1:])]
        prio = [f"priority {q.prio}"] if p.algo.startswith("priority") else []
        procs.append(f"P{q.pid}: " + ", ".join([f"AT {q.at}"] + bursts + prio) + ".")
    return f"{name}. " + " ".join(procs)


def render(s: Solution) -> str:
    p = s.problem
    n = len(s.rows)
    out = [f"Algorithm: {title(p)}", "", "Gantt chart:",
           "| " + " | ".join(f"{lab} {a}-{b}" for lab, a, b in s.gantt) + " |", ""]
    has_prio = p.algo.startswith("priority")
    head = ["Process", "AT", "BT"] + (["Priority"] if has_prio else []) + ["CT", "TAT", "WT", "RT"]
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "---|" * len(head))
    for r in s.rows:
        cells = [f"P{r['pid']}", r["at"], r["bt"]] + ([r["prio"]] if has_prio else []) \
            + [r["ct"], r["tat"], r["wt"], r["rt"]]
        out.append("| " + " | ".join(str(c) for c in cells) + " |")
    out.append("")
    for name, key in (("TAT", "tat"), ("WT", "wt"), ("RT", "rt")):
        total = sum(r[key] for r in s.rows)
        out.append(f"Average {name} = {total}/{n} = {_num(total / n)}")
    busy = sum(r["bt"] for r in s.rows)
    out.append(f"CPU utilisation = {busy}/{s.end} = {_num(100 * busy / s.end)}%"
               f"    Throughput = {n}/{s.end} = {_num(n / s.end)} processes per time unit")
    out.append("")
    out.append("Formulas: TAT = CT - AT, WT = TAT - BT, RT = first time on CPU - AT."
               + (" BT is the total CPU burst, so time spent in I/O counts towards WT, as in the"
                  " teacher's worked example." if any(q.io for q in p.procs) else ""))
    out += ["", "Decisions:"] + [f"  {line}" for line in s.trace]
    return "\n".join(out)


# ---------------------------------------------------------------------------------------
# Reading a problem out of a question
# ---------------------------------------------------------------------------------------
_PID = re.compile(r"\bP(\d+)\b", re.I)
_KEY = re.compile(
    r"\b(arrival(?:\s*time)?|at(?:\s*time)?|cpu\s*burst(?:\s*time)?|burst(?:\s*time)?|bt|"
    r"i\s*/?\s*o(?:\s*(?:time|burst))?|priority|prio|pri|pr)\b\s*(?:[:=]|is|of)?\s*(\d+)",
    re.I,
)
_QUANTUM = re.compile(r"\b(?:time\s*)?(?:quantum|slice)\b[^\d\n]{0,15}(\d+)|\b(?:tq\s*[:=]?|q\s*[:=])\s*(\d+)", re.I)
_BARE = re.compile(r"^[\s:=\-,(|]*((?:\d+[\s,|;]*){1,3})")
_SAME_ARRIVAL = re.compile(r"(all|same|every|each)\b[^.]{0,40}\b(arriv\w*|time 0|time zero|t\s*=\s*0)", re.I)


def detect_algorithm(text: str) -> str | None:
    """Which scheduling algorithm a piece of text names, if any (a key of ALGOS)."""
    t = text.lower()
    preemptive = bool(re.search(r"(?<!non)(?<!non-)(?<!non )pre-?emptive", t))
    if re.search(r"round[\s-]*robin|\brr\b", t):
        return "rr"
    if re.search(r"\bsrtf?\b|shortest[\s-]*remaining", t):
        return "srtf"
    if re.search(r"\bsj[fn]\b|shortest[\s-]*job", t):
        return "srtf" if preemptive else "sjf"
    if re.search(r"\bfcfs\b|first[\s-]*come", t):
        return "fcfs"
    # "priority" used as a word, not just as a per-process value such as "priority 2"
    if re.search(r"priority(?!\s*(?:[:=]|is|of)?\s*\d)", t):
        return "priority_p" if preemptive else "priority"
    return None


_SCHEDULING = re.compile(r"\b(burst|arrival|arrives|quantum|gantt|turnaround|cpu[- ]?scheduling|"
                         r"round[\s-]*robin|srtf|sj[fn]|preemptive|waiting time|completion time)\b", re.I)
_OTHER = re.compile(r"\b(banker\w*|deadlock\w*|allocation|available|need|resources?|disk|cylinders?|tracks?|"
                    r"heads?|seek|sstf|c-?scan|c-?look|pages?|frames?|reference string|page fault\w*|"
                    r"semaphores?|philosophers?|segments?|tlb|partitions?|holes?|(?:first|best|worst|next)[\s-]*fit|"
                    r"kb|mb)\b", re.I)


def about_other_topic(text: str) -> bool:
    """True for numericals on other topics (Banker's algorithm, disk scheduling, page replacement,
    ...) that name processes P0, P1 ... but are not CPU-scheduling problems."""
    return bool(_OTHER.search(text)) and not _SCHEDULING.search(text)


def parse_problem(text: str) -> Problem | None:
    """Problem if `text` states a scheduling numerical, None if it is an ordinary question.

    Raises ProblemError when it clearly is a numerical but something needed is missing.
    Accepted process formats:  "P1: AT 0, BT 5, priority 2"  or bare columns  "P1 0 5 2"
    (arrival, burst, priority; just burst[, priority] when all arrive at time 0).
    """
    if about_other_topic(text):
        return None
    qm = _QUANTUM.search(text)
    quantum = int(qm.group(1) or qm.group(2)) if qm else None
    body = _QUANTUM.sub(" ", text)
    same_arrival = bool(_SAME_ARRIVAL.search(body))
    marks = list(_PID.finditer(body))
    procs: dict[int, Proc] = {}
    for i, m in enumerate(marks):
        pid = int(m.group(1))
        seg = body[m.end(): marks[i + 1].start() if i + 1 < len(marks) else len(body)]
        at, prio, cpu, io = None, None, [], []
        for key, val in _KEY.findall(seg):
            k, v = re.sub(r"\s+", "", key.lower()), int(val)
            if k.startswith(("arrival", "at")):
                at = v
            elif k.startswith("pr"):
                prio = v
            elif k.startswith("i"):
                io.append(v)
            else:
                cpu.append(v)
        if not cpu:
            bare = _BARE.match(seg)
            nums = [int(x) for x in re.findall(r"\d+", bare.group(1))] if bare else []
            if same_arrival or len(nums) == 1:
                nums = [0] + nums
            if len(nums) >= 2:
                at, cpu = nums[0], [nums[1]]
                prio = nums[2] if len(nums) > 2 else prio
        if cpu and pid not in procs:
            procs[pid] = Proc(pid, at or 0, cpu, io, prio)
    if len(procs) < 2:
        return None
    algo = detect_algorithm(text)
    if algo is None:
        raise ProblemError("say which algorithm to use (FCFS, SJF, SRTF, priority or Round Robin).")
    prob = Problem(algo, [procs[k] for k in sorted(procs)], quantum, bool(_HIGH_IS_BETTER.search(text)))
    for q in prob.procs:
        if len(q.io) != len(q.cpu) - 1:
            raise ProblemError(f"P{q.pid}: write I/O between two CPU bursts, e.g. 'BT 2, I/O 3, BT 2'.")
    return prob


_HIGH_IS_BETTER = re.compile(
    r"(high|larg|bigg|great)\w*\s+(number|integer|value)s?\b[^.]{0,40}high\w*\s+priority", re.I)
_AGAIN = re.compile(r"\b(now|instead|again|redo|re-?solve|repeat|what if|change|do it|try (it )?with|"
                    r"same (process\w*|problem|question|data|numbers))\b", re.I)


def follow_up(text: str, previous: Problem | None) -> Problem | None:
    """A request to re-run the previous numerical with a change ("now with quantum 2",
    "what if we use SRTF instead") -> the changed problem; None if `text` is not one."""
    if previous is None or not _AGAIN.search(text):
        return None
    qm = _QUANTUM.search(text)
    algo = detect_algorithm(text)
    if not qm and algo is None:
        return None
    quantum = int(qm.group(1) or qm.group(2)) if qm else previous.quantum
    return Problem(algo or previous.algo, previous.procs, quantum,
                   previous.high_is_better or bool(_HIGH_IS_BETTER.search(text)))


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    try:
        prob = parse_problem(" ".join(sys.argv[1:]))
    except ProblemError as e:
        sys.exit(f"Cannot solve this: {e}")
    if prob is None:
        sys.exit("No process list found. Write it like:  P1: AT 0, BT 5. P2: AT 1, BT 4.")
    print(render(solve(prob)))


if __name__ == "__main__":
    main()
