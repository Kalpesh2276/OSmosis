"""Reasoning Debugger: check a student's own working against the exact solvers.

    diagnose(problem, answer) -> Diagnosis

`problem` is any solver's Problem (rag.numericals); `answer` is a dict of what the student wrote:
  CPU scheduling  {"gantt": "P1 0-3, P2 3-7, ..." or [(label, start, end)],
                   "table": {pid: {"ct": .., "tat": .., "wt": .., "rt": ..}}, "averages": {"tat": .., "wt": ..}}
  Banker's        {"need": [[..]], "safe": True/False, "sequence": [pid, ...], "request": "granted"|"wait"|...}
  Page replacement{"marks": "FFFH..." (one F or H per reference), "faults": n}
  Disk scheduling {"order": [cylinder, ...], "total": n}
Every part is optional; only what was given is judged.

The verdict is deterministic: the reference comes from the solver, and every finding is a checked
difference. A misconception id (rag/misconceptions.py) is attached only when the answer shows the
specific pattern that defines it; `certain` says whether that cause is established or only likely.
Findings are ordered by where they occur in the working, so findings[0] is the earliest verifiable
error. Differences that follow from an earlier mistake are listed as its consequences, not as new
mistakes. Each finding carries staged feedback (observation, hint, explanation, correction); the
observation and hint never give the correct value away.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from . import banker, disk, paging, solver
from .numericals import kind as kind_of


class AnswerError(ValueError):
    """Part of the answer could not be read; the message says how to write it."""


@dataclass
class Finding:
    where: str                       # "Gantt chart at t=5", "WT of P2"
    observation: str                 # stage 1: what is off, without the answer
    hint: str                        # stage 2: the rule to reconsider, without the answer
    explanation: str                 # stage 3: why the step is wrong
    correction: str                  # stage 4: the correct step and its effect
    misconception: str | None = None
    certain: bool = True             # False: the cause is likely, not established
    consequences: list[str] = field(default_factory=list)


@dataclass
class Diagnosis:
    kind: str                        # "scheduling" | "banker" | "paging" | "disk"
    problem: object
    solution: object
    findings: list[Finding] = field(default_factory=list)
    correct: list[str] = field(default_factory=list)   # parts that were right
    unclear: list[str] = field(default_factory=list)   # parts that could not be read

    @property
    def verdict(self) -> str:
        if self.findings:
            return "incorrect"
        if self.unclear or not self.correct:
            return "unclear"
        return "correct"

    @property
    def first(self) -> Finding | None:
        return self.findings[0] if self.findings else None


def diagnose(problem, answer: dict) -> Diagnosis:
    return {"scheduling": _scheduling, "banker": _banker, "paging": _paging, "disk": _disk}[kind_of(problem)](
        problem, answer)


def _same(a, b, tol=0.011) -> bool:
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


def _num(x) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".") if isinstance(x, float) else str(x)


# =======================================================================================
# CPU scheduling
# =======================================================================================
_SEG = re.compile(r"(P\d+|idle)\s*[:(\[]?\s*(\d+)\s*(?:-|–|to|,|\s)\s*(\d+)\s*[)\]]?", re.I)
_SEG_REV = re.compile(r"(\d+)\s*(?:-|–|to)\s*(\d+)\s*:?\s*(P\d+|idle)", re.I)
GANTT_FORMAT = "Write the chart like: P1 0-3, P2 3-7, Idle 7-8, P1 8-10."


def parse_gantt(text) -> list[tuple[str, int, int]]:
    """'P1 0-3, P2 3-7' (also 'P1(0-3)', '0-3 P1', or '| P1 | P2 |' then '0 3 7') -> segments."""
    if not isinstance(text, str):
        return [(str(l), int(a), int(b)) for l, a, b in text]
    segs = [(l, int(a), int(b)) for l, a, b in _SEG.findall(text)]
    if not segs:
        segs = [(l, int(a), int(b)) for a, b, l in _SEG_REV.findall(text)]
    if not segs:
        labels = re.findall(r"P\d+|idle", text, re.I)
        times = [int(x) for x in re.findall(r"\d+", re.sub(r"P\d+", " ", text, flags=re.I))]
        if labels and len(times) == len(labels) + 1:
            segs = [(l, times[i], times[i + 1]) for i, l in enumerate(labels)]
    if not segs:
        raise AnswerError(GANTT_FORMAT)
    out = []
    for l, a, b in segs:
        if b <= a:
            raise AnswerError(f"'{l} {a}-{b}': a slice must end after it starts. {GANTT_FORMAT}")
        out.append(("Idle" if l.lower() == "idle" else l.upper(), a, b))
    return out


def timeline(segs) -> list[int | None]:
    """One entry per time unit: the process id running, None when idle."""
    tl: list[int | None] = []
    for label, a, b in sorted(segs, key=lambda s: s[1]):
        if a < len(tl):
            raise AnswerError(f"two slices overlap at t={a}. {GANTT_FORMAT}")
        tl += [None] * (a - len(tl))              # a gap in the chart is idle time
        tl += [None if label == "Idle" else int(label[1:])] * (b - a)
    return tl


def _segments(tl, start=0) -> list[tuple[str, int, int]]:
    out = []
    for t in range(start, len(tl)):
        label = "Idle" if tl[t] is None else f"P{tl[t]}"
        if out and out[-1][0] == label:
            out[-1] = (label, out[-1][1], t + 1)
        else:
            out.append((label, t, t + 1))
    return out


def _chart(segs) -> str:
    return ", ".join(f"{l} {a}-{b}" for l, a, b in segs)


def rule(p: solver.Problem) -> str:
    better = "larger" if p.high_is_better else "smaller"
    return {
        "fcfs": "FCFS runs the ready process that arrived first (ties: lower process number) and never preempts.",
        "sjf": "Non-preemptive SJF chooses, whenever the CPU becomes free, the ready process with the shortest "
               "burst; a running process is never interrupted.",
        "srtf": "SRTF always runs the process with the least remaining time; a newcomer with less remaining time "
                "preempts the running process.",
        "priority": f"Non-preemptive priority chooses, whenever the CPU becomes free, the ready process with the "
                    f"highest priority ({better} number = higher priority); it never interrupts a running process.",
        "priority_p": f"Preemptive priority always runs the highest-priority process ({better} number = higher); a "
                      f"newcomer with higher priority preempts the running one.",
        "rr": f"Round Robin runs the head of the ready queue for at most one quantum ({p.quantum}); a process whose "
              f"quantum expires goes to the tail, after any process arriving at that same instant.",
    }[p.algo]


def _show(p: solver.Problem, step: dict, pid: int) -> str:
    if p.algo in ("sjf", "srtf"):
        return f"P{pid} ({step['remaining'].get(pid, '?')} left)"
    if p.algo.startswith("priority"):
        return f"P{pid} (priority {next(q.prio for q in p.procs if q.pid == pid)})"
    if p.algo == "fcfs":
        return f"P{pid} (arrived at {next(q.at for q in p.procs if q.pid == pid)})"
    return f"P{pid}"


def _candidates(step: dict) -> list[int]:
    """Processes that could run right after this decision point."""
    cands = list(step["ready"])
    before = step["before"]
    if before is not None and before not in step["completed"] + step["to_io"] and before != step["expired"] \
            and before not in cands:
        cands.append(before)
    return cands


def _other_schedules(p: solver.Problem) -> dict[str, list]:
    """The same processes under the other algorithms that apply to them."""
    out = {}
    for algo in solver.ALGOS:
        if algo == p.algo or (algo.startswith("priority") and any(q.prio is None for q in p.procs)):
            continue
        if algo == "rr" and not p.quantum:
            continue
        try:
            s = solver.solve(solver.Problem(algo, p.procs, p.quantum, p.high_is_better))
            out[algo] = timeline(s.gantt)
        except solver.ProblemError:
            continue
    return out


def _rr_preempted_first(p: solver.Problem) -> list | None:
    """Round Robin with the opposite tie rule (the preempted process re-joins the queue before a
    process arriving at the same instant), for recognising that misconception. CPU bursts only."""
    if p.algo != "rr" or any(q.io for q in p.procs):
        return None
    rem = {q.pid: q.cpu[0] for q in p.procs}
    arrive = sorted(p.procs, key=lambda q: (q.at, q.pid))
    queue, tl, t, k = [], [], 0, 0
    while any(rem.values()):
        while k < len(arrive) and arrive[k].at <= t:
            queue.append(arrive[k].pid)
            k += 1
        if not queue:
            tl.append(None)
            t += 1
            continue
        pid = queue.pop(0)
        run = min(p.quantum, rem[pid])
        for _ in range(run):
            tl.append(pid)
            t += 1
        rem[pid] -= run
        while k < len(arrive) and arrive[k].at < t:    # arrived during the quantum: already queued
            queue.append(arrive[k].pid)
            k += 1
        if rem[pid]:
            queue.append(pid)                     # before this instant's arrivals: the opposite rule
        while k < len(arrive) and arrive[k].at <= t:
            queue.append(arrive[k].pid)
            k += 1
    return tl


def _gantt_finding(p: solver.Problem, s: solver.Solution, stu: list) -> Finding | None:
    ref = timeline(s.gantt)
    t = next((i for i in range(max(len(ref), len(stu))) if (ref[i] if i < len(ref) else "end")
              != (stu[i] if i < len(stu) else "end")), None)
    if t is None:
        return None
    S = stu[t] if t < len(stu) else "end"
    R = ref[t] if t < len(ref) else "end"
    nm = lambda x: "the CPU idle" if x is None else ("nothing (the chart ends)" if x == "end" else f"P{x}")
    step = next((st for st in s.steps if st["t"] == t), None)
    procs = {q.pid: q for q in p.procs}
    fix = f"Correct chart from t={t}: {_chart(_segments(ref, t))}."
    obs = f"Your Gantt chart first differs from the correct schedule at t={t}; everything before t={t} is right."
    mis, certain = None, True
    hint = f"At t={t}, write down which processes are ready and apply the rule. {rule(p)}"

    # the whole chart is another algorithm's schedule
    for algo, tl in _other_schedules(p).items():
        if tl == stu:
            return Finding(f"Gantt chart at t={t}", obs + " In fact the whole chart follows one consistent rule.",
                           f"Check which algorithm the question asks for, and whether it is preemptive. {rule(p)}",
                           f"Your chart is exactly the {solver.ALGOS[algo]} schedule for these processes, but the "
                           f"question asks for {solver.ALGOS[p.algo]}.",
                           fix, "sched.wrong_algorithm", True)
    if _rr_preempted_first(p) == stu:
        return Finding(f"Gantt chart at t={t}", obs, "When a quantum expires at the same instant a process arrives, "
                       "which of the two joins the ready queue first?",
                       "Your chart is exactly Round Robin with the preempted process put back in the queue before a "
                       "process arriving at the same instant. The course's convention is the other way round: the "
                       "new arrival joins first, then the preempted process goes to the tail.",
                       fix, "sched.arrival_boundary", True)

    if S == "end" or R == "end":
        short = [pid for pid in procs if stu.count(pid) != sum(procs[pid].cpu)]
        detail = ", ".join(f"P{pid} runs {stu.count(pid)} units instead of {sum(procs[pid].cpu)}" for pid in short)
        return Finding(f"Gantt chart at t={t}", obs + f" Your chart {'ends' if S == 'end' else 'continues'} there.",
                       "Add up each process's slices in your chart and compare with its total CPU burst.",
                       f"In your chart {detail or 'the total length is wrong'}; every process must get exactly its "
                       f"CPU burst.", fix, None, False)

    if step is None:
        # no event at t in the correct schedule: the running process simply continues
        if p.algo == "rr":
            mis = "sched.rr_queue"
            why = f"No quantum expires at t={t}: P{R} has run for fewer than {p.quantum} units since it started."
        elif S is None or stu.count(R) < sum(procs[R].cpu):
            why = f"Nothing happens at t={t}: no arrival, and P{R} has not finished its burst, so it keeps the CPU."
            certain = False
        else:
            mis = "sched.preemption" if p.algo not in solver.PREEMPTIVE else "sched.selection_rule"
            why = f"Nothing arrives at t={t}, so nothing can take the CPU from P{R}."
        return Finding(f"Gantt chart at t={t}", obs, hint, f"{why} You have {nm(S)} there.", fix, mis, certain)

    cands = _candidates(step)
    listing = ", ".join(_show(p, step, x) for x in cands) or "none"
    events = ([f"P{x} arrives" for x in step["arrived"]] + [f"P{x} returns from I/O" for x in step["returned"]]
              + [f"P{x} completes" for x in step["completed"]] + [f"P{x} starts I/O" for x in step["to_io"]]
              + ([f"P{step['expired']}'s quantum expires"] if step["expired"] else []))
    at_t = f"At t={t}: {'; '.join(events) if events else 'nothing new happens'}."
    if S is None:
        mis = "sched.arrival_boundary"
        why = f"{at_t} P{R} is ready, so the CPU must not be idle."
    elif S not in cands:
        if procs[S].at > t:
            mis = "sched.arrival_boundary"
            why = f"{at_t} P{S} only arrives at t={procs[S].at}, so it cannot run yet."
        elif S in step["to_io"] or (S not in step["remaining"]):
            mis = "sched.io_burst" if S in step["to_io"] or procs[S].io else None
            why = (f"{at_t} P{S} is doing I/O, so it is not in the ready queue." if mis else
                   f"{at_t} P{S} has already finished.")
            certain = mis is not None
        else:
            mis = "sched.io_burst" if procs[S].io else None
            why = f"{at_t} P{S} is not ready at this moment (it is in I/O)."
            certain = mis is not None
    elif p.algo == "rr":
        if step["expired"] == S and stu[t - 1] == S:
            mis = "sched.rr_queue"
            why = f"{at_t} P{S} has used its quantum of {p.quantum}, so it must go to the tail of the queue."
        elif step["expired"] == S and R in step["arrived"]:
            mis = "sched.arrival_boundary"
            why = (f"{at_t} P{R} arrives at the same instant P{S}'s quantum expires. In this course the arrival "
                   f"joins the queue first, so the queue is {', '.join(f'P{x}' for x in step['ready'])}.")
        else:
            mis = "sched.rr_queue"
            why = f"{at_t} The ready queue (head first) is {', '.join(f'P{x}' for x in step['ready'])}."
    elif p.algo == "fcfs":
        mis = "sched.selection_rule"
        why = f"{at_t} Ready: {listing}. FCFS takes the one that arrived first."
    else:
        before = step["before"]
        if before == S and step["preempted"] == S:
            mis = "sched.preemption"
            why = f"{at_t} Ready now: {listing}. P{R} beats the running P{S}, so it preempts it."
        elif before == R and p.algo not in solver.PREEMPTIVE and R in cands:
            mis = "sched.preemption"
            why = f"{at_t} {solver.ALGOS[p.algo]} is non-preemptive: P{R} keeps the CPU until it finishes its burst."
        else:
            mis = "sched.selection_rule"
            why = f"{at_t} Ready: {listing}. The rule picks P{R}."
    return Finding(f"Gantt chart at t={t}", obs, hint, f"{why} You have {nm(S)} there.", fix, mis, certain)


def _from_chart(p: solver.Problem, tl: list) -> dict[int, dict]:
    """CT and first start of each process according to a (student's) chart."""
    out = {}
    for q in p.procs:
        times = [t for t, x in enumerate(tl) if x == q.pid]
        if times:
            out[q.pid] = {"ct": times[-1] + 1, "start": times[0]}
    return out


def _slice_ends(tl: list, pid: int) -> list[int]:
    return [t + 1 for t in range(len(tl)) if tl[t] == pid and (t + 1 == len(tl) or tl[t + 1] != pid)]


def _scheduling(p: solver.Problem, answer: dict) -> Diagnosis:
    s = solver.solve(p)
    d = Diagnosis("scheduling", p, s)
    ref_tl = timeline(s.gantt)
    stu_tl = None
    if answer.get("gantt"):
        try:
            stu_tl = timeline(parse_gantt(answer["gantt"]))
        except AnswerError as e:
            d.unclear.append(f"Gantt chart: {e}")
    if stu_tl is not None:
        f = _gantt_finding(p, s, stu_tl)
        if f:
            mine, theirs = _from_chart(p, stu_tl), {r["pid"]: r["ct"] for r in s.rows}
            f.consequences = [f"CT of P{pid} becomes {theirs[pid]} (yours: {mine.get(pid, {}).get('ct', '-')})"
                              for pid in theirs if mine.get(pid, {}).get("ct") != theirs[pid]]
            d.findings.append(f)
        else:
            d.correct.append("Gantt chart")
    basis = _from_chart(p, stu_tl) if stu_tl is not None else \
        {r["pid"]: {"ct": r["ct"], "start": r["rt"] + r["at"]} for r in s.rows}
    _scheduling_table(p, s, answer.get("table") or {}, basis, stu_tl, d)
    _scheduling_averages(p, s, answer, d)
    return d


def _scheduling_table(p, s, table, basis, stu_tl, d: Diagnosis):
    rows = {r["pid"]: r for r in s.rows}
    procs = {q.pid: q for q in p.procs}
    own_chart = stu_tl is not None and d.first is not None    # values judged against the student's own chart
    ok = {"ct": [], "tat": [], "wt": [], "rt": []}
    for pid in sorted(procs):
        given = {k: v for k, v in (table.get(pid) or table.get(str(pid)) or table.get(f"P{pid}") or {}).items()
                 if v is not None and v != ""}
        if not given:
            continue
        q, r = procs[pid], rows[pid]
        at, bt, io = q.at, sum(q.cpu), sum(q.io)
        ct = given.get("ct")
        exp_ct = basis.get(pid, {}).get("ct", r["ct"])
        if ct is not None:
            if _same(ct, exp_ct):
                ok["ct"].append(pid)
            else:
                ends = _slice_ends(stu_tl if stu_tl is not None else timeline(s.gantt), pid)
                early = len(ends) > 1 and any(_same(ct, e) for e in ends[:-1])
                src = "your Gantt chart" if own_chart else "the schedule"
                d.findings.append(Finding(
                    f"CT of P{pid}", f"The completion time of P{pid} does not match {src}.",
                    "CT is the time at which the process's last slice ends.",
                    (f"{_num(ct)} is the end of an earlier slice of P{pid}; it runs again later." if early else
                     f"In {src}, P{pid}'s last slice ends at {exp_ct}, not {_num(ct)}."),
                    f"CT of P{pid} = {exp_ct}.", "sched.ct_reading" if early else None, early))
        use_ct = ct if ct is not None else exp_ct
        tat = given.get("tat")
        exp_tat = use_ct - at
        if tat is not None:
            if _same(tat, exp_tat):
                ok["tat"].append(pid)
            else:
                pattern = ("you wrote CT itself (AT was not subtracted)" if _same(tat, use_ct) else
                           "that is the waiting time (TAT - BT)" if _same(tat, exp_tat - bt) else None)
                d.findings.append(Finding(
                    f"TAT of P{pid}", f"The turnaround time of P{pid} does not follow from its CT.",
                    "TAT = CT - AT.", f"TAT = {_num(use_ct)} - {at} = {_num(exp_tat)}, not {_num(tat)}"
                    + (f"; {pattern}." if pattern else "."), f"TAT of P{pid} = {_num(exp_tat)}.",
                    "sched.metric_confusion" if pattern else None, bool(pattern)))
        use_tat = tat if tat is not None else exp_tat
        wt = given.get("wt")
        exp_wt = use_tat - bt
        first_start = basis.get(pid, {}).get("start", r["rt"] + at)
        if wt is not None:
            # right if it follows from the student's own TAT, or from the TAT their CT gives
            if _same(wt, exp_wt) or _same(wt, exp_tat - bt):
                ok["wt"].append(pid)
            else:
                if io and _same(wt, exp_wt - io):
                    mis, pattern = "sched.io_burst", "you also subtracted the I/O time; in this course WT = TAT - total CPU burst"
                elif _same(wt, first_start - at) and not _same(first_start - at, exp_wt):
                    mis, pattern = "sched.metric_confusion", "that is the response time (first time on CPU - AT)"
                elif _same(wt, use_ct - bt) and at:
                    mis, pattern = "sched.metric_confusion", "that is CT - BT; the arrival time was not subtracted"
                else:
                    mis, pattern = None, None
                d.findings.append(Finding(
                    f"WT of P{pid}", f"The waiting time of P{pid} does not follow from its TAT.",
                    "WT = TAT - BT, where BT is the total CPU burst.",
                    f"WT = {_num(use_tat)} - {bt} = {_num(exp_wt)}, not {_num(wt)}" + (f"; {pattern}." if pattern else "."),
                    f"WT of P{pid} = {_num(exp_wt)}.", mis, bool(mis)))
        rt = given.get("rt")
        exp_rt = first_start - at
        if rt is not None:
            if _same(rt, exp_rt):
                ok["rt"].append(pid)
            else:
                pattern = _same(rt, exp_wt) and not _same(exp_rt, exp_wt)
                d.findings.append(Finding(
                    f"RT of P{pid}", f"The response time of P{pid} does not match when it first runs.",
                    "RT = the time the process first gets the CPU - AT.",
                    f"P{pid} first runs at t={first_start}, so RT = {first_start} - {at} = {exp_rt}, not {_num(rt)}"
                    + ("; that is the waiting time." if pattern else "."),
                    f"RT of P{pid} = {exp_rt}.", "sched.metric_confusion" if pattern else None, pattern))
    names = {"ct": "CT", "tat": "TAT", "wt": "WT", "rt": "RT"}
    for k, pids in ok.items():
        if pids:
            d.correct.append(f"{names[k]} of " + ", ".join(f"P{x}" for x in pids))


def _scheduling_averages(p, s, answer, d: Diagnosis):
    avgs = answer.get("averages") or {}
    table = answer.get("table") or {}
    n = len(s.rows)
    for key, name in (("tat", "TAT"), ("wt", "WT"), ("rt", "RT")):
        given = avgs.get(key)
        if given in (None, ""):
            continue
        mine = [((table.get(r["pid"]) or table.get(str(r["pid"])) or table.get(f"P{r['pid']}") or {}).get(key))
                for r in s.rows]
        own = all(v not in (None, "") for v in mine)
        exp = sum(float(v) for v in mine) / n if own else sum(r[key] for r in s.rows) / n
        if _same(given, exp):
            d.correct.append(f"average {name}")
        else:
            d.findings.append(Finding(
                f"average {name}", f"Your average {name} does not match {'your own column' if own else 'the schedule'}.",
                f"Average = sum of the {name} column / number of processes ({n}).",
                f"The {name} values add up to {_num(exp * n)}, and {_num(exp * n)} / {n} = {_num(round(exp, 2))}, "
                f"not {_num(float(given))}.", f"Average {name} = {_num(round(exp, 2))}.", None, False))


# =======================================================================================
# Banker's algorithm
# =======================================================================================
def parse_sequence(text) -> list[int]:
    if not isinstance(text, str):
        return [int(x) for x in text]
    pids = [int(x) for x in re.findall(r"[PT]\s?(\d+)", text, re.I)] or [int(x) for x in re.findall(r"\d+", text)]
    if not pids:
        raise AnswerError("Write the sequence like: P1, P3, P4, P0, P2.")
    return pids


def _banker(p: banker.Problem, answer: dict) -> Diagnosis:
    s = banker.solve(p)
    d = Diagnosis("banker", p, s)
    name = lambda i: f"{p.prefix}{p.pids[i]}"
    vec = lambda v: "(" + " ".join(map(str, v)) + ")"
    need_given = answer.get("need")
    if need_given and s.need:
        bad = next(((i, j) for i, row in enumerate(need_given) for j, v in enumerate(row)
                    if v is not None and v != s.need[i][j]), None)
        if bad:
            i, j = bad
            v, a, mx = need_given[i][j], p.alloc[i][j], p.maxm[i][j]
            pattern = ("Allocation - Max" if v == a - mx else "Max itself" if v == mx else
                       "Max + Allocation" if v == mx + a else None)
            d.findings.append(Finding(
                f"Need of {name(i)}", f"Row {name(i)} of your Need matrix has a wrong entry ({p.resources[j]}).",
                "Need = Max - Allocation, entry by entry.",
                f"Need[{name(i)}][{p.resources[j]}] = {mx} - {a} = {mx - a}; you wrote {v}"
                + (f", which is {pattern}." if pattern else "."),
                f"Need of {name(i)} = {vec(s.need[i])}.", "banker.need_formula" if pattern else None, bool(pattern)))
        else:
            d.correct.append("Need matrix")
    if answer.get("sequence") and not p.detect:
        try:
            seq = parse_sequence(answer["sequence"])
        except AnswerError as e:
            d.unclear.append(f"Safe sequence: {e}")
            seq = None
        if seq is not None:
            _banker_sequence(p, s, seq, d, name, vec)
    safe = answer.get("safe")
    if safe is not None and not p.detect:
        if bool(safe) == s.check.safe:
            d.correct.append("safe / unsafe verdict")
        else:
            d.findings.append(Finding(
                "safe / unsafe verdict", f"The system is not {'safe' if safe else 'unsafe'} as you said.",
                "Run the safety algorithm: Work = Available; let any process whose Need fits in Work finish and add "
                "its Allocation to Work; repeat until no process can go.",
                banker.state(p, s.check), "The safety steps are listed in the solution.",
                "banker.work_update" if not safe else "banker.need_vs_work", False))
    req = answer.get("request")
    if req and s.request:
        verdict = s.request["verdict"]
        if req == verdict:
            d.correct.append("request verdict")
        else:
            step = {"error": "Step 1 (Request <= Need)", "wait": "Step 2 (Request <= Available)",
                    "unsafe": "Step 3 (is the state safe after pretending to allocate?)",
                    "granted": "all three checks"}[verdict]
            d.findings.append(Finding(
                "request verdict", "Your decision on the request is not what the resource-request algorithm gives.",
                "Do the three checks in order: Request <= Need, Request <= Available, then the safety algorithm on "
                "the state after pretending to allocate.",
                f"The decision is made at {step}: " + s.request["reason"],
                " ".join(s.request["steps"][:2]), "banker.request_steps", True))
    return d


def _banker_sequence(p, s, seq, d, name, vec):
    index = {pid: i for i, pid in enumerate(p.pids)}
    unknown = [x for x in seq if x not in index]
    if unknown or len(set(seq)) != len(seq):
        d.unclear.append(f"Safe sequence: list each process once ({', '.join(name(i) for i in range(len(p.pids)))}).")
        return
    work = list(p.available)
    for k, pid in enumerate(seq):
        i = index[pid]
        if any(n > w for n, w in zip(s.need[i], work)):
            j = next(j for j in range(len(work)) if s.need[i][j] > work[j])
            used_alloc = all(a <= w for a, w in zip(p.alloc[i], work))
            d.findings.append(Finding(
                f"safe sequence, position {k + 1}", f"{name(i)} cannot finish at position {k + 1} of your sequence.",
                "At each turn, compare the process's Need (not its Allocation or Max) with the current Work.",
                f"Work before {name(i)} is {vec(work)}, but Need({name(i)}) = {vec(s.need[i])}: "
                f"{p.resources[j]} needs {s.need[i][j]} > {work[j]}."
                + (" Its Allocation would fit - check you compared Need." if used_alloc else ""),
                f"A safe sequence is {banker.sequence(p, s.check)}." if s.check.safe else banker.state(p, s.check),
                "banker.need_vs_work", True))
            return
        work = [w + a for w, a in zip(work, p.alloc[i])]
    if len(seq) < len(p.pids):
        d.unclear.append("Safe sequence: it does not include every process.")
    else:
        same = [p.pids[i] for i in s.check.order] == seq
        d.correct.append("safe sequence" + ("" if same else " (a valid order, different from the slides' scan order)"))


# =======================================================================================
# Page replacement
# =======================================================================================
def parse_marks(text, n: int) -> list[bool]:
    """'F F F H F' or 'FFFHF' (F = fault, H = hit) -> [True, True, True, False, True]."""
    if not isinstance(text, str):
        return [bool(x) for x in text]
    marks = re.findall(r"[FHX✓*-]", text.upper().replace("HIT", "H").replace("FAULT", "F"))
    marks = [m in "FX*" for m in marks]
    if len(marks) != n:
        raise AnswerError(f"Give one mark per reference ({n} in all): F for a fault, H for a hit.")
    return marks


def _paging(p: paging.Problem, answer: dict) -> Diagnosis:
    s = paging.solve(paging.Problem(p.refs, p.frames[:1], p.algos[:1]))
    d = Diagnosis("paging", p, s)
    run = s.runs[0]
    n = len(p.refs)
    marks = None
    if answer.get("marks"):
        try:
            marks = parse_marks(answer["marks"], n)
        except AnswerError as e:
            d.unclear.append(f"Fault marks: {e}")
    if marks is not None:
        t = next((i for i in range(n) if marks[i] != run.faults[i]), None)
        if t is None:
            d.correct.append("hits and faults")
        else:
            others = {a: paging._simulate(p.refs, run.frames, a).faults for a in paging.ALGOS if a != run.algo}
            same = [a for a, f in others.items() if f == marks]
            prefix = [a for a, f in others.items() if f[:t + 1] == marks[:t + 1]]
            before = run.table[t - 1] if t else [None] * run.frames
            resident = p.refs[t] in before
            frames = " ".join("-" if x is None else str(x) for x in before)
            fix = (f"Ref {t + 1} (page {p.refs[t]}) is a {'hit' if resident else 'fault'}. "
                   + run.steps[t].split(";", 1)[-1].strip())
            if same:
                mis, certain = "paging.wrong_algorithm", True
                why = f"Your marks are exactly what {paging.SHORT[same[0]]} gives, but the question asks for " \
                      f"{paging.SHORT[run.algo]}."
            else:
                mis = "paging.wrong_algorithm" if prefix else "paging.hit_fault"
                certain = False
                why = (f"Just before reference {t + 1}, the frames hold {frames}; page {p.refs[t]} is "
                       f"{'there' if resident else 'not there'}."
                       + (f" Your frames had probably drifted: up to here your marks also match "
                          f"{paging.SHORT[prefix[0]]}." if prefix else ""))
            d.findings.append(Finding(
                f"reference {t + 1} (page {p.refs[t]})", f"Your hit/fault mark first goes wrong at reference {t + 1}.",
                f"Before marking, check whether the page is in a frame; on a fault with full frames, "
                f"{paging.ALGOS[run.algo]} picks the victim.", why, fix, mis, certain,
                [f"{sum(marks) - run.n_faults:+d} faults in total"] if sum(marks) != run.n_faults else []))
    faults = answer.get("faults")
    if faults not in (None, ""):
        if int(faults) == run.n_faults:
            d.correct.append("number of page faults")
        elif marks is not None and int(faults) == sum(marks):
            pass   # consistent with the marks, already explained there
        else:
            others = {a: paging._simulate(p.refs, run.frames, a).n_faults for a in paging.ALGOS if a != run.algo}
            match = [a for a, f in others.items() if f == int(faults) and f != run.n_faults]
            d.findings.append(Finding(
                "number of page faults", "Your number of page faults is not right.",
                "Go through the string one reference at a time and keep the frame contents written down.",
                f"{paging.SHORT[run.algo]} gives {run.n_faults} faults, not {faults}."
                + (f" {faults} is what {paging.SHORT[match[0]]} gives." if match else ""),
                f"{run.n_faults} page faults ({n - run.n_faults} hits).",
                "paging.wrong_algorithm" if match else None, False))
    return d


# =======================================================================================
# Disk scheduling
# =======================================================================================
def _disk(p: disk.Problem, answer: dict) -> Diagnosis:
    q = disk.Problem(p.requests, p.head, p.algos[:1], p.up, p.cylinders, p.count_return, p.assumed)
    s = disk.solve(q)
    d = Diagnosis("disk", q, s)
    run = s.runs[0]
    order = None
    if answer.get("order"):
        order = [int(x) for x in re.findall(r"\d+", answer["order"])] if isinstance(answer["order"], str) \
            else [int(x) for x in answer["order"]]
        if order and order[0] == p.head and len(order) == len(run.served) + 1:
            order = order[1:]                      # the student included the starting position
        if sorted(order) != sorted(run.served):
            d.unclear.append("Service order: list every requested cylinder once, in the order served.")
            order = None
    others = {a: disk._simulate(q, a) for a in disk.ALGOS if a != run.algo}
    if order is not None:
        t = next((i for i in range(len(order)) if order[i] != run.served[i]), None)
        if t is None:
            d.correct.append("service order")
        else:
            same = [a for a, r in others.items() if r.served == order]
            prev = p.head if t == 0 else run.served[t - 1]
            why = (f"Your order is exactly {disk.SHORT[same[0]]}'s, but the question asks for "
                   f"{disk.SHORT[run.algo]}." if same else
                   f"From {prev}, {disk.ALGOS[run.algo]} serves {run.served[t]} next, not {order[t]}.")
            d.findings.append(Finding(
                f"service order, request {t + 1}", f"Your order first differs at the {t + 1}th request served.",
                f"From cylinder {prev}, apply the rule of {disk.ALGOS[run.algo]} "
                f"(head moving {'towards larger cylinders' if p.up else 'towards 0'}).", why,
                "Correct order: " + ", ".join(map(str, run.served)) + ".",
                "disk.wrong_algorithm" if same else None, bool(same)))
    total = answer.get("total")
    if total not in (None, ""):
        total = int(total)
        exp = s.total(run)
        if total == exp:
            d.correct.append("total head movement")
        else:
            moves = sum(run.moves)
            if run.jumps and total in (moves - run.return_seek, moves) and total != exp:
                mis, certain = "disk.return_seek", True
                why = (f"{total} {'leaves out' if p.count_return else 'includes'} the return seek of "
                       f"{run.return_seek} cylinders; this question {'counts' if p.count_return else 'does not count'} it.")
            elif order is not None and d.first is None:
                mis, certain = "disk.arithmetic", True
                why = f"Your order is right, but the moves add up to {exp}, not {total}."
            else:
                match = [a for a, r in others.items() if sum(r.moves) == total]
                mis, certain = ("disk.wrong_algorithm", False) if match else (None, False)
                why = f"{disk.SHORT[run.algo]} gives {exp} cylinders, not {total}." + \
                    (f" {total} is what {disk.SHORT[match[0]]} gives." if match else "")
            d.findings.append(Finding(
                "total head movement", "Your total head movement is not right.",
                "Write every move as |next - current| (including any trip to the end of the disk) and add them.",
                why, "Total = " + " + ".join(map(str, run.moves)) + f" = {sum(run.moves)}"
                + (f" ({exp} counted)" if exp != sum(run.moves) else "") + ".", mis, certain))
    return d


# =======================================================================================
# Practice: a similar problem
# =======================================================================================
def similar_problem(problem, rng: random.Random | None = None):
    """A fresh problem of the same kind and algorithm, to check the mistake is fixed."""
    rng = rng or random.Random()
    k = kind_of(problem)
    if k == "scheduling":
        n = len(problem.procs)
        arrivals = sorted([0] + rng.sample(range(1, 2 * n), n - 1))
        prio = rng.sample(range(1, n + 2), n)
        procs = [solver.Proc(i + 1, arrivals[i], [rng.randint(2, 7)], [],
                             prio[i] if problem.algo.startswith("priority") else None) for i in range(n)]
        return solver.Problem(problem.algo, procs, problem.quantum, problem.high_is_better)
    if k == "paging":
        pages = max(problem.refs) + 1
        return paging.Problem([rng.randrange(pages) for _ in problem.refs], problem.frames[:1], problem.algos[:1])
    if k == "disk":
        reqs = rng.sample([c for c in range(problem.cylinders) if c != problem.head], len(problem.requests))
        return disk.Problem(reqs, problem.head, problem.algos[:1], problem.up, problem.cylinders, problem.count_return)
    from .quiz import _banker_state
    p, _ = _banker_state(rng, len(problem.pids), len(problem.available))
    return p
