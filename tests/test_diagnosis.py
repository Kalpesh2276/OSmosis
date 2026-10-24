"""Checks the Reasoning Debugger: correct work is accepted, the first wrong step is found, causes are
named only on evidence, consequences are not counted twice, and hints do not give answers away.

    python -m tests.test_diagnosis        (or: pytest tests)
"""
import random

from rag import banker, disk, paging, solver
from rag.diagnosis import AnswerError, diagnose, parse_gantt, similar_problem, timeline
from rag.misconceptions import MISCONCEPTIONS

RR = solver.parse_problem("Round Robin, time quantum 2. P1: AT 0, BT 5. P2: AT 1, BT 3. P3: AT 2, BT 1. "
                          "P4: AT 3, BT 2. P5: AT 4, BT 3.")                       # slides 78-85
SRTF = solver.parse_problem("SRTF. P1: AT 0, BT 8. P2: AT 1, BT 4. P3: AT 2, BT 9. P4: AT 3, BT 5.")   # slides 41-42
FCFS = solver.parse_problem("FCFS. P1: AT 4, BT 5. P2: AT 6, BT 4. P3: AT 0, BT 3. P4: AT 6, BT 2. P5: AT 5, BT 4.")
IO = solver.parse_problem("SRTF. P1: AT 2, BT 4. P2: AT 3, BT 2, I/O 3, BT 2. P3: AT 5, BT 1.")


def full_table(s):
    return {r["pid"]: {k: r[k] for k in ("ct", "tat", "wt", "rt")} for r in s.rows}


def check_feedback_hides_answer(d):
    """Stage 1 and 2 must not contain the correct value or process."""
    for f in d.findings:
        assert f.misconception is None or f.misconception in MISCONCEPTIONS
        if f.where.startswith("Gantt chart at t="):
            t = int(f.where.split("t=")[1])
            ref = timeline(d.solution.gantt)
            if t < len(ref) and ref[t] is not None:
                assert f"P{ref[t]}" not in f.observation + f.hint, f
        assert "Correct chart" not in f.hint and "Correct" not in f.observation


def test_correct_work_is_accepted():
    for p in (RR, SRTF, FCFS, IO):
        s = solver.solve(p)
        d = diagnose(p, {"gantt": ", ".join(f"{l} {a}-{b}" for l, a, b in s.gantt), "table": full_table(s),
                         "averages": {"wt": sum(r["wt"] for r in s.rows) / len(s.rows)}})
        assert d.verdict == "correct" and not d.findings, (p.algo, d.findings)


def test_gantt_formats():
    assert parse_gantt("P1 0-3, P2 3-7") == [("P1", 0, 3), ("P2", 3, 7)]
    assert parse_gantt("| P1 (0-3) | idle 3-4 |") == [("P1", 0, 3), ("Idle", 3, 4)]
    assert parse_gantt("0-3 P1, 3-7 P2") == [("P1", 0, 3), ("P2", 3, 7)]
    assert parse_gantt("| P1 | P2 | P1 |\n0  3  7  9") == [("P1", 0, 3), ("P2", 3, 7), ("P1", 7, 9)]
    assert timeline(parse_gantt("P1 2-4")) == [None, None, 1, 1]           # a gap is idle time
    for bad in ("first P1 then P2", "P1 3-3"):
        try:
            parse_gantt(bad)
        except AnswerError:
            continue
        raise AssertionError(bad)


def test_missed_preemption_in_srtf():
    d = diagnose(SRTF, {"gantt": "P1 0-8, P4 8-13, P2 13-17, P3 17-26"})
    f = d.first
    assert f.where == "Gantt chart at t=1" and f.misconception == "sched.preemption" and f.certain
    assert "P2" in f.explanation and "preempt" in f.explanation
    assert f.consequences                                  # later CT differences follow from it
    assert len(d.findings) == 1                            # and are not separate mistakes
    check_feedback_hides_answer(d)


def test_whole_chart_of_another_algorithm():
    sjf = solver.solve(solver.Problem("sjf", SRTF.procs))
    d = diagnose(SRTF, {"gantt": sjf.gantt})
    assert d.first.misconception == "sched.wrong_algorithm" and "Shortest Job First" in d.first.explanation


def test_round_robin_queue_errors():
    s = solver.solve(RR)
    # the course rule: a process arriving when a quantum expires queues before the preempted one.
    # Applying the opposite rule throughout (at t=2 and t=4) gives this chart:
    d = diagnose(RR, {"gantt": "P1 0-2, P2 2-4, P1 4-6, P3 6-7, P4 7-9, P2 9-10, P5 10-12, P1 12-13, P5 13-14"})
    assert d.first.where == "Gantt chart at t=4"
    assert d.first.misconception == "sched.arrival_boundary" and d.first.certain
    assert "arrival joins first" in d.first.explanation
    # applied at one tie only: not one consistent rule, so only the queue-order step is reported
    d = diagnose(RR, {"gantt": "P1 0-2, P2 2-4, P3 4-5, P1 5-7, P4 7-9, P2 9-10, P5 10-12, P1 12-13, P5 13-14"})
    assert d.first.where == "Gantt chart at t=9" and d.first.misconception == "sched.rr_queue"
    # a process kept past its quantum
    d = diagnose(RR, {"gantt": "P1 0-3, P2 3-5, P3 5-6, P4 6-8, P5 8-10, P1 10-12, P2 12-13, P5 13-14"})
    assert d.first.where == "Gantt chart at t=2" and d.first.misconception == "sched.rr_queue"
    check_feedback_hides_answer(d)
    assert diagnose(RR, {"gantt": s.gantt}).verdict == "correct"


def test_running_before_arrival_and_idle():
    d = diagnose(FCFS, {"gantt": "P1 0-5, P3 5-8"})
    assert d.first.where == "Gantt chart at t=0" and d.first.misconception == "sched.arrival_boundary"
    d = diagnose(FCFS, {"gantt": "P3 0-3, P1 4-9, P5 9-13, P2 13-17, P4 17-19"})   # idle 3-4 left implicit
    assert d.verdict == "correct"


def test_io_burst():
    d = diagnose(IO, {"gantt": "Idle 0-2, P1 2-3, P2 3-5, P3 5-6, P2 6-8, P1 8-11"})
    assert d.first.where == "Gantt chart at t=6" and d.first.misconception == "sched.io_burst"
    s = solver.solve(IO)
    table = full_table(s)
    table[2] = {**table[2], "wt": table[2]["wt"] - 3}                       # also subtracted the 3 units of I/O
    d = diagnose(IO, {"gantt": s.gantt, "table": table})
    assert [f.where for f in d.findings] == ["WT of P2"] and d.first.misconception == "sched.io_burst"


def test_table_formula_errors_and_consequences():
    s = solver.solve(SRTF)
    table = full_table(s)
    table[1] = {**table[1], "wt": table[1]["rt"]}          # WT written as RT (P1 waits later, after preemption)
    table[3] = {**table[3], "tat": table[3]["ct"]}         # TAT written as CT
    d = diagnose(SRTF, {"gantt": s.gantt, "table": table})
    # P3's WT is right (it follows from P3's CT), so the wrong TAT is reported once, not twice
    assert [f.where for f in d.findings] == ["WT of P1", "TAT of P3"]
    assert d.findings[0].misconception == "sched.metric_confusion" and d.findings[1].misconception == \
        "sched.metric_confusion"
    assert "response time" in d.findings[0].explanation and "AT was not subtracted" in d.findings[1].explanation
    check_feedback_hides_answer(d)


def test_values_consistent_with_a_wrong_chart_are_consequences():
    wrong = "P1 0-8, P4 8-13, P2 13-17, P3 17-26"           # missed preemption
    mine = {1: {"ct": 8, "tat": 8, "wt": 0}, 2: {"ct": 17, "tat": 16, "wt": 12}, 4: {"ct": 13, "tat": 10, "wt": 5}}
    d = diagnose(SRTF, {"gantt": wrong, "table": mine})
    assert len(d.findings) == 1 and d.first.misconception == "sched.preemption"
    mine[2]["tat"] = 17                                      # an independent slip after the chart error
    d = diagnose(SRTF, {"gantt": wrong, "table": mine})
    assert [f.where for f in d.findings] == ["Gantt chart at t=1", "TAT of P2"]


def test_ambiguous_answers_are_not_diagnosed():
    d = diagnose(SRTF, {"gantt": "P1 then P2 then P4"})
    assert d.verdict == "unclear" and not d.findings and "P1 0-3" in d.unclear[0]
    d = diagnose(SRTF, {})
    assert d.verdict == "unclear"
    d = diagnose(SRTF, {"table": {1: {"ct": 17}}})          # a single right value: right, nothing else judged
    assert d.verdict == "correct" and d.correct == ["CT of P1"]


def test_wrong_ct_reading():
    s = solver.solve(SRTF)
    table = full_table(s)
    table[1] = {"ct": 1}                                     # P1's first slice ends at 1; it finishes at 17
    d = diagnose(SRTF, {"gantt": s.gantt, "table": table})
    assert d.first.where == "CT of P1" and d.first.misconception == "sched.ct_reading"


def test_banker():
    p = banker.parse_problem("Banker's algorithm. Allocation: P0 0 1 0, P1 2 0 0, P2 3 0 2, P3 2 1 1, P4 0 0 2. "
                             "Max: P0 7 5 3, P1 3 2 2, P2 9 0 2, P3 2 2 2, P4 4 3 3. Available: 3 3 2. "
                             "Request from P1: (1 0 2).")
    s = banker.solve(p)
    d = diagnose(p, {"need": s.need, "sequence": "P1, P3, P4, P0, P2", "safe": True, "request": "granted"})
    assert d.verdict == "correct"
    d = diagnose(p, {"sequence": "P3, P1, P4, P2, P0"})      # another valid order is accepted
    assert d.verdict == "correct" and "different from the slides" in d.correct[0]
    d = diagnose(p, {"sequence": "P0, P1, P2, P3, P4"})
    assert d.first.where == "safe sequence, position 1" and d.first.misconception == "banker.need_vs_work"
    need = [row[:] for row in s.need]
    need[0][0] = p.alloc[0][0] - p.maxm[0][0]
    d = diagnose(p, {"need": need})
    assert d.first.misconception == "banker.need_formula" and d.first.certain
    d = diagnose(p, {"request": "unsafe"})
    assert d.first.misconception == "banker.request_steps"


def test_paging():
    p = paging.Problem([7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1], [3], ["lru"])
    lru = paging.solve(p).runs[0]
    fifo = paging._simulate(p.refs, 3, "fifo")
    marks = "".join("F" if f else "H" for f in fifo.faults)
    d = diagnose(p, {"marks": marks, "faults": fifo.n_faults})
    assert d.first.misconception == "paging.wrong_algorithm" and d.first.certain and "FIFO" in d.first.explanation
    assert len(d.findings) == 1                              # the total follows from the marks
    good = " ".join("F" if f else "H" for f in lru.faults)
    assert diagnose(p, {"marks": good, "faults": 12}).verdict == "correct"
    assert diagnose(p, {"marks": "FFH"}).verdict == "unclear"


def test_disk():
    q = "Requests: 98, 183, 37, 122, 14, 124, 65, 67. Head at 53."
    p = disk.parse_problem("C-SCAN, moving towards larger cylinders, 200 cylinders. " + q)
    s = disk.solve(p)
    run = s.runs[0]
    assert diagnose(p, {"order": run.served, "total": 382}).verdict == "correct"
    d = diagnose(p, {"order": run.served, "total": 183})     # left out the return seek
    assert d.first.misconception == "disk.return_seek" and d.first.certain
    d = diagnose(p, {"order": run.served, "total": 380})
    assert d.first.misconception == "disk.arithmetic"
    sstf = disk.parse_problem("SSTF. " + q)
    fcfs_order = [98, 183, 37, 122, 14, 124, 65, 67]
    d = diagnose(sstf, {"order": fcfs_order})
    assert d.first.misconception == "disk.wrong_algorithm" and "FCFS" in d.first.explanation


def test_similar_problem_and_solvers_unchanged():
    rng = random.Random(3)
    before = solver.solve(SRTF).rows
    for p in (RR, SRTF, FCFS):
        q = similar_problem(p, rng)
        assert q.algo == p.algo and len(q.procs) == len(p.procs) and q != p
        solver.solve(q)
        diagnose(p, {"gantt": "P1 0-1"})
    assert solver.solve(SRTF).rows == before


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print("ok  ", fn.__name__)
