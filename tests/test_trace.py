"""Checks OS Lab (Predict, Run, Explain): replaying a lab's events reproduces the solver's result,
events come in order, and the session controls (predict, run, next, previous, replay, reset) behave.

    python -m tests.test_trace        (or: pytest tests)
"""
from rag import banker, disk, paging, solver
from rag.examples import EXAMPLES
from rag.numericals import kind, parse_problem
from rag.trace import Session, events


def schedule_from(lab) -> list:
    """The timeline the lab's events describe: each decision holds until the next one."""
    evs = lab.events
    tl = [None] * evs[0].data["step"]["t"]          # idle until the first arrival (no decision before it)
    for e, nxt in zip(evs, evs[1:] + [None]):
        t0 = e.data["step"]["t"]
        t1 = nxt.data["step"]["t"] if nxt else lab.solution.end
        run = e.options[e.answer]
        tl += [None if run == "CPU idle" else int(run[1:])] * (t1 - t0)
    return tl


def test_scheduling_traces_match_the_solver():
    problems = [parse_problem(e.question) for e in EXAMPLES if kind(parse_problem(e.question)) == "scheduling"]
    problems.append(solver.parse_problem("SRTF. P1: AT 2, BT 4. P2: AT 3, BT 2, I/O 3, BT 2. P3: AT 5, BT 1."))
    for p in problems:
        lab = events(p)
        ref = [x for l, a, b in lab.solution.gantt for x in [None if l == "Idle" else int(l[1:])] * (b - a)]
        assert schedule_from(lab) == ref, p.algo
        times = [e.data["step"]["t"] for e in lab.events]
        assert times == sorted(times) and len(set(times)) == len(times)
        for e in lab.events:
            assert e.options[e.answer] == e.after["running"] and e.explanation.startswith(e.title.replace(" = ", "="))
            assert "CPU idle" in e.options


def test_round_robin_lab_state():
    p = solver.parse_problem("Round Robin, time quantum 2. P1: AT 0, BT 5. P2: AT 1, BT 3. P3: AT 2, BT 1. "
                             "P4: AT 3, BT 2. P5: AT 4, BT 3.")
    lab = events(p)
    at = {e.data["step"]["t"]: e for e in lab.events}
    e = at[4]                                   # P2's quantum expires as P5 arrives
    assert e.before["used"] == 2 and e.before["quantum"] == 2
    assert e.before["ready"] == ["P3", "P1", "P4", "P5", "P2"]      # the arrival is queued before P2
    assert e.options[e.answer] == "P3" and e.after["ready"] == ["P1", "P4", "P5", "P2"]
    assert at[1].before["used"] == 1 and at[1].options[at[1].answer] == "P1"   # no expiry yet: P1 continues
    assert lab.events[-1].after["running"] == "P5" and "quantum" in lab.reflection


def test_banker_trace():
    p = banker.parse_problem("Banker's algorithm. Allocation: P0 0 0 1 2, P1 1 0 0 0, P2 1 3 5 4, P3 0 6 3 2, "
                             "P4 0 0 1 4. Max: P0 0 0 1 2, P1 1 7 5 0, P2 2 3 5 6, P3 0 6 5 2, P4 0 6 5 6. "
                             "Available: 1 5 2 0.")
    lab = events(p)
    finished = [e.before["process"] for e in lab.events if e.answer == 0]
    assert finished == ["P0", "P2", "P3", "P4", "P1"]                      # slide 29
    assert lab.events[-1].after["work"] == "(3 14 12 12)"
    assert lab.events[1].answer == 1 and "waits" in lab.events[1].explanation   # P1 must wait in pass 1


def test_paging_trace():
    p = paging.Problem([7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1], [3], ["lru"])
    lab = events(p)
    assert len(lab.events) == 22
    assert sum(e.answer != 0 for e in lab.events) == 12                    # LRU: 12 faults (slide 79)
    e = lab.events[3]                                                      # page 2: replaces 7
    assert e.options[e.answer] == "Fault: replace page 7" and e.after["frames"] == "2 0 1"


def test_disk_trace():
    for algo, total in (("sstf", 236), ("cscan", 382), ("clook", 322), ("scan", 331)):
        p = disk.parse_problem(f"{algo.upper().replace('CS', 'C-S').replace('CL', 'C-L')}, head moving towards "
                               f"larger cylinders, 200 cylinders. Requests 98, 183, 37, 122, 14, 124, 65, 67. "
                               f"Head at 53.")
        lab = events(p)
        path = [lab.events[0].before["head"]] + [int(e.options[e.answer].split()[0]) for e in lab.events]
        assert path == lab.solution.runs[0].path, algo
        assert lab.events[-1].after["moved so far"] == total and lab.events[-1].after["pending"] == ""


def test_session_controls():
    lab = events(solver.parse_problem("FCFS. P1: AT 0, BT 3. P2: AT 1, BT 2."))
    s = Session(lab)
    s.next()
    assert s.pos == 0                                    # cannot skip an event before running it
    s.predict(lab.events[0].answer)
    s.run()
    s.predict(1 - lab.events[0].answer)                  # locked after running
    assert s.predictions[0] == lab.events[0].answer and s.score() == (1, 1)
    s.next()
    s.predict(len(lab.events[1].options) - 1)            # predicts "CPU idle": wrong
    s.run()
    assert s.score() == (1, 2)
    while not s.done:
        s.run()
        s.next()
    assert s.done and s.event is None
    s.previous()
    assert s.pos == len(lab.events) - 1
    s.replay()
    assert s.pos == 0 and s.score() == (1, 2)            # replay keeps what was seen
    s.reset()
    assert s.pos == 0 and s.score() == (0, 0) and not s.ran


def test_opening_the_lab_where_the_working_went_wrong():
    from rag.diagnosis import diagnose
    from rag.trace import event_for
    p = solver.parse_problem("SRTF. P1: AT 0, BT 8. P2: AT 1, BT 4. P3: AT 2, BT 9. P4: AT 3, BT 5.")
    f = diagnose(p, {"gantt": "P1 0-8, P4 8-13, P2 13-17, P3 17-26"}).first
    lab = events(p)
    i = event_for(lab, f.where)
    assert lab.events[i].title == "t = 1"                     # the missed preemption
    s = Session(lab)
    s.jump(i)
    assert s.pos == i and s.ran == set(range(i)) and s.score() == (0, 0)
    s.run()
    s.next()
    assert s.pos == i + 1
    pg = paging.Problem([1, 2, 3, 4, 1, 2], [3], ["fifo"])
    assert event_for(events(pg), "reference 4 (page 4)") == 3
    assert event_for(events(pg), "number of page faults") == 0


def test_solver_results_unchanged():
    p = solver.parse_problem("SRTF. P1: AT 0, BT 8. P2: AT 1, BT 4. P3: AT 2, BT 9. P4: AT 3, BT 5.")
    lab = events(p)
    assert [r["wt"] for r in lab.solution.rows] == [9, 0, 15, 2]           # slides 41-43
    assert lab.solution.rows == solver.solve(p).rows


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print("ok  ", fn.__name__)
