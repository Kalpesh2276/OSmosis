"""Checks the solver against the worked examples in the teacher's slides.

    python -m tests.test_solver        (or: pytest tests)
"""
from rag.solver import ALGOS, Problem, ProblemError, Proc, parse_problem, solve, to_text


def run(question):
    s = solve(parse_problem(question))
    gantt = ", ".join(f"{lab} {a}-{b}" for lab, a, b in s.gantt)
    return gantt, {k: [r[k] for r in s.rows] for k in ("ct", "tat", "wt", "rt")}


# (name, question, expected Gantt chart, expected column values)
CASES = [
    ("FCFS example 2 (slides 25-28)",
     "FCFS. P1: AT 4, BT 5; P2: AT 6, BT 4; P3: AT 0, BT 3; P4: AT 6, BT 2; P5: AT 5, BT 4.",
     "P3 0-3, Idle 3-4, P1 4-9, P5 9-13, P2 13-17, P4 17-19",
     {"ct": [9, 17, 3, 19, 13], "tat": [5, 11, 3, 13, 8], "wt": [0, 7, 0, 11, 4]}),
    ("FCFS example 4 (slides 29-31)",
     "Use FCFS. All processes arrive at time 0. P1 burst 6, P2 burst 3, P3 burst 8.",
     "P1 0-6, P2 6-9, P3 9-17",
     {"tat": [6, 9, 17], "wt": [0, 6, 9]}),
    ("SJF example 1 (slides 39-40)",
     "SJF, non-preemptive. All arrive at time 0. Burst times: P1 6, P2 8, P3 7, P4 3 ms.",
     "P4 0-3, P1 3-9, P3 9-16, P2 16-24",
     {"wt": [3, 16, 9, 0]}),
    ("SJF example 6 (slides 47-49)",
     "Non-preemptive SJF. P1: AT 0, BT 10; P2: AT 1, BT 3; P3: AT 2, BT 6; P4: AT 3, BT 4; P5: AT 4, BT 2.",
     "P1 0-10, P5 10-12, P2 12-15, P4 15-19, P3 19-25",
     {"tat": [10, 14, 23, 16, 8], "wt": [0, 11, 17, 12, 6]}),
    ("SRTF example 2 (slides 41-43)",
     "Preemptive SJF. P1: AT 0, BT 8; P2: AT 1, BT 4; P3: AT 2, BT 9; P4: AT 3, BT 5.",
     "P1 0-1, P2 1-5, P4 5-10, P1 10-17, P3 17-26",
     {"wt": [9, 0, 15, 2]}),
    ("SRTF example 3 (slides 44-45)",
     "SRTF. P1: AT 0, BT 12; P2: AT 2, BT 4; P3: AT 3, BT 6; P4: AT 8, BT 5.",
     "P1 0-2, P2 2-6, P3 6-12, P4 12-17, P1 17-27",
     {"wt": [15, 0, 3, 4]}),
    ("SRTF example 4 (slides 46-47)",
     "Shortest remaining time first. P1: AT 0, BT 10; P2: AT 3, BT 6; P3: AT 7, BT 1; P4: AT 8, BT 3.",
     "P1 0-3, P2 3-7, P3 7-8, P2 8-10, P4 10-13, P1 13-20",
     {"tat": [20, 7, 1, 5]}),
    ("SRTF with I/O burst (slides 112-113)",
     "SRTF. P1: AT 2, BT 4. P2: AT 3, BT 2, I/O 3, BT 2. P3: AT 5, BT 1.",
     "Idle 0-2, P1 2-3, P2 3-5, P3 5-6, P1 6-9, P2 9-11",
     {"ct": [9, 11, 6], "tat": [7, 8, 1], "wt": [3, 4, 0], "rt": [0, 0, 0]}),
    ("Priority example 1 (slides 59-60)",
     "Non-preemptive priority scheduling, smallest integer = highest priority. All arrive at time 0. "
     "P1: BT 10, priority 3; P2: BT 1, priority 1; P3: BT 2, priority 4; P4: BT 1, priority 5; P5: BT 5, priority 2.",
     "P2 0-1, P5 1-6, P1 6-16, P3 16-18, P4 18-19",
     {"wt": [6, 0, 16, 18, 1]}),
    ("Priority example 2, preemptive (slides 60-61)",
     "Preemptive priority scheduling, 0 is the highest priority. P1: AT 0, BT 11, pri 2; P2: AT 5, BT 28, pri 0; "
     "P3: AT 12, BT 2, pri 3; P4: AT 2, BT 10, pri 1; P5: AT 9, BT 16, pri 4.",
     "P1 0-2, P4 2-5, P2 5-33, P4 33-40, P1 40-49, P3 49-51, P5 51-67",
     {"wt": [38, 0, 37, 28, 42]}),
    ("Priority example 3, higher number = higher priority (slides 62-64)",
     "Priority non-preemptive; a higher number represents a higher priority. P1: AT 0, BT 4, pri 2; "
     "P2: AT 1, BT 3, pri 3; P3: AT 2, BT 1, pri 4; P4: AT 3, BT 5, pri 5; P5: AT 4, BT 2, pri 5.",
     "P1 0-4, P4 4-9, P5 9-11, P3 11-12, P2 12-15",
     {"ct": [4, 15, 12, 9, 11], "tat": [4, 14, 10, 6, 7], "wt": [0, 11, 9, 1, 5]}),
    ("Round Robin example 1 (slides 75-77)",
     "Round Robin, time quantum 4 ms. All arrive at time 0. P1 BT 24, P2 BT 3, P3 BT 3.",
     "P1 0-4, P2 4-7, P3 7-10, P1 10-30",
     {"ct": [30, 7, 10], "wt": [6, 4, 7]}),
    ("Round Robin example 2 (slides 78-85)",
     "RR with time quantum = 2 units. P1: AT 0, BT 5; P2: AT 1, BT 3; P3: AT 2, BT 1; P4: AT 3, BT 2; P5: AT 4, BT 3.",
     "P1 0-2, P2 2-4, P3 4-5, P1 5-7, P4 7-9, P5 9-11, P2 11-12, P1 12-13, P5 13-14",
     {"ct": [13, 12, 5, 9, 14], "tat": [13, 11, 3, 6, 10], "wt": [8, 8, 2, 4, 7]}),
    # The slide's own Gantt chart for this one runs P5 before P1 at t=12, which contradicts the
    # queueing rule of example 2. Expected values follow the rule (knowledge-base note, slides 83-85).
    ("Round Robin example 3, by the example-2 rule (slides 83-85)",
     "Round Robin, time quantum 3 ms. P1: AT 0, BT 8; P2: AT 1, BT 4; P3: AT 2, BT 9; P4: AT 3, BT 5; P5: AT 4, BT 2.",
     "P1 0-3, P2 3-6, P3 6-9, P4 9-12, P1 12-15, P5 15-17, P2 17-18, P3 18-21, P4 21-23, P1 23-25, P3 25-28",
     {"ct": [25, 18, 28, 23, 17]}),
    ("Arrival exactly when a quantum expires (bare columns)",
     "Solve using Round Robin with time quantum 3.\nP1 0 5\nP2 1 4\nP3 3 2\nP4 6 3",
     "P1 0-3, P2 3-6, P3 6-8, P1 8-10, P4 10-13, P2 13-14",
     {"ct": [10, 14, 8, 13], "tat": [10, 13, 5, 7], "wt": [5, 9, 3, 4]}),
]

NOT_NUMERICAL = [
    "What is the convoy effect?",
    "Explain SRTF with an example",
    "How does round robin work and how to choose the time quantum?",
    "In MLFQ with Q1 = 2, Q2 = 7 and Q3 = 12, in which queue does a 40 second job finish?",
    "Why would P1 starve under priority scheduling?",
]


def test_worked_examples():
    for name, question, gantt, cols in CASES:
        got_gantt, got = run(question)
        assert got_gantt == gantt, f"{name}: Gantt chart\n  got      {got_gantt}\n  expected {gantt}"
        for key, want in cols.items():
            assert got[key] == want, f"{name}: {key.upper()} got {got[key]}, expected {want}"


def test_ordinary_questions_are_left_alone():
    for q in NOT_NUMERICAL:
        assert parse_problem(q) is None, q


def test_missing_details_are_reported():
    for q in ("P1: AT 0, BT 5. P2: AT 1, BT 4. Find the average waiting time.",   # no algorithm
              "Round Robin. P1: AT 0, BT 5. P2: AT 1, BT 4.",                       # no quantum
              "Priority scheduling. P1: AT 0, BT 5. P2: AT 1, BT 4."):              # no priorities
        try:
            solve(parse_problem(q))
        except ProblemError:
            continue
        raise AssertionError(f"expected a ProblemError for: {q}")


def test_to_text_round_trip():
    procs = [Proc(1, 0, [5], [], 2), Proc(2, 1, [2, 2], [3], 1), Proc(3, 4, [1], [], 3)]
    for algo in ALGOS:
        for high in (False, True):
            p = Problem(algo, procs, 3 if algo == "rr" else None, high and algo.startswith("priority"))
            back = parse_problem(to_text(p))
            if not algo.startswith("priority"):
                for q in back.procs:
                    q.prio = None
                want = Problem(p.algo, [Proc(q.pid, q.at, q.cpu, q.io) for q in procs], p.quantum, False)
            else:
                want = p
            assert back == want, f"{algo}: {to_text(p)!r} read back as {back}"


if __name__ == "__main__":
    for fn in (test_worked_examples, test_ordinary_questions_are_left_alone, test_missing_details_are_reported,
               test_to_text_round_trip):
        fn()
        print("ok  ", fn.__name__)
    print(f"{len(CASES)} worked examples match.")
