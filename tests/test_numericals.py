"""Checks the Banker's-algorithm, page-replacement and disk-scheduling solvers against the
teacher's slides and the textbook, and how they read problems out of typed questions.

    python -m tests.test_numericals        (or: pytest tests)
"""
from rag import banker, disk, paging
from rag.numericals import ProblemError, follow_up, kind, parse_problem, solve, to_text, topic_of

# ---------------------------------------------------------------------------------------
# Banker's algorithm (teacher slides, Module 3 deadlocks)
# ---------------------------------------------------------------------------------------
EX1 = ("Banker's algorithm. Allocation: P0 0 0 1 2, P1 1 0 0 0, P2 1 3 5 4, P3 0 6 3 2, P4 0 0 1 4. "
       "Max: P0 0 0 1 2, P1 1 7 5 0, P2 2 3 5 6, P3 0 6 5 2, P4 0 6 5 6. Available: 1 5 2 0.")
# Example 2 as a pasted table. The slide text gives P3's Max as 1 4 2 1, but its Need row (0 1 1 2)
# and totals (12 12 8 10) need 1 4 2 4.
EX2_TABLE = """Process  Allocation  Max       Available
         A B C D     A B C D   A B C D
P0       2 0 0 1     4 2 1 2   3 3 2 1
P1       3 1 2 1     5 2 5 2
P2       2 1 0 3     2 3 1 6
P3       1 3 1 2     1 4 2 4
P4       1 4 3 2     3 6 6 5
1. Calculate the NEED matrix. 2. Is the system in a safe state?"""
EX2_INLINE = ("P0: Allocation 2 0 0 1, Max 4 2 1 2. P1: Allocation 3 1 2 1, Max 5 2 5 2. P2: Allocation 2 1 0 3, "
              "Max 2 3 1 6. P3: Allocation 1 3 1 2, Max 1 4 2 4. P4: Allocation 1 4 3 2, Max 3 6 6 5. "
              "Available 3 3 2 1. Use the Banker's algorithm.")
DETECT = ("Deadlock detection. Allocation: P0 0 1 0, P1 2 0 0, P2 3 0 3, P3 2 1 1, P4 0 0 2. "
          "Request: P0 0 0 0, P1 2 0 2, P2 0 0 {c}, P3 1 0 0, P4 0 0 2. Available: 0 0 0.")


def seq(s):
    return [s.problem.pids[i] for i in s.check.order]


def test_banker_example_1():
    s = banker.solve(banker.parse_problem(EX1))
    assert s.need == [[0, 0, 0, 0], [0, 7, 5, 0], [1, 0, 0, 2], [0, 0, 2, 0], [0, 6, 4, 2]]
    assert s.total == [3, 14, 12, 12]
    assert s.check.safe and seq(s) == [0, 2, 3, 4, 1]                            # slide 29
    assert s.check.works == [[1, 5, 3, 2], [2, 8, 8, 6], [2, 14, 11, 8], [2, 14, 12, 12], [3, 14, 12, 12]]


def test_banker_example_2_layouts_and_requests():
    table = banker.parse_problem(EX2_TABLE)
    assert table == banker.parse_problem(EX2_INLINE)
    s = banker.solve(table)
    assert s.need == [[2, 2, 1, 1], [2, 1, 3, 1], [0, 2, 1, 3], [0, 1, 1, 2], [2, 2, 3, 3]]   # slide 33
    assert s.total == [12, 12, 8, 10]
    # the slide's own sequence is <P0, P3, P1, P2, P4>; it lists ours among the other safe sequences (slide 36)
    assert s.check.safe and seq(s) == [0, 3, 4, 1, 2]
    p1 = banker.solve(banker.parse_problem(EX2_TABLE + " If a request from P1 arrives for (1,1,0,0), can it be "
                                                       "granted immediately?"))
    assert p1.request["verdict"] == "granted" and p1.request["available"] == [2, 2, 2, 1]
    assert [p1.problem.pids[i] for i in p1.request["check"].order] == [0, 3, 4, 1, 2]       # slide 53
    p4 = banker.solve(banker.parse_problem(EX2_INLINE + " P4 requests (0,0,2,0)."))
    assert p4.request["verdict"] == "unsafe" and not p4.request["check"].order            # slide 57
    for text, verdict in ((" P2 requests (1, 0, 0, 0).", "error"), (" Request from P2: (0 2 1 3).", "wait"),
                          (" Can P1's request (1, 1, 0, 0) be granted?", "granted"),
                          (" Is a request (0, 0, 2, 0) by P4 safe to grant?", "unsafe")):
        assert banker.solve(banker.parse_problem(EX2_INLINE + text)).request["verdict"] == verdict, text


def test_banker_detection():
    s = banker.solve(banker.parse_problem(DETECT.format(c=0)))
    assert s.check.safe and seq(s) == [0, 2, 3, 4, 1]                                      # slide 61
    s = banker.solve(banker.parse_problem(DETECT.format(c=1)))
    assert not s.check.safe and seq(s) == [0]                                              # slide 62
    assert "P1, P2, P3, P4" in banker.verdict(s)


def test_banker_input_variants():
    textbook = ("Five threads T0 to T4. Total instances: A = 10, B = 5, C = 7. Allocation: T0 0 1 0, T1 2 0 0, "
                "T2 3 0 2, T3 2 1 1, T4 0 0 2. Max: T0 7 5 3, T1 3 2 2, T2 9 0 2, T3 2 2 2, T4 4 3 3. Is it safe?")
    p = banker.parse_problem(textbook)
    assert p.available == [3, 3, 2] and p.prefix == "T"                                     # Silberschatz 8.6.3
    assert banker.solve(p).check.safe
    with_need = ("Banker's algorithm. Allocation: P0 0 1 0, P1 2 0 0, P2 3 0 2. Need: P0 7 4 3, P1 1 2 2, "
                 "P2 6 0 0. Available: 3 3 2.")
    assert banker.parse_problem(with_need).maxm == [[7, 5, 3], [3, 2, 2], [9, 0, 2]]
    resources_named_r = ("Banker. Allocation R1 R2 R3: P0 0 1 0, P1 2 0 0. Max R1 R2 R3: P0 7 5 3, P1 3 2 2. "
                         "Available R1 R2 R3: 3 3 2.")
    assert banker.parse_problem(resources_named_r).available == [3, 3, 2]


# ---------------------------------------------------------------------------------------
# Page replacement (teacher slides, Module 5 part 2)
# ---------------------------------------------------------------------------------------
REF = "7,0,1,2,0,3,0,4,2,3,0,3,0,3,2,1,2,0,1,7,0,1"


def faults(text):
    return {(r.algo, r.frames): r.n_faults for r in paging.solve(paging.parse_problem(text)).runs}


def test_page_replacement_slides():
    got = faults(f"Find the page faults with 3 frames. Reference string: {REF}")
    assert got == {("fifo", 3): 15, ("lru", 3): 12, ("optimal", 3): 9}                     # slides 76, 78, 79
    belady = faults("Show Belady's anomaly for 1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5 with 3 and 4 frames")
    assert belady == {("fifo", 3): 9, ("fifo", 4): 10}                                     # slides 76-77
    s = paging.solve(paging.parse_problem(f"FIFO with 3 frames, reference string {REF}"))
    assert s.runs[0].table[3] == [2, 0, 1] and s.runs[0].victims[3] == 7


def test_page_replacement_parsing():
    assert faults("Consider the page reference string 1 2 3 4 2 1 5 6 2 1 2 3 7 6 3 2 1 2 3 6. How many page "
                  "faults occur for the LRU algorithm with four frames? Use 4 frames.") == {("lru", 4): 10}
    try:
        paging.parse_problem("LRU page replacement, reference string 7 0 1 2 0 3")
    except ProblemError:
        pass
    else:
        raise AssertionError("a reference string without a frame count should be refused")


# ---------------------------------------------------------------------------------------
# Disk scheduling (Silberschatz 10e, 11.2 and exercise 11.13)
# ---------------------------------------------------------------------------------------
QUEUE = "requests for cylinders 98, 183, 37, 122, 14, 124, 65, 67. The head is initially at cylinder 53."


def totals(text):
    s = disk.solve(disk.parse_problem(text))
    return {r.algo: s.total(r) for r in s.runs}


def test_disk_textbook_example():
    assert totals(f"FCFS. {QUEUE}") == {"fcfs": 640}                                       # 11.2.1
    assert totals(f"SCAN, the arm moving towards 0, disk of 200 cylinders. {QUEUE}") == {"scan": 236}
    assert totals(f"SSTF. {QUEUE}") == {"sstf": 236}
    assert totals(f"C-SCAN, moving from 0 to 199. {QUEUE}") == {"cscan": 382}               # 183 + return of 199
    assert totals(f"C-SCAN, moving from 0 to 199, not counting the return seek. {QUEUE}") == {"cscan": 183}
    s = disk.solve(disk.parse_problem(f"C-SCAN, moving from 0 to 199. {QUEUE}"))
    assert s.runs[0].path == [53, 65, 67, 98, 122, 124, 183, 199, 0, 14, 37]


def test_disk_textbook_exercise():
    text = ("Suppose that a disk drive has 5,000 cylinders, numbered 0 to 4,999. The drive is currently serving a "
            "request at cylinder 2,150, and the previous request was at cylinder 1,805. The queue of pending "
            "requests, in FIFO order, is: 2,069; 1,212; 2,296; 2,800; 544; 1,618; 356; 1,523; 4,965; 3,681. What "
            "is the total distance that the disk arm moves for FCFS, SSTF, SCAN, LOOK, C-SCAN and C-LOOK?")
    assert totals(text.replace(",", "").replace(";", ",")) == {
        "fcfs": 13011, "sstf": 7586, "scan": 7492, "cscan": 9917, "look": 7424, "clook": 9137}


def test_disk_assumptions_are_stated():
    p = disk.parse_problem(f"Compare SCAN and C SCAN. {QUEUE}")
    assert p.algos == ["scan", "cscan"] and p.up and len(p.assumed) == 2
    assert "Assumed" in disk.render(disk.solve(p))


# ---------------------------------------------------------------------------------------
# All solvers: routing, round trips, follow-ups
# ---------------------------------------------------------------------------------------
NOT_NUMERICAL = [
    "Explain the Banker's algorithm with an example",
    "What is a safe state?",
    "Explain LRU page replacement",
    "What is Belady's anomaly?",
    "Compare the SCAN and C-SCAN disk scheduling algorithms",
    "Explain paging with a TLB",
    "Page table: page 0 -> frame 5, page 1 -> frame 9, page 2 -> frame 3. Find the physical address of 2500.",
    "What is the convoy effect?",
    "First fit: memory partitions 100, 500, 200, 300, 600 KB; processes P1 212 KB, P2 417 KB, P3 112 KB, P4 426 KB",
]


def test_routing():
    assert kind(parse_problem(EX1)) == "banker"
    assert kind(parse_problem(f"LRU with 3 frames. Reference string {REF}")) == "paging"
    assert kind(parse_problem(f"SSTF. {QUEUE}")) == "disk"
    assert kind(parse_problem("Round robin, quantum 2. P1: AT 0, BT 5. P2: AT 1, BT 3.")) == "scheduling"
    for q in NOT_NUMERICAL:
        assert parse_problem(q) is None, q
    assert topic_of("Explain the Banker's algorithm with an example") == "banker"
    assert topic_of("Explain LRU with an example") == "paging"
    assert topic_of("Explain FCFS disk scheduling with an example") == "disk"
    assert topic_of("Explain FCFS with an example") == "fcfs"


def test_to_text_round_trip():
    problems = [
        banker.parse_problem(EX2_INLINE + " P1 requests (1, 1, 0, 0)."),
        banker.parse_problem(DETECT.format(c=1)),
        paging.Problem([1, 2, 3, 4, 1, 2, 5], [3, 4], ["fifo", "optimal"]),
        disk.Problem([98, 183, 37], 53, ["cscan", "clook"], False, 300, False),
        disk.Problem([98, 183, 37], 53, ["fcfs"], True, 200, True),
    ]
    for p in problems:
        assert parse_problem(to_text(p)) == p, to_text(p)
        solve(p)


def test_follow_ups():
    prev = parse_problem(EX2_INLINE)
    again = follow_up("what if P4 requests (0, 0, 2, 0)?", prev)
    assert again.request == (4, [0, 0, 2, 0]) and again.alloc == prev.alloc
    pr = parse_problem(f"FIFO with 3 frames. Reference string {REF}")
    assert follow_up("now do it with 4 frames", pr).frames == [4]
    assert follow_up("what about LRU?", pr).algos == ["lru"]
    assert follow_up("Explain LRU", pr) is None
    dk = parse_problem(f"SCAN. {QUEUE}")
    assert follow_up("now with the head moving towards 0", dk).up is False
    assert follow_up("what about C-LOOK instead?", dk).algos == ["clook"]


def test_missing_details_are_reported():
    for q in ("Banker's algorithm. Allocation: P0 0 1 0, P1 2 0 0. Max: P0 7 5 3, P1 3 2 2.",      # no Available
              "Banker's algorithm. Allocation: P0 0 1 0, P1 2 0 0. Available: 3 3 2.",             # no Max / Need
              "Banker's algorithm. Allocation: P0 0 9 0, P1 2 0 0. Max: P0 7 5 3, P1 3 2 2. Available 3 3 2",
              "SSTF disk scheduling. Requests 98, 183, 37, 122.",                                  # no head
              "LRU page replacement. Reference string 1, 2, 3, 4, 1, 2"):                          # no frames
        try:
            solve(parse_problem(q))
        except ProblemError:
            continue
        raise AssertionError(f"expected a ProblemError for: {q}")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print("ok  ", fn.__name__)
