"""Checks the evidence log and the Misconception Graph: uncertain evidence stays tentative, repeats
become recurring, later correct work resolves a concern, and old progress files still load.

    python -m tests.test_learning        (or: pytest tests)
"""
import json

from rag import learning, solver
from rag.diagnosis import diagnose
from rag.quiz import Progress, Question
from rag.trace import events

SRTF = "SRTF. P1: AT 0, BT 8. P2: AT 1, BT 4. P3: AT 2, BT 9. P4: AT 3, BT 5."
SRTF2 = "SRTF. P1: AT 0, BT 12. P2: AT 2, BT 4. P3: AT 3, BT 6. P4: AT 8, BT 5."     # slides 44-45
# both miss the first preemption without being exactly another algorithm's schedule (that would be
# diagnosed as "applying a different algorithm" instead)
MISSED = {SRTF: "P1 0-8, P4 8-13, P2 13-17, P3 17-26", SRTF2: "P1 0-12, P3 12-18, P2 18-22, P4 22-27"}


def attempt(progress, text, gantt=None):
    p = solver.parse_problem(text)
    s = solver.solve(p)
    d = diagnose(p, {"gantt": gantt or s.gantt})
    return learning.record_diagnosis(progress, d, text)


def status(progress, m_id):
    return learning.assess(progress.evidence)["misconceptions"].get(m_id, {}).get("status")


def test_uncertain_evidence_is_not_confirmed():
    pr = Progress()
    lab = events(solver.parse_problem(SRTF))
    e = next(i for i, ev in enumerate(lab.events) if ev.data["step"]["preempted"])
    wrong = lab.events[e].options.index(f"P{lab.events[e].data['step']['preempted']}")
    rec = learning.record_prediction(pr, lab, e, wrong, SRTF)
    assert rec["misconception"] == "sched.preemption" and not rec["certain"]
    learning.record_prediction(pr, lab, e, wrong, SRTF2)
    assert status(pr, "sched.preemption") == "tentative"          # uncertain evidence never becomes recurring
    assert learning.record_prediction(pr, lab, e, None, SRTF) is None


def test_repeated_mistakes_become_recurring_and_correct_work_resolves_them():
    pr = Progress()
    attempt(pr, SRTF, MISSED[SRTF])
    assert status(pr, "sched.preemption") == "tentative"
    attempt(pr, SRTF, MISSED[SRTF])                              # the same problem again: not a new problem
    assert status(pr, "sched.preemption") == "tentative"
    attempt(pr, SRTF2, MISSED[SRTF2])
    a = learning.assess(pr.evidence)
    assert a["misconceptions"]["sched.preemption"]["status"] == "recurring"
    assert a["concepts"]["sched.preemption"]["status"] == "weak (recurring)"
    rec = learning.recommendations(a)
    assert rec[0]["misconception"] == "sched.preemption" and rec[0]["topic"] == "preemption"
    assert "sched.fcfs" in rec[0]["prerequisites"]                 # not yet shown to be understood
    attempt(pr, SRTF)                                              # correct
    assert status(pr, "sched.preemption") == "recurring"           # one correct attempt is not enough
    attempt(pr, SRTF2)
    assert status(pr, "sched.preemption") == "resolved"
    assert not learning.recommendations(learning.assess(pr.evidence))


def test_misconceptions_attach_to_the_algorithm_where_they_were_seen():
    pr = Progress()
    rr = "Round Robin, time quantum 2. P1: AT 0, BT 5. P2: AT 1, BT 3. P3: AT 2, BT 1."
    p = solver.parse_problem(rr)
    d = diagnose(p, {"gantt": "P1 0-2, P3 2-3, P2 3-5, P1 5-8, P2 8-9"})
    learning.record_diagnosis(pr, d, rr)
    a = learning.assess(pr.evidence)
    assert all(not c.startswith("sched.sjf") for v in a["misconceptions"].values() for c in v["concepts"])
    assert a["concepts"]["sched.sjf"]["status"] == "needs more evidence"


def test_concept_status_needs_enough_evidence():
    pr = Progress()
    fcfs = "FCFS. P1: AT 0, BT 3. P2: AT 1, BT 2."
    attempt(pr, fcfs)
    a = learning.assess(pr.evidence)
    assert a["concepts"]["sched.fcfs"]["status"] == "needs more evidence"
    attempt(pr, "FCFS. P1: AT 0, BT 4. P2: AT 2, BT 2.")
    attempt(pr, "FCFS. P1: AT 0, BT 5. P2: AT 1, BT 1.")
    assert learning.assess(pr.evidence)["concepts"]["sched.fcfs"]["status"] == "understood"


def test_quiz_answers_are_evidence():
    pr = Progress()
    q = Question("numeric", "rr", 1, "a question", answer=1.0)
    assert learning.record_quiz(pr, q, 1.0)["concept"] == "sched.rr"
    assert learning.record_quiz(pr, Question("mcq", "threads", 1, "q"), 0) is None   # no concept for it yet
    assert pr.evidence[-1]["outcome"] == "correct" and [e["n"] for e in pr.evidence] == [1]


def test_unreadable_answers_record_nothing():
    pr = Progress()
    p = solver.parse_problem(SRTF)
    assert learning.record_diagnosis(pr, diagnose(p, {"gantt": "P1 then P2"}), SRTF) == []
    assert pr.evidence == []


def test_progress_files_old_new_and_broken():
    old = json.dumps({"rr": {"asked": 2, "points": 1.0, "level": 1, "streak": 0}})
    pr = Progress.from_json(old)                                    # a file written before the evidence log
    assert pr.stats["rr"]["asked"] == 2 and pr.evidence == []
    assert Progress.from_json("").stats == {} and Progress.from_json("{}").evidence == []
    attempt(pr, SRTF, MISSED[SRTF])
    again = Progress.from_json(pr.to_json())
    assert again.stats == pr.stats and again.evidence == pr.evidence
    assert json.loads(pr.to_json())["rr"]["asked"] == 2               # topic stats keep their place
    assert Progress.from_json(json.dumps({**json.loads(old), "_evidence": "junk"})).evidence == []
    try:
        Progress.from_json("[1, 2]")
    except ValueError:
        pass
    else:
        raise AssertionError("a non-object file should be refused")


def test_graph_and_sample():
    a = learning.assess(learning.sample_evidence())
    assert a["misconceptions"]["sched.preemption"]["status"] == "recurring"
    assert a["misconceptions"]["banker.need_vs_work"]["status"] == "resolved"
    assert a["misconceptions"]["sched.metric_confusion"]["status"] == "tentative"   # uncertain only
    g = learning.dot(a)
    assert g.startswith("digraph") and '"m:sched.preemption" -> "sched.srtf"' in g and "sched.fcfs" in g
    # the misconception "sched.preemption" and the concept "sched.preemption" are two separate nodes
    assert '"sched.preemption" [label="Preemptive vs non-preemptive' in g and '"m:sched.preemption" [label=' in g
    assert "disk.scan" not in g                                     # nothing seen there: not drawn
    assert "disk.scan" in learning.dot(a, only_seen=False)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print("ok  ", fn.__name__)
