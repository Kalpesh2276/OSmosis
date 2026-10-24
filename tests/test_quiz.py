"""Checks quiz mode, follow-ups and the worked-example registry (no LLM or index needed).

    python -m tests.test_quiz        (or: pytest tests)
"""
import random

from rag import quiz
from rag.ask import clean_answer, is_follow_up
from rag.examples import EXAMPLES, find_example
from rag.numericals import kind, parse_problem as parse_any, solve as solve_any
from rag.quiz import BANK, TOPICS, Progress, bank_question, check_numeric, next_question, numeric_question
from rag.solver import ProblemError, follow_up, parse_problem, solve


class Hit:  # stands in for retriever.Hit
    def __init__(self, a, b, kind="teacher_notes", source="keerthika_m3"):
        source = "silberschatz10" if kind == "textbook" else source
        self.item = {"source_kind": kind, "source_id": source, "page_start": a, "page_end": b,
                     "text": "passage text"}
        self.cite = f"Teacher slides (Module 3), slides {a}-{b}"


class FakeRetriever:
    def search(self, query, k=4):
        return [Hit(21, 21), Hit(72, 74)]


def test_numeric_questions_have_a_computed_answer():
    rng = random.Random(1)
    for topic, (_, _, algos) in TOPICS.items():
        if not algos:
            continue
        for level in (1, 2, 3):
            for _ in range(40):
                q = numeric_question(topic, level, rng)
                p = q.solution.problem
                if kind(p) == "scheduling":
                    assert p.algo == algos[level - 1]
                elif kind(p) in ("paging", "disk"):
                    assert p.algos[0] in algos[level - 1].split()
                assert parse_any(q.prompt.split("\n")[0]) == p, q.prompt   # the question states the whole problem
                if q.options:          # computed options: exactly one is right
                    assert len(set(q.options)) == len(q.options) and 0 <= q.answer < len(q.options)
                else:
                    assert check_numeric(q, q.answer) and not check_numeric(q, q.answer + 0.5)


def test_banker_quiz_options_are_right():
    rng = random.Random(8)
    for _ in range(30):
        q = numeric_question("banker", 2, rng)        # which of these is a safe sequence?
        s = q.solution
        for i, option in enumerate(q.options):
            order = [s.problem.pids.index(int(x)) for x in option.strip("<>").replace("P", "").split(", ")]
            work, ok = list(s.problem.available), True
            for k in order:
                ok = ok and all(n <= w for n, w in zip(s.need[k], work))
                work = [w + a for w, a in zip(work, s.problem.alloc[k])]
            assert ok == (i == q.answer), (q.prompt, option)
        q = numeric_question("banker", 3, rng)        # can the request be granted?
        assert list(quiz.REQUEST_OPTIONS)[q.answer] == q.solution.request["verdict"]


def test_bank_covers_every_topic_and_is_well_formed():
    banked = {b[0] for b in BANK}
    assert banked <= set(TOPICS)
    for topic in banked:    # topics outside the bank are offered only with the LLM (test_new_topics_offline)
        assert sum(b[0] == topic for b in BANK) >= 2, topic
    rng = random.Random(2)
    for topic, _, prompt, options, answer, why, cite in BANK:
        assert len(options) == 4 == len(set(options)) and answer == 0 and why and cite, prompt
        q = bank_question(topic, 1, rng, seen=[b[2] for b in BANK if b[2] != prompt])
        assert q.prompt == prompt and q.options[q.answer] == options[0]


def test_progress_adapts():
    p = Progress()
    assert p.level("rr") == 1
    p.record("rr", 1); p.record("rr", 1)
    assert p.level("rr") == 2                      # two correct in a row -> harder
    p.record("rr", 1); p.record("rr", 0.5)
    assert p.level("rr") == 2                      # partly correct breaks the streak, no change
    p.record("rr", 0)
    assert p.level("rr") == 1                      # wrong -> easier
    p.record("rr", 0)
    assert p.level("rr") == 1                      # never below level 1 (rr is now at 3.5 / 6)
    for _ in range(3):
        p.record("fcfs", 1)
    rng = random.Random(3)
    picks = [p.pick_topic(rng, ["rr", "fcfs"]) for _ in range(400)]
    assert picks.count("rr") > picks.count("fcfs")  # the weaker topic comes up more often
    assert "rr" in p.weak_topics() and "fcfs" not in p.weak_topics()
    assert Progress.from_json(p.to_json()).stats == p.stats


def test_llm_questions_are_parsed_and_validated(monkeypatch=None):
    replies = iter([
        '<p>Here you go</p> {"question": "Q?", "options": ["a", "b", "c", "d"], "answer": 2, '
        '"explanation": "because", "source": "S2"}',
        '{"question": "Q?", "options": ["a", "a", "c", "d"], "answer": 2}',          # duplicate option
        '{"question": "Explain X", "key_points": ["p1", "p2"], "source": "S1"}',
        '{"score": 5, "feedback": "good"}',
    ])
    real = quiz.llm_chat
    quiz.llm_chat = lambda messages, temperature=0: next(replies)
    try:
        rng = random.Random(4)
        q = quiz.llm_mcq(FakeRetriever(), "rr", 1, rng)
        assert q.options[q.answer] == "c" and q.cite.endswith("slides 72-74") and q.origin == "llm"
        note = next_question(FakeRetriever(), Progress(), rng, ["mcq"], "rr", use_llm=True)[1]
        assert "built-in bank" in note                                            # malformed -> bank
        v = quiz.llm_viva(FakeRetriever(), "rr", 1)
        assert v.kind == "viva" and v.key_points == ["p1", "p2"] and v.passage == "passage text"
        assert quiz.grade_viva(v, "my answer") == (2, "good")                    # score clamped to 0-2
    finally:
        quiz.llm_chat = real


def test_next_question_without_llm():
    rng = random.Random(5)
    for _ in range(60):
        q, note = next_question(None, Progress(), rng, ["mcq", "numeric", "viva"], None, use_llm=False)
        assert q.kind in ("mcq", "numeric") and not note
    q, _ = next_question(None, Progress(), rng, ["numeric"], None, use_llm=False)
    assert q.kind == "numeric"


def test_follow_up_numericals():
    prev = parse_problem("Round Robin, time quantum 3. P1: AT 0, BT 5. P2: AT 1, BT 4. P3: AT 3, BT 2.")
    again = follow_up("now do it with time quantum 2", prev)
    assert again.algo == "rr" and again.quantum == 2 and again.procs == prev.procs
    assert follow_up("what if we use SRTF instead?", prev).algo == "srtf"
    assert follow_up("How does round robin work?", prev) is None
    assert follow_up("Explain SRTF with an example", prev) is None
    assert follow_up("now with quantum 2", None) is None
    try:
        solve(follow_up("now solve it with priority scheduling", prev))
    except ProblemError:
        pass
    else:
        raise AssertionError("priority without priorities should be refused")


def test_follow_up_detection():
    assert is_follow_up("why does it happen?") and is_follow_up("explain that step again")
    assert not is_follow_up("What is the convoy effect?")
    assert not is_follow_up("What is the convoy effect and why does it happen in FCFS scheduling?")


def test_worked_examples():
    for e in EXAMPLES:
        solve_any(parse_any(e.question))
    hits = [Hit(112, 114), Hit(44, 46), Hit(206, 207, "textbook")]
    assert find_example("Explain SRTF with an example", hits).name == "SRTF with an I/O burst"
    assert find_example("Explain SRTF with an example", hits[1:]).name == "SRTF Example 3"
    assert find_example("Give an example of round robin", []).name == "Round Robin Example 2"
    assert find_example("Show a preemptive priority example", []).name == "Priority Example 2"
    assert find_example("Explain SRTF", hits) is None                 # no example asked for
    assert find_example("Give an example of the convoy effect", hits) is None   # no algorithm named
    other_deck = [Hit(44, 46, source="keerthika_m4")]                 # same slide numbers, other deck
    assert find_example("Explain SRTF with an example", other_deck).name == "SRTF Example 3"  # default, not slide match
    # the other solvers' examples: the topic, then the wording ("request", "detection", "C-SCAN") picks one
    assert find_example("Explain FCFS disk scheduling with an example", hits).deck == "silberschatz10"
    assert find_example("Explain C-SCAN with an example", []).name.startswith("C-SCAN")
    assert find_example("Explain the Banker's algorithm with an example", []).name == "Banker's Algorithm Example 1"
    assert "request from P1" in find_example("Banker's algorithm resource request example", []).name
    assert find_example("Explain deadlock detection with an example", []).name == "Deadlock detection example"
    assert find_example("Explain LRU page replacement with an example", []).name.startswith("Page replacement")
    assert find_example("Show Belady's anomaly with an example", []).name.startswith("Belady")
    textbook_front_matter = [Hit("P-45", "P-45", "textbook")]
    assert find_example("Explain SSTF disk scheduling with an example", textbook_front_matter).deck == "silberschatz10"


def test_other_topics_are_not_scheduling_problems():
    for q in ("Banker's algorithm. P0: Allocation 0 1 0, Max 7 5 3. P1: Allocation 2 0 0, Max 3 2 2. "
              "Available 3 3 2. Is the system safe?",
              "P1 requests resources (1, 0, 2). Can it be granted? P0 0 1 0 P1 2 0 0",
              "FCFS disk scheduling: head at 53, requests 98 183 37 122 14 124 65 67",
              "LRU page replacement with 3 frames, reference string 7 0 1 2 0 3 0 4",
              "Dining philosophers P0 P1 P2 P3 P4 each pick up the left chopstick"):
        assert parse_problem(q) is None, q
    # a real scheduling numerical is still read, even if it mentions waiting
    assert parse_problem("Round robin, quantum 2. P1: AT 0, BT 5. P2: AT 1, BT 3.").algo == "rr"


def test_new_topics_offline():
    rng = random.Random(6)
    offline = quiz.offline_topics(["mcq", "numeric"])
    assert "banker" in offline and "disk" in offline and "rr" in offline and "process" in offline
    assert "banker" not in quiz.offline_topics(["mcq"]) and "sync" not in offline
    for _ in range(80):        # adaptive choice never lands on a topic with no built-in questions
        q, _ = next_question(None, Progress(), rng, ["mcq", "numeric"], None, use_llm=False)
        assert q.topic in offline
    try:
        next_question(None, Progress(), rng, ["mcq"], "sync", use_llm=False)
    except ValueError as e:
        assert "need the LLM" in str(e)
    else:
        raise AssertionError("a topic with no built-in questions should be refused offline")
    # a topic with numericals but no built-in MCQs falls back to a numerical
    q, _ = next_question(None, Progress(), rng, ["mcq"], "banker", use_llm=False)
    assert q.kind == "numeric" and q.origin == "solver"


def test_answer_clean_up():
    assert clean_answer("a 【S1】 b [**S2**] c [S9]", 6) == "a [S1] b [S2] c "


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print("ok  ", fn.__name__)
