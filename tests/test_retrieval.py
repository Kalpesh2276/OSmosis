"""Checks retrieval fusion, the evaluation metrics, query intent, citations and follow-ups.

    python -m tests.test_retrieval        (or: pytest tests)

Fusion is tested on synthetic rankings, so no embedding model or vector index is needed; the
figure and follow-up checks use the real passages with keyword search only.
"""
from rag import ask, retrieval_eval as ev
from rag.intent import classify, is_follow_up
from rag.retriever import Retriever


def fake(chunks, back_matter=()):
    """A Retriever over the given chunks, without loading the index."""
    r = Retriever.__new__(Retriever)
    r.chunks, r.back_matter = chunks, set(back_matter)
    return r


def chunk(n, kind="textbook", section=None, text=None):
    return {"id": f"c{n}", "source_kind": kind, "source_id": "book" if kind == "textbook" else "slides",
            "section": section or f"S{n}", "text": text or f"passage number {n} " + "x" * 130}


BASE = {**ev.BASELINE, "depth": 50}


def test_fusion_is_weighted_rrf_and_deterministic():
    r = fake([chunk(i) for i in range(4)])
    # chunk 0: first by BM25 only; chunk 1: first by embeddings only; tie at 1:1 weights
    bm, dense = [0, 2], [1, 2]
    order = [i for i, _ in r.fuse(bm, dense, fusion={**BASE, "teacher_boost": 1.0})]
    assert order == [2, 0, 1]                      # 2 is in both lists; the 0/1 tie keeps keyword order
    assert order == [i for i, _ in r.fuse(bm, dense, fusion={**BASE, "teacher_boost": 1.0})]
    s = dict(r.fuse(bm, dense, fusion={**BASE, "teacher_boost": 1.0, "rrf_k": 60}))
    assert abs(s[2] - (1 / 62 + 1 / 62)) < 1e-12 and abs(s[0] - 1 / 61) < 1e-12
    emb = [i for i, _ in r.fuse(bm, dense, fusion={**BASE, "teacher_boost": 1.0, "bm25_weight": 0.5})]
    assert emb.index(1) < emb.index(0)             # embedding-heavy: the embedding favourite first
    only_bm = [i for i, _ in r.fuse(bm, dense, fusion={**BASE, "dense_weight": 0.0})]
    assert only_bm == [0, 2]                       # a zero-weight ranking contributes nothing


def test_teacher_boost_and_example_intent():
    # the "memory stall" case found in the audit: the textbook passage is 1st in both rankings,
    # an unrelated teacher slide 3rd and 6th. The old 1.15 boost put the slide first; 1.05 does not.
    r = fake([chunk(0), chunk(1), chunk(2), chunk(3, "teacher_notes"), chunk(4), chunk(5)])
    bm, dense = [0, 1, 3, 2, 4, 5], [0, 1, 2, 4, 5, 3]
    for boost, first in ((1.0, 0), (1.05, 0), (1.15, 3)):
        assert r.fuse(bm, dense, fusion={**BASE, "teacher_boost": boost})[0][0] == first, boost
    r = fake([chunk(0), chunk(1, "teacher_notes")])
    near = ([0, 1], [1, 0])                        # a tie: the boost decides
    assert r.fuse(*near, fusion={**BASE, "teacher_boost": 1.05})[0][0] == 1
    assert r.fuse(*near, fusion={**BASE, "teacher_boost": 1.0})[0][0] == 0
    # 1st vs 6th in both lists: 1.05 alone does not flip it; for an example question 1.05 x 1.15 does
    r6 = fake([chunk(i) for i in range(5)] + [chunk(5, "teacher_notes")])
    lists = ([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5])
    cfg = {**BASE, "teacher_boost": 1.05, "example_teacher_boost": 1.15}
    assert r6.fuse(*lists, fusion=cfg, intent="conceptual")[0][0] == 0
    assert r6.fuse(*lists, fusion=cfg, intent="example")[0][0] == 5
    assert r6.fuse(*lists, fusion=cfg, intent="numerical")[0][0] == 5


def test_back_matter_and_duplicates():
    r = fake([chunk(0, section="Further Reading"), chunk(1)], back_matter=[0])
    lists = ([0, 1], [1, 0])
    assert r.fuse(*lists, fusion={**BASE, "teacher_boost": 1.0})[0][0] == 0
    assert r.fuse(*lists, fusion={**BASE, "teacher_boost": 1.0, "back_matter_weight": 0.5})[0][0] == 1
    dup = fake([chunk(0, text="same text " * 20), chunk(1, text="same text " * 20), chunk(2)])
    assert [i for i, _ in dup.fuse([0, 1, 2], [0, 1, 2], fusion=BASE)] == [0, 2]   # the copy is dropped
    sec = fake([chunk(0, section="5.3"), chunk(1, section="5.3"), chunk(2, section="5.3"), chunk(3)])
    capped = [i for i, _ in sec.fuse([0, 1, 2, 3], [0, 1, 2, 3], fusion={**BASE, "max_per_section": 2})]
    assert capped == [0, 1, 3]


def test_kind_slot_brings_both_sources_forward():
    r = fake([chunk(0), chunk(1), chunk(2), chunk(3, "teacher_notes")])
    lists = ([0, 1, 2, 3], [0, 1, 2, 3])
    plain = [i for i, _ in r.fuse(*lists, fusion={**BASE, "teacher_boost": 1.0})]
    slotted = [i for i, _ in r.fuse(*lists, fusion={**BASE, "teacher_boost": 1.0, "kind_slot": 2})]
    assert plain == [0, 1, 2, 3] and slotted == [0, 3, 1, 2]


def test_metrics():
    m = ev.metrics([1, 2, None, 6])
    assert m["hit@1"] == 0.25 and m["hit@3"] == 0.5 and m["hit@6"] == 0.75 and m["n"] == 4
    assert abs(m["mrr"] - (1 + 0.5 + 0 + 1 / 6) / 4) < 1e-12


def test_evaluation_uses_the_same_labelled_questions():
    import eval_retrieval as labels
    items = ev.cases(labels.GROUPS)
    assert len(items) == 78 and len({c.question for c in items}) == 78
    held = [c for c in items if c.held_out]
    assert 15 <= len(held) <= 22 and held == [c for c in ev.cases(labels.GROUPS) if c.held_out]
    # every variant is scored on the same two rankings per question
    r = fake([chunk(i) for i in range(3)])
    c = ev.Case("q", ["S1"], [], "g", "t", False, "conceptual", [1, 0], [1, 2])
    c.sections = []
    c.ranges = []
    assert ev.first_relevant(r, c, BASE) is None
    r.chunks[1]["section"] = "5.3 x"
    c.sections = ["5.3"]
    assert ev.first_relevant(r, c, BASE) == 1
    assert ev.topic_of(["5.3.1"], []) == "CPU scheduling" and ev.topic_of([], [("lab_shell_slides", 1, 2)]) == \
        "Linux and shell"


def test_intent():
    cases = {
        "Explain SRTF with an example": "example",
        "Write a shell script to find the factorial of a number": "example",
        "Solve using Round Robin with quantum 2. P1: AT 0, BT 5.": "numerical",
        "How many page faults occur with LRU and 3 frames?": "numerical",
        "What is starvation and how does aging solve it?": "definition",   # "solve" here is not a command
        "Difference between paging and segmentation": "comparison",
        "What is the convoy effect?": "definition",
        "How does round robin work?": "conceptual",
    }
    for q, want in cases.items():
        assert classify(q) == want, (q, classify(q))
    assert classify("why does it happen?", has_history=True) == "follow_up"
    assert classify("why does it happen?") != "follow_up" and is_follow_up("explain that step again")


def test_citations_match_the_context():
    class H:
        def __init__(self, n):
            self.cite, self.item = f"Source {n}", {"text": f"text {n}"}
    hits = [H(1), H(2)]
    context = ask.build_context(hits, [])
    assert "[S1] (Source 1)" in context and "[S2] (Source 2)" in context and "[S3]" not in context
    assert ask.clean_answer("a [S1] b [S2] c [S3] d [S7]", len(hits)) == "a [S1] b [S2] c  d "


class Recorder:
    """Stands in for the retriever and records what it was asked."""
    def __init__(self):
        self.queries = []

    def retrieve(self, query, k=6):
        self.queries.append(query)
        return [], []


def test_follow_up_retrieval_keeps_the_topic():
    rec = Recorder()
    history = [("Explain the convoy effect in FCFS", "answer ...")]
    ask.answer(rec, "why does it happen?", use_llm=False, history=history)
    ask.answer(rec, "What is thrashing?", use_llm=False, history=history)
    assert "convoy effect" in rec.queries[0]          # the short follow-up carries the earlier question
    assert rec.queries[1] == "What is thrashing?"     # a new question does not


def test_explaining_a_mistake_keeps_the_solver_facts():
    import os
    from rag import solver
    from rag.diagnosis import diagnose
    p = solver.parse_problem("SRTF. P1: AT 0, BT 8. P2: AT 1, BT 4. P3: AT 2, BT 9. P4: AT 3, BT 5.")
    f = diagnose(p, {"gantt": "P1 0-8, P4 8-13, P2 13-17, P3 17-26"}).first
    rec = Recorder()
    key = os.environ.pop("LLM_API_KEY", None)
    try:
        reply = ask.explain_mistake(rec, p, f.where, f.hint, f.explanation)
    finally:
        if key is not None:
            os.environ["LLM_API_KEY"] = key
    assert not reply.llm_used and "LLM_API_KEY" in reply.error          # no key: no LLM, no crash
    assert "COMPUTED SOLUTION (exact)" in reply.body and "P2 1-5" in reply.body
    assert f"{f.where}: {f.explanation}" in reply.body
    assert "P2" not in rec.queries[0] and rec.queries[0].startswith("Shortest Remaining Time First")   # the rule only


def test_figures_stay_attached():
    r = Retriever(use_dense=False)                    # keyword search over the real passages
    hits, figs = r.retrieve("What is the convoy effect in FCFS scheduling?")
    assert hits and any(h.item["section"].startswith("5.3.1") or h.item["source_kind"] == "teacher_notes"
                        for h in hits)
    linked = {f["id"] for h in hits[:2] for f in h.figures}
    assert linked and linked <= {f["id"] for f in figs} | linked
    assert all(f["id"] in r.fig_by_id for f in figs)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print("ok  ", fn.__name__)
