"""Retrieval evaluation and fusion experiments on the hand-labelled question set.

The labels live in eval_retrieval.py: (question, textbook sections, [(source id, first, last)]),
where first/last are slide or page numbers, or a section-title prefix with last = None. A passage
is relevant if it comes from a labelled section or range.

Each question is ranked once by BM25 and once by the embeddings; every fusion variant is then
scored on those same two rankings, so variants differ only in how the rankings are combined.

Tuning procedure (--tune): the questions are split deterministically - every 4th question of each
group is held out - and a small grid of fusion settings is scored on the other 3/4. Settings that
lower Hit@3 or Hit@6 there are ruled out; of the rest, the best MRR is chosen (ties: higher
Hit@1, then the setting closest to the current one), then reported on the held-out quarter,
which played no part in the choice. With 78 questions, a difference of one or two questions is
within noise; read the numbers as indications.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass

from .config import FUSION, TOP_K
from .intent import classify
from .retriever import Retriever

KS = (1, 3, TOP_K)

# topic of a question, from the source its answer lives in
DECK_TOPIC = {
    "keerthika_m3": "CPU scheduling", "keerthika_m2": "Processes and threads", "keerthika_m1": "Processes and threads",
    "keerthika_m4": "Synchronization", "keerthika_m3_deadlock": "Deadlocks and Banker's algorithm",
    "keerthika_m5a": "Memory and page replacement", "notes_m5b": "Memory and page replacement",
    "lab_linux_commands": "Linux and shell", "lab_shell_slides": "Linux and shell",
    "lab_shell_exercises": "Linux and shell",
}
CHAPTER_TOPIC = {"2": "Processes and threads", "3": "Processes and threads", "4": "Processes and threads",
                 "5": "CPU scheduling", "6": "Synchronization", "7": "Synchronization",
                 "8": "Deadlocks and Banker's algorithm", "9": "Memory and page replacement",
                 "10": "Memory and page replacement", "11": "Storage and file systems",
                 "13": "Storage and file systems", "14": "Storage and file systems",
                 "20.4": "Processes and threads", "20.5": "CPU scheduling", "20.7": "Storage and file systems"}


def topic_of(sections, ranges) -> str:
    for deck, _, _ in ranges:
        return DECK_TOPIC[deck]
    for s in sections:
        for prefix in sorted(CHAPTER_TOPIC, key=len, reverse=True):
            if s == prefix or s.startswith(prefix + "."):
                return CHAPTER_TOPIC[prefix]
    return "Other"


def relevant(chunk, sections, ranges) -> bool:
    if chunk["source_kind"] == "textbook":   # a labelled section includes its subsections
        number = chunk["section"].split(" ")[0]
        return any(number == s or number.startswith(s + ".") for s in sections)
    for deck, lo, hi in ranges:
        if chunk["source_id"] != deck:
            continue
        if isinstance(lo, str):              # section title
            if chunk["section"].startswith(lo):
                return True
        elif lo <= int(chunk["page_end"]) and int(chunk["page_start"]) <= hi:
            return True
    return False


@dataclass
class Case:
    question: str
    sections: list
    ranges: list
    group: str
    topic: str
    held_out: bool
    intent: str = "conceptual"
    bm_rank: list[int] | None = None
    dense_rank: list[int] | None = None


def cases(groups: dict[str, list]) -> list[Case]:
    out = []
    for group, questions in groups.items():
        for n, (q, sec, rng) in enumerate(questions):
            out.append(Case(q, sec, rng, group, topic_of(sec, rng), n % 4 == 3, classify(q)))
    return out


def rank_all(r: Retriever, items: list[Case]):
    for c in items:
        c.bm_rank, c.dense_rank = r.rankings(c.question)


def first_relevant(r: Retriever, c: Case, fusion: dict, sections=None, ranges=None) -> int | None:
    sections = c.sections if sections is None else sections
    ranges = c.ranges if ranges is None else ranges
    for rank, (i, _) in enumerate(r.fuse(c.bm_rank, c.dense_rank, fusion=fusion, intent=c.intent), 1):
        if relevant(r.chunks[i], sections, ranges):
            return rank
    return None


def metrics(ranks: list[int | None]) -> dict:
    n = len(ranks) or 1
    out = {f"hit@{k}": sum(x is not None and x <= k for x in ranks) / n for k in KS}
    out["mrr"] = sum(1 / x for x in ranks if x) / n
    out["n"] = len(ranks)
    return out


def evaluate(r: Retriever, items: list[Case], fusion: dict) -> dict:
    return metrics([first_relevant(r, c, fusion) for c in items])


# the fusion settings before tuning (2026-10-23), kept so the baseline stays reproducible
BASELINE = {"bm25_weight": 1.0, "dense_weight": 1.0, "rrf_k": 60, "teacher_boost": 1.15, "example_teacher_boost": 1.0,
            "back_matter_weight": 1.0, "max_per_section": None, "kind_slot": None}
VARIANTS = {   # the controlled comparison
    "A. BM25 only": {**BASELINE, "dense_weight": 0.0, "teacher_boost": 1.0},
    "B. Embeddings only": {**BASELINE, "bm25_weight": 0.0, "teacher_boost": 1.0},
    "C. Baseline RRF (1:1, boost 1.15)": BASELINE,
    "D. Embedding-heavy (BM25 0.5)": {**BASELINE, "bm25_weight": 0.5},
    "E. BM25-heavy (embeddings 0.5)": {**BASELINE, "dense_weight": 0.5},
    "G. Baseline without teacher boost": {**BASELINE, "teacher_boost": 1.0},
    "H. No boost + both source kinds in top 2": {**BASELINE, "teacher_boost": 1.0, "kind_slot": 2},
    "I. BM25 0.5, no boost + both kinds in top 2": {**BASELINE, "bm25_weight": 0.5, "teacher_boost": 1.0,
                                                    "kind_slot": 2},
    "J. Baseline + back matter x0.5": {**BASELINE, "back_matter_weight": 0.5},
    "K. No boost, boost 1.15 only for example questions": {**BASELINE, "teacher_boost": 1.0,
                                                           "example_teacher_boost": 1.15},
    "F. Selected = config.FUSION": {},
}
GRID = {"bm25_weight": [0.5, 0.75, 1.0], "teacher_boost": [1.0, 1.05], "rrf_k": [10, 30, 60],
        "max_per_section": [None, 2], "kind_slot": [None, 2], "back_matter_weight": [1.0, 0.5, 0.0],
        "example_teacher_boost": [1.0, 1.15, 1.3]}


def tune(r: Retriever, items: list[Case], base: dict = BASELINE) -> tuple[dict, list]:
    """Grid search on the tuning questions -> (best setting, all results sorted best first).

    A setting qualifies only if it keeps Hit@3 and Hit@6 at least as high as `base` on the
    tuning questions (no regression); among those, the highest MRR wins, then the higher Hit@1,
    then the setting closest to `base`."""
    tuning = [c for c in items if not c.held_out]
    now = evaluate(r, tuning, base)
    results = []
    for values in itertools.product(*GRID.values()):
        setting = {**base, **dict(zip(GRID, values))}
        m = evaluate(r, tuning, setting)
        ok = m["hit@3"] >= now["hit@3"] and m["hit@6"] >= now["hit@6"]
        distance = sum(abs((setting[k] or 0) - (base[k] or 0)) / (60 if k == "rrf_k" else 1) for k in GRID)
        results.append((setting, m, distance, ok))
    results.sort(key=lambda x: (not x[3], -x[1]["mrr"], -x[1]["hit@1"], x[2]))
    return results[0][0], [(s, m, d) for s, m, d, ok in results if ok] or [(s, m, d) for s, m, d, _ in results]


def per_question(r: Retriever, items: list[Case], fusion: dict) -> list[dict]:
    """Ranked passage ids, scores and source kinds for each question (for --save)."""
    rows = []
    for c in items:
        ranked = r.fuse(c.bm_rank, c.dense_rank, fusion=fusion, intent=c.intent)
        rows.append({
            "question": c.question, "group": c.group, "topic": c.topic, "held_out": c.held_out,
            "first_relevant": first_relevant(r, c, fusion),
            "results": [{"id": r.chunks[i]["id"], "score": round(s, 6), "source_kind": r.chunks[i]["source_kind"],
                         "relevant": relevant(r.chunks[i], c.sections, c.ranges)} for i, s in ranked],
        })
    return rows


def source_kind_breakdown(r: Retriever, items: list[Case], fusion: dict) -> dict:
    """For questions labelled with both a textbook section and teacher material: how often each
    kind of source is found, and which kind the top passage is."""
    both = [c for c in items if c.sections and c.ranges]
    book = metrics([first_relevant(r, c, fusion, ranges=[]) for c in both])
    notes = metrics([first_relevant(r, c, fusion, sections=[]) for c in both])
    top_teacher = sum(r.chunks[r.fuse(c.bm_rank, c.dense_rank, k=1, fusion=fusion, intent=c.intent)[0][0]]["source_kind"]
                      == "teacher_notes" for c in items) / len(items)
    return {"questions_with_both": len(both), "textbook_found": book, "teacher_found": notes,
            "top1_is_teacher_material": top_teacher}


def table(rows: list[tuple[str, dict]]) -> str:
    head = f"{'':<44}" + "".join(f"Hit@{k:<5}" for k in KS) + "MRR     n"
    lines = [head]
    for name, m in rows:
        lines.append(f"{name:<44}" + "".join(f"{m[f'hit@{k}']:<9.1%}" for k in KS) + f"{m['mrr']:<8.3f}{m['n']}")
    return "\n".join(lines)


def save(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
