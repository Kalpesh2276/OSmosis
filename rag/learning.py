"""Evidence of what a student understands, and the Misconception Graph built from it.

Evidence records come from three places and are kept in the progress file (Progress.evidence):
  debugger  a "Check my working" diagnosis: each finding with a misconception id is an error
            record (deterministic or uncertain, as the validator said); a fully correct answer is a
            correct record for the concepts it exercised
  lab       an OS Lab prediction: a wrong prediction is uncertain evidence (a guess can be wrong for
            many reasons); a right one is a correct record
  quiz      a marked quiz answer: a correct or error record for the topic's concept (no misconception)

Record: {"n", "time", "source", "concept", "misconception" (id or None), "certain", "outcome"
("error" | "correct"), "problem" (short id of the problem, so repeats of one attempt are not counted
twice)}.

The rules are transparent and deliberately coarse; a handful of attempts is not a precise mastery
score:
  misconception  recurring  - a certain error on 2 or more different problems, not yet resolved
                 tentative  - one certain error, or only uncertain evidence, not yet resolved
                 resolved   - 2 correct attempts on the concerned concepts after the last error
  concept        understood - 3+ attempts, at least 75% correct, and no active misconception
                 weak       - an active misconception (tentative or recurring) concerns it
                 needs more evidence - otherwise (fewer than 3 attempts, or a mixed record)
"""
from __future__ import annotations

import datetime as _dt
import zlib

from .misconceptions import CONCEPTS, MISCONCEPTIONS

ALGO_CONCEPT = {"fcfs": "sched.fcfs", "sjf": "sched.sjf", "srtf": "sched.srtf", "priority": "sched.priority",
                "priority_p": "sched.priority", "rr": "sched.rr"}
PAGING_CONCEPT = {"fifo": "paging.fifo", "lru": "paging.lru", "optimal": "paging.optimal"}
DISK_CONCEPT = {"fcfs": "disk.movement", "sstf": "disk.sstf", "scan": "disk.scan", "look": "disk.scan",
                "cscan": "disk.cscan", "clook": "disk.cscan"}
# misconceptions about whichever algorithm was asked: recorded against that algorithm's concept
ABOUT_THE_ALGORITHM = {"sched.selection_rule", "sched.wrong_algorithm", "paging.wrong_algorithm",
                       "disk.wrong_algorithm"}
TOPIC_CONCEPT = {}          # quiz topic id -> its main concept (the first concept listed for it)
for _c in CONCEPTS.values():
    TOPIC_CONCEPT.setdefault(_c.topic, _c.id)


def problem_id(text: str) -> str:
    return format(zlib.crc32(text.encode("utf-8")), "08x")


def _add(progress, **rec):
    rec.setdefault("time", _dt.datetime.now().isoformat(timespec="seconds"))
    rec["n"] = (progress.evidence[-1]["n"] + 1) if progress.evidence else 1
    progress.evidence.append(rec)
    return rec


def _main_concept(d) -> str:
    p = d.problem
    if d.kind == "scheduling":
        return ALGO_CONCEPT[p.algo]
    if d.kind == "paging":
        return PAGING_CONCEPT[p.algos[0]]
    if d.kind == "disk":
        return DISK_CONCEPT[p.algos[0]]
    return "banker.request" if p.request else "banker.safety"


def record_diagnosis(progress, d, problem_text: str) -> list[dict]:
    """Evidence from a Reasoning Debugger diagnosis (nothing is recorded for an unreadable answer)."""
    pid, added = problem_id(problem_text), []
    main = _main_concept(d)
    if d.verdict == "correct":
        concepts = {main}
        joined = " ".join(d.correct)
        if d.kind == "scheduling":
            if any(k in joined for k in ("CT of", "TAT of", "WT of", "RT of", "average")):
                concepts.add("sched.metrics")
            if "Gantt chart" in joined:
                concepts.add("sched.gantt")
                steps = d.solution.steps
                # a correct chart where some arrival met a running process shows the preemption decision
                if any(st["arrived"] and st["before"] is not None and st["action"] in ("continue", "preempt")
                       for st in steps):
                    concepts.add("sched.preemption")
                if any(q.io for q in d.problem.procs):
                    concepts.add("sched.io")
        if d.kind == "banker" and "Need matrix" in joined:
            concepts.add("banker.need")
        for c in sorted(concepts):
            added.append(_add(progress, source="debugger", concept=c, misconception=None, certain=True,
                               outcome="correct", problem=pid))
        return added
    for f in d.findings:
        if f.misconception:
            concept = main if f.misconception in ABOUT_THE_ALGORITHM else MISCONCEPTIONS[f.misconception].concept
            added.append(_add(progress, source="debugger", concept=concept, misconception=f.misconception,
                              certain=f.certain, outcome="error", problem=pid, where=f.where))
    if d.findings and not added:      # a wrong answer with no recognised cause still counts against the concept
        added.append(_add(progress, source="debugger", concept=main, misconception=None, certain=True,
                          outcome="error", problem=pid))
    return added


def record_prediction(progress, lab, pos: int, guess: int | None, problem_text: str) -> dict | None:
    """Evidence from one OS Lab prediction (none if the student did not predict)."""
    if guess is None:
        return None
    e = lab.events[pos]
    pid = problem_id(problem_text)
    if lab.kind == "scheduling":
        concept = ALGO_CONCEPT[lab.problem.algo]
    elif lab.kind == "paging":
        concept = PAGING_CONCEPT[lab.problem.algos[0]]
    elif lab.kind == "disk":
        concept = DISK_CONCEPT[lab.problem.algos[0]]
    else:
        concept = "banker.safety"
    if guess == e.answer:
        return _add(progress, source="lab", concept=concept, misconception=None, certain=True, outcome="correct",
                    problem=pid)
    mis = None
    if lab.kind == "scheduling":
        st = e.data["step"]
        picked = e.options[guess]
        if st["preempted"] is not None and picked == f"P{st['preempted']}":
            mis = "sched.preemption"                        # expected the running process to keep the CPU
        elif lab.problem.algo == "rr":
            mis = "sched.arrival_boundary" if st["expired"] and st["arrived"] and picked == f"P{st['expired']}" \
                else "sched.rr_queue"
        elif lab.problem.algo in ("sjf", "priority") and st["before"] is not None and st["action"] == "continue":
            mis = "sched.preemption"                        # expected a preemption in a non-preemptive algorithm
        else:
            mis = "sched.selection_rule"
    elif lab.kind == "banker":
        mis = "banker.need_vs_work"
    elif lab.kind == "paging":
        mis = "paging.hit_fault" if (guess == 0) != (e.answer == 0) else "paging.wrong_algorithm"
    elif lab.kind == "disk":
        mis = "disk.wrong_algorithm"
    return _add(progress, source="lab", concept=concept, misconception=mis, certain=False, outcome="error",
                problem=pid, where=e.title)


def record_quiz(progress, q, points: float) -> dict | None:
    """Evidence from a marked quiz answer, for the topic's concept (topics without one are skipped)."""
    concept = TOPIC_CONCEPT.get(q.topic)
    if concept is None:
        return None
    return _add(progress, source="quiz", concept=concept, misconception=None, certain=True,
                outcome="correct" if points >= 1 else "error", problem=problem_id(q.prompt))


# ---------------------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------------------
def assess(evidence: list[dict]) -> dict:
    """{"misconceptions": {id: {...}}, "concepts": {id: {...}}} from the evidence, by the rules above."""
    mis = {}
    for m_id in MISCONCEPTIONS:
        errs = [e for e in evidence if e.get("misconception") == m_id and e["outcome"] == "error"]
        if not errs:
            continue
        concepts = {e["concept"] for e in errs}           # the concepts where it was actually seen
        last = max(e["n"] for e in errs)
        later = [e for e in evidence if e["n"] > last and e["outcome"] == "correct" and e["concept"] in concepts]
        certain_problems = {e["problem"] for e in errs if e["certain"]}
        if len(later) >= 2:
            status = "resolved"
        elif len(certain_problems) >= 2:
            status = "recurring"
        else:
            status = "tentative"
        mis[m_id] = {"status": status, "certain": sum(e["certain"] for e in errs),
                     "uncertain": sum(not e["certain"] for e in errs), "problems": len({e["problem"] for e in errs}),
                     "correct_since": len(later), "concepts": sorted(concepts), "evidence": errs}
    concepts = {}
    for c_id in CONCEPTS:
        recs = [e for e in evidence if e["concept"] == c_id]
        right = sum(e["outcome"] == "correct" for e in recs)
        active = [m for m, v in mis.items() if c_id in v["concepts"] and v["status"] != "resolved"]
        if any(mis[m]["status"] == "recurring" for m in active):
            status = "weak (recurring)"
        elif active:
            status = "weak (tentative)"
        elif len(recs) >= 3 and right / len(recs) >= 0.75:
            status = "understood"
        else:
            status = "needs more evidence"
        concepts[c_id] = {"status": status, "attempts": len(recs), "correct": right, "active": active}
    return {"misconceptions": mis, "concepts": concepts}


SEVERITY = {"recurring": 0, "tentative": 1, "resolved": 2}


def recommendations(a: dict) -> list[dict]:
    """What to practise next: active misconceptions first (recurring before tentative), each with its
    remediation, a practice activity, and any prerequisite concept that is not yet understood."""
    out = []
    for m_id, v in sorted(a["misconceptions"].items(), key=lambda kv: (SEVERITY[kv[1]["status"]], -kv[1]["certain"])):
        if v["status"] == "resolved":
            continue
        m = MISCONCEPTIONS[m_id]
        c = CONCEPTS[m.concept]
        prereq = [p for p in c.prerequisites if a["concepts"][p]["status"] != "understood"]
        out.append({"misconception": m_id, "status": v["status"], "concept": c.id, "topic": c.topic,
                    "do": m.remediation, "practice": m.practice, "refs": list(c.refs),
                    "prerequisites": prereq})
    return out


STATUS_COLOUR = {"understood": "#b7e4c7", "needs more evidence": "#e6e6e6", "weak (tentative)": "#fdd49e",
                 "weak (recurring)": "#f4a6a6", "recurring": "#f4a6a6", "tentative": "#fdd49e", "resolved": "#b7e4c7"}


def dot(a: dict, only_seen: bool = True) -> str:
    """Graphviz DOT: concepts (boxes, coloured by status), prerequisite arrows, and the student's
    misconceptions (ellipses) attached to the concepts they concern."""
    seen = {c for c, v in a["concepts"].items() if v["attempts"]} | \
        {c for v in a["misconceptions"].values() for c in v["concepts"]}
    shown = set(CONCEPTS) if not only_seen else set(seen)
    for c in list(shown):                                    # include prerequisites of what is shown
        stack = [c]
        while stack:
            for p in CONCEPTS[stack.pop()].prerequisites:
                if p not in shown:
                    shown.add(p)
                    stack.append(p)
    lines = ['digraph G {', 'bgcolor="transparent"; rankdir=LR; node [fontname="Helvetica", fontsize=11, '
             'style="filled,rounded", shape=box, color="#888888", fontcolor="#1b1b1b"]; edge [color="#999999"];']
    for c_id in sorted(shown):
        v = a["concepts"][c_id]
        label = f"{CONCEPTS[c_id].name}\\n{v['status']}" + (f" ({v['correct']}/{v['attempts']})" if v["attempts"] else "")
        lines.append(f'"{c_id}" [label="{label}", fillcolor="{STATUS_COLOUR[v["status"]]}"];')
    for c_id in sorted(shown):
        for p in CONCEPTS[c_id].prerequisites:
            if p in shown:
                lines.append(f'"{p}" -> "{c_id}";')
    for m_id, v in a["misconceptions"].items():          # "m:" keeps them apart from concepts of the same id
        lines.append(f'"m:{m_id}" [label="{MISCONCEPTIONS[m_id].name}\\n{v["status"]}", shape=ellipse, '
                     f'fillcolor="{STATUS_COLOUR[v["status"]]}"];')
        for c in v["concepts"]:
            if c in shown:
                lines.append(f'"m:{m_id}" -> "{c}" [style=dashed, color="#cc6666", arrowhead=none];')
    lines.append("}")
    return "\n".join(lines)


def sample_evidence() -> list[dict]:
    """Made-up evidence for the demonstration view (always labelled as sample data in the app)."""
    base = {"source": "debugger", "certain": True, "time": "sample"}
    recs = [
        {"concept": "sched.srtf", "misconception": "sched.preemption", "outcome": "error", "problem": "s1"},
        {"concept": "sched.srtf", "misconception": "sched.preemption", "outcome": "error", "problem": "s2"},
        {"concept": "sched.rr", "misconception": "sched.arrival_boundary", "outcome": "error", "problem": "s3"},
        {"concept": "sched.metrics", "misconception": "sched.metric_confusion", "outcome": "error", "problem": "s4",
         "certain": False},
        {"concept": "sched.fcfs", "misconception": None, "outcome": "correct", "problem": "s5"},
        {"concept": "sched.fcfs", "misconception": None, "outcome": "correct", "problem": "s6"},
        {"concept": "sched.fcfs", "misconception": None, "outcome": "correct", "problem": "s7"},
        {"concept": "banker.safety", "misconception": "banker.need_vs_work", "outcome": "error", "problem": "s8"},
        {"concept": "banker.safety", "misconception": None, "outcome": "correct", "problem": "s9"},
        {"concept": "banker.safety", "misconception": None, "outcome": "correct", "problem": "s10"},
    ]
    return [{**base, **r, "n": i + 1} for i, r in enumerate(recs)]
