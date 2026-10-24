"""Measure retrieval quality on a hand-labelled question set.

    python eval_retrieval.py                 # current configuration, by question group and topic
    python eval_retrieval.py --tune          # controlled variants + grid search with a held-out check
    python eval_retrieval.py --save run.json # also save each question's ranked passages and scores

Each question is labelled with where its answer lives: textbook section numbers and/or slide
(or page, or section-title) ranges of a given source. A retrieved passage is relevant if it comes
from one of them. Reported: Hit@k (a relevant passage among the first k) and MRR (mean of
1 / rank of the first relevant passage, within the top 6). The scoring and the experiments are in
rag/retrieval_eval.py; this file holds the labels.
"""
import argparse
import sys

from rag import retrieval_eval as ev
from rag.config import FUSION
from rag.retriever import Retriever

M3, M2, DL, M4, M5A, M5B = "keerthika_m3", "keerthika_m2", "keerthika_m3_deadlock", "keerthika_m4", \
    "keerthika_m5a", "notes_m5b"

# (question, textbook section numbers, [(deck, first slide, last slide)])
SCHEDULING = [
    ("What is the convoy effect?", ["5.3.1"], [(M3, 21, 24)]),
    ("Why do short jobs get stuck waiting behind one long job?", ["5.3.1"], [(M3, 21, 24)]),
    ("Difference between preemptive and non-preemptive scheduling", ["5.1.3"], [(M3, 14, 15)]),
    ("What does the dispatcher do?", ["5.1.4"], [(M3, 5, 5)]),
    ("Which measures are used to compare scheduling algorithms?", ["5.2"], [(M3, 16, 19)]),
    ("How is the length of the next CPU burst predicted?", ["5.3.2"], [(M3, 41, 43)]),
    ("How does shortest remaining time first work?", ["5.3.2"], [(M3, 22, 22), (M3, 41, 49)]),
    ("How should the time quantum be chosen in round robin?", ["5.3.3"], [(M3, 72, 74)]),
    ("What happens to a process when its time slice ends?", ["5.3.3"], [(M3, 72, 74)]),
    ("What is starvation and how does aging solve it?", ["5.3.4"], [(M3, 57, 58)]),
    ("How do processes move between queues in a multilevel feedback queue?", ["5.3.6"], [(M3, 155, 157)]),
    ("What is multilevel queue scheduling?", ["5.3.5"], [(M3, 148, 154)]),
    ("What information is stored in a process control block?", ["3.1.3"], [(M2, 39, 39)]),
    ("What are the states a process can be in?", ["3.1.2"], [(M2, 38, 38)]),
    ("What happens during a context switch?", ["3.2.3"], [(M2, 43, 44)]),
    ("What is the CPU-I/O burst cycle?", ["5.1.1"], [(M3, 3, 4)]),
    ("What is processor affinity?", ["5.5.4"], []),
    ("What is load balancing in multiprocessor scheduling?", ["5.5.3"], [(M3, 182, 201)]),
    ("What is a memory stall on a multicore processor?", ["5.5.2"], []),
    ("Explain rate-monotonic scheduling", ["5.6.3"], []),
    ("How does earliest-deadline-first scheduling pick a process?", ["5.6.4"], []),
    ("How does the Linux completely fair scheduler work?", ["5.7.1", "20.5.1"], []),
    ("How does Windows schedule threads?", ["5.7.2"], []),
    ("What is deterministic modeling for evaluating a scheduling algorithm?", ["5.8.1"], []),
    ("Difference between process contention scope and system contention scope", ["5.4.1"], []),
    ("Round robin worked example with time quantum 2 and five processes", [], [(M3, 78, 85)]),
    ("In the MLFQ example a 40 second process: in which queue does it finish?", [], [(M3, 158, 158)]),
    ("SRTF worked example where a process has an I/O burst", [], [(M3, 101, 114)]),
    ("Priority scheduling example where a higher number means higher priority", [], [(M3, 62, 64)]),
    ("FCFS example where the CPU is idle before the next process arrives", [], [(M3, 25, 28)]),
]
NEW_TOPICS = [
    ("What happens when a process calls fork()?", ["3.3.1", "20.4.1"], [(M2, 46, 57)]),
    ("What are the types of system calls?", ["2.3.3"], [(M2, 10, 16)]),
    ("Difference between user threads and kernel threads", ["4.3"], [(M2, 69, 74)]),
    ("What is the many-to-many threading model?", ["4.3.3"], [(M2, 70, 74)]),
    ("What is the critical-section problem and what must a solution satisfy?", ["6.2"], [(M4, 26, 27)]),
    ("How does Peterson's solution work?", ["6.3"], []),
    ("How are semaphores used to solve the producer-consumer problem?", ["6.6", "7.1.1"], [(M4, 16, 22)]),
    ("What is a race condition?", ["6.1"], [(M4, 23, 25)]),
    ("How can the dining philosophers problem lead to deadlock?", ["7.1.3"], [(M4, 71, 74)]),
    ("What are the four necessary conditions for deadlock?", ["8.3.1"], [(DL, 5, 5)]),
    ("Explain the safety check in the Banker's algorithm", ["8.6.3"], [(DL, 15, 57)]),
    ("How is deadlock detected with a wait-for graph?", ["8.7.1"], [(DL, 8, 8), (DL, 58, 64)]),
    ("Difference between internal and external fragmentation", ["9.2.3"], [(M5B, 19, 20), (M5A, 5, 10)]),
    ("Compare first fit, best fit and worst fit allocation", ["9.2.2"], [(M5B, 18, 18), (M5A, 5, 10)]),
    ("How is a logical address translated to a physical address with paging?", ["9.3.1"],
     [(M5B, 21, 26), (M5A, 2, 4)]),
    ("What is a TLB and how does it speed up paging?", ["9.3.2"], [(M5B, 28, 32)]),
    ("What is segmentation?", [], [(M5B, 91, 96)]),
    ("What happens on a page fault?", ["10.2.1"], [(M5B, 58, 62)]),
    ("Explain Belady's anomaly with FIFO page replacement", ["10.4.2"], [(M5B, 76, 77)]),
    ("How does LRU page replacement work?", ["10.4.4"], [(M5B, 79, 81)]),
    ("What is thrashing?", ["10.6"], [(M5B, 85, 88)]),
    ("Explain the SCAN disk scheduling algorithm", ["11.2.2"], []),
    ("How does C-SCAN differ from SCAN?", ["11.2.3"], []),
    ("Compare contiguous, linked and indexed file allocation", ["14.4"], []),
    ("What is a tree-structured directory?", ["13.3.3"], []),
    ("How is free disk space tracked with a bit vector?", ["14.5.1"], []),
    ("What is journaling in the Linux ext3 file system?", ["20.7.3"], []),
    ("What attributes does a file have?", ["13.1.1"], []),
]
# lab material: Linux commands handout (pages), shell lab slides (slides), shell exercise sheet
# (a Word file without pages: labelled by section title, written (deck, "title start", None))
LC, SS, SX = "lab_linux_commands", "lab_shell_slides", "lab_shell_exercises"
LINUX_SHELL = [
    ("How do I change file permissions with chmod?", [], [(LC, 6, 7)]),
    ("What does the grep command do?", [], [(LC, 7, 8)]),
    ("How do you write a while loop in a shell script?", [], [(SS, 27, 27), (SX, "Loops", None)]),
    ("What are special variables like $# and $0 in shell scripts?", [], [(SS, 33, 36), (SX, "Using Arguments", None),
                                                                        (SX, "Changing your prompt", None)]),
    ("How to read input from the user in a bash script?", [], [(SS, 7, 8)]),
    ("What is the difference between single and double quotes in bash?", [], [(SX, "Changing your prompt", None)]),
    ("How do pipes and redirection work?", [], [(LC, 11, 12)]),
    ("What is a shell?", ["2.2.1"], [(SS, 2, 4), (LC, 8, 8)]),
    ("Write a shell script to find the factorial of a number", [], [(SS, 29, 29)]),
    ("How do I write a function in bash?", [], [(SX, "Shell Functions", None), (SX, "Version 2", None),
                                               (SX, "Version 3", None), (SX, "Version 4", None)]),
    ("How to list files in a directory?", [], [(LC, 3, 4)]),
    ("How do I run a process in the background?", [], [(LC, 12, 12)]),
    ("case statement syntax in shell", [], [(SS, 25, 25)]),
    ("How do I write an if-else statement in a shell script?", [], [(SS, 17, 24), (SX, "An Overview of Bash", None)]),
    ("How do I delete a directory?", [], [(LC, 4, 6)]),
    ("What does the man command do?", [], [(LC, 3, 3)]),
    ("How are arrays used in shell scripts?", [], [(SS, 37, 37), (SX, "Version 5", None)]),
    ("How do wildcards like * and ? match file names?", [], [(LC, 10, 11)]),
    ("How do I do arithmetic in a shell script?", [], [(SS, 8, 8), (SS, 12, 13), (SX, "Arithmetic", None)]),
    ("What are environment variables such as PATH?", [], [(LC, 9, 10)]),
]
GROUPS = {"CPU scheduling and processes": SCHEDULING, "Newer topics": NEW_TOPICS,
          "Linux commands and shell programming": LINUX_SHELL}
relevant = ev.relevant   # kept importable from here


def report(r, items, fusion, label):
    print(f"\n== {label} ==")
    rows = [(f"{g} ({len([c for c in items if c.group == g])})", ev.evaluate(r, [c for c in items if c.group == g],
                                                                           fusion)) for g in GROUPS]
    rows.append(("All questions", ev.evaluate(r, items, fusion)))
    print(ev.table(rows))
    topics = sorted({c.topic for c in items})
    print("\nBy topic")
    print(ev.table([(t, ev.evaluate(r, [c for c in items if c.topic == t], fusion)) for t in topics]))
    kinds = ev.source_kind_breakdown(r, items, fusion)
    print(f"\nTextbook vs teacher material ({kinds['questions_with_both']} questions labelled with both): "
          f"textbook passage found in top {ev.KS[-1]} for {kinds['textbook_found']['hit@6']:.0%}, teacher "
          f"material for {kinds['teacher_found']['hit@6']:.0%}; top passage is teacher material for "
          f"{kinds['top1_is_teacher_material']:.0%} of all questions.")
    misses = [c.question for c in items if ev.first_relevant(r, c, fusion) is None]
    for q in misses:
        print(f"  not found in the top {ev.KS[-1]}: {q}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true", help="compare variants and grid-search the fusion settings")
    ap.add_argument("--save", help="write each question's ranked passages and scores to this JSON file")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    r = Retriever()
    if r.mode != "hybrid":
        raise SystemExit("Build the index first: python -m rag.build_index")
    items = ev.cases(GROUPS)
    ev.rank_all(r, items)
    print(f"{len(r.chunks)} passages in the index, {len(items)} questions, top {ev.KS[-1]} retrieved")
    print("Current configuration:", {k: v for k, v in FUSION.items()})
    report(r, items, {}, "Current configuration")
    if a.save:
        ev.save(a.save, ev.per_question(r, items, {}))
        print(f"\nPer-question results saved to {a.save}")
    if not a.tune:
        return

    print("\n== Controlled variants (all 78 questions) ==")
    print(ev.table([(name, ev.evaluate(r, items, v)) for name, v in ev.VARIANTS.items()]))
    sched = [c for c in items if c.topic == "CPU scheduling"]
    print(f"\nSame variants, CPU-scheduling questions only ({len(sched)})")
    print(ev.table([(name, ev.evaluate(r, sched, v)) for name, v in ev.VARIANTS.items()]))

    best, results = ev.tune(r, items)
    tuning = [c for c in items if not c.held_out]
    held = [c for c in items if c.held_out]
    print(f"\n== Grid search on {len(tuning)} tuning questions ({len(results)} settings), best 8 ==")
    for s, m, _ in results[:8]:
        print("  " + ", ".join(f"{k}={s[k]}" for k in ev.GRID))
        print(f"      Hit@1 {m['hit@1']:.1%}  Hit@3 {m['hit@3']:.1%}  Hit@6 {m['hit@6']:.1%}  MRR {m['mrr']:.3f}")
    print("\nSelected:", {k: best[k] for k in ev.GRID})
    print(f"\n== Held-out check ({len(held)} questions not used for tuning) ==")
    print(ev.table([("C. Baseline", ev.evaluate(r, held, ev.BASELINE)), ("F. Selected", ev.evaluate(r, held, best))]))
    print(f"\n== Selected vs baseline on all {len(items)} questions ==")
    print(ev.table([("C. Baseline", ev.evaluate(r, items, ev.BASELINE)), ("F. Selected", ev.evaluate(r, items, best))]))
    topics = sorted({c.topic for c in items})
    print("\nBy topic, baseline -> selected (Hit@1, MRR)")
    for t in topics:
        sub = [c for c in items if c.topic == t]
        m0, m1 = ev.evaluate(r, sub, ev.BASELINE), ev.evaluate(r, sub, best)
        print(f"  {t:<36} n={len(sub):<3} Hit@1 {m0['hit@1']:.0%} -> {m1['hit@1']:.0%}   "
              f"MRR {m0['mrr']:.3f} -> {m1['mrr']:.3f}")


if __name__ == "__main__":
    main()
