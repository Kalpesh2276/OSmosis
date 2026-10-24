"""Concepts and misconceptions with stable ids, shared by the Reasoning Debugger, the evidence log
and the Misconception Graph.

A misconception is only recorded when a validator in rag/diagnosis.py sees evidence for it (a
specific, checkable pattern in the student's answer). Ids never change once used, because saved
progress refers to them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

_M3 = "Teacher slides (Module 3), "
_DL = "Teacher slides (Module 3, deadlocks), "
_M5 = "Teacher slides (Module 5, part 2), "


@dataclass(frozen=True)
class Concept:
    id: str
    name: str
    topic: str                      # quiz topic id (rag.quiz.TOPICS) used for practice
    prerequisites: tuple = ()       # concept ids
    refs: tuple = ()                # where the course explains it


@dataclass(frozen=True)
class Misconception:
    id: str
    name: str
    concept: str                    # Concept.id
    description: str
    evidence: str                   # what in an answer shows it
    remediation: str                # what to revise / do
    practice: str                   # kind of practice question that tests it
    refs: tuple = field(default=())


CONCEPTS = {c.id: c for c in [
    Concept("sched.gantt", "Reading and drawing Gantt charts", "fcfs", (), (_M3 + "slides 25-28",)),
    Concept("sched.metrics", "CT, TAT, WT and RT", "criteria", ("sched.gantt",), (_M3 + "slides 16-19",)),
    Concept("sched.fcfs", "FCFS scheduling", "fcfs", ("sched.gantt",), (_M3 + "slides 21-31",)),
    Concept("sched.sjf", "SJF selection rule", "sjf", ("sched.fcfs",), (_M3 + "slides 37-40", _M3 + "slides 47-49")),
    Concept("sched.preemption", "Preemptive vs non-preemptive decisions", "preemption", ("sched.fcfs",),
            (_M3 + "slides 14-15",)),
    Concept("sched.srtf", "SRTF (preemptive SJF)", "sjf", ("sched.sjf", "sched.preemption"), (_M3 + "slides 41-47",)),
    Concept("sched.priority", "Priority scheduling", "priority", ("sched.fcfs", "sched.preemption"),
            (_M3 + "slides 57-64",)),
    Concept("sched.rr", "Round Robin and the ready queue", "rr", ("sched.fcfs", "sched.preemption"),
            (_M3 + "slides 72-85",)),
    Concept("sched.io", "CPU and I/O bursts", "sjf", ("sched.gantt",), (_M3 + "slides 101-114",)),
    Concept("banker.need", "Need = Max - Allocation", "banker", (), (_DL + "slides 18, 24",)),
    Concept("banker.safety", "Safety algorithm (Work and Finish)", "banker", ("banker.need",), (_DL + "slides 19, 25-31",)),
    Concept("banker.request", "Resource-request algorithm", "banker", ("banker.safety",), (_DL + "slides 20, 39-57",)),
    Concept("paging.faults", "Page hits and page faults", "vm", (), (_M5 + "slide 74",)),
    Concept("paging.fifo", "FIFO replacement", "vm", ("paging.faults",), (_M5 + "slide 76",)),
    Concept("paging.lru", "LRU replacement", "vm", ("paging.faults",), (_M5 + "slides 79-81",)),
    Concept("paging.optimal", "Optimal replacement", "vm", ("paging.faults",), (_M5 + "slide 78",)),
    Concept("disk.movement", "Head movement arithmetic", "disk", (), ("Silberschatz 10e, 11.2.1, p. 458",)),
    Concept("disk.sstf", "SSTF", "disk", ("disk.movement",), ("Silberschatz 10e, 11.2",)),
    Concept("disk.scan", "SCAN and LOOK", "disk", ("disk.movement",), ("Silberschatz 10e, 11.2.2, pp. 458-459",)),
    Concept("disk.cscan", "C-SCAN and C-LOOK", "disk", ("disk.scan",), ("Silberschatz 10e, 11.2.3, p. 460",)),
]}

MISCONCEPTIONS = {m.id: m for m in [
    # CPU scheduling
    Misconception("sched.selection_rule", "Misapplied selection rule", "sched.sjf",
                  "Picks a ready process the algorithm would not pick (e.g. not the shortest burst in SJF, not "
                  "the highest priority).",
                  "At a decision point, the chosen process was ready but another ready process wins under the rule.",
                  "List every ready process at each decision point and compare them with the algorithm's key.",
                  "numerical with several processes ready at once"),
    Misconception("sched.preemption", "Incorrect preemption reasoning", "sched.preemption",
                  "Misses a preemption in a preemptive algorithm, or preempts in a non-preemptive one.",
                  "The student keeps a process running when a better one arrives (preemptive), or switches away "
                  "from an unfinished process (non-preemptive).",
                  "On every arrival, ask: is the algorithm preemptive, and does the newcomer beat the running process?",
                  "SRTF or preemptive-priority numerical with mid-burst arrivals"),
    Misconception("sched.rr_queue", "Incorrect Round Robin queue handling", "sched.rr",
                  "Runs processes out of ready-queue order, or not for exactly one quantum.",
                  "The student runs a process other than the head of the queue, or for more or less than the quantum.",
                  "Write the ready queue down after every event; the preempted process goes to the tail.",
                  "Round Robin numerical with a small quantum"),
    Misconception("sched.arrival_boundary", "Arrivals at scheduling boundaries", "sched.rr",
                  "Mishandles a process that arrives exactly when another event happens (the course puts a new "
                  "arrival before the process whose quantum just expired), or runs a process before it arrives.",
                  "The student schedules a process before its arrival time, leaves the CPU idle while one is ready, "
                  "or queues the preempted process ahead of a simultaneous arrival.",
                  "Mark arrivals on the time line first; at a tie, arrivals join the queue before the preempted process.",
                  "Round Robin numerical where an arrival coincides with a quantum expiry"),
    Misconception("sched.io_burst", "Handling I/O bursts", "sched.io",
                  "Runs a process while it is doing I/O, or leaves I/O time out of the waiting time.",
                  "The student schedules a process during its I/O, or computes WT = TAT - BT - I/O.",
                  "During I/O the process is not in the ready queue; in this course WT = TAT - total CPU burst.",
                  "SRTF numerical with an I/O burst"),
    Misconception("sched.metric_confusion", "Confusing CT, TAT, WT and RT", "sched.metrics",
                  "Uses one metric's formula for another (e.g. WT = RT, TAT = CT).",
                  "A value matches another metric's formula instead of its own.",
                  "TAT = CT - AT, WT = TAT - BT, RT = first time on CPU - AT.",
                  "table-completion numerical"),
    Misconception("sched.ct_reading", "Reading CT from the Gantt chart", "sched.gantt",
                  "Takes a process's completion time from the end of its first slice instead of its last.",
                  "A preempted process's CT equals the end of an earlier slice in the chart.",
                  "CT is where the process's last bar ends.",
                  "preemptive numerical where processes run in several slices"),
    Misconception("sched.wrong_algorithm", "Applying a different algorithm", "sched.preemption",
                  "The whole schedule follows another algorithm (e.g. SJF where SRTF was asked).",
                  "The student's Gantt chart is exactly another algorithm's schedule for the same processes.",
                  "Re-read which algorithm is asked, and whether it is preemptive.",
                  "same processes under two algorithms"),
    # Banker's algorithm
    Misconception("banker.need_formula", "Need computed wrongly", "banker.need",
                  "Need is not Max - Allocation (e.g. Allocation - Max, or Max itself).",
                  "A Need entry equals Allocation - Max, Max, or Max + Allocation.",
                  "Need[i][j] = Max[i][j] - Allocation[i][j], never negative.",
                  "Need-matrix numerical"),
    Misconception("banker.need_vs_work", "Comparing the wrong vectors", "banker.safety",
                  "Lets a process finish although its Need exceeds Work (often comparing Allocation or Max).",
                  "In the student's sequence a process's Need > Work at its turn.",
                  "A process can finish only if every entry of its Need fits in Work.",
                  "which-is-a-safe-sequence question"),
    Misconception("banker.work_update", "Not updating Work", "banker.safety",
                  "Forgets that a finished process releases its Allocation (Work = Work + Allocation).",
                  "The student's sequence would be possible only if Work grew by something other than the "
                  "Allocation of finished processes, or the student calls a safe state unsafe.",
                  "After each process finishes, add its Allocation to Work before the next check.",
                  "safety-algorithm numerical"),
    Misconception("banker.request_steps", "Skipping a resource-request check", "banker.request",
                  "Grants a request without all three checks (Request <= Need, Request <= Available, safe after).",
                  "The student's verdict on a request differs from the three-step algorithm's.",
                  "Check Need, then Available, then pretend to allocate and run the safety algorithm.",
                  "can-this-request-be-granted question"),
    # Page replacement
    Misconception("paging.wrong_algorithm", "Applying a different replacement algorithm", "paging.faults",
                  "Uses another algorithm's victim rule (e.g. FIFO when LRU is asked).",
                  "The student's hits and faults are exactly another algorithm's.",
                  "FIFO: loaded first; LRU: used least recently; Optimal: used farthest in the future.",
                  "same reference string under two algorithms"),
    Misconception("paging.hit_fault", "Hit or fault misjudged", "paging.faults",
                  "Marks a hit when the page is not in a frame, or a fault when it is.",
                  "At a reference the student's mark disagrees with the frame contents at that moment.",
                  "Before marking, check whether the page is in any frame right now.",
                  "frame-table numerical"),
    # Disk scheduling
    Misconception("disk.wrong_algorithm", "Applying a different disk algorithm", "disk.movement",
                  "Serves requests in another algorithm's order (e.g. LOOK for SCAN, SSTF for FCFS).",
                  "The student's service order is exactly another algorithm's.",
                  "SCAN goes to the disk's end, LOOK turns at the last request; C-variants jump back.",
                  "same queue under two algorithms"),
    Misconception("disk.return_seek", "C-SCAN / C-LOOK return seek", "disk.cscan",
                  "Counts the return jump when it should not, or leaves it out when it should be counted.",
                  "The student's total equals the total with the return seek removed (or added).",
                  "In this course the return seek is counted unless the question says otherwise.",
                  "C-SCAN numerical"),
    Misconception("disk.arithmetic", "Head-movement arithmetic", "disk.movement",
                  "Right order, wrong total.",
                  "The service order is right but the sum of |next - current| is not.",
                  "Write each move as |next - current| and add them up.",
                  "FCFS disk numerical"),
]}


def concept_of(misconception_id: str) -> Concept:
    return CONCEPTS[MISCONCEPTIONS[misconception_id].concept]
