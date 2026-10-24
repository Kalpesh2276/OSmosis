"""Quiz mode: questions on the OS course, marked, with difficulty that adapts to the student.

Three kinds of question:
  numeric  a random CPU-scheduling, Banker's-algorithm, page-replacement or disk-scheduling
           problem; the exact solvers compute the answer key, so marking is exact. Some have a
           computed set of options ("which of these is a safe sequence?")
  mcq      written by the LLM from retrieved passages (falls back to a small built-in bank)
  viva     an open question written by the LLM, which also marks the typed answer

Progress keeps, per topic, how the student is doing and a level 1-3: two correct answers in a
row raise the level, a wrong one lowers it, and weaker topics are asked more often.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from itertools import permutations

from . import banker, disk, paging
from .ask import build_context, llm_chat
from .solver import Problem, Proc, Solution, solve, to_text

# topic id -> name, retrieval query, and (for numericals) what is asked at levels 1, 2, 3: the
# algorithm (several separated by spaces: one is picked), or for "banker" the kind of question
TOPICS = {
    "fcfs": ("FCFS", "first come first served FCFS scheduling convoy effect", ("fcfs", "fcfs", "fcfs")),
    "sjf": ("SJF and SRTF", "shortest job first SJF SRTF shortest remaining time", ("sjf", "sjf", "srtf")),
    "priority": ("Priority scheduling", "priority scheduling starvation aging",
                 ("priority", "priority", "priority_p")),
    "rr": ("Round Robin", "round robin time quantum context switch", ("rr", "rr", "rr")),
    "mlfq": ("Multilevel queues", "multilevel queue multilevel feedback queue MLFQ", None),
    "criteria": ("Scheduling criteria", "scheduling criteria turnaround waiting response time throughput", None),
    "preemption": ("Preemptive vs non-preemptive", "preemptive non-preemptive scheduling dispatcher", None),
    "process": ("Process concepts", "process control block PCB process state context switch", None),
    "syscalls": ("System calls", "system calls types fork exec wait system call interface", None),
    "threads": ("Threads", "threads multithreading models user threads kernel threads", None),
    "sync": ("Synchronization", "critical section problem race condition mutex lock semaphore Peterson", None),
    "prodcons": ("Producer-consumer problem", "producer consumer bounded buffer problem", None),
    "dining": ("Dining philosophers", "dining philosophers problem chopsticks", None),
    "deadlock": ("Deadlocks", "deadlock necessary conditions prevention detection resource allocation graph", None),
    "banker": ("Banker's algorithm", "banker's algorithm safety algorithm safe state need matrix",
               ("need", "safe", "request")),
    "memory": ("Memory allocation", "contiguous memory allocation fragmentation first fit best fit", None),
    "paging": ("Paging and segmentation", "paging page table TLB segmentation address translation", None),
    "vm": ("Virtual memory", "demand paging page fault page replacement FIFO LRU optimal", ("fifo", "lru", "optimal")),
    "fs": ("File systems", "file system directory structure allocation methods free space", None),
    "disk": ("Disk scheduling", "disk scheduling FCFS SCAN C-SCAN SSTF head movement",
             ("fcfs sstf", "scan look", "cscan clook")),
    "linux": ("Linux commands", "Linux commands ls cd cp mv rm chmod grep wc pipes redirection job control", None),
    "shell": ("Shell programming", "bash shell script variables read echo if case while for loop functions "
                                   "special variables arrays", None),
}
LEVELS = {1: "easy: recall one definition or fact",
          2: "medium: compare two ideas or apply a rule to a short situation",
          3: "hard: reason about a scenario, a trade-off or an edge case"}


@dataclass
class Question:
    kind: str                       # "numeric" | "mcq" | "viva"
    topic: str
    level: int
    prompt: str
    options: list[str] = field(default_factory=list)   # mcq
    answer: float | int | None = None                   # numeric: value, mcq: index of the right option
    explanation: str = ""
    cite: str = ""
    solution: Solution | None = None                    # numeric: the full worked solution
    key_points: list[str] = field(default_factory=list)  # viva
    passage: str = ""                                    # viva: the source text it was written from
    origin: str = ""                                     # "solver" | "llm" | "bank"


# ---------------------------------------------------------------------------------------
# Numerical questions (answer key from the solver)
# ---------------------------------------------------------------------------------------
def numeric_question(topic: str, level: int, rng: random.Random) -> Question:
    maker = {"banker": banker_question, "vm": paging_question, "disk": disk_question}.get(topic)
    if maker:
        return maker(level, rng)
    algo = TOPICS[topic][2][level - 1]
    n = {1: 3, 2: 4, 3: 5}[level]
    arrivals = [0] * n if level == 1 else sorted(rng.sample(range(0, 2 * n), n))
    if level > 1:
        arrivals[0] = 0 if rng.random() < 0.7 else arrivals[0]
    priorities = rng.sample(range(1, n + 2), n)
    procs = [Proc(i + 1, arrivals[i], [rng.randint(2, 4 + 2 * level)], [],
                  priorities[i] if algo.startswith("priority") else None) for i in range(n)]
    problem = Problem(algo, procs, rng.choice([2, 3] if level < 3 else [2, 3, 4]) if algo == "rr" else None)
    solution = solve(problem)
    what, key = rng.choice([("average waiting time", "wt"), ("average turnaround time", "tat")])
    if level == 3 and rng.random() < 0.4:
        row = rng.choice(solution.rows)
        ask, value = f"What is the completion time of P{row['pid']}?", float(row["ct"])
    else:
        ask, value = f"What is the {what}?", round(sum(r[key] for r in solution.rows) / n, 2)
    return Question("numeric", topic, level, f"{to_text(problem)}\n\n**{ask}** (to 2 decimal places)",
                    answer=value, solution=solution, origin="solver",
                    explanation="Worked out by the scheduling simulator with the course conventions.")


def paging_question(level: int, rng: random.Random) -> Question:
    algo = TOPICS["vm"][2][level - 1]
    refs = [rng.randrange({1: 5, 2: 6, 3: 7}[level]) for _ in range({1: 12, 2: 14, 3: 16}[level])]
    problem = paging.Problem(refs, [3 if level < 3 else rng.choice([3, 4])], [algo])
    solution = paging.solve(problem)
    run = solution.runs[0]
    hits = level > 1 and rng.random() < 0.3
    ask, value = ("How many page hits are there?", len(refs) - run.n_faults) if hits else \
        ("How many page faults occur?", run.n_faults)
    return Question("numeric", "vm", level, f"{paging.to_text(problem)}\n\n**{ask}** (frames start empty)",
                    answer=float(value), solution=solution, origin="solver",
                    explanation="Worked out by the page-replacement simulator.")


def disk_question(level: int, rng: random.Random) -> Question:
    algo = rng.choice(TOPICS["disk"][2][level - 1].split())
    head = rng.randrange(20, 180)
    requests = rng.sample([c for c in range(200) if c != head], {1: 6, 2: 7, 3: 8}[level])
    problem = disk.Problem(requests, head, [algo], rng.random() < 0.5, 200, True)
    solution = disk.solve(problem)
    note = " Count the return seek." if algo in ("cscan", "clook") else ""
    return Question("numeric", "disk", level, f"{disk.to_text(problem)}\n\n**What is the total head movement, in "
                                              f"cylinders?**{note}",
                    answer=float(solution.total(solution.runs[0])), solution=solution, origin="solver",
                    explanation="Worked out by the disk-scheduling simulator.")


REQUEST_OPTIONS = {  # verdict of banker.solve -> option
    "granted": "Yes: the state after granting it is safe, so it is granted immediately",
    "unsafe": "No: the state after granting it would be unsafe, so the process must wait",
    "wait": "No: the resources are not available, so the process must wait",
    "error": "No: it exceeds the process's maximum claim, so it is an error",
}


def _banker_state(rng: random.Random, n: int, m: int) -> tuple[banker.Problem, banker.Solution]:
    """A random safe state."""
    while True:
        maxm = [[rng.randint(0, 7) for _ in range(m)] for _ in range(n)]
        alloc = [[rng.randint(0, x) for x in row] for row in maxm]
        problem = banker.Problem(list(range(n)), alloc, maxm, [rng.randint(0, 4) for _ in range(m)])
        solution = banker.solve(problem)
        if solution.check.safe:
            return problem, solution


def _unsafe_orders(s: banker.Solution, rng: random.Random, k: int) -> list[list[int]]:
    """k orders of the processes that are NOT safe sequences."""
    p, out = s.problem, []
    orders = list(permutations(range(len(p.pids))))
    rng.shuffle(orders)
    for order in orders:
        work = list(p.available)
        for i in order:
            if any(x > w for x, w in zip(s.need[i], work)):
                out.append(list(order))
                break
            work = [w + a for w, a in zip(work, p.alloc[i])]
        if len(out) == k:
            break
    return out


def banker_question(level: int, rng: random.Random) -> Question:
    what = TOPICS["banker"][2][level - 1]
    if what == "need":
        problem, solution = _banker_state(rng, 4, 3)
        if rng.random() < 0.7:
            i, j = rng.randrange(4), rng.randrange(3)
            ask, value = f"What is the Need of P{problem.pids[i]} for resource {problem.resources[j]}?", \
                solution.need[i][j]
        else:
            j = rng.randrange(3)
            ask, value = f"How many instances of resource {problem.resources[j]} are there in total?", \
                solution.total[j]
        return Question("numeric", "banker", level, f"{banker.to_text(problem)}\n\n**{ask}**", answer=float(value),
                        solution=solution, origin="solver", explanation=banker.verdict(solution))
    if what == "safe":
        while True:
            problem, solution = _banker_state(rng, 5, 3)
            wrong = _unsafe_orders(solution, rng, 3)
            if len(wrong) == 3:
                break
        name = lambda order: "<" + ", ".join(f"P{problem.pids[i]}" for i in order) + ">"
        q = Question("numeric", "banker", level, f"{banker.to_text(problem)}\n\n**Which of these is a safe sequence?**",
                     [name(solution.check.order)] + [name(o) for o in wrong], 0, banker.verdict(solution),
                     solution=solution, origin="solver")
        return _shuffled(q, rng)
    target = rng.choice(list(REQUEST_OPTIONS))
    while True:
        problem, solution = _banker_state(rng, 5, 3)
        for _ in range(60):
            i = rng.randrange(5)
            need = solution.need[i]
            req = [rng.randint(0, x) for x in need]
            if target == "error":
                j = rng.randrange(3)
                req[j] = need[j] + rng.randint(1, 2)
            elif target == "wait":
                j = rng.randrange(3)
                if need[j] <= problem.available[j]:
                    continue
                req[j] = rng.randint(problem.available[j] + 1, need[j])
            if not any(req):
                continue
            asked = banker.Problem(problem.pids, problem.alloc, problem.maxm, problem.available, (i, req))
            result = banker.solve(asked)
            if result.request["verdict"] == target:
                options = list(REQUEST_OPTIONS.values())
                return Question("numeric", "banker", level,
                                f"{banker.to_text(asked)}\n\n**Can P{i}'s request be granted immediately?**",
                                options, list(REQUEST_OPTIONS).index(target), banker.verdict(result),
                                solution=result, origin="solver")


def check_numeric(q: Question, given: float) -> bool:
    return abs(given - q.answer) <= 0.011


# ---------------------------------------------------------------------------------------
# Built-in MCQ bank (used when the LLM is off or its reply cannot be read)
# ---------------------------------------------------------------------------------------
_SLIDES = "Teacher slides (Module 3), "
_BOOK = "Silberschatz 10e, "
_LINUX = "Linux commands notes, "
_SHELL = "Shell programming lab slides, "
BANK = [
    ("fcfs", 1, "Which non-preemptive algorithm suffers most from the convoy effect?",
     ["FCFS", "SJF", "Priority scheduling", "None of them"], 0,
     "The slide's comparison rates the convoy effect High for FCFS, Very Low for SJF and Moderate for priority.",
     _SLIDES + "slide 21"),
    ("fcfs", 2, "In the convoy effect, what happens while the CPU-bound process holds the CPU?",
     ["The I/O-bound processes wait in the ready queue and the I/O devices sit idle",
      "The I/O-bound processes preempt it one after another",
      "The CPU-bound process is moved to a lower-priority queue",
      "The scheduler shortens the CPU-bound process's burst"], 0,
     "The short processes finish their I/O and queue up behind the big one, so the devices are idle.",
     _BOOK + "5.3.1, pp. 206-207"),
    ("sjf", 1, "Among the ready processes, SJF selects the one with the",
     ["smallest CPU burst time", "earliest arrival time", "highest priority number", "longest waiting time"], 0,
     "SJF picks the process with the smallest CPU burst; once started it runs to completion.",
     _SLIDES + "slide 21"),
    ("sjf", 2, "What is the real difficulty with implementing SJF?",
     ["Knowing the length of the next CPU burst", "It needs a time quantum", "It causes the convoy effect",
      "It cannot be made preemptive"], 0,
     "There is no way to know the next CPU burst, so it can only be predicted from earlier bursts.",
     _SLIDES + "slide 43"),
    ("sjf", 3, "In the exponential average τ(n+1) = α·t(n) + (1 − α)·τ(n), what does α = 0 mean?",
     ["Recent history has no effect on the prediction", "Only the most recent burst matters",
      "Recent and past history are weighted equally", "The prediction is always zero"], 0,
     "With α = 0 the prediction stays τ(n): the latest burst is ignored.",
     _BOOK + "5.3.2, p. 208"),
    ("priority", 1, "Low-priority processes may never execute under priority scheduling. The solution is",
     ["aging", "a smaller time quantum", "swapping", "the convoy effect"], 0,
     "Aging raises the priority of a process as it waits, so it eventually runs.",
     _SLIDES + "slide 58"),
    ("priority", 2, "A higher-priority process arrives while another is running under NON-preemptive priority "
                    "scheduling. What happens?",
     ["It is put at the head of the ready queue and runs when the current process finishes",
      "It preempts the running process immediately", "It is discarded",
      "The running process's priority is raised"], 0,
     "A non-preemptive priority scheduler only places the new process at the head of the ready queue.",
     _SLIDES + "slide 57"),
    ("rr", 1, "A Round Robin time quantum is generally",
     ["10 to 100 milliseconds", "1 to 5 microseconds", "1 to 2 seconds", "equal to the longest burst"], 0,
     "The slides and the textbook both give 10 to 100 milliseconds.",
     _SLIDES + "slide 72"),
    ("rr", 2, "If the time quantum is larger than every CPU burst, Round Robin behaves like",
     ["FCFS", "SJF", "SRTF", "Priority scheduling"], 0,
     "No process is ever preempted, so they simply run in arrival order.",
     _SLIDES + "slide 73"),
    ("rr", 3, "In Round Robin, P3 arrives at the same instant P1's quantum expires. In the course's worked "
              "examples, where does P3 go?",
     ["Into the ready queue before the preempted P1", "Behind P1 at the tail", "Straight onto the CPU",
      "It waits for the next quantum to join"], 0,
     "The new arrival is queued before the preempted process (Round Robin Example 2).",
     _SLIDES + "slide 79"),
    ("mlfq", 1, "In a multilevel feedback queue, a process that keeps using its whole time slice",
     ["may be demoted to a lower-priority queue", "is promoted to a higher-priority queue",
      "is terminated", "stays permanently in its first queue"], 0,
     "CPU-heavy processes move down; I/O-bound or short processes move up.",
     _SLIDES + "slide 156"),
    ("mlfq", 2, "A 40-second CPU-bound process enters an MLFQ whose quanta are 2, 7, 12, 17 and 22 seconds. "
                "How many times is it interrupted, and where does it finish?",
     ["4 times, in queue 5", "5 times, in queue 5", "3 times, in queue 4", "4 times, in queue 4"], 0,
     "2 + 7 + 12 + 17 = 38 s over four queues (four interruptions); the last 2 s run in queue 5.",
     _SLIDES + "slide 158"),
    ("criteria", 1, "Turnaround time is",
     ["the total time from a process's submission to its completion",
      "the time spent waiting in the ready queue", "the time until the first response",
      "the number of processes completed per time unit"], 0,
     "TAT = completion time − arrival time.",
     _SLIDES + "slide 16"),
    ("criteria", 2, "In non-preemptive scheduling, the response time of a process equals its",
     ["waiting time", "turnaround time", "burst time", "completion time"], 0,
     "The process starts executing only once, so its first response comes when its waiting ends.",
     _SLIDES + "slide 19"),
    ("preemption", 1, "Under non-preemptive scheduling, a running process gives up the CPU only when it",
     ["terminates or moves to the waiting state", "has used one time quantum",
      "is outranked by a new arrival", "has run for 100 ms"], 0,
     "Once running, it is not disturbed until it finishes or blocks.",
     _SLIDES + "slide 15"),
    ("preemption", 2, "Which kind of scheduling do Windows, macOS, Linux and UNIX use?",
     ["Preemptive", "Non-preemptive", "Cooperative", "FCFS only"], 0,
     "Virtually all modern operating systems use preemptive scheduling algorithms.",
     _SLIDES + "slide 15"),
    ("process", 1, "Which structure represents a process inside the operating system?",
     ["The process control block (PCB)", "The ready queue", "The Gantt chart", "The dispatcher"], 0,
     "Each process is represented by a PCB holding its state, program counter, registers and more.",
     _BOOK + "3.1.3, p. 109"),
    ("process", 2, "Context-switch time is best described as",
     ["pure overhead, because the system does no useful work while switching",
      "part of the incoming process's burst time", "part of the outgoing process's waiting time only",
      "negligible and therefore ignored by every scheduler"], 0,
     "The kernel saves one context and loads another; nothing useful runs meanwhile.",
     _BOOK + "3.2.3, pp. 114-115"),
    ("linux", 1, "Which command changes the access permissions of a file?",
     ["chmod", "grep", "wc", "diff"], 0,
     "chmod changes who may read, write or execute a file or directory.",
     _LINUX + "13 chmod - Changing Access Permissions, pp. 6-7"),
    ("linux", 1, "What does the -i option of grep do?",
     ["Ignores case in the search", "Prints only the number of matching lines",
      "Searches every file in a directory tree", "Prints the lines that do not match"], 0,
     "grep [-i] 'string' file...: -i ignores case in the search.",
     _LINUX + "15 grep - Searching for Strings in Files, pp. 7-8"),
    ("linux", 2, "What does chmod g-r o-r phone do?",
     ["Removes read permission on phone for the group and for others",
      "Gives read permission on phone to the group and to others",
      "Removes every permission on phone from its owner", "Moves phone into the group's directory"], 0,
     "g-r means group remove read and o-r others remove read; the owner's permissions are unchanged.",
     _LINUX + "13 chmod - Changing Access Permissions, pp. 6-7"),
    ("shell", 1, "How do you get the value stored in the shell variable NAME?",
     ["$NAME", "NAME", "%NAME%", "&NAME"], 0,
     "To access the value stored in a variable, prefix its name with the dollar sign ($).",
     _SHELL + "slide 10"),
    ("shell", 2, "In a shell script, what does $# hold?",
     ["The number of arguments passed to the script", "The process ID of the shell",
      "The name of the script", "The first argument"], 0,
     "$# is the number of arguments; $$ is the shell's process ID, $0 the script name, $1 the first argument.",
     _SHELL + "slides 33-35"),
    ("shell", 2, "What happens to $what inside single quotes, as in echo 'testing $what'?",
     ["It is printed literally, without expansion", "It is replaced by the value of the variable what",
      "It is run as a command", "The shell reports a syntax error"], 0,
     "Within single quotes all characters are taken literally; double quotes expand variables.",
     "Shell scripting exercises, Changing your prompt"),
]


def bank_question(topic: str, level: int, rng: random.Random, seen=()) -> Question | None:
    """A question from the bank on `topic`, nearest to `level`, not asked before if possible."""
    pool = [b for b in BANK if b[0] == topic]
    fresh = [b for b in pool if b[2] not in seen] or pool
    if not fresh:
        return None
    _, lvl, prompt, options, answer, why, cite = min(fresh, key=lambda b: (abs(b[1] - level), rng.random()))
    return _shuffled(Question("mcq", topic, lvl, prompt, list(options), answer, why, cite, origin="bank"), rng)


def _shuffled(q: Question, rng: random.Random) -> Question:
    order = list(range(len(q.options)))
    rng.shuffle(order)
    q.options, q.answer = [q.options[i] for i in order], order.index(q.answer)
    return q


# ---------------------------------------------------------------------------------------
# LLM-written questions
# ---------------------------------------------------------------------------------------
_WRITER = ("You write exam questions for an Operating Systems course. Use ONLY the "
           "numbered SOURCES; every question must be answerable from them. Reply with one JSON object and "
           "nothing else.")


def _json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("the model did not return JSON")
    return json.loads(m.group(0))


def _sources(r, topic: str):
    hits = r.search(TOPICS[topic][1], k=4)
    return hits, build_context(hits, [])


def _cite(hits, tag) -> tuple[str, str]:
    m = re.search(r"\d+", str(tag))
    i = int(m.group(0)) - 1 if m else 0
    hit = hits[i] if 0 <= i < len(hits) else hits[0]
    return hit.cite, hit.item["text"]


def llm_mcq(r, topic: str, level: int, rng: random.Random, seen=()) -> Question:
    hits, context = _sources(r, topic)
    avoid = "".join(f"\n- {s}" for s in list(seen)[-8:])
    reply = llm_chat([
        {"role": "system", "content": _WRITER},
        {"role": "user", "content":
            f"SOURCES:\n{context}\n\nWrite one multiple-choice question on \"{TOPICS[topic][0]}\". "
            f"Difficulty - {LEVELS[level]}. Exactly four options, exactly one correct; the wrong options must "
            f"be plausible but clearly wrong according to the SOURCES."
            + (f"\nDo not repeat these earlier questions:{avoid}" if avoid else "") +
            '\nJSON keys: "question" (string), "options" (array of 4 strings), "answer" (0-3, index of the '
            'correct option), "explanation" (one or two sentences), "source" (the tag it is based on, e.g. "S2").'},
    ], temperature=0.7)
    d = _json(reply)
    options = [str(o) for o in d["options"]]
    if len(options) != 4 or len(set(options)) != 4 or int(d["answer"]) not in range(4):
        raise ValueError("the model's question is malformed")
    cite, _ = _cite(hits, d.get("source"))
    return _shuffled(Question("mcq", topic, level, str(d["question"]), options, int(d["answer"]),
                              str(d.get("explanation", "")), cite, origin="llm"), rng)


def llm_viva(r, topic: str, level: int, seen=()) -> Question:
    hits, context = _sources(r, topic)
    avoid = "".join(f"\n- {s}" for s in list(seen)[-8:])
    reply = llm_chat([
        {"role": "system", "content": _WRITER},
        {"role": "user", "content":
            f"SOURCES:\n{context}\n\nWrite one viva (oral exam) question on \"{TOPICS[topic][0]}\" that a "
            f"student answers in two to four sentences. Difficulty - {LEVELS[level]}."
            + (f"\nDo not repeat these earlier questions:{avoid}" if avoid else "") +
            '\nJSON keys: "question" (string), "key_points" (array of 2-4 short strings a full answer must '
            'contain, each taken from the SOURCES), "source" (the tag it is based on, e.g. "S2").'},
    ], temperature=0.7)
    d = _json(reply)
    points = [str(p) for p in d["key_points"]][:4]
    if not points:
        raise ValueError("the model's question has no key points")
    cite, passage = _cite(hits, d.get("source"))
    return Question("viva", topic, level, str(d["question"]), cite=cite, key_points=points,
                    passage=passage, origin="llm")


def grade_viva(q: Question, given: str) -> tuple[int, str]:
    """Mark a typed viva answer -> (score 0, 1 or 2, feedback for the student)."""
    points = "".join(f"\n- {p}" for p in q.key_points)
    reply = llm_chat([
        {"role": "system", "content":
            "You mark a student's short oral-exam answer for an Operating Systems course. Judge only against "
            "the KEY POINTS and the SOURCE. Be fair: accept correct paraphrases, do not reward padding. Reply "
            "with one JSON object and nothing else."},
        {"role": "user", "content":
            f"SOURCE:\n{q.passage}\n\nQUESTION: {q.prompt}\n\nKEY POINTS:{points}\n\nSTUDENT ANSWER: {given}\n\n"
            'JSON keys: "score" (2 = all key points correct, 1 = partly correct or incomplete, 0 = wrong or '
            'irrelevant), "feedback" (two or three sentences to the student: what was right, what was '
            'missing or wrong).'},
    ])
    d = _json(reply)
    return max(0, min(2, int(d["score"]))), str(d.get("feedback", ""))


# ---------------------------------------------------------------------------------------
# Progress and adaptation
# ---------------------------------------------------------------------------------------
@dataclass
class Progress:
    stats: dict = field(default_factory=dict)   # topic -> {"asked", "points", "level", "streak"}
    evidence: list = field(default_factory=list)   # learning evidence records (rag/learning.py)

    def _t(self, topic: str) -> dict:
        return self.stats.setdefault(topic, {"asked": 0, "points": 0.0, "level": 1, "streak": 0})

    def level(self, topic: str) -> int:
        return self._t(topic)["level"]

    def accuracy(self, topic: str) -> float | None:
        t = self._t(topic)
        return t["points"] / t["asked"] if t["asked"] else None

    def record(self, topic: str, points: float):
        """points: 1 correct, 0.5 partly correct, 0 wrong. Two correct in a row -> harder; wrong -> easier."""
        t = self._t(topic)
        t["asked"] += 1
        t["points"] += points
        if points >= 1:
            t["streak"] += 1
            if t["streak"] >= 2 and t["level"] < 3:
                t["level"], t["streak"] = t["level"] + 1, 0
        else:
            t["streak"] = 0
            if points == 0 and t["level"] > 1:
                t["level"] -= 1

    def pick_topic(self, rng: random.Random, topics=None, avoid: str | None = None) -> str:
        """Weaker and untried topics are more likely; `avoid` (the last topic) is skipped if possible."""
        pool = [t for t in (topics or TOPICS) if t != avoid] or list(topics or TOPICS)
        weights = [2.0 if self.accuracy(t) is None else 1 + 2 * (1 - self.accuracy(t)) for t in pool]
        return rng.choices(pool, weights)[0]

    def weak_topics(self) -> list[str]:
        return [t for t in TOPICS if self._t(t)["asked"] >= 2 and self.accuracy(t) < 0.6]

    @property
    def asked(self) -> int:
        return sum(t["asked"] for t in self.stats.values())

    @property
    def points(self) -> float:
        return sum(t["points"] for t in self.stats.values())

    def to_json(self) -> str:
        """Topic stats at the top level, as before; the evidence log under "_evidence" (files written
        before it existed simply have no such key)."""
        return json.dumps({**self.stats, **({"_evidence": self.evidence} if self.evidence else {})}, indent=1)

    @classmethod
    def from_json(cls, text: str) -> "Progress":
        data = json.loads(text) if text.strip() else {}
        if not isinstance(data, dict):
            raise ValueError("progress file is not a JSON object")
        evidence = data.get("_evidence")
        return cls({k: v for k, v in data.items() if k in TOPICS},
                   [e for e in evidence if isinstance(e, dict)] if isinstance(evidence, list) else [])


def next_question(r, progress: Progress, rng: random.Random, kinds, topic: str | None = None,
                  use_llm: bool = False, seen=(), last_topic: str | None = None) -> tuple[Question, str]:
    """The next question for this student -> (question, note about any fallback that happened).

    `kinds` is the subset of {"mcq", "numeric", "viva"} the student enabled; `topic` fixes the
    topic, otherwise it is chosen adaptively.
    """
    kinds = set(kinds) or {"mcq"}
    if not use_llm:
        kinds = (kinds - {"viva"}) or {"mcq"}
    numeric_topics = [t for t in TOPICS if TOPICS[t][2]]
    if topic is None:
        pool = numeric_topics if kinds == {"numeric"} else list(TOPICS)
        if not use_llm:
            pool = [t for t in pool if t in offline_topics(kinds)]
        topic = progress.pick_topic(rng, pool, avoid=last_topic)
    level = progress.level(topic)
    usable = [k for k in kinds if k != "numeric" or TOPICS[topic][2]] or ["mcq"]
    kind = rng.choice(sorted(usable))
    if kind == "numeric":
        return numeric_question(topic, level, rng), ""
    note = ""
    if use_llm:
        try:
            return (llm_viva(r, topic, level, seen) if kind == "viva" else llm_mcq(r, topic, level, rng, seen)), ""
        except Exception as e:
            note = f"The LLM could not write a question ({e}); this one is from the built-in bank."
    q = bank_question(topic, level, rng, seen)
    if q is None:  # no built-in questions on this topic
        if TOPICS[topic][2]:
            return numeric_question(topic, level, rng), note.replace("from the built-in bank", "a numerical")
        raise ValueError(note.replace("; this one is from the built-in bank", "") if note else
                         f"There are no built-in questions on {TOPICS[topic][0]}; they need the LLM. "
                         "Pick another topic or set LLM_API_KEY.")
    return q, note


def offline_topics(kinds) -> list[str]:
    """Topics that have questions without the LLM: in the built-in bank, or numericals."""
    banked = {b[0] for b in BANK}
    return [t for t in TOPICS if t in banked or ("numeric" in kinds and TOPICS[t][2])]
