"""The course's worked examples, as problems the solvers can run.

When a student asks for an example ("Explain SRTF with an example", "Explain the Banker's
algorithm with an example"), the tutor solves one of these in code, so the table and every
step come from the solver and the LLM only narrates them. tests/test_solver.py and
tests/test_numericals.py check each one against the slide's (or the textbook's) answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .config import SOURCES
from .numericals import kind, parse_problem, topic_of

DECK = "keerthika_m3"   # the scheduling deck
_SOURCES = {s["id"]: s for s in SOURCES}


@dataclass(frozen=True)
class Example:
    name: str
    slides: tuple[int, int]     # first and last slide (textbook: printed page) of the example
    question: str               # readable by numericals.parse_problem
    deck: str = DECK            # source id the slide numbers refer to
    prefer: str = ""            # regex: questions matching it get this example before others of its topic

    @property
    def topic(self) -> str:
        """The CPU-scheduling algorithm key, or 'banker' / 'paging' / 'disk'."""
        p = parse_problem(self.question)
        return p.algo if kind(p) == "scheduling" else kind(p)

    @property
    def algo(self) -> str:
        return self.topic

    @property
    def where(self) -> str:
        src = _SOURCES[self.deck]
        a, b = self.slides
        span = f"{a}-{b}" if a != b else str(a)
        if src["kind"] == "textbook":
            return f"{src['short']}, pp. {span}"
        return f"{src['short']}, slide{'s' * (a != b)} {span}"


_DL, _M5B, _BOOK = "keerthika_m3_deadlock", "notes_m5b", "silberschatz10"
_DISK_QUEUE = "Requests: 98, 183, 37, 122, 14, 124, 65, 67."
EXAMPLES = [
    Example("FCFS Example 2", (26, 28),
            "FCFS. P1: AT 4, BT 5. P2: AT 6, BT 4. P3: AT 0, BT 3. P4: AT 6, BT 2. P5: AT 5, BT 4."),
    Example("SJF Example 6", (49, 49),
            "SJF (non-preemptive). P1: AT 0, BT 10. P2: AT 1, BT 3. P3: AT 2, BT 6. P4: AT 3, BT 4. P5: AT 4, BT 2."),
    Example("SJF Example 1", (39, 40),
            "SJF (non-preemptive). P1: AT 0, BT 6. P2: AT 0, BT 8. P3: AT 0, BT 7. P4: AT 0, BT 3."),
    Example("SRTF Example 3", (44, 45),
            "SRTF. P1: AT 0, BT 12. P2: AT 2, BT 4. P3: AT 3, BT 6. P4: AT 8, BT 5."),
    Example("SRTF Example 2", (41, 42),
            "SRTF. P1: AT 0, BT 8. P2: AT 1, BT 4. P3: AT 2, BT 9. P4: AT 3, BT 5."),
    Example("SRTF Example 4", (46, 47),
            "SRTF. P1: AT 0, BT 10. P2: AT 3, BT 6. P3: AT 7, BT 1. P4: AT 8, BT 3."),
    Example("SRTF with an I/O burst", (101, 114),
            "SRTF. P1: AT 2, BT 4. P2: AT 3, BT 2, I/O 3, BT 2. P3: AT 5, BT 1."),
    Example("Priority Example 1", (59, 60),
            "Priority scheduling (non-preemptive). P1: AT 0, BT 10, priority 3. P2: AT 0, BT 1, priority 1. "
            "P3: AT 0, BT 2, priority 4. P4: AT 0, BT 1, priority 5. P5: AT 0, BT 5, priority 2."),
    Example("Priority Example 2", (60, 61),
            "Priority scheduling (preemptive). P1: AT 0, BT 11, priority 2. P2: AT 5, BT 28, priority 0. "
            "P3: AT 12, BT 2, priority 3. P4: AT 2, BT 10, priority 1. P5: AT 9, BT 16, priority 4."),
    Example("Round Robin Example 2", (78, 85),
            "Round Robin, time quantum 2. P1: AT 0, BT 5. P2: AT 1, BT 3. P3: AT 2, BT 1. P4: AT 3, BT 2. "
            "P5: AT 4, BT 3."),
    Example("Round Robin Example 1", (75, 77),
            "Round Robin, time quantum 4. P1: AT 0, BT 24. P2: AT 0, BT 3. P3: AT 0, BT 3."),
    Example("Banker's Algorithm Example 1", (21, 30),
            "Banker's algorithm. Allocation: P0 0 0 1 2, P1 1 0 0 0, P2 1 3 5 4, P3 0 6 3 2, P4 0 0 1 4. "
            "Max: P0 0 0 1 2, P1 1 7 5 0, P2 2 3 5 6, P3 0 6 5 2, P4 0 6 5 6. Available: 1 5 2 0.", _DL),
    # Max of P3 is 1 4 2 4: the slide's Need row (0 1 1 2) and totals (12 12 8 10) require it
    Example("Banker's Algorithm Example 2, request from P1", (32, 54),
            "Banker's algorithm. Allocation: P0 2 0 0 1, P1 3 1 2 1, P2 2 1 0 3, P3 1 3 1 2, P4 1 4 3 2. "
            "Max: P0 4 2 1 2, P1 5 2 5 2, P2 2 3 1 6, P3 1 4 2 4, P4 3 6 6 5. Available: 3 3 2 1. "
            "Request from P1: (1 1 0 0).", _DL, r"\brequest"),
    Example("Banker's Algorithm Example 2, request from P4", (55, 57),
            "Banker's algorithm. Allocation: P0 2 0 0 1, P1 3 1 2 1, P2 2 1 0 3, P3 1 3 1 2, P4 1 4 3 2. "
            "Max: P0 4 2 1 2, P1 5 2 5 2, P2 2 3 1 6, P3 1 4 2 4, P4 3 6 6 5. Available: 3 3 2 1. "
            "Request from P4: (0 0 2 0).", _DL, r"\bunsafe\b|\bdenied\b|not\s+(?:be\s+)?granted|\brefus"),
    Example("Deadlock detection example", (60, 61),
            "Deadlock detection (Banker's algorithm). Allocation: P0 0 1 0, P1 2 0 0, P2 3 0 3, P3 2 1 1, "
            "P4 0 0 2. Request: P0 0 0 0, P1 2 0 2, P2 0 0 0, P3 1 0 0, P4 0 0 2. Available: 0 0 0.", _DL,
            r"\bdetect"),
    Example("Deadlock detection example, P2 asks for one more C", (62, 62),
            "Deadlock detection (Banker's algorithm). Allocation: P0 0 1 0, P1 2 0 0, P2 3 0 3, P3 2 1 1, "
            "P4 0 0 2. Request: P0 0 0 0, P1 2 0 2, P2 0 0 1, P3 1 0 0, P4 0 0 2. Available: 0 0 0.", _DL,
            r"\bdetect\w*\b.*\bdeadlocked\b|\bdeadlocked\b.*\bdetect"),
    Example("Page replacement example (FIFO, Optimal, LRU)", (76, 79),
            "Page replacement with FIFO, LRU, Optimal. Reference string: 7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 0, 3, "
            "2, 1, 2, 0, 1, 7, 0, 1. Number of frames: 3.", _M5B),
    Example("Belady's anomaly (FIFO with 3 and 4 frames)", (76, 77),
            "Page replacement with FIFO. Reference string: 1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5. "
            "Number of frames: 3 and 4.", _M5B, r"belady|anomal"),
    Example("Disk scheduling example (head at 53, moving towards 0)", (458, 459),
            f"Disk scheduling with FCFS, SSTF, SCAN, LOOK. The disk has 200 cylinders (0-199). Head at 53, moving "
            f"towards 0. {_DISK_QUEUE}", _BOOK),
    Example("C-SCAN example (head at 53, moving towards 199)", (460, 460),
            f"Disk scheduling with C-SCAN, C-LOOK. The disk has 200 cylinders (0-199). Head at 53, moving towards "
            f"larger cylinders. {_DISK_QUEUE}", _BOOK, r"\bc[\s-]?(?:scan|look)\b|circular"),
]

_WANTS_EXAMPLE = re.compile(r"\b(examples?|worked|illustrat\w*|demonstrat\w*|numerical|gantt)\b", re.I)


def find_example(question: str, hits) -> Example | None:
    """The worked example to show for `question`, or None if it does not ask for one.

    Among the examples of the topic asked about, prefers one whose `prefer` pattern matches
    the question, then one whose slides are among the retrieved passages (best passage first),
    so the solution shown is the one the cited slides contain.
    """
    topic = topic_of(question)
    if topic is None or not _WANTS_EXAMPLE.search(question):
        return None
    pool = [e for e in EXAMPLES if e.topic == topic]
    preferred = [e for e in pool if e.prefer and re.search(e.prefer, question, re.I)]
    pool = preferred or [e for e in pool if not e.prefer] or pool
    for h in hits:
        c = h.item
        if not any(c["source_id"] == e.deck for e in pool):
            continue
        try:
            a = int(c.get("page_start", c.get("page")))
            b = int(c.get("page_end", a))
        except (TypeError, ValueError):     # textbook front matter and exercises: "P-45", "EX-14"
            continue
        for e in pool:
            if c["source_id"] == e.deck and e.slides[0] <= b and a <= e.slides[1]:
                return e
    return pool[0] if pool else None
