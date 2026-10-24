"""One entry point for every exact solver, so chat, the solver page and the quiz treat them alike.

    CPU scheduling     rag.solver   FCFS, SJF, SRTF, priority, Round Robin
    Banker's algorithm rag.banker   safety, resource request, deadlock detection
    Page replacement   rag.paging   FIFO, LRU, Optimal
    Disk scheduling    rag.disk     FCFS, SSTF, SCAN, C-SCAN, LOOK, C-LOOK

Each module has parse_problem(text), solve(problem), render(solution), to_text(problem) and
follow_up(text, previous); the functions here pick the module from the question or the object.
"""
from __future__ import annotations

from . import banker, disk, paging, solver
from .solver import ProblemError

KINDS = {"scheduling": solver, "banker": banker, "paging": paging, "disk": disk}
LABELS = {"scheduling": "CPU scheduling", "banker": "Banker's algorithm", "paging": "Page replacement",
          "disk": "Disk scheduling"}


def kind(obj) -> str:
    """'scheduling' | 'banker' | 'paging' | 'disk' for a Problem or Solution of any solver."""
    for name, module in KINDS.items():
        if isinstance(obj, (module.Problem, module.Solution)):
            return name
    raise TypeError(f"not a numerical: {type(obj).__name__}")


def parse_problem(text: str):
    """The numerical stated in `text` (any kind), or None for an ordinary question.

    Raises ProblemError when it clearly is a numerical but cannot be solved as written."""
    for module in (banker, paging, disk):
        problem = module.parse_problem(text)
        if problem is not None:
            return problem
    return solver.parse_problem(text)


def solve(problem):
    return KINDS[kind(problem)].solve(problem)


def render(solution) -> str:
    return KINDS[kind(solution)].render(solution)


def to_text(problem) -> str:
    return KINDS[kind(problem)].to_text(problem)


def title(problem) -> str:
    return KINDS[kind(problem)].title(problem)


def follow_up(text: str, previous):
    """The previous numerical re-run with the change `text` asks for, or None."""
    return None if previous is None else KINDS[kind(previous)].follow_up(text, previous)


def topic_of(text: str) -> str | None:
    """Which kind of numerical a question is about, from its wording alone (no data needed).

    CPU scheduling questions give the algorithm key ("srtf", "rr", ...)."""
    if banker.mentions(text):
        return "banker"
    if paging.mentions(text):
        return "paging"
    if disk.mentions(text):
        return "disk"
    if solver.about_other_topic(text):
        return None
    return solver.detect_algorithm(text)


def narrow(problem, text: str):
    """A worked example restricted to the algorithm(s) the question names (paging and disk)."""
    module = KINDS[kind(problem)]
    return module.narrow(problem, text) if hasattr(module, "narrow") else problem


__all__ = ["KINDS", "LABELS", "ProblemError", "follow_up", "kind", "narrow", "parse_problem", "render", "solve",
           "title", "to_text", "topic_of"]
