"""What kind of question a student is asking - transparent rules, checked in tests/test_retrieval.py.

    numerical    a calculation: "Solve using SRTF ...", "find the average waiting time", "how many
                 page faults ..."
    example      asks for a worked example or a script: "Explain SRTF with an example",
                 "Write a shell script to ..."
    comparison   "difference between", "compare", "X vs Y"
    definition   "what is", "define", "meaning of"
    follow_up    a short question leaning on the previous turn ("why does it happen?")
    conceptual   everything else

Retrieval uses the numerical and example intents: the teacher's slides and lab notes hold the
course's worked examples and scripts, so teacher material is preferred for those questions
(config.FUSION["example_teacher_boost"]).
"""
from __future__ import annotations

import re

# "solve" etc. only as a command, so "how does aging solve it?" is not a calculation
_NUMERICAL = re.compile(r"^\s*(?:solve|calculate|compute|find|determine|draw)\b|\bgantt\b|\bnumericals?\b|"
                        r"\bfind the (?:average|total|waiting|turnaround|completion|safe|number|physical)\b|"
                        r"\bhow many (?:page faults|page hits|cylinders)\b", re.I)
_EXAMPLE = re.compile(r"\b(examples?|worked|illustrat\w*|demonstrat\w*)\b|^\s*write an? (?:\w+ )?(?:script|program)\b",
                      re.I)
EXAMPLE_INTENTS = ("numerical", "example")
_COMPARISON = re.compile(r"\b(difference|differ\w*|compare\w*|comparison|versus|vs\.?|distinguish|contrast)\b", re.I)
_DEFINITION = re.compile(r"^\s*(what\s+(?:is|are|does)\b|define\b|definition\b|meaning of\b|what do you mean)", re.I)
_REFERS_BACK = re.compile(r"\b(that|this|it|its|those|these|them|above|previous|earlier|again|same|more|why|"
                          r"elaborate|simpler|step)\b", re.I)


def is_follow_up(question: str) -> bool:
    """A short question that leans on the previous turn ("why does it happen?", "explain that again")."""
    return len(question.split()) <= 8 and bool(_REFERS_BACK.search(question))


def classify(question: str, has_history: bool = False) -> str:
    if has_history and is_follow_up(question):
        return "follow_up"
    if _NUMERICAL.search(question):
        return "numerical"
    if _EXAMPLE.search(question):
        return "example"
    if _COMPARISON.search(question):
        return "comparison"
    if _DEFINITION.search(question):
        return "definition"
    return "conceptual"
