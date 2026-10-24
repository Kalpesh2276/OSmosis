"""Reading lists of numbers out of a typed question (shared by the numerical solvers)."""
from __future__ import annotations

import re

# a number | a lone capital letter used as a column name ("A B C") | separators | any other word
_TOKEN = re.compile(r"(\d+)|([A-Z])(?![A-Za-z0-9])|([\s,;:=()\[\]<>{}|/\\-]+)|([A-Za-z0-9_']+|.)", re.S)


def read_vector(text: str, skip_words: int = 3) -> list[int]:
    """The numbers at the start of `text`: "(3, 3, 2)", "A 3 B 3 C 2", "= 3 3 2".

    Up to `skip_words` words before the first number are skipped ("Available vector is 3 3 2");
    after it, the first word ends the list.
    """
    nums, words = [], 0
    for m in _TOKEN.finditer(text):
        if m.group(1):
            nums.append(int(m.group(1)))
        elif m.group(2) or m.group(3):
            continue
        else:
            if nums or re.search(r"\d", m.group(4)):   # a word like P1 or T0 starts the next row
                break
            words += 1
            if words > skip_words:
                break
    return nums


_RUN = re.compile(r"\d+(?:\s*[,;\s]\s*\d+)+")


def longest_run(text: str, at_least: int = 2) -> list[int]:
    """The longest list of numbers separated by commas or spaces ("98, 183, 37, 122")."""
    best: list[int] = []
    for m in _RUN.finditer(text):
        nums = [int(x) for x in re.findall(r"\d+", m.group(0))]
        if len(nums) > len(best):
            best = nums
    return best if len(best) >= at_least else []


def vec(v) -> str:
    """(1 5 2 0) - how the slides write a vector."""
    return "(" + " ".join(str(x) for x in v) + ")"
