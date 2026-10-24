from __future__ import annotations

import re

from .config import CHUNK_MAX_CHARS, CHUNK_TARGET_CHARS

_LIGATURE_FIXES = [
    (re.compile(r"\bfirs\b"), "first"),       # bold-font ligature loss in OSC 10e
    (re.compile("ﬁ"), "fi"),
    (re.compile("ﬂ"), "fl"),
    (re.compile("ﬀ"), "ff"),
    (re.compile("ﬃ"), "ffi"),
    (re.compile("ﬄ"), "ffl"),
]


def clean_text(s: str) -> str:
    s = s.replace("­", "")
    # join hyphenated line breaks: "illus-\ntrates" -> "illustrates"
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)
    s = s.replace("\n", " ")
    for rx, rep in _LIGATURE_FIXES:
        s = rx.sub(rep, s)
    # subscripts split by extraction: "P 1" -> "P1", "T 0" -> "T0"
    s = re.sub(r"\b([PTQ])\s+(\d{1,2})\b", r"\1\2", s)
    s = re.sub(r"[ \t ]+", " ", s)
    return s.strip()


def clean_lines(s: str) -> str:
    """Like clean_text, but keeps line breaks: shell scripts and command examples need them."""
    s = s.replace("­", "").replace("\r", "\n").replace("\x0b", "\n")
    for rx, rep in _LIGATURE_FIXES:
        s = rx.sub(rep, s)
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    lines = []
    for line in s.replace("\t", "    ").replace(" ", " ").split("\n"):
        body = line.lstrip(" ")          # keep the indentation, collapse spaces inside the line
        lines.append(" " * (len(line) - len(body)) + re.sub(r" +", " ", body).rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def split_paragraphs_into_chunks(paras: list[dict]) -> list[dict]:
    """Greedy paragraph packing with one-paragraph overlap.

    paras: [{"text", "page", "pdf_page"}] in reading order (one section).
    """
    out: list[dict] = []
    cur: list[dict] = []
    size = 0

    def emit():
        if not cur:
            return
        out.append(
            {
                "text": "\n".join(p["text"] for p in cur),
                "page_start": cur[0]["page"],
                "page_end": cur[-1].get("page_end", cur[-1]["page"]),
                "pdf_page_start": cur[0]["pdf_page"],
                "pdf_page_end": cur[-1].get("pdf_page_end", cur[-1]["pdf_page"]),
            }
        )

    # Merge fragments: a paragraph that does not end a sentence (page break
    # mid-sentence, table header separated from its rows) absorbs the next one.
    merged: list[dict] = []
    for p in paras:
        if merged and (
            not re.search(r"[.:?!)\]]\s*$", merged[-1]["text"]) or len(merged[-1]["text"]) < 60
        ) and len(merged[-1]["text"]) + len(p["text"]) < CHUNK_MAX_CHARS:
            merged[-1] = {**merged[-1], "text": merged[-1]["text"] + " " + p["text"],
                          "page_end": p["page"], "pdf_page_end": p["pdf_page"]}
        else:
            merged.append(dict(p))
    paras = merged

    for p in paras:
        # very long single paragraphs: split on sentences
        pieces = [p]
        if len(p["text"]) > CHUNK_MAX_CHARS:
            sents = re.split(r"(?<=[.!?])\s+", p["text"])
            pieces, buf = [], ""
            for s in sents:
                if len(buf) + len(s) > CHUNK_TARGET_CHARS and buf:
                    pieces.append({**p, "text": buf.strip()})
                    buf = ""
                buf += s + " "
            if buf.strip():
                pieces.append({**p, "text": buf.strip()})
        for piece in pieces:
            if size + len(piece["text"]) > CHUNK_TARGET_CHARS and cur:
                emit()
                last = cur[-1]
                cur = [last] if len(last["text"]) < 350 else []
                size = sum(len(x["text"]) for x in cur)
            cur.append(piece)
            size += len(piece["text"])
    emit()
    return out
