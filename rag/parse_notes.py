"""Parse lab notes and handouts (PDF documents, Word 97-2003 .doc files) into section chunks.

Unlike slide decks, these are continuous documents: they are split at their section headings,
and line breaks are kept, because shell scripts and command examples depend on them.

  headings="numbered"  numbered sections ("15" / "grep - Searching for Strings in Files"), as in
                       the Linux commands handout; the table of contents is skipped
  headings="lines"     short title lines followed by prose or an example, as in the shell
                       scripting exercise sheet
"""
from __future__ import annotations

import re
from pathlib import Path

import pymupdf

from .config import CHUNK_MAX_CHARS
from .textutil import clean_lines


# ---------------------------------------------------------------------------------------
# Reading the text
# ---------------------------------------------------------------------------------------
def read_doc_text(path: Path) -> str:
    """The text of a Word 97-2003 .doc file.

    Such files keep the document's characters as one 8-bit run (here from byte 0xA00); the
    run ends where Word's binary tables start. Field codes (HYPERLINK ...) and Word's control
    characters are removed.
    """
    data = path.read_bytes()
    runs = list(re.finditer(rb"[\x09\x0a\x0d\x20-\x7e\x91-\x97\xa0-\xfe]{20,}", data))
    first = next((k for k, m in enumerate(runs) if re.search(rb"[A-Za-z]{3,} [A-Za-z]{2,} [A-Za-z]{2,}", m.group(0))),
                 None)
    if first is None:
        return ""
    runs = runs[first:]
    start, end = runs[0].start(), runs[0].end()
    for m in runs[1:]:                     # small gaps are field marks and special characters
        if m.start() - end > 64:
            break
        end = m.end()
    text = data[start:end].decode("cp1252", errors="replace")
    text = re.sub(r'\s*HYPERLINK\s+"[^"]*"(?:\s*\\[a-z]\s*"[^"]*")*\s*', " ", text)
    text = re.sub(r"[\x00-\x08\x0e-\x1f\x7f-\x9f]", "", text.replace("\x07", "\n"))
    return clean_lines(text)


def _pdf_pages(path: Path, skip) -> list[tuple[int, list[str]]]:
    pages = []
    for i, page in enumerate(pymupdf.open(path), start=1):
        if i in skip:
            continue
        lines = clean_lines(page.get_text()).split("\n")
        if lines and lines[-1].strip() == str(i):          # page-number footer
            lines = lines[:-1]
        pages.append((i, lines))
    return pages


# ---------------------------------------------------------------------------------------
# Finding the sections
# ---------------------------------------------------------------------------------------
def _numbered_sections(pages) -> list[dict]:
    """Sections headed "N" + "Title" (or "N Title"), numbered 1, 2, 3 ... in order.

    A heading in the table of contents is followed by its page number, so it is ignored.
    """
    flat = [(no, line) for no, lines in pages for line in lines]
    sections, expected, i = [], 1, 0
    cur = None
    while i < len(flat):
        no, line = flat[i]
        title, skip = None, 1
        one = re.fullmatch(r"(\d{1,2})\s+(\S.{2,70})", line.strip())
        if line.strip() == str(expected) and i + 1 < len(flat):
            title, skip = flat[i + 1][1].strip(), 2
        elif one and int(one.group(1)) == expected:
            title = one.group(2).strip()
        nxt = flat[i + skip][1].strip() if i + skip < len(flat) else ""
        if title and re.match(r"[A-Za-z]", title) and len(title) <= 70 and not re.fullmatch(r"\d+", nxt):
            cur = {"title": f"{expected} {title}", "start": no, "end": no, "lines": []}
            sections.append(cur)
            expected += 1
            i += skip
            continue
        if cur is not None:
            cur["lines"].append(line)
            cur["end"] = no
        i += 1
    return sections


_CODEISH = re.compile(r"[$#=;{}()\[\]<>|`\\/*]|^\s*(do|done|fi|then|else|esac|echo)\b")


def _line_sections(text: str) -> list[dict]:
    """Sections headed by a short title line that is followed by prose or an example."""
    lines = text.split("\n")
    sections = [{"title": "", "start": 1, "end": 1, "lines": []}]
    for k, line in enumerate(lines):
        s = line.strip()
        nxt = next((x.strip() for x in lines[k + 1:] if x.strip()), "")
        heading = (3 <= len(s) <= 70 and s[0].isupper() and not _CODEISH.search(s)
                   and not s.endswith((".", ":", ",")) and not s.startswith("Example")
                   and (len(nxt) >= 60 or nxt.startswith("Example") or re.match(r"(Version|Part|Exercise)\s+\d+", s)))
        if heading:
            sections.append({"title": s, "start": len(sections) + 1, "end": len(sections) + 1, "lines": []})
        else:
            sections[-1]["lines"].append(line)
    if not sections[0]["lines"] or not "".join(sections[0]["lines"]).strip():
        sections = sections[1:]
    else:
        sections[0]["title"] = "Introduction"
    for n, sec in enumerate(sections, 1):
        sec["start"] = sec["end"] = n
    return sections


def _pack(lines: list[str]) -> list[str]:
    """Split a section into pieces of at most CHUNK_MAX_CHARS, at blank lines where possible."""
    paras, cur = [], []
    for line in lines:
        if not line.strip() and cur:
            paras.append("\n".join(cur))
            cur = []
        elif line.strip() or cur:
            cur.append(line)
    if cur:
        paras.append("\n".join(cur))
    pieces, buf = [], ""
    for p in paras:
        while len(p) > CHUNK_MAX_CHARS:                    # one very long block: cut at a line break
            cut = p.rfind("\n", 0, CHUNK_MAX_CHARS)
            cut = cut if cut > 200 else CHUNK_MAX_CHARS
            if buf:
                pieces.append(buf)
                buf = ""
            pieces.append(p[:cut])
            p = p[cut:].lstrip("\n")
        if buf and len(buf) + len(p) + 2 > CHUNK_MAX_CHARS:
            pieces.append(buf)
            buf = ""
        buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        pieces.append(buf)
    return pieces


def parse_notes(src: dict, root: Path, fig_dir: Path):
    path = root / src["file"]
    if path.suffix.lower() == ".doc":
        sections = _line_sections(read_doc_text(path))
    elif src.get("headings") == "numbered":
        sections = _numbered_sections(_pdf_pages(path, set(src.get("skip_pages", []))))
    else:
        pages = _pdf_pages(path, set(src.get("skip_pages", [])))
        sections = [{"title": lines[0] if lines else f"Page {no}", "start": no, "end": no, "lines": lines[1:]}
                    for no, lines in pages]
    unit = "page" if path.suffix.lower() == ".pdf" else "section"
    chunks = []
    for sec in sections:
        for piece in _pack(sec["lines"]):
            chunks.append({
                "source_id": src["id"],
                "source_kind": src["kind"],
                "source_short": src["short"],
                "chapter": "",
                "section": sec["title"],
                "kind": "example" if re.search(r"\bexample", piece + sec["title"], re.I) else "explanation",
                "page_start": str(sec["start"]),
                "page_end": str(sec["end"]),
                "pdf_page_start": sec["start"],
                "unit": unit,
                "fig_refs": [],
                "text": f"{sec['title']}\n{piece}",
            })
    return chunks, []
