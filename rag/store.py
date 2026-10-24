from __future__ import annotations

import json
from functools import lru_cache

from .config import PROCESSED_DIR


def _load(name):
    path = PROCESSED_DIR / name
    if not path.exists():
        raise SystemExit(f"{path} not found - run `python -m rag.ingest` first.")
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


@lru_cache(maxsize=1)
def load_chunks():
    return _load("chunks.jsonl")


@lru_cache(maxsize=1)
def load_figures():
    figs = _load("figures.jsonl")
    # several Gantt charts can share a page and so a label; number them so each can be named
    by_label = {}
    for f in figs:
        by_label.setdefault((f["source_id"], f["label"]), []).append(f)
    for group in by_label.values():
        if len(group) > 1:
            for n, f in enumerate(group, 1):
                f["label"] = f"{f['label']} #{n}"
    return figs


def figure_search_text(f: dict) -> str:
    return f"{f['label']}: {f['caption']}. {f['section']}. {f['context']}"


def citation(item: dict) -> str:
    """Human-readable source reference for a chunk or figure."""
    if item.get("unit") == "section":   # a document without page numbers (Word lab sheet)
        return f"{item['source_short']}, {item['section']}"
    if item["source_kind"] == "textbook" or item.get("unit") == "page":
        a = item.get("page_start", item.get("page"))
        b = item.get("page_end", a)
        pages = f"p. {a}" if a == b else f"pp. {a}-{b}"
        sec = item.get("section", "")
        return f"{item['source_short']}, {sec}, {pages}".replace(", ,", ",")
    a = item.get("page_start", item.get("page"))
    b = item.get("page_end", a)
    sl = f"slide {a}" if a == b else f"slides {a}-{b}"
    return f"{item['source_short']}, {sl}"
