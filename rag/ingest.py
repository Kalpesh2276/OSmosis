"""Step 1: PDFs/PPTX/DOC -> data/processed/{chunks,figures}.jsonl + data/figures/*

Usage (from the project root):
    python -m rag.ingest
"""
from __future__ import annotations

import json
import shutil
import sys

from .config import FIGURES_DIR, PROCESSED_DIR, ROOT, SOURCES, TEACHER_NOTES_DIR
from .parse_book import parse_book
from .parse_notes import parse_notes
from .parse_slides import parse_slides


def discover_sources():
    sources = list(SOURCES)
    known = {s["file"] for s in sources}
    if TEACHER_NOTES_DIR.exists():
        for p in sorted(TEACHER_NOTES_DIR.iterdir()):
            if p.suffix.lower() not in (".pdf", ".pptx", ".doc"):
                continue
            rel = str(p.relative_to(ROOT)).replace("\\", "/")
            if rel in known:
                continue
            sid = "notes_" + "".join(ch if ch.isalnum() else "_" for ch in p.stem.lower())[:40]
            sources.append(
                {
                    "id": sid,
                    "kind": "teacher_notes",
                    "title": f"Teacher notes - {p.stem}",
                    "short": f"Teacher notes ({p.stem[:30]})",
                    "file": rel,
                    **({"format": "notes"} if p.suffix.lower() == ".doc" else {}),
                }
            )
    return sources


def main():
    if FIGURES_DIR.exists():
        shutil.rmtree(FIGURES_DIR)
    FIGURES_DIR.mkdir(parents=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_chunks, all_figs = [], []
    for src in discover_sources():
        path = ROOT / src["file"]
        if not path.exists():
            print(f"[skip] {src['file']} not found", file=sys.stderr)
            continue
        print(f"[parse] {src['short']} ...", flush=True)
        if src["kind"] == "textbook":
            chunks, figs = parse_book(src, ROOT, FIGURES_DIR)
        elif src.get("format") == "notes":
            chunks, figs = parse_notes(src, ROOT, FIGURES_DIR)
        else:
            chunks, figs = parse_slides(src, ROOT, FIGURES_DIR)
        print(f"        {len(chunks)} chunks, {len(figs)} figures/slides")
        all_chunks += chunks
        all_figs += figs

    all_chunks = [c for c in all_chunks if len(c["text"]) >= 40]
    for i, c in enumerate(all_chunks):
        c["id"] = f"{c['source_id']}_c{i:04d}"

    with open(PROCESSED_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(PROCESSED_DIR / "figures.jsonl", "w", encoding="utf-8") as f:
        for x in all_figs:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"[done] {len(all_chunks)} chunks, {len(all_figs)} figures -> {PROCESSED_DIR}")
    print("Next: python -m rag.build_index")


if __name__ == "__main__":
    main()
