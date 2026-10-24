"""Parse the Silberschatz textbook PDF into text chunks and figure images.

Layout facts this relies on (OSC 10e):
  * body text        : PalatinoLTStd-* fonts
  * section headings : HelveticaNeueLTStd-BdE
  * figure captions  : HelveticaNeueLTStd-Bd, text starts "Figure N.M", sits BELOW the figure
  * inline Gantt charts: process labels in PalatinoLTStd-LightIta, time labels in Helvetica
  * running header   : top ~50pt of every page
"""
from __future__ import annotations

import re
from pathlib import Path

import pymupdf

from .textutil import clean_text, split_paragraphs_into_chunks

HEADER_Y = 50
BODY_LEFT, BODY_RIGHT = 80, 470
FIG_RE = re.compile(r"^Figure\s+(\d+\.\d+)\s+(.*)", re.S)
FIG_REF_RE = re.compile(r"Figure\s+(\d+\.\d+)")


def _block_info(b):
    spans = [s for l in b["lines"] for s in l["spans"] if s["text"].strip()]
    if not spans:
        return None
    lines = []
    for l in b["lines"]:
        t = "".join(s["text"] for s in l["spans"])
        if t.strip():
            lines.append(t)
    text = "\n".join(lines)
    return {
        "bbox": b["bbox"],
        "font": spans[0]["font"],
        "size": spans[0]["size"],
        "text": text,
    }


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _is_heading(bi) -> bool:
    return bi["font"].startswith("HelveticaNeueLTStd-BdE") or (
        bi["font"].startswith("HelveticaNeueLTStd-Bd") and bi["size"] >= 12
    )


def _is_caption(bi) -> bool:
    return bi["font"].startswith("HelveticaNeueLTStd-Bd") and FIG_RE.match(
        clean_text(bi["text"])
    ) is not None


def _is_body(bi) -> bool:
    return bi["font"].startswith(("PalatinoLTStd", "STIX", "Courier", "LucidaSans"))


def parse_book(src: dict, root: Path, fig_dir: Path):
    doc = pymupdf.open(root / src["file"])
    toc = doc.get_toc()
    pages = []
    for a, b in src["pdf_pages"]:
        pages.extend(range(a, b + 1))
    page_set = set(pages)

    toc_by_page: dict[int, list[str]] = {}
    for lvl, title, p in toc:
        toc_by_page.setdefault(p, []).append(title)

    def section_at(pno: int):
        chap, sec = "", ""
        for lvl, title, p in toc:
            if p > pno:
                break
            if lvl == 2:
                chap, sec = title, title
            elif lvl >= 3:
                sec = title
        return chap, sec

    chunks, figures = [], []
    fig_dir.mkdir(parents=True, exist_ok=True)

    # paragraph accumulator per section
    cur_paras: list[dict] = []
    cur_chap, cur_sec = section_at(pages[0])

    def flush():
        nonlocal cur_paras
        if not cur_paras:
            return
        for piece in split_paragraphs_into_chunks(cur_paras):
            text = piece["text"]
            refs = sorted(set(FIG_REF_RE.findall(text)))
            kind = "exercise" if "exercise" in cur_sec.lower() else "explanation"
            chunks.append(
                {
                    "source_id": src["id"],
                    "source_kind": src["kind"],
                    "source_short": src["short"],
                    "chapter": cur_chap,
                    "section": cur_sec,
                    "kind": kind,
                    "page_start": piece["page_start"],
                    "page_end": piece["page_end"],
                    "pdf_page_start": piece["pdf_page_start"],
                    "pdf_page_end": piece["pdf_page_end"],
                    "fig_refs": refs,
                    "text": text,
                }
            )
        cur_paras = []

    last_pno = None
    for pno in pages:
        page = doc[pno - 1]
        label = page.get_label() or str(pno)
        if last_pno is not None and pno != last_pno + 1:
            flush()
            cur_chap, cur_sec = section_at(pno)
        last_pno = pno
        toc_titles = {_norm(t): t for t in toc_by_page.get(pno, [])}

        d = page.get_text("dict", sort=False)
        infos = [bi for b in d["blocks"] if b["type"] == 0 for bi in [_block_info(b)] if bi]
        infos = [bi for bi in infos if bi["bbox"][3] > HEADER_Y]

        # ---- figures (captioned) ------------------------------------------------
        body_blocks = [
            bi for bi in infos
            if bi["font"].startswith("PalatinoLTStd-Roman") and bi["bbox"][2] - bi["bbox"][0] > 200
        ] + [bi for bi in infos if _is_heading(bi)]
        for bi in infos:
            if not _is_caption(bi):
                continue
            m = FIG_RE.match(clean_text(bi["text"]))
            fig_no, caption = m.group(1), m.group(2).strip()
            cap_top = bi["bbox"][1]
            above = [x["bbox"][3] for x in body_blocks if x["bbox"][3] <= cap_top - 4]
            top = max(above) + 3 if above else HEADER_Y + 4
            if cap_top - top < 25:  # nothing graphical found; skip
                continue
            clip = pymupdf.Rect(BODY_LEFT, top, BODY_RIGHT, bi["bbox"][3] + 4) & page.rect
            fid = f"{src['id']}_fig_{fig_no.replace('.', '_')}"
            if any(f["id"] == fid for f in figures):
                # the end-of-chapter exercises reuse figure numbers from the chapter body
                fid += "_ex"
            out = fig_dir / f"{fid}.png"
            page.get_pixmap(clip=clip, dpi=150).save(out)
            figures.append(
                {
                    "id": fid,
                    "source_id": src["id"],
                    "source_kind": src["kind"],
                    "source_short": src["short"],
                    "label": f"Figure {fig_no}" + (" (exercises)" if fid.endswith("_ex") else ""),
                    "caption": caption,
                    "chapter": cur_chap,
                    "section": cur_sec,
                    "page": label,
                    "pdf_page": pno,
                    "image": f"figures/{out.name}",
                    "context": "",
                    "_y": top,
                }
            )

        # ---- inline Gantt charts (uncaptioned) ----------------------------------
        # anchor: a Helvetica row of time stamps with italic "P.." labels just above
        gantt_label_ids = set()
        gi = 0
        for nb in infos:
            nt_ = clean_text(nb["text"])
            if not (nb["font"].startswith("HelveticaLTStd") and nb["size"] <= 9
                    and re.fullmatch(r"[\d\s]+", nt_) and len(nt_.split()) >= 3):
                continue
            labels = [
                bi for bi in infos
                if "Ita" in bi["font"] and bi["font"].startswith("PalatinoLTStd")
                and bi["bbox"][3] <= nb["bbox"][1] + 3
                and nb["bbox"][1] - bi["bbox"][3] <= 25
                and clean_text(bi["text"]).startswith("P")
            ]
            if not labels:
                continue
            gantt_label_ids.update(id(b) for b in labels)
            gi += 1
            top = min(b["bbox"][1] for b in labels) - 10
            clip = pymupdf.Rect(BODY_LEFT, top, BODY_RIGHT, nb["bbox"][3] + 5) & page.rect
            fid = f"{src['id']}_gantt_p{label}_{gi}"
            out = fig_dir / f"{fid}.png"
            page.get_pixmap(clip=clip, dpi=150).save(out)
            figures.append(
                {
                    "id": fid,
                    "source_id": src["id"],
                    "source_kind": src["kind"],
                    "source_short": src["short"],
                    "label": f"Gantt chart (p. {label})",
                    "caption": "",
                    "chapter": cur_chap,
                    "section": "",
                    "page": label,
                    "pdf_page": pno,
                    "image": f"figures/{out.name}",
                    "context": "",
                    "_y": top,
                }
            )

        # ---- body text -------------------------------------------------------------
        page_secs = [(-1.0, cur_chap, cur_sec)]
        for bi in sorted(infos, key=lambda x: (x["bbox"][1], x["bbox"][0])):
            txt = clean_text(bi["text"])
            if not txt:
                continue
            nt = _norm(txt)
            if nt in toc_titles or _is_heading(bi):
                flush()
                title = toc_titles.get(nt, txt)
                if re.match(r"^Chapter\s+\d+\s", title):
                    cur_chap = title
                cur_sec = title
                page_secs.append((bi["bbox"][1], cur_chap, cur_sec))
                continue
            if _is_caption(bi):
                continue
            if not _is_body(bi):
                continue  # figure-internal labels, Gantt time axes, etc.
            if id(bi) in gantt_label_ids:
                continue  # Gantt process labels
            cur_paras.append({"text": txt, "page": label, "pdf_page": pno})
        for f in figures:
            if f["pdf_page"] == pno and "_y" in f:
                for y, ch, sec in page_secs:
                    if y <= f["_y"]:
                        f["chapter"], f["section"] = ch, sec
                if not f["caption"]:
                    f["caption"] = f"Worked-example Gantt chart in {f['section']}"
    flush()

    # attach context text to figures: the chunk that references the figure or
    # (for Gantt charts) the chunk on the same page in the same section.
    for f in figures:
        f.pop("_y", None)
        best = None
        if f["label"].startswith("Figure"):
            no = f["label"].split()[1]
            for c in chunks:
                if no in c["fig_refs"] and (c["kind"] == "exercise") == f["id"].endswith("_ex"):
                    best = c
                    break
        if best is None:
            for c in chunks:
                if c["section"] == f["section"] and c["pdf_page_start"] <= f["pdf_page"]:
                    best = c
        if best is not None:
            f["context"] = best["text"][:700]
            f["section"] = f["section"] or best["section"]
    # chunk -> figure ids (referenced figures + Gantt charts on its pages)
    numbered = [f for f in figures if f["label"].startswith("Figure")]
    by_no = {f["label"].split()[1]: f["id"] for f in numbered if not f["id"].endswith("_ex")}
    by_no_ex = {f["label"].split()[1]: f["id"] for f in numbered if f["id"].endswith("_ex")}
    for c in chunks:
        lookup = by_no_ex if c["kind"] == "exercise" else {}
        ids = [lookup.get(n) or by_no[n] for n in c["fig_refs"] if n in lookup or n in by_no]
        for f in figures:
            if f["label"].startswith("Gantt") and f["section"] == c["section"] \
                    and c["pdf_page_start"] <= f["pdf_page"] <= c["pdf_page_end"]:
                ids.append(f["id"])
        c["fig_refs"] = ids
    # The running chapter label drifts after each "Chapter N Exercises" heading, so take the
    # chapter from the TOC instead; chapter opening pages get the chapter as their section.
    for item, pno in [(c, c["pdf_page_start"]) for c in chunks] + [(f, f["pdf_page"]) for f in figures]:
        item["chapter"] = section_at(pno)[0] or item["chapter"]
        if item["section"].startswith("C H A P T E R"):
            item["section"] = item["chapter"]
    return chunks, figures
