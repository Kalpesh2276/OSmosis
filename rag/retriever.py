"""Hybrid retrieval: BM25 (keywords) + BGE dense vectors, fused with RRF.

BM25 catches exact terms students type ("convoy effect", "SRTF", "aging");
dense search catches paraphrases ("why do short jobs wait behind long ones").
If the Chroma index has not been built yet, retrieval falls back to BM25 only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from .config import CHROMA_DIR, FIGURE_HITS, FIGURE_POOL, FUSION, TOP_K
from .intent import EXAMPLE_INTENTS, classify
from .store import citation, figure_search_text, load_chunks, load_figures

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set(
    "the a an of to in and or is are be for on with as by it this that at from we which "
    "what how why when explain describe give example".split()
)
_SYNONYMS = {
    "srtf": "srtf shortest remaining time first preemptive sjf",
    "sjf": "sjf shortest job first",
    "fcfs": "fcfs first come first served",
    "rr": "rr round robin",
    "mlfq": "mlfq multilevel feedback queue",
    "mlq": "mlq multilevel queue",
    "tat": "tat turnaround time",
    "wt": "wt waiting time",
    "pcb": "pcb process control block",
    "lru": "lru least recently used",
    "lfu": "lfu least frequently used",
    "mfu": "mfu most frequently used",
    "fifo": "fifo first in first out",
    "sstf": "sstf shortest seek time first",
    "tlb": "tlb translation look aside buffer",
    "mmu": "mmu memory management unit",
    "ipc": "ipc interprocess communication",
    "rag": "rag resource allocation graph",
    "fcb": "fcb file control block",
}


_BACK_MATTER = re.compile(r"(Programming Projects|Practice Exercises|Programming Problems|Bibliography|"
                          r"Further Reading|Chapter \d+ Exercises)\b")


def tokenize(s: str) -> list[str]:
    toks = _TOKEN.findall(s.lower())
    out = []
    for t in toks:
        out.extend(_SYNONYMS.get(t, t).split())
    return [t for t in out if t not in _STOP]


@dataclass
class Hit:
    item: dict
    score: float
    cite: str = ""
    figures: list[dict] = field(default_factory=list)


class Retriever:
    def __init__(self, use_dense: bool = True):
        self.chunks = load_chunks()
        self.figures = load_figures()
        self.fig_by_id = {f["id"]: f for f in self.figures}
        self.bm25 = BM25Okapi([tokenize(f"{c['section']} {c['text']}") for c in self.chunks])
        self.fbm25 = BM25Okapi([tokenize(figure_search_text(f)) for f in self.figures])
        self.dense = None
        if use_dense and (CHROMA_DIR / "chroma.sqlite3").exists():
            try:
                import chromadb

                client = chromadb.PersistentClient(path=str(CHROMA_DIR))
                self.dense = (client.get_collection("chunks"), client.get_collection("figures"))
            except Exception as e:  # index missing/corrupt -> keyword only
                print(f"[retriever] dense index unavailable ({e}); using BM25 only")
        self.mode = "hybrid" if self.dense else "bm25"
        # end-of-chapter material: it names topics but rarely explains them
        self.back_matter = {i for i, c in enumerate(self.chunks)
                            if c["source_kind"] == "textbook" and _BACK_MATTER.match(c["section"])}

    # -- helpers -----------------------------------------------------------------
    @staticmethod
    def _rrf(rank_lists: list[list[int]], k: int = 60) -> dict[int, float]:
        scores: dict[int, float] = {}
        for ranks in rank_lists:
            for r, idx in enumerate(ranks):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + r + 1)
        return scores

    def _dense_ranks(self, col, items, q_vec, n):
        res = col.query(query_embeddings=[q_vec], n_results=min(n, len(items)))
        pos = {it["id"]: i for i, it in enumerate(items)}
        return [pos[i] for i in res["ids"][0] if i in pos]

    def _pick_figures(self, chunk: dict, limit: int = 2) -> list[dict]:
        figs = [self.fig_by_id[i] for i in chunk.get("fig_refs", []) if i in self.fig_by_id]
        if chunk["source_kind"] != "textbook":
            # slides: only those that carry a diagram / worked example image
            figs = [f for f in figs if f.get("visual")]
        return figs[:limit]

    # -- public API ------------------------------------------------------------------
    def rankings(self, query: str, depth: int = FUSION["depth"]) -> tuple[list[int], list[int]]:
        """Chunk indices ranked by BM25 (only those sharing a term with the query) and by the
        embeddings (empty without the dense index), best first."""
        bm = self.bm25.get_scores(tokenize(query))
        bm_rank = [i for i in sorted(range(len(bm)), key=lambda i: -bm[i]) if bm[i] > 0][:depth]
        dense_rank = []
        if self.dense:
            from .embeddings import embed_query

            self._q_vec = embed_query(query)
            dense_rank = self._dense_ranks(self.dense[0], self.chunks, self._q_vec, depth)
        return bm_rank, dense_rank

    def fuse(self, bm_rank: list[int], dense_rank: list[int], k: int = TOP_K, fusion: dict | None = None,
             source_kind: str | None = None, intent: str = "conceptual") -> list[tuple[int, float]]:
        """Weighted Reciprocal Rank Fusion -> the top k (chunk index, score), deduplicated.

        `intent` is rag.intent.classify(question). Ties keep the order in which passages were
        first seen (keyword ranking first), so the result is deterministic for fixed rankings."""
        f = {**FUSION, **(fusion or {})}
        scores: dict[int, float] = {}
        for weight, ranks in ((f["bm25_weight"], bm_rank), (f["dense_weight"], dense_rank)):
            for r, idx in enumerate(ranks):
                scores[idx] = scores.get(idx, 0.0) + weight / (f["rrf_k"] + r + 1)
        teacher = f["teacher_boost"] * (f["example_teacher_boost"] if intent in EXAMPLE_INTENTS else 1.0)
        for i in scores:
            if self.chunks[i]["source_kind"] == "teacher_notes":
                scores[i] *= teacher
            elif i in self.back_matter:
                scores[i] *= f["back_matter_weight"]
        out, seen_text, per_section = [], set(), {}
        window = 2 * k if f.get("kind_slot") else k      # candidates considered for the kind slots
        for i in sorted(scores, key=lambda i: -scores[i]):
            c = self.chunks[i]
            if scores[i] <= 0 or (source_kind and c["source_kind"] != source_kind):
                continue
            key = c["text"][:120]                        # the same passage indexed twice
            section = (c["source_id"], c["section"])
            if key in seen_text or (f["max_per_section"] and per_section.get(section, 0) >= f["max_per_section"]):
                continue
            seen_text.add(key)
            per_section[section] = per_section.get(section, 0) + 1
            out.append((i, scores[i]))
            if len(out) >= window:
                break
        if f.get("kind_slot"):
            # the best textbook passage and the best teacher passage among the leading candidates
            # both appear within the first kind_slot places (the rest keep their order)
            for kind in ("textbook", "teacher_notes"):
                pos = next((p for p, (i, _) in enumerate(out) if self.chunks[i]["source_kind"] == kind), None)
                if pos is not None and pos >= f["kind_slot"]:
                    out.insert(f["kind_slot"] - 1, out.pop(pos))
        return out[:k]

    def search(self, query: str, k: int = TOP_K, source_kind: str | None = None,
               fusion: dict | None = None) -> list[Hit]:
        ranked = self.fuse(*self.rankings(query), k=k, fusion=fusion, source_kind=source_kind,
                           intent=classify(query))
        return [Hit(item=self.chunks[i], score=s, cite=citation(self.chunks[i]),
                    figures=self._pick_figures(self.chunks[i])) for i, s in ranked]

    def search_figures(self, query: str, k: int = 3) -> list[Hit]:
        sc = self.fbm25.get_scores(tokenize(query))
        lists = [[i for i in sorted(range(len(sc)), key=lambda i: -sc[i]) if sc[i] > 0][:30]]
        if self.dense:
            from .embeddings import embed_query

            qv = getattr(self, "_q_vec", None) or embed_query(query)
            lists.append(self._dense_ranks(self.dense[1], self.figures, qv, 30))
        fused = self._rrf(lists)
        # a figure only one ranker likes is usually a look-alike, so require all of them
        strong = set.intersection(*(set(ranks[:FIGURE_POOL]) for ranks in lists))
        order = sorted(strong, key=lambda i: -fused[i])
        out = []
        for i in order:
            f = self.figures[i]
            if f["source_kind"] != "textbook" and not f.get("visual"):
                continue
            out.append(Hit(item=f, score=fused[i], cite=citation(f)))
            if len(out) >= k:
                break
        return out

    def retrieve(self, query: str, k: int = TOP_K, max_figures: int = 4):
        """Passages + the diagrams that go with them (deduplicated)."""
        self._q_vec = None
        hits = self.search(query, k)
        figs, seen = [], set()
        for h in hits[:FIGURE_HITS]:
            for f in h.figures:
                if f["id"] not in seen:
                    seen.add(f["id"])
                    figs.append(f)
        for fh in self.search_figures(query, k=3):
            if fh.item["id"] not in seen:
                seen.add(fh.item["id"])
                figs.append(fh.item)
        # balance: alternate textbook / teacher figures so neither crowds out the other
        book = [f for f in figs if f["source_kind"] == "textbook"]
        notes = [f for f in figs if f["source_kind"] != "textbook"]
        mixed = []
        while (book or notes) and len(mixed) < max_figures:
            for lst in (notes, book):
                if lst and len(mixed) < max_figures:
                    mixed.append(lst.pop(0))
        return hits, mixed
