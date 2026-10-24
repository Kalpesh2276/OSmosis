"""Live, step-by-step demonstration of the OSmosis pipeline, from the source files to the answer.

Every stage calls the project's real functions and prints what they produce. Press Enter to go
on to the next stage. Nothing is rebuilt or written to disk, and no API key is needed (the LLM
is called only if LLM_API_KEY is set).

    python demo_pipeline.py
    python demo_pipeline.py "What is a deadlock?"
    python demo_pipeline.py --no-pause
"""
from __future__ import annotations

import os
import sys
import textwrap

import numpy as np

from rag.config import CHROMA_DIR, EMBED_MODEL, FUSION, ROOT, SOURCES, TOP_K
from rag.store import citation, load_chunks, load_figures

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

args = [a for a in sys.argv[1:] if not a.startswith("--")]
PAUSE = "--no-pause" not in sys.argv
QUESTION = args[0] if args else "What is the convoy effect in FCFS?"
NUMERICAL = "FCFS. P1: AT 0, BT 5. P2: AT 1, BT 3. P3: AT 2, BT 1."
step_no = 0


def step(title: str, where: str):
    global step_no
    step_no += 1
    if PAUSE and step_no > 1:
        input("\n[Enter] next step ...")
    print("\n" + "=" * 90)
    print(f"STEP {step_no}: {title}")
    print(f"code: {where}")
    print("=" * 90)


def show(text: str, width: int = 88, limit: int | None = None):
    text = text if limit is None or len(text) <= limit else text[:limit] + " ..."
    for line in text.split("\n"):
        print(textwrap.fill(line, width, initial_indent="  ", subsequent_indent="  ") if line else "")


def source_path(file: str):
    """The course files may sit in the project folder or in datasets/."""
    for p in (ROOT / file, ROOT / "datasets" / file):
        if p.exists():
            return p
    return None


# =========================================================================================
# OFFLINE PART: how the knowledge base was built (shown on real data, nothing is rebuilt)
# =========================================================================================
step("Sources: the dataset is the course material", "rag/config.py -> SOURCES")
for s in SOURCES:
    print(f"  {s['kind']:<14} {s['short']:<38} {s['file'][:40]}")
print(f"\n  {len(SOURCES)} sources. 'kind' decides the teacher boost later.")

step("Parse: read the PDF with fonts", "rag/parse_book.py -> parse_book(), page.get_text('dict')")
chunks = load_chunks()
sample = next(c for c in chunks if c["source_kind"] == "textbook" and "5.3.1" in c["section"])
book = source_path(SOURCES[0]["file"])
if book:
    import pymupdf

    page = pymupdf.open(book)[sample["pdf_page_start"] - 1]
    blocks = [b for b in page.get_text("dict")["blocks"] if b["type"] == 0]
    print(f"  Textbook PDF page {sample['pdf_page_start']} (printed page {sample['page_start']}): "
          f"{len(blocks)} text blocks. The font tells what each block is:\n")
    for b in blocks[:8]:
        span = next(s for l in b["lines"] for s in l["spans"] if s["text"].strip())
        text = " ".join(s["text"] for l in b["lines"] for s in l["spans"]).strip()
        role = ("header" if b["bbox"][3] <= 50 else
                "HEADING" if span["font"].startswith("HelveticaNeueLTStd-Bd") else
                "body text" if span["font"].startswith("PalatinoLTStd") else "other")
        print(f"  [{role:<9}] font={span['font']:<26} {text[:45]!r}")
    print("\n  header = running page header (top 50 pt), skipped; HEADING starts a new section; "
          "body text becomes chunks")
else:
    print("  (textbook PDF not found in the project folder or datasets/; skipping the live read)")

step("Clean: fix PDF text problems", "rag/textutil.py -> clean_text()")
from rag.textutil import clean_text

raw = "The scheduler illus-\ntrates the conﬁguration of P 1 and P 2   with    extra spaces."
print("  before:", repr(raw))
print("  after: ", repr(clean_text(raw)))

step("Chunk: cut sections into ~1,100-character pieces", "rag/textutil.py -> split_paragraphs_into_chunks()")
from rag.textutil import split_paragraphs_into_chunks

paras = [{"text": t, "page": "206", "pdf_page": 273} for t in
         [s.strip() + "." for s in " ".join(c["text"] for c in chunks
                                             if c["section"] == sample["section"]).split(". ") if s.strip()]]
demo = split_paragraphs_into_chunks(paras)
print(f"  {len(paras)} sentences of section {sample['section']!r} -> {len(demo)} chunks, "
      f"sizes {[len(c['text']) for c in demo]} characters")
kinds = {}
for c in chunks:
    kinds[c["source_kind"]] = kinds.get(c["source_kind"], 0) + 1
print(f"\n  The whole knowledge base (data/processed/chunks.jsonl): {len(chunks)} chunks {kinds}, "
      f"{len(load_figures())} figures")
print("\n  One chunk as stored:")
for k in ("id", "source_short", "section", "page_start", "page_end", "kind", "fig_refs"):
    print(f"    {k:<13} {sample[k]}")
print(f"    {'text':<13} {sample['text'][:150]!r} ...")

step("Embed: BGE turns text into 768 numbers", "rag/embeddings.py -> embed_passages(), embed_query()")
from rag.embeddings import embed_passages, embed_query

print(f"  model: {EMBED_MODEL} (loading ...)")
vec = np.array(embed_passages([f"{sample['section']}\n{sample['text']}"])[0])
print(f"  chunk vector: {len(vec)} numbers, first five {np.round(vec[:5], 4).tolist()}, length {np.linalg.norm(vec):.3f}")
other = next(c for c in chunks if "Dining" in c["section"])
ovec = np.array(embed_passages([f"{other['section']}\n{other['text']}"])[0])
qvec = np.array(embed_query(QUESTION))
print(f"\n  Similar meaning -> similar direction (cosine similarity = dot product of unit vectors):")
print(f"    question vs {sample['section'][:40]!r:<44} {qvec @ vec:.3f}")
print(f"    question vs {other['section'][:40]!r:<44} {qvec @ ovec:.3f}")

step("Store: vectors live in ChromaDB", "rag/build_index.py -> col.add(...)")
import chromadb

client = chromadb.PersistentClient(path=str(CHROMA_DIR))
col = client.get_collection("chunks")
rec = col.get(ids=[sample["id"]], include=["embeddings", "metadatas", "documents"])
print(f"  collection 'chunks': {col.count()} vectors, distance = {col.metadata.get('hnsw:space')}")
print(f"  stored record {sample['id']}: metadata {rec['metadatas'][0]}, "
      f"vector of {len(rec['embeddings'][0])} numbers")
print(f"  same as the vector just computed: {np.allclose(rec['embeddings'][0], vec, atol=1e-4)}")

# =========================================================================================
# ONLINE PART: one question, from input to output
# =========================================================================================
step("Question arrives -> Retriever loads, answer() starts", "app.py chat_page() -> rag/ask.py answer()")
from rag.retriever import Retriever, tokenize

r = Retriever()
print(f"  QUESTION: {QUESTION!r}")
print(f"  Retriever ready: mode = {r.mode}, BM25 index over {len(r.chunks)} chunks, ChromaDB connected")
from rag.numericals import parse_problem

print(f"  numericals.parse_problem(question) -> {parse_problem(QUESTION)}  (not a numerical, so no solver)")

step("Intent and tokens", "rag/intent.py classify(),  rag/retriever.py tokenize()")
from rag.intent import classify

intent = classify(QUESTION)
print(f"  intent: {intent!r}  (teacher boost x{FUSION['teacher_boost']}"
      f"{', plus x' + str(FUSION['example_teacher_boost']) if intent in ('numerical', 'example') else ''})")
print(f"  tokens: {tokenize(QUESTION)}")

step("Keyword search (BM25)", "rag/retriever.py rankings() -> self.bm25.get_scores()")
bm_scores = r.bm25.get_scores(tokenize(QUESTION))
bm_rank, dense_rank = r.rankings(QUESTION)
for n, i in enumerate(bm_rank[:5], 1):
    print(f"  {n}. score {bm_scores[i]:6.2f}  {citation(r.chunks[i])}")

step("Vector search (BGE + ChromaDB)", "rag/retriever.py rankings() -> embed_query() -> col.query()")
res = r.dense[0].query(query_embeddings=[r._q_vec], n_results=5)
for n, (cid, dist) in enumerate(zip(res["ids"][0], res["distances"][0]), 1):
    c = next(x for x in r.chunks if x["id"] == cid)
    print(f"  {n}. similarity {1 - dist:.3f}  {citation(c)}")

step("Fuse: weighted Reciprocal Rank Fusion -> top 6", "rag/retriever.py fuse(), line 132")
fused = r.fuse(bm_rank, dense_rank, intent=intent)
print(f"  score = sum of 1/({FUSION['rrf_k']} + rank), teacher material x{FUSION['teacher_boost']}\n")
for n, (i, s) in enumerate(fused, 1):
    b = bm_rank.index(i) + 1 if i in bm_rank else None
    d = dense_rank.index(i) + 1 if i in dense_rank else None
    parts = " + ".join(f"1/{60 + x}" for x in (b, d) if x)
    boost = " x1.05" if r.chunks[i]["source_kind"] == "teacher_notes" else ""
    print(f"  [S{n}] {s:.5f} = ({parts}){boost:<6} BM25 #{b}, BGE #{d}  {citation(r.chunks[i])}")

step("Context: number the passages [S1]..[S6]", "rag/ask.py build_context()")
from rag.ask import SYSTEM_PROMPT, answer, build_context, clean_answer, is_grounded

hits, figs = r.retrieve(QUESTION)
context = build_context(hits, figs)
show(context, limit=1200)
print(f"\n  ({len(context)} characters in total, {len(figs)} figures listed)")

step("Prompt: system rules + SOURCES + QUESTION", "rag/ask.py SYSTEM_PROMPT, call_llm()")
messages = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"SOURCES:\n{context}\n\nQUESTION: {QUESTION}"}]
print("  messages[0] role=system (first lines):")
for line in SYSTEM_PROMPT.split("\n")[:12]:
    print("    " + line)
print("    ...")
print(f"\n  messages[1] role=user: 'SOURCES:\\n' + {len(context)} characters + '\\n\\nQUESTION: {QUESTION}'")

step("LLM call", "rag/ask.py llm_chat() -> POST {LLM_BASE_URL}/chat/completions")
if os.environ.get("LLM_API_KEY"):
    from rag.ask import llm_chat

    print("  calling the LLM (temperature 0) ...")
    raw_answer = llm_chat(messages)
    print("\n  raw answer from the LLM:")
    show(raw_answer, limit=1500)
else:
    raw_answer = ("**FCFS** is non-preemptive [S1]. In the **convoy effect**, short processes wait behind one "
                  "long CPU-bound process [S2][S74].")
    print("  LLM_API_KEY is not set in this terminal, so the request is not sent.")
    print("  It would be: POST https://api.groq.com/openai/v1/chat/completions")
    print('  body = {"model": "openai/gpt-oss-120b", "temperature": 0, "messages": messages}')
    print("\n  For the next step, this sample answer stands in for the LLM's reply:")
    show(raw_answer)

step("Check the answer: remove invalid citations, test grounding", "rag/ask.py clean_answer(), is_grounded()")
cleaned = clean_answer(raw_answer, len(hits))
print("  after clean_answer() (tags outside [S1]-[S6] removed):")
show(cleaned, limit=1500)
print(f"\n  is_grounded() = {is_grounded(cleaned)}  -> the source list is "
      f"{'shown' if is_grounded(cleaned) else 'hidden'}")

step("Output: what the app shows", "app.py show_reply()")
reply = answer(r, QUESTION, use_llm=bool(os.environ.get("LLM_API_KEY")))
print(f"  answer written by the LLM: {reply.llm_used}" + ("" if reply.llm_used else "  (no key: the passages are shown)"))
print("  Sources:")
for n, h in enumerate(reply.hits, 1):
    print(f"    [S{n}] {h.cite}")
print("  Figures:")
for f in reply.figs:
    print(f"    {f['label']:<26} data/{f['image']}")

# =========================================================================================
# SIDE BRANCH: numericals are computed by code, never by the LLM
# =========================================================================================
step("Numerical branch: exact solver", "rag/numericals.py parse_problem() -> rag/solver.py solve()")
from rag.numericals import kind, render, solve

problem = parse_problem(NUMERICAL)
print(f"  QUESTION: {NUMERICAL!r}")
print(f"  parse_problem() -> a {kind(problem)} problem, algorithm {problem.algo!r}, "
      f"{len(problem.procs)} processes")
print()
show(render(solve(problem)), limit=1600)
print("\n  This text is added to the prompt as 'COMPUTED SOLUTION (exact)'; the LLM only explains it.")
print("\nEnd of the demonstration.")
