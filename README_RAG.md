# OSmosis – an Operating Systems tutor (RAG)

OSmosis answers Operating Systems questions from the course's own material, solves numericals
exactly (CPU scheduling, Banker's algorithm, page replacement, disk scheduling), and quizzes the
student. This file covers the knowledge base it searches
before answering, and how to run everything. Answers are grounded in **Silberschatz 10e** and
**your teacher's slides for Modules 1 to 5**, cite page/slide numbers, and come with the
matching textbook figure or slide image.

## What's in the knowledge base
`data/processed/` and `data/figures/` are built from the PDFs in this folder by `python -m rag.ingest`:

| Source | What is included | Passages |
|---|---|---|
| Silberschatz 10e | Ch. 2 (system calls), 3 (processes, IPC), 4 (threads), 5 (CPU scheduling), 6–7 (synchronization, producer-consumer, dining philosophers), 8 (deadlocks, Banker's algorithm), 9–10 (memory, paging, virtual memory), 11 (disk scheduling), 13–15 (file systems), 20 (the Linux system) | 1,783 |
| Teacher slides, Module 1 | Introduction | 30 |
| Teacher slides, Module 2 | System calls, interrupts, processes, fork/exec, threads | 46 |
| Teacher slides, Module 3 | CPU scheduling (200 slides) | 81 |
| Teacher slides, Module 3 | Deadlocks, Banker's algorithm | 36 |
| Teacher slides, Module 4 | IPC, producer-consumer, race conditions, synchronization, classical problems | 47 |
| Teacher slides, Module 5 (parts 1 and 2) | Memory allocation, paging, TLB, virtual memory, page replacement, segmentation | 106 |
| Lab notes, `Linux commands.pdf` | Basic Unix commands: man, ls, cd, cp, mv, rm, chmod, grep, wc, wildcards, redirection and pipes, job control (33 sections) | 39 |
| Lab slides, `Shell programming.pptx` | Bash: variables, read/echo, operators, if/case, while/for/until loops, special variables, arrays | 29 |
| Lab exercises, `Shell programming - sample  excersice.doc` | Worked bash scripts: quoting, arithmetic, control structures, functions, a line-count script in seven versions | 25 |

Together that is 2,222 passages and 928 images (297 textbook figures and Gantt charts, 631 slides).

Some slides are pictures with almost no text. Where they hold content found nowhere else
(worked examples), they are transcribed by hand in `data/annotations/<source id>.json`:
33 scheduling slides, 4 memory slides and the 2 shell diagrams. The other picture slides repeat
textbook figures, whose text is already indexed.

The lab handout and exercise sheet are documents, not slides: `rag/parse_notes.py` splits them at
their section headings and keeps line breaks, so scripts stay readable. The `.doc` sheet is read
directly (no Word needed); its citations name the section, since it has no page numbers.
File systems and disk scheduling come from the textbook only.

## Setup (once, on your laptop)
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python -m rag.build_index       # downloads BGE (~440 MB) the first time; embedding ~3,000
                                # passages takes about 45 min on a laptop CPU. Embeddings are
                                # cached in data/embed_cache.npz, so later rebuilds only embed
                                # new or changed passages.
```

## Ask a question
```bash
# without an LLM - shows exactly what was retrieved (good for testing)
python -m rag.ask "What is the convoy effect?" --no-llm

# with an LLM (free Groq key from console.groq.com)
set LLM_API_KEY=gsk_...                       # Windows  (macOS/Linux: export ...)
python -m rag.ask "Explain SRTF with an example" --html answer.html
```
Open `answer.html` in a browser to see the answer with the figures.
In PowerShell, set the key with `$env:LLM_API_KEY = 'gsk_...'` (`set` only works in Command Prompt).

## Chat interface
```bash
streamlit run app.py
```
Opens the tutor in the browser with six pages and a Dark mode switch in the sidebar:
* **Chat** - answers with their figures and sources. It remembers the last few turns, so
  "why does it happen?", "now do it with time quantum 2" or "what if P4 requests (0, 0, 2, 0)?"
  work. Asking for an example ("Explain SRTF with an example", "Explain the Banker's algorithm
  with an example", "Explain LRU with an example") shows the course's worked example solved in code.
* **Numerical solver** - four solvers: CPU scheduling (Gantt chart, CT/TAT/WT/RT table), Banker's
  algorithm (Need matrix, safe sequence, resource request, deadlock detection), page replacement
  (frame table, faults and hits) and disk scheduling (head-movement chart and total). Fill in the
  form, press Solve; every step is listed.
* **Quiz** - multiple-choice questions written by the LLM from the knowledge base, numerical
  questions whose answer key comes from the solvers (scheduling, Banker's algorithm, page
  replacement, disk scheduling), and viva questions the LLM marks. Each topic
  has a level 1-3: two correct answers in a row raise it, a wrong one lowers it, and weaker
  topics come up more often. Progress is kept in `data/quiz_progress.json`.
* **Check my working** (Reasoning Debugger) - the student enters their own working (a Gantt
  chart and CT/TAT/WT/RT table, a safe sequence, hit/fault marks, or a head order and total).
  It is compared step by step with the exact solver: the first wrong step is found, the cause is
  named only when the answer shows its pattern, later differences that follow from it are not
  counted again, and the feedback comes in stages (hint, why, correction) before the full
  solution. "See it run in OS Lab" opens the lab at that step; "Try a similar problem" gives a
  fresh one. With the LLM on, "Explain with the course notes" adds a cited explanation.
* **OS Lab** (Predict, Run, Explain) - step through a real run of any of the four solvers. At each
  decision (Round Robin: who runs next, with the ready queue, remaining bursts, quantum used and
  the Gantt chart so far) the student predicts, runs the step, and sees the solver's reason and
  the new state; Previous / Next / Replay / Reset, a prediction score, and a closing question.
* **My learning map** (Misconception Graph) - concepts and the student's misconceptions as a graph,
  built only from recorded attempts (debugger, lab, quiz). A mistake seen once is *tentative*; the
  same mistake with certainty in two different problems is *recurring*; two correct attempts
  afterwards mark it *resolved*. Each misconception shows the evidence, the slides to revise, a
  course passage, and buttons to practise it or take a quiz on it. With no data yet it offers a
  sample map, clearly labelled as made-up.

Set `LLM_API_KEY` in the same terminal first to get written
answers; without it the app shows the retrieved passages and still solves numericals, and the
debugger, lab, learning map and offline quiz all work. The first answer of a session takes about
half a minute while the embedding model loads.

## Numericals
Scheduling problems are computed by `rag/solver.py`, not by the LLM, so the Gantt chart and the
CT / TAT / WT / RT table are exact; the LLM only explains the result.
```bash
# through the tutor (solution + explanation)
python -m rag.ask "Round Robin, time quantum 3. P1: AT 0, BT 5. P2: AT 1, BT 4. P3: AT 3, BT 2."

# solver alone - no index, no LLM, instant
python -m rag.solver "SRTF. P1: AT 2, BT 4. P2: AT 3, BT 2, I/O 3, BT 2. P3: AT 5, BT 1."

# check the solvers against the teacher's and the textbook's worked examples, and the quiz / follow-up logic
python -m tests.test_solver
python -m tests.test_numericals
python -m tests.test_quiz
```
All test suites (none needs an API key; `test_retrieval` reads the passages but not the vector index):
```bash
python -m tests.test_solver
python -m tests.test_numericals
python -m tests.test_quiz
python -m tests.test_retrieval
python -m tests.test_diagnosis
python -m tests.test_trace
python -m tests.test_learning
```
Supported: FCFS, SJF, SRTF, priority (preemptive and non-preemptive), Round Robin, with optional
I/O between CPU bursts. Name the algorithm, give whole-number times, and write each process as
`P1: AT 0, BT 5, priority 2` (or bare columns `P1 0 5 2`). It follows the course conventions in
`rag/config.py`; say "higher number = higher priority" in the question to flip the priority rule.
### Banker's algorithm, page replacement, disk scheduling
These are solved exactly too, by `rag/banker.py`, `rag/paging.py` and `rag/disk.py`; type them in
the chat or use the Numerical solver page.
```bash
python -m rag.banker "Allocation: P0 0 1 0, P1 2 0 0, P2 3 0 2, P3 2 1 1, P4 0 0 2. Max: P0 7 5 3, P1 3 2 2, P2 9 0 2, P3 2 2 2, P4 4 3 3. Available: 3 3 2. P1 requests (1, 0, 2)."
python -m rag.paging "FIFO, LRU and Optimal with 3 frames. Reference string: 7, 0, 1, 2, 0, 3, 0, 4, 2, 3"
python -m rag.disk "SSTF. Head at 53. Requests: 98, 183, 37, 122, 14, 124, 65, 67"
```
* **Banker's algorithm**: Allocation plus Max (or Need), and Available (or the total instances).
  The matrices can be written one after the other, side by side under a header row
  (`Process Allocation Max Available`), or one process at a time (`P0: Allocation 0 1 0, Max 7 5 3`).
  Add a request ("P1 requests (1, 0, 2)") to run the resource-request algorithm, or give a
  Request matrix with "deadlock detection". The safety algorithm checks P0, P1, ... in order,
  pass after pass, as the slides do; other safe sequences can also be correct.
* **Page replacement**: a reference string and the number of frames ("3 and 4 frames" solves
  both, showing Belady's anomaly). With no algorithm named, FIFO, LRU and Optimal are compared.
* **Disk scheduling**: the request queue and the head position, plus the direction and disk size
  for SCAN / C-SCAN (if not given: towards larger cylinders, 200 cylinders, and the answer says it
  assumed this). C-SCAN and C-LOOK count the return seek unless the question says not to.
  With no algorithm named, all six are compared.

The default model is `openai/gpt-oss-120b` on Groq; set `LLM_MODEL` to use another.
Offline instead of Groq: install Ollama, `ollama pull qwen2.5:3b`, then
`set LLM_BASE_URL=http://localhost:11434/v1`, `set LLM_API_KEY=ollama`, `set LLM_MODEL=qwen2.5:3b`.

## Adding more material
* Teacher notes / past papers (PDF, PPTX or Word .doc): put them in `teacher_notes/`, then
  `python -m rag.ingest` and `python -m rag.build_index`. For a proper citation name
  ("Teacher slides (Module 6)"), add the deck to `SOURCES` in `rag/config.py` instead, as the
  Module 1–5 decks are.
* More textbook pages: edit `pdf_pages` in `rag/config.py` (PDF page numbers; the PDF's own
  table of contents gives each chapter's first page).
* After adding a deck with image-only slides, add transcriptions to `data/annotations/<source_id>.json`.

## How it works
```
PDFs ──ingest──► chunks.jsonl + figures.jsonl + images
                      │
                build_index (BGE embeddings → ChromaDB)
                      │
question ─► BM25 (keywords) + BGE (meaning) ─► Reciprocal Rank Fusion
          ─► top passages + their figures ─► LLM prompt with [S1]… citations
```
* **Hybrid retrieval**: BM25 catches exact terms (SRTF, convoy effect); BGE catches paraphrases.
  The two rankings are combined with Reciprocal Rank Fusion; the weights are in `FUSION` in
  `rag/config.py` and were chosen by a measured experiment (below).
* **Teacher material preferred, within limits**: teacher passages get a small ×1.05 boost, and
  ×1.15 more when the question asks for an example or a numerical (`rag/intent.py`), because the
  worked examples are in the slides. Textbook exercises, projects and reading lists count half.
  The prompt tells the LLM to follow the teacher's convention where it differs from the book.

### Retrieval evaluation
```bash
python eval_retrieval.py            # the current configuration, by question group and topic
python eval_retrieval.py --tune     # all variants, the grid search and the held-out check
```
78 hand-labelled questions. Before → after the tuning (Hit@1 / Hit@3 / Hit@6 / MRR):
71.8% / 93.6% / 98.7% / 0.830 → 76.9% / 94.9% / 98.7% / 0.858; on the 19 held-out questions
84.2% → 94.7% Hit@1. CPU-scheduling Hit@1 rose from 48% to 63%. Details, the defects found and
the limitations are in `docs/retrieval_evaluation.md`.
* **Figures linked to text**: book chunks link to the figures they mention ("Figure 5.5")
  and to Gantt charts on the same pages; slide chunks link to their slide images.

| File | Role |
|---|---|
| `rag/config.py` | sources, page ranges, course conventions, model names |
| `rag/parse_book.py` | textbook → section-tagged chunks, figure + Gantt crops |
| `rag/parse_slides.py` | slide deck → grouped chunks + slide images (+ transcriptions) |
| `rag/parse_notes.py` | lab handouts (PDF, Word .doc) → section chunks with line breaks kept |
| `rag/ingest.py` | runs the parsers → `data/processed/` |
| `rag/build_index.py` | embeddings → `data/chroma/` |
| `rag/retriever.py` | hybrid search (configurable weighted RRF), returns passages + figures |
| `rag/intent.py` | question intent (numerical, example, comparison, definition, follow-up, conceptual) |
| `rag/retrieval_eval.py` + `eval_retrieval.py` | retrieval metrics, controlled variants, tuning; the labelled questions |
| `rag/solver.py` | exact scheduling simulator for numericals (Gantt chart, CT/TAT/WT/RT) |
| `rag/banker.py` | Banker's algorithm: Need, safety, resource request, deadlock detection |
| `rag/paging.py` | page replacement: FIFO, LRU, Optimal |
| `rag/disk.py` | disk scheduling: FCFS, SSTF, SCAN, C-SCAN, LOOK, C-LOOK |
| `rag/numericals.py` | one entry point that routes a question to the right solver |
| `rag/ask.py` | CLI: retrieval → (solver) → LLM → answer with citations (+ HTML view) |
| `rag/examples.py` | the course's worked examples (slides and textbook) as solvable problems |
| `rag/quiz.py` | quiz questions (LLM, solvers, built-in bank), marking, adaptive progress, progress file |
| `rag/diagnosis.py` | Reasoning Debugger: compares a student's working with the solver, staged feedback |
| `rag/trace.py` | OS Lab: a solver run as events to predict and step through |
| `rag/misconceptions.py` | concepts, prerequisites and misconceptions with stable ids |
| `rag/learning.py` | evidence log, transparent assessment rules, recommendations, the graph |
| `app.py` | Streamlit interface over all of the above |
| `tests/test_solver.py` | scheduling solver checked against the teacher's worked examples |
| `tests/test_numericals.py` | Banker's, page-replacement and disk solvers checked against the slides and textbook |
| `tests/test_retrieval.py` | fusion, metrics, intent, citations, follow-ups, figures |
| `tests/test_diagnosis.py` | the debugger on correct and mistaken working for all four solvers |
| `tests/test_trace.py` | each lab replays to the solver's exact result; session controls |
| `tests/test_learning.py` | evidence rules, old and new progress files, the graph |
| `faq_check.py` | runs a set of common questions and saves the answers to `faq_answers.md` |
