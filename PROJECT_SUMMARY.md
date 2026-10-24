# OSmosis – project summary

## What it is
OSmosis is an AI tutor for the Operating Systems course VIT BCSE303L. It is a student course project built against a problem statement titled "LLM specifically for learning and teaching Operating System concepts" (proposed name OSTutorLLM). That statement offers two options: Option 1, RAG + an existing instruction-tuned LLM, and Option 2, fine-tuning. The project follows **Option 1 only**.

OSmosis answers OS questions only from the course's own material and cites the page or slide for each fact. It solves numericals exactly in code instead of letting the LLM guess, and it quizzes the student with adaptive difficulty. It runs locally as a Streamlit web app.

## Fixed decisions (please don't suggest changing these)
- No fine-tuning (Option 2 is out of scope).
- Runs locally only; no public hosting or mobile app.
- LLM: `openai/gpt-oss-120b` through Groq's free OpenAI-compatible API (any OpenAI-compatible endpoint works, including Ollama for offline use). The API key is set in the user's own terminal.
- Without an API key the app still works: it shows the retrieved passages instead of a written answer, and the numericals and offline quiz work as normal.

## Tech stack
Python 3.12, Streamlit 1.65, PyMuPDF (PDF text and images), python-pptx, rank_bm25, ChromaDB, sentence-transformers with BAAI/bge-base-en-v1.5 embeddings, and pandas.

## Knowledge base (2,222 passages, 928 images)
| Source | Content |
|---|---|
| Silberschatz, *Operating System Concepts* 10e | Ch. 2–11, 13–15, 20 (system calls, processes, threads, scheduling, synchronization, deadlocks, memory, virtual memory, disk scheduling, file systems, Linux) – 1,783 passages, 297 figures and Gantt charts |
| Teacher's slides, Modules 1–5 | Introduction; system calls and processes; CPU scheduling (200 slides); deadlocks and the Banker's algorithm; IPC and synchronization; memory, paging, virtual memory and page replacement |
| Lab notes: Linux commands (PDF) | 33 sections: man, ls, cp, mv, rm, chmod, grep, wildcards, pipes and redirection, job control… |
| Lab slides: shell programming (PPTX) | bash variables, read/echo, operators, if/case, loops, special variables, arrays |
| Lab exercises: bash scripting (old Word .doc) | worked scripts; quoting, arithmetic, functions, a line-count script in 7 versions |

Slides that are mostly pictures (worked examples) were transcribed by hand into JSON so they can be searched.

## How it works
1. **Ingest:** separate parsers for the textbook (section-aware chunks, figure and Gantt-chart crops, links from passages to the figures they mention), slide decks (consecutive slides with the same title are grouped, and every slide is rendered as an image), and lab documents (split at section headings with line breaks kept so scripts stay readable; the .doc file is read directly without Word).
2. **Index:** BGE embeddings stored in ChromaDB, with an embedding cache so a rebuild only embeds new text. A full build takes about 45 minutes on a laptop CPU.
3. **Retrieval:** hybrid search, combining BM25 keyword search and BGE embedding search with Reciprocal Rank Fusion. Teacher material gets a 1.15× score boost. Related figures are attached, balanced between textbook and slides.
4. **Answer:** the LLM sees the top 6 passages with tags [S1]…[S6] plus strict rules:
   - answer only from the sources and cite every fact;
   - prefer the teacher's conventions where they differ from the book;
   - give exam-style answers that end with "one common mistake to avoid".

   It keeps the last 3 turns, so follow-up questions work.

## Exact solvers (the main feature that sets it apart)
Numericals are computed in code. The LLM only explains the computed result and is told never to recompute it.
- **CPU scheduling:** FCFS, SJF, SRTF, priority (preemptive and non-preemptive) and Round Robin, including I/O bursts. It produces the Gantt chart, the CT/TAT/WT/RT table and the reason for every scheduling decision. It follows the teacher's tie-breaking and Round Robin queueing rules, and matches all 15 of the teacher's worked examples.
- **Banker's algorithm:** Need matrix, safety algorithm, resource-request algorithm and deadlock detection. Accepts Max or Need, and Available or total instances, written in several table layouts. Matches the slides' examples and the textbook's.
- **Page replacement:** FIFO, LRU and Optimal, with a frame table, fault and hit counts, and Belady's anomaly when several frame counts are given. Matches the slides (15 / 12 / 9 faults).
- **Disk scheduling:** FCFS, SSTF, SCAN, C-SCAN, LOOK and C-LOOK, with a head-movement chart. Assumptions (direction, disk size) are stated when the question doesn't give them. Matches the textbook example and all six answers of a textbook exercise.
- **In chat:** typed or pasted numericals are detected automatically. Follow-ups like "now with quantum 2", "what if P4 requests (0,0,2,0)" or "now with 4 frames" re-run the solver. "Explain X with an example" solves the teacher's own worked example from the slides in code.

## App (3 pages)
- **Chat:** cited answers, figures and slide images, the solver output shown above the explanation, and dark mode.
- **Numerical solver:** forms for all four solvers (pre-filled with slide examples), visual output, and an optional "Explain this step by step" button that uses the LLM.
- **Quiz:**
  - Question types: MCQs and viva (typed-answer) questions written by the LLM from retrieved passages; numerical questions with answer keys computed by the solvers (some as computed multiple choice, e.g. "which is a safe sequence?"); and a small built-in MCQ bank for offline use.
  - 22 topics, each with a level from 1 to 3: two correct answers in a row move it up, a wrong answer moves it down, and weaker topics are asked more often.
  - The LLM marks viva answers 0–2 with feedback, and the app shows a per-topic progress table with "Revise:" suggestions. Progress is saved to a JSON file.

## Evaluation
- **Retrieval:** a hand-labelled set of 78 questions, comparing keyword-only, embedding-only and hybrid search.
  - Hybrid, all questions: the right passage is first for 71.8%, in the top 3 for 93.6%, and in the top 6 for 98.7% (MRR 0.83).
  - Linux and shell questions: 80% first, 95% top 3, 100% top 6.
  - Known weakness: on CPU scheduling questions, embedding-only search gets 80% first, but the hybrid only 53% (keyword search pulls the best passage down). A weighted fusion would likely fix this.
- **Tests:** three suites covering solver correctness against the slide and textbook examples, text parsing, quiz logic and follow-ups. All pass.

## Known gaps and to-dos
- The written project report still describes an old prototype and needs rewriting.
- No lab-exercise generator yet (the problem statement mentions "automatic generation of labs").
- No past question papers in the knowledge base (optional in the problem statement).
- Using bge-base rather than bge-large (switching is one config line, but means a ~45-minute rebuild).
- The built-in offline MCQ bank is small (scheduling, processes, Linux and shell only).

## Ideas already on the list (please suggest beyond or improve on these)
1. "Check my working": the student enters their own Gantt chart, safe sequence or fault table, and the solver finds the first wrong step.
2. Step-by-step animations of every solver (Gantt chart filling in, frames being replaced, disk head moving, Work growing in the Banker's algorithm).
3. Timed mock-exam mode with a report card and the slide pages to revise.
4. Lab-exercise generator from the shell notes, plus a checker for common bash mistakes.
5. Hint mode (Socratic questions instead of direct answers).
6. "What-if" comparison of all CPU-scheduling algorithms on one input.
7. Tracking specific misconceptions (e.g. "confuses WT and TAT"), not just weak topics.

## What I'd like from you
Suggest unique, demo-worthy features that would make this project stand out from other student RAG/LLM projects. They should be feasible locally in Python/Streamlit by one student, fit OS teaching, and stay within the decisions above. For each, give the idea, why it stands out, and a rough implementation sketch.
