# Retrieval evaluation (2026-10-23)

## Set-up
- **Questions**: 78 hand-labelled questions in `eval_retrieval.py` (30 CPU scheduling and processes, 28 newer
  topics, 20 Linux and shell). Each label names where the answer lives: textbook section numbers and/or slide,
  page or section-title ranges. The labels were not changed during this work.
- **Metrics**: Hit@1, Hit@3, Hit@6 (a relevant passage among the first k of the 6 passages given to the LLM) and
  MRR (mean of 1 / rank of the first relevant passage; 0 if none in the top 6).
- **Topics** are derived from each question's labels (`rag/retrieval_eval.py: topic_of`).
- **Controlled comparison**: each question is ranked once by BM25 and once by the embeddings; every variant is
  scored on those same two rankings (`Retriever.rankings` + `Retriever.fuse`), so variants differ only in fusion.
- **Tuning**: every 4th question of each group (19) is held out. A grid of 147 settings was scored on the other
  59. Settings that lowered Hit@3 or Hit@6 there were ruled out; the best MRR won (ties: Hit@1, then the setting
  closest to the baseline). The held-out 19 played no part in the choice.

Reproduce: `python eval_retrieval.py --tune` (full output in `retrieval_eval_output.txt`; each question's ranked
passage ids, scores and source kinds in `retrieval_run.json`).

## What the audit found
| Checked | Result |
|---|---|
| Metrics reproducible? | Yes: `eval_retrieval.py` regenerates them; the baseline was reproduced exactly (71.8 / 93.6 / 98.7, MRR 0.830) after refactoring. |
| Passage ids: BM25 vs ChromaDB | Consistent: 2,222 chunks, 2,222 vectors, no missing or extra ids, no duplicate ids. |
| Query vs document preprocessing | Consistent: BM25 tokenises queries and passages the same way; BGE queries get the BGE query prefix, passages do not (as the model expects). |
| Citations vs context | `[S1]..[S6]` are numbered from the passages actually sent; tags beyond them are removed (tested). |
| **Teacher boost** (confirmed defect) | RRF scores of nearby ranks differ by only a few per cent, so the ×1.15 boost let a teacher slide ranked 3rd and 6th beat a textbook passage ranked **1st by both** rankers. Four of the eight scheduling questions where the hybrid lost rank 1 to the embedding-only ranker were this. |
| **Back matter** (confirmed) | ~260 textbook passages are exercises, programming projects, bibliography and further reading. They name topics without explaining them and reached rank 1 (e.g. "Further Reading" for *convoy effect*). |
| Keyword over-ranking | Three of the eight losses: BM25 ranked a neighbouring section or an exercise first. Not fixable by fusion weights without losing Hit@3/6 elsewhere. |
| Duplicates | 25% of top-6 slots were a second passage of an already-listed section (overlapping chunks). A per-section cap was tested and did not change the metrics; left off. |
| Source balance | Before: the textbook answer reached the top 6 for 68% of questions labelled with both sources, teacher material for 89%. After: 92% and 86%. |
| Intent rule bug (fixed) | "how does aging **solve** it" was read as a calculation; "solve / calculate / find" now count only as commands. |

## Results
| Variant (all 78 questions) | Hit@1 | Hit@3 | Hit@6 | MRR |
|---|---|---|---|---|
| A. BM25 only | 51.3% | 83.3% | 91.0% | 0.674 |
| B. Embeddings only | 78.2% | 89.7% | 93.6% | 0.842 |
| C. **Baseline** RRF 1:1, teacher boost 1.15 | 71.8% | 93.6% | 98.7% | 0.830 |
| D. Embedding-heavy (BM25 weight 0.5) | 75.6% | 94.9% | 97.4% | 0.843 |
| E. BM25-heavy (embedding weight 0.5) | 71.8% | 94.9% | 98.7% | 0.825 |
| G. Baseline without teacher boost | 71.8% | 87.2% | 98.7% | 0.817 |
| H. No boost, both source kinds in top 2 | 71.8% | 92.3% | 98.7% | 0.827 |
| I. BM25 0.5, no boost, both kinds in top 2 | 76.9% | 92.3% | 97.4% | 0.848 |
| J. Baseline, back matter ×0.5 | 74.4% | 94.9% | 98.7% | 0.846 |
| K. No boost; boost 1.15 only for example questions | 71.8% | 89.7% | 98.7% | 0.825 |
| **F. Selected** (now `config.FUSION`) | **76.9%** | **94.9%** | **98.7%** | **0.858** |

| Held-out 19 questions | Hit@1 | Hit@3 | Hit@6 | MRR |
|---|---|---|---|---|
| Baseline | 84.2% | 100% | 100% | 0.921 |
| Selected | 94.7% | 100% | 100% | 0.965 |

| Topic (n) | Hit@1 baseline → selected | MRR baseline → selected |
|---|---|---|
| CPU scheduling (27) | 48% → **63%** | 0.691 → 0.775 |
| Synchronization (5) | 40% → 60% | 0.567 → 0.717 |
| Storage and file systems (7) | 86% → 100% | 0.929 → 1.000 |
| Memory and page replacement (9) | 100% → 100% | 1.000 → 1.000 |
| Processes and threads (7) | 100% → 100% | 1.000 → 1.000 |
| Linux and shell (20) | 80% → 75% | 0.887 → 0.846 |
| Deadlocks and Banker's (3) | 100% → 67% | 1.000 → 0.833 |

## Selected configuration
`bm25_weight 1.0, dense_weight 1.0, rrf_k 60, teacher_boost 1.05, example_teacher_boost 1.15,
back_matter_weight 0.5` (per-section cap and kind slot off).
- The teacher boost stays (without it Hit@3 falls to 87%), but at 1.05 it can no longer override a passage
  both rankers put clearly ahead. For example and numerical questions (`rag/intent.py`) teacher material gets
  ×1.15 more, because the course's worked examples and scripts are there.
- Back-matter passages count half.
- It is the simplest setting that improves every overall metric and the held-out set: the same 1:1 fusion,
  two re-weightings.

## Limitations
- 78 questions: a change of one question moves a topic's Hit@1 by 4–33 points. The small topics (deadlocks 3,
  synchronization 5) are noise-level; the overall and held-out figures are the meaningful ones.
- CPU-scheduling Hit@1 (63%) did not reach the embedding-only 78%: embeddings alone win rank 1 more often but
  miss the answer entirely more often (Hit@6 85% vs 96%). The remaining misses are mostly adjacent sections or a
  related slide range at rank 1 with the labelled one at rank 2 (e.g. 5.8 vs 5.2, slides 72–74 vs 78–85).
- Linux and Deadlocks lose one question each at rank 1 (the answer moves to rank 2).
- One question is still not found in the top 6 ("What happens to a process when its time slice ends?").
- "What is the convoy effect?" (the app's first example question) ranks its answer 3rd (the baseline
  ranked it 4th). Keyword search has the right passages 1st and 2nd, but the embedding model does not put
  them in its top 50, and RRF gives nothing for a missing ranking, so passages ranked moderately by both
  (4.2.1 Programming Challenges, 9.3.2) score higher. A smaller RRF constant (k = 10) fixes this question but
  lowers Hit@3/Hit@6 elsewhere, so it was not adopted. The right passages are still in the six given to the
  LLM, but figures (taken from the top two passages) are not shown for this question.
- The grid had many exact ties (e.g. back-matter weight 0 and 0.5 score the same); the tie-break picks the
  setting closest to the baseline.

## Next experiment
A cross-encoder reranker over the top ~20 fused candidates is the most likely way to separate adjacent
sections. It needs a second model download (about 100 MB–1 GB), so it was not added; evaluate it with
`eval_retrieval.py` before adopting it. Switching to bge-large is not indicated by this evaluation: the
embedding ranking is already the stronger signal, and the losses are in fine ordering, not recall.
