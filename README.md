# OSmosis

OSmosis is a tutor for an Operating Systems course. It does four things:

- **Answers questions from the course material.** It uses hybrid retrieval: BM25 keyword search plus BGE embeddings in ChromaDB, combined with reciprocal rank fusion. An LLM then writes an answer that cites its sources.
- **Solves numerical problems exactly in code.** Supported: CPU scheduling, the Banker's algorithm, page replacement and disk scheduling.
- **Finds the first wrong step in a student's working.** This is the Reasoning Debugger.
- **Lets students step through algorithms** in OS Lab, and tracks recurring misconceptions on a learning map.

- How to install, build the knowledge base and run the app: [README_RAG.md](README_RAG.md)
- Project report and slides: [deliverables/](deliverables/)
- Retrieval evaluation: [docs/retrieval_evaluation.md](docs/retrieval_evaluation.md)

## Not included in this repository

The course material is **not** in this repository, because it is copyrighted:

- the Silberschatz *Operating System Concepts* (10th ed.) PDF
- the teacher's lecture slides
- the lab handouts
- the knowledge base built from them (`data/`), including the hand transcriptions in `data/annotations/`

To run OSmosis, put your own copies of these files in the project folder and build the index with `python -m rag.ingest` and `python -m rag.build_index`. [README_RAG.md](README_RAG.md) gives the exact file names.

The LLM is any OpenAI-compatible endpoint, set through the environment variables `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL`. No API key is stored in this repository. Without a key, every feature works except written answers.

## Tests

```bash
python -m tests.test_solver
```
The suites need no API key.
