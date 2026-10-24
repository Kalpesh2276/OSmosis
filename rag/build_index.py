"""Step 2: embed chunks + figure captions with BGE and store them in ChromaDB.

Usage (from the project root, after `python -m rag.ingest`):
    python -m rag.build_index
"""
from __future__ import annotations

import time

from . import embeddings
from .config import CHROMA_DIR, EMBED_MODEL
from .store import figure_search_text, load_chunks, load_figures


def _recreate(client, name):
    try:
        client.delete_collection(name)
    except Exception:
        pass
    return client.create_collection(name, metadata={"hnsw:space": "cosine", "embed_model": EMBED_MODEL})


def main():
    import chromadb

    chunks = load_chunks()
    figures = load_figures()
    print(f"[index] {len(chunks)} chunks, {len(figures)} figures, model={EMBED_MODEL}")

    t0 = time.time()
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    col = _recreate(client, "chunks")
    docs = [f"{c['section']}\n{c['text']}" for c in chunks]
    vecs = embeddings.embed_passages(docs)
    for i in range(0, len(chunks), 256):
        col.add(
            ids=[c["id"] for c in chunks[i:i + 256]],
            embeddings=vecs[i:i + 256],
            documents=docs[i:i + 256],
            metadatas=[{"source_kind": c["source_kind"], "section": c["section"]} for c in chunks[i:i + 256]],
        )

    fcol = _recreate(client, "figures")
    fdocs = [figure_search_text(f) for f in figures]
    fvecs = embeddings.embed_passages(fdocs)
    for i in range(0, len(figures), 256):
        fcol.add(
            ids=[f["id"] for f in figures[i:i + 256]],
            embeddings=fvecs[i:i + 256],
            documents=fdocs[i:i + 256],
            metadatas=[{"source_kind": f["source_kind"]} for f in figures[i:i + 256]],
        )
    print(f"[done] index built in {time.time() - t0:.0f}s -> {CHROMA_DIR}")
    print('Try: python -m rag.ask "Explain round robin with an example"')


if __name__ == "__main__":
    main()
