from __future__ import annotations

import hashlib
from functools import lru_cache

from .config import BGE_QUERY_PREFIX, DATA_DIR, EMBED_MODEL

CACHE_FILE = DATA_DIR / "embed_cache.npz"


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    # First run downloads the model (~440 MB for bge-base) from Hugging Face.
    return SentenceTransformer(EMBED_MODEL)


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Embeddings for `texts`, reusing cached ones so a rebuild only embeds new or changed text
    (embedding is the slow step: tens of minutes for a few thousand passages on a laptop CPU)."""
    import numpy as np

    keys = [hashlib.sha1(f"{EMBED_MODEL}\n{t}".encode("utf-8")).hexdigest() for t in texts]
    cache = {}
    if CACHE_FILE.exists():
        with np.load(CACHE_FILE) as z:
            cache = dict(zip(z["keys"].tolist(), z["vecs"]))
    todo = [i for i, k in enumerate(keys) if k not in cache]
    print(f"[embed] {len(texts) - len(todo)} cached, {len(todo)} to embed", flush=True)
    if todo:
        vecs = _model().encode([texts[i] for i in todo], batch_size=16, normalize_embeddings=True,
                               show_progress_bar=True)
        for i, v in zip(todo, vecs):
            cache[keys[i]] = v
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(CACHE_FILE, keys=np.array(list(cache)), vecs=np.stack(list(cache.values())))
    return [cache[k].tolist() for k in keys]


def embed_query(text: str) -> list[float]:
    prefix = BGE_QUERY_PREFIX if "bge" in EMBED_MODEL.lower() else ""
    return _model().encode([prefix + text], normalize_embeddings=True)[0].tolist()
