
"""Semantic retrieval over the existing MedlinePlus chunks."""

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .corpus import build_chunks, load_topics
from .search import coverage_by_topic


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Override with SEMANTIC_CACHE_DIR if needed.
CACHE_DIR = Path(
    os.getenv("SEMANTIC_CACHE_DIR", "data/semantic_cache")
)
CACHE_FILE = CACHE_DIR / "semantic_index.npz"


def _build_documents(chunks: list[dict]) -> list[str]:
    """Build the exact text representation used for embeddings."""
    return [
        f"Title: {chunk['title']}\n"
        f"Aliases: {'; '.join(chunk.get('also_called', []))}\n"
        f"Content: {chunk['text']}"
        for chunk in chunks
    ]


def _cache_key(documents: list[str]) -> str:
    """Identify the model and exact corpus used to build the index."""
    digest = hashlib.sha256()
    digest.update(MODEL_NAME.encode("utf-8"))

    for document in documents:
        digest.update(b"\0")
        digest.update(document.encode("utf-8"))

    return digest.hexdigest()


@lru_cache(maxsize=1)
def _load_semantic_index():
    """Load cached embeddings or build and persist the semantic index."""
    chunks = build_chunks(load_topics())

    if not chunks:
        raise ValueError("No medical chunks available for semantic search.")

    documents = _build_documents(chunks)
    key = _cache_key(documents)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(MODEL_NAME)

    if CACHE_FILE.exists():
        try:
            with np.load(CACHE_FILE, allow_pickle=False) as cached:
                cached_key = str(cached["cache_key"].item())
                embeddings = cached["embeddings"].astype(np.float32)

            if (
                cached_key == key
                and embeddings.ndim == 2
                and embeddings.shape[0] == len(chunks)
                and embeddings.shape[1] == model.get_embedding_dimension()
                and np.isfinite(embeddings).all()
            ):
                print("Loaded cached semantic embeddings.")
                return model, chunks, embeddings

            print("Semantic cache is outdated; rebuilding embeddings.")

        except (OSError, ValueError, KeyError, EOFError):
            print("Semantic cache is invalid; rebuilding embeddings.")

    print(f"Building semantic index for {len(chunks)} chunks...")

    embeddings = model.encode(
        documents,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    # Write atomically so an interrupted save doesn't leave a partial cache.
    temp_file = CACHE_DIR / "semantic_index.tmp.npz"

    np.savez_compressed(
        temp_file,
        cache_key=np.array(key),
        embeddings=embeddings,
    )
    temp_file.replace(CACHE_FILE)

    print(f"Saved semantic embeddings to: {CACHE_FILE}")

    return model, chunks, embeddings



def semantic_search(question: str, k: int = 5) -> list[dict]:
    """Return semantic candidates with independently calculated lexical coverage."""
    question = (question or "").strip()
    if not question or k <= 0:
        return []

    model, chunks, embeddings = _load_semantic_index()

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)[0]

    scores = embeddings @ query_embedding
    top_indices = np.argsort(scores)[::-1]

    # Use the existing BM25 coverage calculation; semantic similarity alone
    # must not satisfy the pipeline's evidence gate.
    lexical_coverage = coverage_by_topic(question, k=50)

    results = []
    seen_topics = set()

    for idx in top_indices:
        chunk = chunks[int(idx)]
        topic = chunk.get("topic_id") or chunk.get("title") or ""
        topic_key = str(topic).strip().casefold()

        if not topic_key or topic_key in seen_topics:
            continue

        seen_topics.add(topic_key)

        results.append({
            "chunk": chunk,
            "score": round(float(scores[idx]), 4),
            "coverage": lexical_coverage.get(topic_key, 0.0),
            "retrieval_method": "semantic",
        })

        if len(results) >= k:
            break

    return results
