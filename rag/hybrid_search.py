
"""Hybrid retrieval combining per-item BM25 and semantic search using RRF."""

import re

from .search import retrieve, search_multiquery, split_items
from .semantic_search import semantic_search


def _looks_like_symptom_list(question: str) -> bool:
    """Detect likely lists without treating every multiword question as a list."""
    q = (question or "").strip().lower()

    # Explicit separators are the strongest signal.
    if re.search(r"[,;]|\band\b|\balong with\b", q):
        return True

    # Recognize common space-separated symptom lists.
    symptom_terms = {
        "headache", "fever", "fatigue", "tiredness", "cough",
        "sore throat", "nausea", "vomiting", "dizziness",
        "diarrhea", "rash", "weakness", "body aches",
        "runny nose", "congestion",
    }

    matches = [
        term for term in symptom_terms
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", q)
    ]
    return len(matches) >= 2


def hybrid_search(
    question: str,
    k: int = 5,
    bm25_k: int = 10,
    semantic_k: int = 10,
    rrf_constant: int = 60,
) -> list[dict]:
    """
    Fuse BM25 and semantic rankings.

    For likely symptom lists, retrieve each item independently. Coverage remains
    lexical evidence metadata; it is not a confidence score or diagnosis.
    """
    if not isinstance(question, str) or not question.strip():
        return []

    if k <= 0 or bm25_k <= 0 or semantic_k <= 0:
        return []

    if rrf_constant < 0:
        raise ValueError("rrf_constant must be >= 0.")

    question = question.strip()

    # Independently retrieve explicit list items to avoid one symptom dominating.
    if _looks_like_symptom_list(question):
        symptom_terms = sorted(
            {
                "abdominal pain", "stomach pain", "belly pain",
                "headache", "fever", "fatigue", "tiredness",
                "cough", "sore throat", "nausea", "vomiting",
                "dizziness", "diarrhea", "diarrhoea", "rash",
                "weakness", "weak", "body aches", "runny nose",
                "congestion", "dehydration", "stomach cramps",
            },
            key=len,
            reverse=True,
        )

        q_lower = question.lower()
        parts = []

        for term in symptom_terms:
            pattern = rf"(?<!\w){re.escape(term)}(?!\w)"
            if re.search(pattern, q_lower):
                parts.append(term)

        bm25_results = search_multiquery(
            [question, *parts],
            k=max(bm25_k, len(parts) + 5),
        )
    else:
        bm25_results = retrieve(
            question,
            k=bm25_k,
            single_k=bm25_k,
        )
    semantic_results = semantic_search(question, k=semantic_k)

    fused = {}
    for method, results in (
        ("bm25", bm25_results),
        ("semantic", semantic_results),
    ):
        for rank, result in enumerate(results, start=1):
            chunk = result.get("chunk", {})
            if not chunk:
                continue

            topic_key = str(
                chunk.get("topic_id") or chunk.get("title") or ""
            ).strip().casefold()

            if not topic_key:
                continue

            if topic_key not in fused:
                fused[topic_key] = {
                    **result,
                    "chunk": chunk,
                    "score": 0.0,
                    "rrf_score": 0.0,
                    "retrieval_methods": [],
                    "lexical_coverage": 0.0,
                    "coverage": 0.0,
                }

            item = fused[topic_key]
            item["rrf_score"] += 1.0 / (rrf_constant + rank)
            item["score"] = item["rrf_score"]

            if method not in item["retrieval_methods"]:
                item["retrieval_methods"].append(method)

            if method == "bm25":
                lexical_coverage = float(result.get("coverage", 0.0))
                item["lexical_coverage"] = max(
                    item["lexical_coverage"], lexical_coverage
                )
                item["coverage"] = item["lexical_coverage"]

    
    ranked = sorted(
        fused.values(),
        key=lambda item: (
            item.get("direct_title_match", False),
            item["rrf_score"],
            item["lexical_coverage"],
            len(item["retrieval_methods"]),
        ),
        reverse=True,
    )

    deduplicated = []
    seen_topics = set()

    for item in ranked:
        chunk = item.get("chunk", {})
        topic_key = str(
            chunk.get("topic_id") or chunk.get("title") or ""
        ).strip().casefold()

        if not topic_key or topic_key in seen_topics:
            continue

        seen_topics.add(topic_key)
        deduplicated.append(item)

        if len(deduplicated) >= k:
            break

    return deduplicated