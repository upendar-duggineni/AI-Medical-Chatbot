"""Question-answering pipeline with optional hybrid retrieval.

BM25 remains the default. Enable hybrid retrieval with:
    RAG_RETRIEVAL_MODE=hybrid

Safety checks always run before retrieval. Hybrid results must still pass the
existing lexical coverage threshold before being sent to answer generation.
"""

import os
import re

from . import safety
from .answer import generate
from .search import retrieve, split_items  # split_items retained for compatibility


ATTRIBUTION = (
    "Information from MedlinePlus.gov, U.S. National Library of Medicine."
)

DISCLAIMER = (
    "General health information only, not medical advice or a diagnosis. "
    "In an emergency call 112."
)


NO_SOURCE = (
    "I'm sorry you're dealing with this. I couldn't find a sufficiently "
    "relevant passage in my MedlinePlus health information to give you a "
    "reliable, source-backed answer right now.\n\n"
    "I can still help you think through the next safe step. If you tell me "
    "the main symptom, how long it has been happening, and whether it is "
    "getting better or worse, I can try a more focused search.\n\n"
    "If symptoms are severe, worsening, or have persisted for a week, "
    "please contact a healthcare professional promptly. If there is "
    "trouble breathing, severe chest pain, confusion, or another "
    "life-threatening symptom, seek emergency medical care."
)


OUT_OF_SCOPE_MESSAGE = (
    "I can help with general health information using MedlinePlus sources, "
    "but I can't answer unrelated questions. Please ask a health-related question."
)


def min_coverage() -> float:
    """Minimum lexical query coverage required before generating an answer."""
    return float(os.getenv("RAG_MIN_COVERAGE", "0.40"))


def retrieval_mode() -> str:
    """Return the configured retrieval strategy."""
    mode = os.getenv("RAG_RETRIEVAL_MODE", "auto").strip().lower()
    return mode if mode in {"auto", "bm25", "semantic", "hybrid"} else "bm25"

def clearly_out_of_scope(question: str) -> bool:
    """Reject a small set of clearly unrelated requests before retrieval."""
    patterns = [
        r"\bweather forecast\b",
        r"\bwhat(?:'s| is) the weather\b",
        r"\btell me a joke\b",
        r"\b(?:write|tell) (?:me )?a joke\b",
        r"\bsports scores\b",
        r"\bstock prices\b",
        r"\bfootball match results\b",
        r"\b(?:cricket|football|soccer|basketball|tennis) match (?:score|result|results)\b",
        r"\bwho won (?:the )?(?:cricket|football|soccer|basketball|tennis) match\b",
        r"\bwrite (?:a )?(?:python|javascript|java) code\b",
        r"\bdebug (?:a )?(?:python|javascript|java) code\b",
    ]
    q = (question or "").lower().strip()
    return any(re.search(pattern, q) for pattern in patterns)


def clearly_nonmedical_howto(question: str) -> bool:
    """Detect clearly unrelated practical repair questions."""
    q = (question or "").lower().strip()
    patterns = [
        r"\b(?:bicycle|bike)\b.*\b(?:repair|fix|tyre|tire|puncture|tube)\b",
        r"\b(?:repair|fix|patch|change|replace)\b.*\b(?:bicycle|bike)\b",
        r"\bhow do i repair a (?:bicycle|bike)\b",
    ]
    return any(re.search(pattern, q) for pattern in patterns)


def _result(mode, answer, sources=None, debug=None):
    sources = sources or []
    out = {
        "mode": mode,
        "answer": answer,
        "sources": sources,
        "attribution": ATTRIBUTION,
        "disclaimer": DISCLAIMER,
    }
    if os.getenv("RAG_DEBUG", "").lower() == "true" and debug is not None:
        out["debug"] = debug
    return out


def _split_symptom_list(question: str) -> list[str]:
    """Compatibility wrapper; the splitting logic lives in search.split_items."""
    return split_items(question)



def _retrieve(question: str, k: int = 5) -> list[dict]:
    """Retrieve candidates using the configured or automatically selected mode."""
    from .router import route_question

    configured_mode = retrieval_mode()
    route = route_question(question, configured_mode)
    selected_mode = route["mode"]

    if selected_mode == "hybrid":
        from .hybrid_search import hybrid_search
        hits = hybrid_search(question, k=k)

    elif selected_mode == "semantic":
        from .semantic_search import semantic_search
        hits = semantic_search(question, k=k)

    else:
        hits = retrieve(question, k=k, single_k=3)

    # Preserve routing metadata for debugging.
    for hit in hits:
        hit["selected_retrieval_mode"] = selected_mode
        hit["routing_reason"] = route["reason"]

    return hits


def ask(question: str, use_llm: bool = True) -> dict:
    """Run safety checks, retrieve evidence, and generate a grounded response."""
    q = (question or "").strip()

    if not q:
        return _result("no_source", NO_SOURCE)

    # 1. Safety checks always run before retrieval or answer generation.
    verdict = safety.check(q)

    if verdict["action"] == "emergency":
        return _result("emergency", verdict["message"])

    if verdict["action"] == "dose":
        return _result("dose_refused", verdict["message"])

    flags = verdict.get("red_flags", [])

    def _r(mode, answer, sources=None, debug=None):
        out = _result(mode, answer, sources, debug)
        out["red_flags"] = flags
        return out

    # 2. Reject clearly unrelated questions.
    if clearly_out_of_scope(q):
        return _r("out_of_scope", OUT_OF_SCOPE_MESSAGE)

    if clearly_nonmedical_howto(q):
        return _r("no_source", NO_SOURCE)

    # 3. Retrieve candidates. Safety has already been checked above.
    hits = _retrieve(q, k=5)

    debug = [
        {
            "title": hit.get("chunk", {}).get("title", ""),
            "score": hit.get("score", 0.0),
            "coverage": hit.get("coverage", 0.0),
            "retrieval_methods": hit.get("retrieval_methods", []),
        }
        for hit in hits
    ]

    threshold = min_coverage()

    # Semantic similarity alone is not sufficient evidence for a medical answer.
    # Every accepted result must retain adequate lexical query coverage.
    if not hits or max(
        (float(hit.get("coverage", 0.0)) for hit in hits),
        default=0.0,
    ) < threshold:
        return _r("no_source", NO_SOURCE, debug=debug)

    # RRF scores are rank-fusion scores, not BM25 scores, so do not compare them
    # using the ordinary BM25 "half of best score" filter.
    fused = any(
        "matched_queries" in hit or "retrieval_methods" in hit
        for hit in hits
    )
    best_score = max(float(hit.get("score", 0.0)) for hit in hits)

    good = [
        hit
        for hit in hits
        if float(hit.get("coverage", 0.0)) >= threshold
        and (
            fused
            or best_score <= 0
            or float(hit.get("score", 0.0)) >= 0.5 * best_score
        )
    ]

    if not good:
        return _r("no_source", NO_SOURCE, debug=debug)

    # 4. Generate an answer using only retained evidence passages.
    answer_text, mode = generate(
        q,
        good,
        verdict.get("personal", False),
        use_llm,
    )

    if mode == "no_answer" or not answer_text or not answer_text.strip():
        return _r("no_source", NO_SOURCE, debug=debug)

    # 5. Keep the existing API source metadata format.
    sources = [
        {
            "n": i,
            "title": hit["chunk"]["title"],
            "url": hit["chunk"]["url"],
        }
        for i, hit in enumerate(good, start=1)
    ]

    return _r(mode, answer_text.strip(), sources=sources, debug=debug)
