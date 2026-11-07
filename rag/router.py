
"""Deterministic routing for the Multi-RAG medical chatbot."""

import re

VALID_MODES = {"bm25", "semantic", "hybrid"}

# Phrases commonly used when listing multiple symptoms.
SYMPTOM_TERMS = {
    "headache",
    "fever",
    "fatigue",
    "tiredness",
    "cough",
    "sore throat",
    "nausea",
    "vomiting",
    "dizziness",
    "diarrhea",
    "rash",
    "chest pain",
    "shortness of breath",
    "weakness",
    "body aches",
    "runny nose",
    "congestion",
}


def route_question(question: str, configured_mode: str = "auto") -> dict:
    """Choose a retrieval strategy for a health-related question."""

    configured_mode = (configured_mode or "auto").strip().lower()

    if configured_mode in VALID_MODES:
        return {
            "mode": configured_mode,
            "reason": "Explicit retrieval mode configured.",
        }

    q = (question or "").strip().lower()

    if not q:
        return {
            "mode": "bm25",
            "reason": "Empty question fallback.",
        }

    # Recognize symptom lists even when symptoms are space-separated.
    matched_symptoms = [
        symptom
        for symptom in sorted(SYMPTOM_TERMS, key=len, reverse=True)
        if re.search(
            rf"(?<!\w){re.escape(symptom)}(?!\w)",
            q,
        )
    ]

    if len(matched_symptoms) >= 2:
        return {
            "mode": "hybrid",
            "reason": "Question contains multiple recognized symptoms.",
        }

    # Also recognize comma-, semicolon-, or "and"-separated topic lists.
    
    # Also recognize comma-, semicolon-, or "and"-separated topic lists.
    parts = [
        part.strip()
        for part in re.split(r"[,;]|\band\b", q)
        if part.strip()
    ]

    if len(parts) >= 2 and re.search(r"[,;]|\band\b", q):
        return {
            "mode": "hybrid",
            "reason": "Question appears to contain multiple topics.",
        }

    # Natural-language symptom descriptions benefit from hybrid retrieval.
    symptom_description_patterns = [
        r"\bi feel\b",
        r"\bi'm feeling\b",
        r"\bi am feeling\b",
        r"\bi've been feeling\b",
        r"\bi have been feeling\b",
        r"\bi am having\b",
        r"\bi'm having\b",
        r"\bi have\b",
        r"\bi've got\b",
        r"\bi am experiencing\b",
        r"\bi've been experiencing\b",
    ]

    if any(re.search(pattern, q) for pattern in symptom_description_patterns):
        return {
            "mode": "hybrid",
            "reason": "Natural-language symptom description benefits from hybrid retrieval.",
        }

    # Short symptom descriptions may omit phrases such as "I feel".
    short_symptom_phrases = {
        "tired",
        "unusually tired",
        "feeling tired",
        "very tired",
        "extremely tired",
        "fatigued",
        "feeling fatigued",
        "weak",
        "feeling weak",
        "dizzy",
        "feeling dizzy",
    }

    if q in short_symptom_phrases:
        return {
            "mode": "hybrid",
            "reason": "Short symptom description benefits from hybrid retrieval.",
        }

    semantic_patterns = [
        r"\bwhat could be causing\b",
        r"\bwhy do i\b",
        r"\bwhy am i\b",
        r"\bwhat does it feel like\b",
        r"\bin simple words\b",
        r"\bexplain like\b",
        r"\bfeels like\b",
        r"\bwhat might be wrong\b",
    ]

    if any(re.search(pattern, q) for pattern in semantic_patterns):
        return {
            "mode": "semantic",
            "reason": (
                "Natural-language phrasing may benefit from semantic retrieval."
            ),
        }

    return {
        "mode": "bm25",
        "reason": "Default keyword retrieval strategy.",
    }
