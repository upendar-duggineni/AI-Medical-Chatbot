"""Tools the agent can use.

Plain functions that can be tested independently and offered to a Gemini
function-calling loop. Risk assessment uses transparent rules, and medical
content comes from retrieved MedlinePlus text.
"""

import os
import re

from . import safety
from .pipeline import min_coverage, retrieval_mode
from .router import SYMPTOM_TERMS, route_question
from .search import get_index, retrieve

SEVERITIES = ("mild", "moderate", "severe")
AGE_GROUPS = ("infant", "child", "adult", "older_adult")
RISK_RANK = {
    "self_care": 0,
    "see_doctor": 1,
    "see_doctor_soon": 2,
    "emergency": 3,
}

NEXT_STEPS = {
    "emergency": "Call 112 or go to the nearest emergency department now.",
    "see_doctor_soon": (
        "Please see a doctor today, or visit a nearby clinic or hospital. "
        "If things suddenly get worse, call 112."
    ),
    "see_doctor": (
        "Consider booking an appointment with a doctor in the next few days."
    ),
    "self_care": (
        "This is general information. See a doctor if symptoms last more "
        "than a few days, get worse, or you are worried."
    ),
}


# ---------------------------------------------------------------------------
# Safety and retrieval tools
# ---------------------------------------------------------------------------

def check_red_flags(text: str) -> dict:
    """Emergency, crisis, and medication-dose rules."""
    return safety.check(text)


def search_medlineplus(query: str, k: int = 3) -> list:
    """Retrieve relevant MedlinePlus passages, including separate symptom pages for lists."""
    query = re.sub(r"\s+", " ", (query or "")).strip()
    if not query or k <= 0:
        return []

    index = get_index()

    # Check relevance of the whole query. For an explicit symptom list, do not
    # reject the query solely because the combined wording scores poorly: each
    # symptom will be searched separately below.
    full_hits = index.search(query, k=3)
    full_coverage = max(
        (float(hit.get("coverage", 0.0)) for hit in full_hits),
        default=0.0,
    )
    coverage_floor = float(os.getenv("RAG_SYMPTOM_FULL_COVERAGE", "0.30"))
    has_separators = bool(
        re.search(r"[,;]|\band\b", query, flags=re.IGNORECASE)
    )
    if full_coverage < coverage_floor and not has_separators:
        return []

    # Split on explicit separators first (commas, semicolons, and conjunctions).
    parts = [
        part.strip(" .?!,:;")
        for part in re.split(r"\s*(?:,|;|\band\b)\s*", query, flags=re.IGNORECASE)
        if part.strip(" .?!,:;")
    ]

    # Handle short, space-separated symptom lists such as "headache fever fatigue".
    # Avoid splitting ordinary questions/sentences into individual words.
    question_words = {
        "a", "about", "am", "an", "and", "are", "can", "cause", "causes",
        "do", "does", "for", "from", "have", "has", "help", "how", "i",
        "is", "me", "my", "of", "or", "please", "should", "the", "to",
        "was", "what", "when", "where", "which", "why", "with", "you",
    }
    
    # Detect short, space-separated symptom lists without splitting
    # ordinary natural-language questions into individual words.
    if len(parts) == 1:
        words = query.replace(",", " ").split()
        single_word_symptoms = {
            symptom.casefold()
            for symptom in SYMPTOM_TERMS
            if " " not in symptom
        }

        if (
             2 <= len(words) <= 8
             and all(word.casefold() in single_word_symptoms for word in words)
        ):
            parts = words

        


    # Always initialize this variable, including queries with commas.
    is_symptom_list = len(parts) > 1

    # Retrieve results using one consistent path for each query type.
    if not is_symptom_list:
        mode = route_question(query, retrieval_mode())["mode"]

        if mode == "hybrid":
            from .hybrid_search import hybrid_search
            hits = hybrid_search(query, k=max(k * 3, 10))

        elif mode == "semantic":
            from .semantic_search import semantic_search
            hits = semantic_search(query, k=max(k * 3, 10))

        else:
            hits = retrieve(
                query,
                k=max(k * 3, 10),
                split_with=True,
                single_k=10,
            )

    else:
        # Prioritize the best result for each individual symptom.
        hits = []
        seen_titles = set()

        for part in parts:
            if not part:
                continue

            part_hits = index.search(part, k=5)

            if part_hits:
                best = part_hits[0]
                title = str(
                    (best.get("chunk") or {}).get("title", "")
                ).strip()
                key = title.casefold()

                if key and key not in seen_titles:
                    hits.append(best)
                    seen_titles.add(key)

        # Add combined-query results without duplicating existing titles.
        combined_hits = retrieve(
            query,
            k=max(k * 5, 20),
            split_with=True,
            single_k=10,
        )

        for hit in combined_hits:
            title = str(
                (hit.get("chunk") or {}).get("title", "")
            ).strip()
            key = title.casefold()

            if key and key not in seen_titles:
                hits.append(hit)
                seen_titles.add(key)

    # Filter results using the configured minimum coverage.
        # Filter results using the configured minimum coverage.
    threshold = min_coverage()

    eligible = [
        hit
        for hit in hits
        if float(hit.get("coverage", 0.0)) >= threshold
    ]

    # For a single symptom, prefer pages whose title or alias directly
    # matches the query, when such pages are available.
        # Prefer exact topic-title matches for single-symptom queries.
    # If none exist, fall back to the broader title/alias match.
    if not is_symptom_list and eligible:
        from .search import tokenize
        from .synonyms import expansion_groups

        query_terms = set(tokenize(query))

        for _, alternatives in expansion_groups(query):
            query_terms.update(tokenize(alternatives))

        exact_topic_hits = []
        direct_topic_hits = []

        for hit in eligible:
            chunk = hit.get("chunk", {})
            title = str(chunk.get("title", "")).strip()
            title_terms = set(tokenize(title))

            alias_terms = set()
            for alias in chunk.get("also_called", []):
                alias_terms.update(tokenize(alias))

            if title_terms and title_terms.issubset(query_terms):
                exact_topic_hits.append(hit)
            elif query_terms & (title_terms | alias_terms):
                direct_topic_hits.append(hit)

        if exact_topic_hits:
            eligible = exact_topic_hits
        elif direct_topic_hits:
            eligible = direct_topic_hits

    # Deduplicate titles while preserving ranking, then respect requested k.
    selected = []
    seen_titles = set()
    for hit in eligible:
        title = str((hit.get("chunk") or {}).get("title", "")).strip()
        key = title.casefold()
        if key and key not in seen_titles:
            selected.append(hit)
            seen_titles.add(key)
        if len(selected) >= k:
            break

    return selected


def assess_risk(state) -> dict:
    """Apply transparent triage rules and return a next step."""
    level, reasons = "self_care", []

    def raise_to(new_level, why):
        nonlocal level
        reasons.append(why)
        if RISK_RANK[new_level] > RISK_RANK[level]:
            level = new_level

    if state.severity == "severe":
        raise_to(
            "see_doctor_soon",
            "you described the symptoms as severe",
        )

    if state.age_group == "infant":
        raise_to(
            "see_doctor_soon",
            "babies can get worse quickly",
        )

    if (
        state.age_group == "older_adult"
        and state.severity in ("moderate", "severe")
    ):
        raise_to(
            "see_doctor_soon",
            "symptoms in older adults deserve prompt attention",
        )

    if (state.duration_days or 0) >= 7:
        raise_to(
            "see_doctor",
            "symptoms lasting a week or more should be checked",
        )

    if getattr(state, "flags_shown", None):
        raise_to(
            "see_doctor_soon",
            "the symptoms you described include a combination that can sometimes be serious",
        )

    return {
        "level": level,
        "next_step": NEXT_STEPS[level],
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Profile extraction
# ---------------------------------------------------------------------------

_WORD_NUM = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "few": 3,
    "couple of": 2,
    "several": 5,
}

_UNIT_DAYS = {
    "hour": 0,
    "day": 1,
    "week": 7,
    "month": 30,
    "year": 365,
}

_SINCE_DAYS = {
    "yesterday": 1,
    "last night": 1,
    "this morning": 0,
    "morning": 0,
    "last week": 7,
    "last month": 30,
    "last year": 365,
}

_AGE_MONTHS = re.compile(r"\b(\d{1,2})\s*-?\s*months?\s*-?\s*old\b")

_AGE_YEARS = re.compile(
    r"\b(\d{1,3})\s*-?\s*(?:years?|yrs?)\s*-?\s*old\b"
    r"|\bage[d]?\s+(\d{1,3})\b"
    r"|\b(?:i am|i'm|im)\s+(\d{1,3})\b"
    r"(?!\s*(?:days?|weeks?|hours?|months?|years?|percent))"
)

_AGE_WORD = re.compile(r"\b(newborn|infant|baby|toddler)\b")

_NUM = (
    r"(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"few|couple of|several)"
)

_DURATION = re.compile(
    rf"\b{_NUM}\s+(hour|day|week|month|year)s?\b"
    r"(?:\s+(?:now|ago))?"
)

_SINCE = re.compile(
    r"\b(?:since|started|began|from)\s+"
    r"(yesterday|this morning|last night|morning|last week|"
    r"last month|last year|monday|tuesday|wednesday|thursday|"
    r"friday|saturday|sunday)\b"
)

_SEVERE = re.compile(
    r"\b(severe|severely|unbearable|terrible|worst|intense|"
    r"excruciating|very bad|really bad|extreme|awful|horrible)\b"
)

_MODERATE = re.compile(r"\b(moderate|quite bad|pretty bad|bad|painful|annoying)\b")
_MILD = re.compile(r"\b(mild|slight|slightly|minor|little|a bit|bit of)\b")


def _age_group(years):
    if years < 1:
        return "infant"
    if years < 13:
        return "child"
    return "older_adult" if years >= 60 else "adult"


def extract_profile(text: str) -> dict:
    """Extract duration, severity, and age group from free text."""
    t = safety._normalize(text or "")

    prof = {
        "duration": None,
        "duration_days": None,
        "severity": None,
        "age_group": None,
    }

    match = _AGE_MONTHS.search(t)
    if match:
        prof["age_group"] = "infant" if int(match.group(1)) < 12 else "child"
        t = t.replace(match.group(0), " ")
    else:
        match = _AGE_YEARS.search(t)
        if match:
            years = int(next(group for group in match.groups() if group))
            if 0 <= years < 110:
                prof["age_group"] = _age_group(years)
            t = t.replace(match.group(0), " ")
        else:
            match = _AGE_WORD.search(t)
            if match:
                prof["age_group"] = (
                    "child" if match.group(1) == "toddler" else "infant"
                )

    match = _DURATION.search(t)
    if match:
        raw_number = match.group(1)
        number = int(raw_number) if raw_number.isdigit() else _WORD_NUM.get(raw_number, 1)
        prof["duration"] = match.group(0).strip()
        prof["duration_days"] = number * _UNIT_DAYS[match.group(2)]
        t = t.replace(match.group(0), " ")
    else:
        match = _SINCE.search(t)
        if match:
            prof["duration"] = match.group(0).strip()
            prof["duration_days"] = _SINCE_DAYS.get(match.group(1), 3)
            t = t.replace(match.group(0), " ")

    for level, pattern in (
        ("severe", _SEVERE),
        ("moderate", _MODERATE),
        ("mild", _MILD),
    ):
        match = pattern.search(t)
        if match:
            prof["severity"] = level
            t = t.replace(match.group(0), " ")
            break

    prof["cleaned"] = re.sub(r"\s+", " ", t).strip()
    return prof


def apply_profile(state, prof: dict) -> None:
    """Save valid profile values without overwriting existing values with None."""
    if prof.get("severity") in SEVERITIES:
        state.severity = prof["severity"]

    if prof.get("age_group") in AGE_GROUPS:
        state.age_group = prof["age_group"]

    if prof.get("duration"):
        state.duration = str(prof["duration"])[:60]
        days = prof.get("duration_days")
        if not isinstance(days, (int, float)) or days < 0:
            days = extract_profile(state.duration)["duration_days"]
        if days is not None:
            state.duration_days = int(days)
