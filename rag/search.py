"""Keyword search (BM25) over MedlinePlus chunks.

This is a lightweight lexical baseline. It improves retrieval scoring, but it is
not semantic search and cannot by itself verify that a source answers a question.
"""
import math
import os
import re
from collections import Counter, defaultdict
from functools import lru_cache

from .corpus import build_chunks, load_topics
from .synonyms import expansion_groups

STOPWORDS = set("""a about after again all also am an and any are as at be because been before being but by can could
did do does doing during each for from had has have having he her here hers him his how i if in into is it its just me
more most my no nor not now of on once only or other our own same she should so some such than that the their them then
there these they this those through to too until very was we were what when where which while who whom why will with
would you your yours""".split())

# These words describe the type of question, but usually don't identify its
# medical topic. They are ignored for matching/coverage, not removed from text.

QUESTION_WORDS = set("""
common sign signs cause causes causing treatment treatments
manage managing management prevent preventing prevention help helpful
information explain list include including
treat treated treating treats cure cured cause caused diagnose diagnosed diagnosing diagnosis
mean means meaning define definition
feeling feel felt time times lot lots keep keeps getting
""".split())


_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def _stem(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("sses"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    if len(w) > 5 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    return w


_RAW_WORD = re.compile(r"[a-z0-9]+")


def letter_pairs(text: str) -> list[str]:
    """Pair a single letter with its neighbour: "Vitamin A" -> "vitamin_a", "B Vitamins" -> "b_vitamins".

    A lone letter is thrown away by tokenize(), but in medical names it is the part that tells
    topics apart (Hepatitis A / B / C, Vitamin A / C / D). Only pairs found in a topic title or
    alias are ever used, so ordinary text like "what is a seizure" is not affected.
    """
    words = _RAW_WORD.findall((text or "").lower())
    pairs = []
    for w1, w2 in zip(words, words[1:]):
        if (len(w1) == 1 and w1.isalpha() and len(w2) >= 3) or (len(w2) == 1 and w2.isalpha() and len(w1) >= 3):
            pairs.append(f"{w1}_{w2}")
    return pairs


def tokenize(text: str) -> list[str]:
    tokens = []
    for w in _WORD.findall((text or "").lower()):
        w = w.split("'")[0]
        if not w or w in STOPWORDS or w in QUESTION_WORDS:
            continue
        if len(w) < 2 and not w.isdigit():
            continue
        tokens.append(_stem(w))
    return tokens


class BM25Index:
    K1, B = 1.5, 0.75

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.letter_phrases = set()  # "vitamin_a", "hepatitis_b", ... found in titles and aliases
        self.tfs = []
        self.lengths = []
        self.postings = defaultdict(list)
        df = Counter()

        for i, c in enumerate(chunks):
            title_pairs = letter_pairs(c.get("title", ""))
            alias_pairs = letter_pairs(" ".join(c.get("also_called", [])))
            self.letter_phrases.update(title_pairs)
            self.letter_phrases.update(alias_pairs)
            tokens = (
                tokenize(c.get("title", "")) * 3
                + title_pairs * 3
                + tokenize(" ".join(c.get("also_called", []))) * 2
                + alias_pairs * 2
                + tokenize(c.get("text", ""))
            )
            tf = Counter(tokens)
            self.tfs.append(tf)
            self.lengths.append(len(tokens))
            for term, count in tf.items():
                self.postings[term].append((i, count))
                df[term] += 1

        # Title and alias terms, used to favour the topic the question is actually about.
        self.title_terms = [
            set(tokenize(c.get("title", ""))) | set(letter_pairs(c.get("title", ""))) for c in chunks
        ]
        self.alias_terms = [
            [set(tokenize(a)) | set(letter_pairs(a)) for a in (c.get("also_called") or [])] for c in chunks
        ]

        self.df = df
        self.n = len(chunks)
        self.avgdl = sum(self.lengths) / self.n if self.n else 0.0
        self.titles = sorted({c.get("title", "") for c in chunks})

    def idf(self, term: str) -> float:
        """Return zero for terms absent from the index."""
        df = self.df.get(term, 0)
        if df == 0 or self.n == 0:
            return 0.0
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def _title_boost(self, i: int, known: set, boost: float = 1.8) -> float:
        """Boost chunks whose topic name is fully named in the question ("diabetes treated" -> Diabetes)."""
        title = self.title_terms[i]
        if title and title <= known:
            return boost
        for alias in self.alias_terms[i]:
            if alias and alias <= known:
                return 1 + (boost - 1) * 0.75
        if title & known:
            return 1.15
        return 1.0

    def search(self, query: str, k: int = 3, title_boost: bool = True) -> list[dict]:
        """Return up to k distinct topics with BM25 score and query coverage.

        Ranking uses a topic-name boost. Coverage is the share of the question's meaningful
        words that a passage satisfies; words that exist nowhere in the knowledge base count
        AGAINST coverage, so an off-topic question that shares one word with a topic is rejected.
        An everyday phrase ("pass urine") counts as satisfied when the passage has any of its
        medical words ("urinary", "bladder").
        """
        if k <= 0 or not self.n:
            return []

        base = list(dict.fromkeys(tokenize(query)))
        pairs = [p for p in letter_pairs(query) if p in self.letter_phrases]
        original = list(dict.fromkeys(base + pairs))

        groups = []  # (everyday-phrase tokens, medical-word alternatives that exist in the index)
        for phrases, words in expansion_groups(query):
            alternatives = [t for t in dict.fromkeys(tokenize(words)) if self.df.get(t, 0) > 0]
            if alternatives:
                groups.append(({t for p in phrases for t in tokenize(p)}, alternatives))

        terms = list(dict.fromkeys(original + [t for _, alt in groups for t in alt]))
        known_terms = [t for t in terms if self.df.get(t, 0) > 0]
        if not known_terms:
            return []
        known = set(known_terms)

        scores = defaultdict(float)
        for term in known_terms:
            idf = self.idf(term)
            for i, tf in self.postings.get(term, ()):
                if self.avgdl <= 0:
                    continue
                denominator = tf + self.K1 * (
                    1 - self.B + self.B * self.lengths[i] / self.avgdl
                )
                scores[i] += idf * tf * (self.K1 + 1) / denominator
        if not scores:
            return []

        # What a passage must contain to be fully covered: one requirement per plain word,
        # one per everyday phrase (satisfied by any of its medical words).
        explained = set().union(*[e for e, _ in groups]) if groups else set()
        requirements = []
        for phrase_tokens, alternatives in groups:
            members = set(alternatives) | {t for t in phrase_tokens if self.df.get(t, 0) > 0}
            requirements.append((members, max(self.idf(t) for t in alternatives)))
        for t in original:
            if self.df.get(t, 0) > 0 and t not in explained:
                requirements.append(({t}, self.idf(t)))
        unknown_count = sum(1 for t in original if self.df.get(t, 0) == 0)
        weights = [w for _, w in requirements]
        if not weights or sum(weights) <= 0:
            return []
        # Each unknown word weighs as much as an average requirement.
        total_idf = sum(weights) + unknown_count * (sum(weights) / len(weights))

        boost = float(os.getenv("RAG_TITLE_BOOST", "1.8"))  # read once, not once per chunk
        candidates = []
        for i, score in scores.items():
            present = self.tfs[i]
            covered = sum(w for members, w in requirements if any(t in present for t in members))
            matched = [term for term in known_terms if term in present]
            factor = self._title_boost(i, known, boost) if title_boost else 1.0
            candidates.append((score * factor, covered / total_idf, i, matched))
        candidates.sort(key=lambda c: c[0], reverse=True)

        hits = []
        seen_topics = set()
        for adjusted_score, coverage, i, matched in candidates:
            chunk = self.chunks[i]
            topic_key = chunk.get("topic_id") or chunk.get("title", "")
            if topic_key in seen_topics:
                continue
            seen_topics.add(topic_key)
            hits.append({
                "chunk": chunk,
                "score": round(adjusted_score, 3),
                "coverage": round(coverage, 3),
                "matched": matched,
            })
            if len(hits) >= k:
                break

        return hits


@lru_cache(maxsize=1)
def get_index() -> BM25Index:
    return BM25Index(build_chunks(load_topics()))


def coverage_by_topic(question: str, k: int = 50) -> dict[str, float]:
    """Return existing BM25 lexical coverage scores keyed by medical topic."""
    if not isinstance(question, str) or not question.strip() or k <= 0:
        return {}

    hits = get_index().search(question.strip(), k=k)
    coverage = {}

    for hit in hits:
        chunk = hit.get("chunk", {})
        topic_key = str(
            chunk.get("topic_id") or chunk.get("title") or ""
        ).strip().casefold()

        if not topic_key:
            continue

        score = float(hit.get("coverage", 0.0))
        coverage[topic_key] = max(coverage.get(topic_key, 0.0), score)

    return coverage


def index_info() -> dict:
    idx = get_index()
    return {
        "chunks": idx.n,
        "topics": len({c.get("topic_id") or c.get("title", "") for c in idx.chunks}),
    }



def search_multiquery(queries: list[str], k: int = 5) -> list[dict]:
    """Merge symptom-query results while preserving whole-query relevance."""
    if k <= 0 or not queries:
        return []

    unique_queries = list(dict.fromkeys(
        q.strip()
        for q in queries
        if isinstance(q, str) and q.strip()
    ))
    if not unique_queries:
        return []

    index = get_index()
    fused = {}
    rank_constant = 60
    candidate_k = max(k, 30)
    whole_query = unique_queries[0].casefold()

    for query in unique_queries:
        query_key = query.casefold()
        is_whole_query = query_key == whole_query
        hits = index.search(query, k=candidate_k)

        for rank, hit in enumerate(hits, start=1):
            chunk = hit.get("chunk") or {}
            title = str(chunk.get("title") or "").strip()
            topic_id = str(chunk.get("topic_id") or title).strip()
            topic_key = topic_id.casefold()

            if not topic_key:
                continue

            item = fused.get(topic_key)
            if item is None:
                item = {
                    **hit,
                    "chunk": chunk,
                    "score": 0.0,
                    "rrf_score": 0.0,
                    "coverage": 0.0,
                    "whole_query_coverage": 0.0,
                    "whole_query_rank": None,
                    "matched_queries": [],
                    "direct_title_match": False,
                }
                fused[topic_key] = item

            # Give the complete symptom description more weight than its fragments.
            weight = 2.0 if is_whole_query else 1.0
            item["rrf_score"] += weight / (rank_constant + rank)
            item["score"] = item["rrf_score"]
            item["coverage"] = max(
                item["coverage"],
                float(hit.get("coverage", 0.0)),
            )

            if is_whole_query:
                item["whole_query_coverage"] = float(
                    hit.get("coverage", 0.0)
                )
                item["whole_query_rank"] = rank

            if query_key not in item["matched_queries"]:
                item["matched_queries"].append(query_key)

            if title.casefold() == query_key:
                item["direct_title_match"] = True

    ranked = sorted(
        fused.values(),
        key=lambda item: (
            item["direct_title_match"],
            item["whole_query_coverage"],
            item["rrf_score"],
            len(item["matched_queries"]),
            item["coverage"],
        ),
        reverse=True,
    )

    return ranked[:k]


# --- shared retrieval entry point ---------------------------------------------------------------
# pipeline.py and tools.py each had their own copy of this logic, and both mixed raw BM25 scores
# (about 1 to 15) with reciprocal-rank-fusion scores (about 0.03). Raw scores always won, so topics
# found only by the combined query (meningitis, flu, dengue ...) sank to the bottom and were then
# dropped by the "score >= 50% of best" filter. Everything now goes through retrieve(), which keeps
# ONE score scale per result list.
_LIST_SPLIT = re.compile(r"\s*(?:,|;|\balong with\b|\band\b)\s*", re.IGNORECASE)
_LIST_SPLIT_WITH = re.compile(r"\s*(?:,|;|\balong with\b|\band\b|\bwith\b)\s*", re.IGNORECASE)


def split_items(text: str, split_with: bool = False) -> list[str]:
    """Split an explicit list ("headache, fever and nausea") into items, without duplicates."""
    pattern = _LIST_SPLIT_WITH if split_with else _LIST_SPLIT
    items, seen = [], set()
    for part in pattern.split(text or ""):
        part = part.strip(" .?!,:;")
        if part and part.casefold() not in seen:
            seen.add(part.casefold())
            items.append(part)
    return items


def retrieve(question: str, k: int = 5, split_with: bool = False, single_k: int = 3) -> list[dict]:
    """One retrieval path for questions and symptom descriptions.

    - A list of items, or (split_with=True) a short unpunctuated symptom phrase, is searched as the whole
      question plus each item, and merged with search_multiquery (fused scores, original weighted higher).
    - Anything else is an ordinary BM25 search. `single_k` is how many hits that returns.
    """
    index = get_index()
    parts = split_items(question, split_with)
    if len(parts) > 1:
        return search_multiquery([question, *parts], k=k)
    if split_with:
        tokens = list(dict.fromkeys(tokenize(question)))
        if 2 <= len(tokens) <= 8:
            return search_multiquery([question, *tokens], k=k)
    return index.search(question, k=single_k)
