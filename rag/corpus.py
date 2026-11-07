"""Load MedlinePlus topics and split summaries into searchable chunks."""

import json
import os
from pathlib import Path

# Optional override through an environment variable.
DEFAULT_DATA_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "processed"
    / "medlineplus_topics.jsonl"
)


def _as_list(value):
    """Normalise a field that may be None, a string, or a list of strings."""
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v]
    return []


def load_topics():
    """Load health topics from the JSONL file."""
    data_file = Path(
        os.getenv("RAG_DATA_FILE", str(DEFAULT_DATA_FILE))
    )

    if not data_file.exists():
        raise FileNotFoundError(
            f"MedlinePlus data not found at: {data_file}. "
            "Place medlineplus_topics.jsonl in data/processed/."
        )

    topics = []

    with data_file.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                topic = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {data_file}"
                ) from exc

            if not isinstance(topic, dict):
                continue

            title = str(topic.get("title") or "").strip()
            url = str(topic.get("url") or "").strip()
            summary = str(topic.get("summary") or "").strip()

            if title and url and summary:
                topic["title"] = title
                topic["url"] = url
                topic["summary"] = summary
                topic["also_called"] = _as_list(topic.get("also_called"))
                topics.append(topic)

    if not topics:
        raise ValueError(f"No valid topics found in {data_file}")

    return topics


def build_chunks(topics, max_words=180, overlap_words=30):
    """Split summaries into overlapping chunks while retaining source metadata."""
    chunks = []

    if max_words < 1 or overlap_words < 0 or overlap_words >= max_words:
        raise ValueError(
            "max_words must be positive and overlap_words must be "
            "non-negative and smaller than max_words."
        )

    step = max_words - overlap_words

    for topic in topics:
        summary = str(topic.get("summary") or "").strip()
        words = summary.split()

        if not words:
            continue

        for start in range(0, len(words), step):
            text = " ".join(words[start:start + max_words])

            if not text:
                continue

            chunks.append({
                "topic_id": str(topic.get("id") or topic["title"]),
                "title": topic["title"],
                "url": topic["url"],
                "also_called": _as_list(topic.get("also_called")),
                "text": text,
            })

            if start + max_words >= len(words):
                break

    return chunks