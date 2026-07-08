from __future__ import annotations

import re
from typing import Any


COVER_PATTERNS = [
    "book cover",
    "обложка книги",
    "росмэн",
]
TITLE_ONLY_RE = re.compile(
    r"^\s*(?:#+\s*)?(?:гарри поттер|дж\.\s*к\.\s*ролинг|j\.\s*k\.\s*rowling)",
    re.IGNORECASE,
)


def is_heading_only_text(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if not lines:
        return True

    return all(line.startswith("#") for line in lines)


def is_cover_or_title_chunk(text: str) -> bool:
    normalized = text.lower()

    if any(pattern in normalized for pattern in COVER_PATTERNS):
        return True

    lines = [line.strip(" #") for line in text.splitlines() if line.strip()]
    if len(lines) <= 4 and any(TITLE_ONLY_RE.match(line) for line in lines):
        return True

    return False


def is_low_quality_chunk(
    text: str,
    metadata: dict[str, Any] | None = None,
    min_characters: int = 150,
) -> bool:
    return get_low_quality_reason(text, metadata, min_characters) is not None


def get_low_quality_reason(
    text: str,
    metadata: dict[str, Any] | None = None,
    min_characters: int = 150,
) -> str | None:
    metadata = metadata or {}
    stripped_text = text.strip()
    characters = int(metadata.get("characters") or len(stripped_text))

    if not stripped_text:
        return "empty"

    if is_cover_or_title_chunk(stripped_text):
        return "cover_or_title"

    if is_heading_only_text(stripped_text):
        return "heading_only"

    if characters < min_characters:
        chunk_id = int(metadata.get("chunk_id") or 0)
        index = int(metadata.get("index") or 0)

        if chunk_id <= 5 or index <= 4:
            return "short_initial_chunk"

    return None


def filter_quality_chunks(
    chunks: list[dict[str, Any]],
    min_characters: int = 150,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept_chunks: list[dict[str, Any]] = []
    removed_chunks: list[dict[str, Any]] = []

    for chunk in chunks:
        text = str(chunk.get("text", ""))

        if is_low_quality_chunk(text, chunk, min_characters=min_characters):
            removed_chunks.append(chunk)
        else:
            kept_chunks.append(chunk)

    return kept_chunks, removed_chunks
