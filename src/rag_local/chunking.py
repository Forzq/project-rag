from __future__ import annotations

import math
from typing import Protocol


class TextEmbedder(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


def split_fixed_size(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())

        if end == len(text):
            break

        start = end - overlap

    return [chunk for chunk in chunks if chunk]


def split_recursive(text: str, chunk_size: int, overlap: int) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def split_markdown_units(text: str) -> list[str]:
    units: list[str] = []
    current: list[str] = []
    in_code_block = False

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code_block = not in_code_block

        starts_new_unit = (
            not in_code_block
            and current
            and (stripped.startswith("#") or stripped.startswith("<!-- Page "))
        )

        if starts_new_unit:
            units.append("\n".join(current).strip())
            current = []

        if not in_code_block and not stripped:
            if current:
                units.append("\n".join(current).strip())
                current = []
            continue

        current.append(line)

    if current:
        units.append("\n".join(current).strip())

    return [unit for unit in units if unit]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot_product / (left_norm * right_norm)


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 1.0

    sorted_values = sorted(values)
    index = round((percentile_value / 100) * (len(sorted_values) - 1))
    return sorted_values[index]


def build_semantic_chunks(
    units: list[str],
    distances: list[float],
    threshold: float,
    max_chars: int,
) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    for index, unit in enumerate(units):
        extra_length = len(unit) + (2 if current else 0)
        should_break_by_size = current and current_length + extra_length > max_chars
        should_break_by_meaning = index > 0 and distances[index - 1] >= threshold

        if current and (should_break_by_size or should_break_by_meaning):
            chunks.append("\n\n".join(current))
            current = [unit]
            current_length = len(unit)
        else:
            current.append(unit)
            current_length += extra_length

    if current:
        chunks.append("\n\n".join(current))

    return [chunk for chunk in chunks if chunk]


def split_semantic(
    text: str,
    max_chars: int,
    embedder: TextEmbedder,
    break_percentile: float,
) -> list[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero.")

    if not 0 <= break_percentile <= 100:
        raise ValueError("break_percentile must be between 0 and 100.")

    units = split_markdown_units(text)

    if len(units) <= 1:
        return units

    embeddings = embedder.embed_texts(units)
    distances = [
        1 - cosine_similarity(left, right)
        for left, right in zip(embeddings, embeddings[1:])
    ]
    threshold = percentile(distances, break_percentile)

    return build_semantic_chunks(units, distances, threshold, max_chars)

