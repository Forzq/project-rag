from __future__ import annotations

import math
from typing import Protocol


class TextEmbedder(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


def is_markdown_heading(line: str) -> bool:
    return line.lstrip().startswith("#")


def is_heading_only_unit(unit: str) -> bool:
    lines = [line.strip() for line in unit.splitlines() if line.strip()]
    return bool(lines) and all(is_markdown_heading(line) for line in lines)


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

        previous_non_empty = next(
            (previous.strip() for previous in reversed(current) if previous.strip()),
            "",
        )
        starts_new_heading = is_markdown_heading(stripped) and not is_markdown_heading(
            previous_non_empty
        )
        starts_new_unit = not in_code_block and current and (
            starts_new_heading or stripped.startswith("<!-- Page ")
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
    force_heading_breaks: bool = True,
) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    for index, unit in enumerate(units):
        extra_length = len(unit) + (2 if current else 0)
        should_break_by_size = current and current_length + extra_length > max_chars
        should_break_by_meaning = index > 0 and distances[index - 1] >= threshold
        should_break_by_heading = (
            force_heading_breaks and current and unit.lstrip().startswith("#")
        )
        should_keep_heading_with_text = (
            current
            and len(current) == 1
            and is_heading_only_unit(current[0])
            and not unit.lstrip().startswith("#")
        )

        if should_keep_heading_with_text:
            should_break_by_meaning = False

        if current and (
            should_break_by_size or should_break_by_meaning or should_break_by_heading
        ):
            chunks.append("\n\n".join(current))
            current = [unit]
            current_length = len(unit)
        else:
            current.append(unit)
            current_length += extra_length

    if current:
        chunks.append("\n\n".join(current))

    return [chunk for chunk in chunks if chunk]


def merge_short_chunks(
    chunks: list[str],
    min_chars: int = 300,
    max_chars: int = 2500,
) -> list[str]:
    if min_chars <= 0:
        raise ValueError("min_chars must be greater than zero.")

    if max_chars < min_chars:
        raise ValueError("max_chars must be greater than or equal to min_chars.")

    merged_chunks: list[str] = []

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        if not merged_chunks:
            merged_chunks.append(chunk)
            continue

        previous = merged_chunks[-1]
        joined = f"{previous}\n\n{chunk}"

        if len(previous) < min_chars or len(chunk) < min_chars:
            if len(joined) <= max_chars:
                merged_chunks[-1] = joined
                continue

        merged_chunks.append(chunk)

    if len(merged_chunks) >= 2 and len(merged_chunks[-1]) < min_chars:
        joined = f"{merged_chunks[-2]}\n\n{merged_chunks[-1]}"
        if len(joined) <= max_chars:
            merged_chunks[-2] = joined
            merged_chunks.pop()

    return merged_chunks


def split_semantic(
    text: str,
    max_chars: int,
    embedder: TextEmbedder,
    break_percentile: float,
    force_heading_breaks: bool = True,
    min_chunk_chars: int = 300,
) -> list[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero.")

    if not 0 <= break_percentile <= 100:
        raise ValueError("break_percentile must be between 0 and 100.")

    units = split_markdown_units(text)

    if len(units) <= 1:
        return merge_short_chunks(units, min_chars=min_chunk_chars, max_chars=max_chars)

    embeddings = embedder.embed_texts(units)
    distances = [
        1 - cosine_similarity(left, right)
        for left, right in zip(embeddings, embeddings[1:])
    ]
    threshold = percentile(distances, break_percentile)

    chunks = build_semantic_chunks(
        units=units,
        distances=distances,
        threshold=threshold,
        max_chars=max_chars,
        force_heading_breaks=force_heading_breaks,
    )
    return merge_short_chunks(
        chunks,
        min_chars=min_chunk_chars,
        max_chars=max(max_chars, min_chunk_chars),
    )
