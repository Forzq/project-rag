from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import requests
from dotenv import load_dotenv


OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split Markdown into semantic chunks with embeddings."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("md_output/my_document.md"),
        help="Path to the input Markdown file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("chunks_output/semantic_chunks.md"),
        help="Path to the output chunks Markdown file.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2000,
        help="Maximum semantic chunk size in characters.",
    )
    parser.add_argument(
        "--model",
        default="qwen/qwen3-embedding-4b",
        help="OpenRouter embedding model.",
    )
    parser.add_argument(
        "--break-percentile",
        type=float,
        default=80.0,
        help="Similarity-drop percentile used to create semantic breaks.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of text units sent per embeddings request.",
    )
    return parser.parse_args()


def require_openrouter_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. "
            "Add OPENROUTER_API_KEY=sk-or-your-key to your .env file."
        )

    return api_key


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


def embed_texts(
    texts: list[str],
    api_key: str,
    model: str,
    batch_size: int,
) -> list[list[float]]:
    embeddings: list[list[float]] = []
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = requests.post(
            OPENROUTER_EMBEDDINGS_URL,
            headers=headers,
            json={
                "model": model,
                "input": batch,
                "encoding_format": "float",
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()["data"]
        data.sort(key=lambda item: item["index"])
        embeddings.extend(item["embedding"] for item in data)

    return embeddings


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


def build_chunks(
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
    model: str,
    break_percentile: float,
    batch_size: int,
) -> list[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero.")

    if not 0 <= break_percentile <= 100:
        raise ValueError("break_percentile must be between 0 and 100.")

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    units = split_markdown_units(text)

    if len(units) <= 1:
        return units

    api_key = require_openrouter_api_key()
    embeddings = embed_texts(units, api_key, model, batch_size)
    distances = [
        1 - cosine_similarity(left, right)
        for left, right in zip(embeddings, embeddings[1:])
    ]
    threshold = percentile(distances, break_percentile)

    return build_chunks(units, distances, threshold, max_chars)


def write_chunks(chunks: list[str], output_path: Path, method: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parts = [f"# {method} Chunks", ""]

    for index, chunk in enumerate(chunks, start=1):
        parts.extend(
            [
                f"## Chunk {index}",
                "",
                f"Characters: {len(chunk)}",
                "",
                "````markdown",
                chunk,
                "````",
                "",
            ]
        )

    output_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Saved {len(chunks)} semantic chunks to {output_path}")


def main() -> None:
    load_dotenv()
    args = parse_args()
    text = args.input.read_text(encoding="utf-8")
    chunks = split_semantic(
        text=text,
        max_chars=args.max_chars,
        model=args.model,
        break_percentile=args.break_percentile,
        batch_size=args.batch_size,
    )
    write_chunks(chunks, args.output, "Semantic")


if __name__ == "__main__":
    main()
