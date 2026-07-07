from __future__ import annotations

import re
from pathlib import Path
from typing import Any


CHUNK_PATTERN = re.compile(
    r"## Chunk (?P<chunk_id>\d+)\s+"
    r"Characters: (?P<characters>\d+)\s+"
    r"````markdown\s*"
    r"(?P<text>.*?)"
    r"\s*````",
    re.DOTALL,
)


def parse_chunks_file(chunks_file: Path) -> list[dict[str, Any]]:
    text = chunks_file.read_text(encoding="utf-8")
    chunks: list[dict[str, Any]] = []

    for index, match in enumerate(CHUNK_PATTERN.finditer(text)):
        chunks.append(
            {
                "index": index,
                "chunk_id": int(match.group("chunk_id")),
                "characters": int(match.group("characters")),
                "source_file": str(chunks_file),
                "text": match.group("text").strip(),
            }
        )

    if not chunks:
        raise ValueError(f"No chunks found in {chunks_file}")

    return chunks


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

