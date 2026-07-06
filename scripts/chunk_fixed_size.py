from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split Markdown into fixed-size chunks.")
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
        default=Path("chunks_output/fixed_size_chunks.md"),
        help="Path to the output chunks Markdown file.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Maximum chunk size in characters.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=120,
        help="Number of characters repeated between neighboring chunks.",
    )
    return parser.parse_args()


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
    print(f"Saved {len(chunks)} fixed-size chunks to {output_path}")


def main() -> None:
    args = parse_args()
    text = args.input.read_text(encoding="utf-8")
    chunks = split_fixed_size(text, args.chunk_size, args.overlap)
    write_chunks(chunks, args.output, "Fixed Size")


if __name__ == "__main__":
    main()
