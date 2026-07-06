from __future__ import annotations

import argparse
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split Markdown with RecursiveCharacterTextSplitter."
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
        default=Path("chunks_output/recursive_chunks.md"),
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


def split_recursive(text: str, chunk_size: int, overlap: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


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
    print(f"Saved {len(chunks)} recursive chunks to {output_path}")


def main() -> None:
    args = parse_args()
    text = args.input.read_text(encoding="utf-8")
    chunks = split_recursive(text, args.chunk_size, args.overlap)
    write_chunks(chunks, args.output, "RecursiveCharacterTextSplitter")


if __name__ == "__main__":
    main()
