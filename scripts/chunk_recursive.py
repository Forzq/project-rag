from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rag_local.chunk_io import write_chunks
from src.rag_local.chunking import split_recursive
from src.rag_local.preprocessing import clean_extracted_markdown


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
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Keep extracted Markdown exactly as-is before chunking.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = args.input.read_text(encoding="utf-8")
    if not args.no_clean:
        text = clean_extracted_markdown(text)

    chunks = split_recursive(text, args.chunk_size, args.overlap)
    write_chunks(chunks, args.output, "RecursiveCharacterTextSplitter")
    print(f"Saved {len(chunks)} recursive chunks to {args.output}")


if __name__ == "__main__":
    main()
