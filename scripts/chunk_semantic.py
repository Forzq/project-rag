from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rag_local.chunk_io import write_chunks
from src.rag_local.chunking import split_semantic
from src.rag_local.config import DEFAULT_SEMANTIC_CHUNKING_MODEL
from src.rag_local.embeddings import OpenRouterEmbedder
from src.rag_local.preprocessing import clean_extracted_markdown


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
        "--min-chars",
        type=int,
        default=300,
        help="Small semantic chunks are merged until they reach this size.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_SEMANTIC_CHUNKING_MODEL,
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
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Keep extracted Markdown exactly as-is before chunking.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    text = args.input.read_text(encoding="utf-8")
    if not args.no_clean:
        text = clean_extracted_markdown(text)

    embedder = OpenRouterEmbedder(model=args.model, batch_size=args.batch_size)
    chunks = split_semantic(
        text=text,
        max_chars=args.max_chars,
        embedder=embedder,
        break_percentile=args.break_percentile,
        min_chunk_chars=args.min_chars,
    )
    write_chunks(chunks, args.output, "Semantic")
    print(f"Saved {len(chunks)} semantic chunks to {args.output}")


if __name__ == "__main__":
    main()
