from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rag_local.chunk_io import parse_chunks_file, write_chunks
from src.rag_local.chunking import split_semantic
from src.rag_local.config import (
    DEFAULT_CHROMA_COLLECTION,
    DEFAULT_CHROMA_DB_PATH,
    DEFAULT_RETRIEVAL_MODEL,
    DEFAULT_SEMANTIC_CHUNKING_MODEL,
)
from src.rag_local.embeddings import (
    OpenRouterEmbedder,
    sanitize_model_name,
    save_embeddings_metadata,
    save_numpy_array,
)
from src.rag_local.preprocessing import clean_extracted_markdown
from src.rag_local.vector_store import ChromaStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full ingestion pipeline for one Markdown document."
    )
    parser.add_argument("input", type=Path, help="Path to the input Markdown file.")
    parser.add_argument("--doc-id", required=True, help="Stable document id.")
    parser.add_argument("--title", required=True, help="Document title.")
    parser.add_argument("--author", required=True, help="Document author.")
    parser.add_argument("--year", type=int, required=True, help="Publication year.")
    parser.add_argument(
        "--document-type",
        default="book",
        help="Document type, for example: book.",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_CHROMA_COLLECTION,
        help="Target Chroma collection.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_CHROMA_DB_PATH,
        help="Directory for persistent ChromaDB data.",
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=Path("chunks_output"),
        help="Base directory for generated chunks.",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=Path("embeddings_output"),
        help="Base directory for generated embeddings.",
    )
    parser.add_argument(
        "--semantic-model",
        default=DEFAULT_SEMANTIC_CHUNKING_MODEL,
        help="OpenRouter model used for semantic chunking.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_RETRIEVAL_MODEL,
        help="OpenRouter model used for retrieval embeddings.",
    )
    parser.add_argument(
        "--semantic-max-chars",
        type=int,
        default=2000,
        help="Maximum semantic chunk size in characters.",
    )
    parser.add_argument(
        "--break-percentile",
        type=float,
        default=80.0,
        help="Similarity-drop percentile used to create semantic breaks.",
    )
    parser.add_argument(
        "--semantic-batch-size",
        type=int,
        default=64,
        help="Batch size for semantic chunking embeddings.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=32,
        help="Batch size for final retrieval embeddings.",
    )
    parser.add_argument(
        "--chroma-batch-size",
        type=int,
        default=500,
        help="Batch size for ChromaDB inserts.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the Chroma collection before loading this document.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Keep extracted Markdown exactly as-is before chunking.",
    )
    return parser.parse_args()


def build_document_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "doc_id": args.doc_id,
        "title": args.title,
        "author": args.author,
        "year": args.year,
        "document_type": args.document_type,
    }


def enrich_chunks(
    chunks: list[dict[str, Any]],
    document_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    return [{**chunk, **document_metadata} for chunk in chunks]


def main() -> None:
    load_dotenv()
    args = parse_args()

    document_metadata = build_document_metadata(args)
    chunks_output_dir = args.chunks_dir / args.doc_id
    embeddings_output_dir = args.embeddings_dir / f"{args.doc_id}_semantic"
    chunks_file = chunks_output_dir / "semantic_chunks.md"
    embeddings_file = (
        embeddings_output_dir / f"{sanitize_model_name(args.embedding_model)}.npy"
    )
    metadata_file = embeddings_output_dir / "chunks_metadata.json"

    print("Step 1/4: semantic chunking")
    source_text = args.input.read_text(encoding="utf-8")
    if not args.no_clean:
        source_text = clean_extracted_markdown(source_text)

    semantic_embedder = OpenRouterEmbedder(
        model=args.semantic_model,
        batch_size=args.semantic_batch_size,
    )
    chunk_texts = split_semantic(
        text=source_text,
        max_chars=args.semantic_max_chars,
        embedder=semantic_embedder,
        break_percentile=args.break_percentile,
    )
    write_chunks(chunk_texts, chunks_file, "Semantic")
    print(f"Saved {len(chunk_texts)} chunks to {chunks_file}")

    print("Step 2/4: retrieval embeddings")
    chunks = parse_chunks_file(chunks_file)
    enriched_chunks = enrich_chunks(chunks, document_metadata)
    texts = [str(chunk["text"]) for chunk in enriched_chunks]
    retrieval_embedder = OpenRouterEmbedder(
        model=args.embedding_model,
        batch_size=args.embedding_batch_size,
    )
    embeddings = retrieval_embedder.embed_texts_numpy(texts)
    save_numpy_array(embeddings, embeddings_file)
    print(f"Saved embeddings to {embeddings_file} with shape {embeddings.shape}")

    print("Step 3/4: metadata")
    save_embeddings_metadata(
        chunks=enriched_chunks,
        output_path=metadata_file,
        generated_files=[
            {
                "provider": "openrouter",
                "model": args.embedding_model,
                "path": str(embeddings_file),
                "shape": list(embeddings.shape),
            }
        ],
    )
    print(f"Saved metadata to {metadata_file}")

    print("Step 4/4: ChromaDB load")
    store = ChromaStore(args.db_path)
    store.load_embeddings(
        collection_name=args.collection,
        chunks=enriched_chunks,
        embeddings=np.asarray(embeddings),
        batch_size=args.chroma_batch_size,
        reset=args.reset,
        document_metadata=document_metadata,
    )
    print(f"Loaded {len(enriched_chunks)} chunks into '{args.collection}'")
    print(f"Document metadata: {document_metadata}")


if __name__ == "__main__":
    main()
