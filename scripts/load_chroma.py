from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rag_local.config import DEFAULT_CHROMA_DB_PATH
from src.rag_local.vector_store import ChromaStore, load_chunks_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load prepared chunks and embeddings into local ChromaDB."
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        required=True,
        help="Path to .npy embeddings file.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("embeddings_output/harry_potter_1_semantic/chunks_metadata.json"),
        help="Path to chunks_metadata.json.",
    )
    parser.add_argument(
        "--collection",
        required=True,
        help="Chroma collection name.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_CHROMA_DB_PATH,
        help="Directory for persistent ChromaDB data.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of records inserted per Chroma add call.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the collection before loading data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embeddings = np.load(args.embeddings)
    chunks = load_chunks_metadata(args.metadata)
    store = ChromaStore(args.db_path)
    store.load_embeddings(
        collection_name=args.collection,
        chunks=chunks,
        embeddings=embeddings,
        batch_size=args.batch_size,
        reset=args.reset,
    )
    print(f"Loaded {len(chunks)} records into Chroma collection '{args.collection}'.")
    print(f"Chroma DB path: {args.db_path}")
    print(f"Vector dimension: {embeddings.shape[1]}")


if __name__ == "__main__":
    main()
