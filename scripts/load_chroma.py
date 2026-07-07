from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import chromadb
import numpy as np


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
        default=Path("chroma_db"),
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


def load_metadata(metadata_path: Path) -> list[dict[str, Any]]:
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    chunks = data["chunks"]

    if not chunks:
        raise ValueError("Metadata file contains no chunks.")

    return chunks


def build_chroma_payloads(
    chunks: list[dict[str, Any]],
    embeddings: np.ndarray,
) -> tuple[list[str], list[str], list[dict[str, Any]], list[list[float]]]:
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Chunks count ({len(chunks)}) does not match embeddings count "
            f"({len(embeddings)})."
        )

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for chunk in chunks:
        chunk_id = int(chunk["chunk_id"])
        ids.append(f"chunk-{chunk_id}")
        documents.append(str(chunk["text"]))
        metadatas.append(
            {
                "chunk_id": chunk_id,
                "index": int(chunk["index"]),
                "characters": int(chunk["characters"]),
                "source_file": str(chunk["source_file"]),
            }
        )

    return ids, documents, metadatas, embeddings.astype(float).tolist()


def get_collection(
    db_path: Path,
    collection_name: str,
    reset: bool,
):
    client = chromadb.PersistentClient(path=str(db_path))

    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def add_in_batches(
    collection,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, Any]],
    embeddings: list[list[float]],
    batch_size: int,
) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end],
        )


def main() -> None:
    args = parse_args()

    embeddings = np.load(args.embeddings)
    chunks = load_metadata(args.metadata)
    ids, documents, metadatas, vectors = build_chroma_payloads(chunks, embeddings)
    collection = get_collection(args.db_path, args.collection, args.reset)

    add_in_batches(
        collection=collection,
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=vectors,
        batch_size=args.batch_size,
    )

    print(f"Loaded {len(ids)} records into Chroma collection '{args.collection}'.")
    print(f"Chroma DB path: {args.db_path}")
    print(f"Vector dimension: {embeddings.shape[1]}")


if __name__ == "__main__":
    main()
