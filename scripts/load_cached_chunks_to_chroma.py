from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import chromadb
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rag_local.config import DEFAULT_CHROMA_COLLECTION, DEFAULT_CHROMA_DB_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load cached chunk metadata and embeddings into a Chroma collection."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("chunking_eval_output/recursive/chunks_metadata.json"),
        help="Path to chunks_metadata.json.",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("chunking_eval_output/recursive/openai_text_embedding_3_small.npy"),
        help="Path to .npy embeddings file.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_CHROMA_DB_PATH,
        help="ChromaDB persistent directory.",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_CHROMA_COLLECTION,
        help="Target Chroma collection.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Chroma upsert batch size.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the target collection before loading cached chunks.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT_DIR / path


def load_chunks(metadata_path: Path) -> tuple[str, list[dict[str, Any]]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    chunks = metadata.get("chunks", [])

    if not chunks:
        raise ValueError(f"No chunks found in {metadata_path}")

    return str(metadata.get("method") or "unknown"), chunks


def build_payloads(
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
        chunk_id = str(chunk["id"])
        text = str(chunk["text"])
        metadata = {
            key: value
            for key, value in chunk.items()
            if key not in {"id", "text"} and value is not None
        }

        ids.append(chunk_id)
        documents.append(text)
        metadatas.append(metadata)

    return ids, documents, metadatas, embeddings.astype(float).tolist()


def get_or_create_collection(client: chromadb.PersistentClient, name: str, reset: bool):
    if reset:
        try:
            client.delete_collection(name)
        except Exception:
            pass

    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero.")

    metadata_path = resolve_path(args.metadata)
    embeddings_path = resolve_path(args.embeddings)
    db_path = resolve_path(args.db_path)
    method, chunks = load_chunks(metadata_path)
    embeddings = np.load(embeddings_path)
    ids, documents, metadatas, vectors = build_payloads(chunks, embeddings)
    client = chromadb.PersistentClient(path=str(db_path))
    collection = get_or_create_collection(client, args.collection, args.reset)

    for start in range(0, len(ids), args.batch_size):
        end = start + args.batch_size
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=vectors[start:end],
        )

    print(
        f"Loaded {len(ids)} {method} chunks into collection '{args.collection}' "
        f"at {db_path}."
    )
    print(f"Collection count: {collection.count()}")


if __name__ == "__main__":
    main()
