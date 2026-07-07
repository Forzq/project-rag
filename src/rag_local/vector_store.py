from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb
import numpy as np


def load_chunks_metadata(metadata_path: Path) -> list[dict[str, Any]]:
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


class ChromaStore:
    def __init__(self, db_path: Path | str) -> None:
        self.client = chromadb.PersistentClient(path=str(db_path))

    def get_or_create_collection(self, collection_name: str, reset: bool = False):
        if reset:
            try:
                self.client.delete_collection(collection_name)
            except Exception:
                pass

        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def get_collection(self, collection_name: str):
        return self.client.get_collection(collection_name)

    def load_embeddings(
        self,
        collection_name: str,
        chunks: list[dict[str, Any]],
        embeddings: np.ndarray,
        batch_size: int,
        reset: bool,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        ids, documents, metadatas, vectors = build_chroma_payloads(chunks, embeddings)
        collection = self.get_or_create_collection(collection_name, reset=reset)

        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            collection.add(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                embeddings=vectors[start:end],
            )


class ChromaRetriever:
    def __init__(self, db_path: Path | str, collection_name: str) -> None:
        self.collection_name = collection_name
        self.collection = ChromaStore(db_path).get_collection(collection_name)

    def count(self) -> int:
        return self.collection.count()

    def search(self, query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        raw_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = raw_results.get("ids", [[]])[0]
        documents = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]

        results: list[dict[str, Any]] = []

        for index, document_id in enumerate(ids):
            metadata = metadatas[index] or {}
            distance = float(distances[index]) if distances[index] is not None else None
            similarity = 1 - distance if distance is not None else None
            results.append(
                {
                    "rank": index + 1,
                    "id": str(document_id),
                    "chunk_id": metadata.get("chunk_id"),
                    "similarity": similarity,
                    "distance": distance,
                    "characters": metadata.get("characters"),
                    "source_file": metadata.get("source_file"),
                    "text": documents[index],
                }
            )

        return results

