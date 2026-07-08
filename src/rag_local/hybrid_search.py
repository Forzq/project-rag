from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any


TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class StoredChunk:
    id: str
    text: str
    metadata: dict[str, Any]


class BM25Index:
    def __init__(
        self,
        chunks: list[StoredChunk],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.term_frequencies = [Counter(tokenize(chunk.text)) for chunk in chunks]
        self.document_lengths = [
            sum(frequencies.values()) for frequencies in self.term_frequencies
        ]
        self.average_document_length = (
            sum(self.document_lengths) / len(self.document_lengths) if chunks else 0.0
        )
        self.idf = self._build_idf()

    def _build_idf(self) -> dict[str, float]:
        document_frequencies: Counter[str] = Counter()

        for frequencies in self.term_frequencies:
            document_frequencies.update(frequencies.keys())

        document_count = len(self.chunks)
        return {
            term: math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequencies.items()
        }

    def score(self, query: str, document_index: int) -> float:
        query_terms = tokenize(query)
        frequencies = self.term_frequencies[document_index]
        document_length = self.document_lengths[document_index]

        if not query_terms or document_length == 0 or self.average_document_length == 0:
            return 0.0

        score = 0.0

        for term in query_terms:
            term_frequency = frequencies.get(term, 0)
            if term_frequency == 0:
                continue

            denominator = term_frequency + self.k1 * (
                1 - self.b + self.b * document_length / self.average_document_length
            )
            score += self.idf.get(term, 0.0) * (
                term_frequency * (self.k1 + 1) / denominator
            )

        return score

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        scored_results: list[dict[str, Any]] = []

        for index, chunk in enumerate(self.chunks):
            score = self.score(query, index)
            if score <= 0:
                continue

            scored_results.append(
                {
                    "id": chunk.id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "bm25_score": score,
                }
            )

        scored_results.sort(key=lambda item: item["bm25_score"], reverse=True)

        for rank, result in enumerate(scored_results[:top_k], start=1):
            result["bm25_rank"] = rank

        return scored_results[:top_k]


def reciprocal_rank_fusion(
    result_lists: list[list[dict[str, Any]]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}
    weights = weights or [1.0 for _ in result_lists]

    if len(weights) != len(result_lists):
        raise ValueError("weights count must match result_lists count.")

    for list_index, results in enumerate(result_lists):
        weight = weights[list_index]

        for rank, result in enumerate(results, start=1):
            document_id = str(result["id"])

            if document_id not in fused:
                fused[document_id] = {**result, "rrf_score": 0.0}

            for key, value in result.items():
                if value is not None:
                    fused[document_id][key] = value

            fused[document_id]["rrf_score"] += weight / (k + rank)

    return sorted(fused.values(), key=lambda item: item["rrf_score"], reverse=True)


def format_chunk_result(
    rank: int,
    document_id: str,
    text: str,
    metadata: dict[str, Any],
    similarity: float | None = None,
    distance: float | None = None,
    bm25_score: float | None = None,
    vector_rank: int | None = None,
    bm25_rank: int | None = None,
    rrf_score: float | None = None,
) -> dict[str, Any]:
    return {
        "rank": rank,
        "id": document_id,
        "chunk_id": metadata.get("chunk_id"),
        "similarity": similarity,
        "distance": distance,
        "bm25_score": bm25_score,
        "vector_rank": vector_rank,
        "bm25_rank": bm25_rank,
        "rrf_score": rrf_score,
        "characters": metadata.get("characters"),
        "source_file": metadata.get("source_file"),
        "doc_id": metadata.get("doc_id"),
        "title": metadata.get("title"),
        "author": metadata.get("author"),
        "year": metadata.get("year"),
        "document_type": metadata.get("document_type"),
        "text": text,
    }
