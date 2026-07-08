from __future__ import annotations

from typing import Any


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int,
    ) -> list[dict[str, Any]]:
        if top_n <= 0:
            raise ValueError("top_n must be greater than zero.")

        if not candidates:
            return []

        pairs = [(query, str(candidate["text"])) for candidate in candidates]
        scores = self.model.predict(pairs)
        ranked_candidates: list[dict[str, Any]] = []

        for candidate, score in zip(candidates, scores):
            ranked_candidates.append(
                {
                    **candidate,
                    "original_rank": candidate.get("rank"),
                    "rerank_score": float(score),
                }
            )

        ranked_candidates.sort(key=lambda item: item["rerank_score"], reverse=True)

        for rank, candidate in enumerate(ranked_candidates[:top_n], start=1):
            candidate["rank"] = rank

        return ranked_candidates[:top_n]
