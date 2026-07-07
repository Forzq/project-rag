from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import requests

from src.rag_local.config import OPENROUTER_EMBEDDINGS_URL


def require_env(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise RuntimeError(f"{name} is not set.")

    return value


def sanitize_model_name(model: str) -> str:
    return model.replace("/", "_").replace("-", "_").replace(".", "_").lower()


class OpenRouterEmbedder:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        batch_size: int = 64,
        timeout: int = 60,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        self.model = model
        self.api_key = api_key or require_env("OPENROUTER_API_KEY")
        self.batch_size = batch_size
        self.timeout = timeout

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = requests.post(
                OPENROUTER_EMBEDDINGS_URL,
                headers=headers,
                json={
                    "model": self.model,
                    "input": batch,
                    "encoding_format": "float",
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()["data"]
            data.sort(key=lambda item: item["index"])
            embeddings.extend(item["embedding"] for item in data)

        return embeddings

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]

    def embed_texts_numpy(self, texts: list[str]) -> np.ndarray:
        return np.array(self.embed_texts(texts), dtype=np.float32)


class HuggingFaceEmbedder:
    def __init__(
        self,
        model_name: str,
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        passage_prefix: str = "passage: ",
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.passage_prefix = passage_prefix

    def embed_texts_numpy(self, texts: list[str]) -> np.ndarray:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.model_name)
        passages = [f"{self.passage_prefix}{text}" for text in texts]
        embeddings = model.encode(
            passages,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=True,
        )
        return embeddings.astype(np.float32)


def save_numpy_array(array: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, array)


def save_embeddings_metadata(
    chunks: list[dict[str, Any]],
    output_path: Path,
    generated_files: list[dict[str, Any]],
) -> None:
    metadata = {
        "chunk_count": len(chunks),
        "generated_files": generated_files,
        "chunks": chunks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

