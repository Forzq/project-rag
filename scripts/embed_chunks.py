from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv


OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate embeddings for prepared Markdown chunks."
    )
    parser.add_argument(
        "chunks_file",
        nargs="?",
        type=Path,
        default=Path("chunks_output/harry_potter_1_embedding/semantic_chunks.md"),
        help="Path to a Markdown file produced by a chunking script.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("embeddings_output"),
        help="Directory where .npy arrays and metadata will be saved.",
    )
    parser.add_argument(
        "--provider",
        choices=["openrouter", "hf", "both"],
        default="both",
        help="Which embeddings to generate.",
    )
    parser.add_argument(
        "--openrouter-model",
        default="openai/text-embedding-3-small",
        help="OpenRouter embeddings model.",
    )
    parser.add_argument(
        "--hf-model",
        default="intfloat/e5-large-v2",
        help="HuggingFace sentence-transformers model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of chunks sent/encoded in one batch.",
    )
    return parser.parse_args()


def parse_chunks(chunks_file: Path) -> list[dict[str, object]]:
    text = chunks_file.read_text(encoding="utf-8")
    pattern = re.compile(
        r"## Chunk (?P<chunk_id>\d+)\s+"
        r"Characters: (?P<characters>\d+)\s+"
        r"````markdown\s*"
        r"(?P<text>.*?)"
        r"\s*````",
        re.DOTALL,
    )

    chunks: list[dict[str, object]] = []

    for index, match in enumerate(pattern.finditer(text)):
        chunk_text = match.group("text").strip()
        chunks.append(
            {
                "index": index,
                "chunk_id": int(match.group("chunk_id")),
                "characters": int(match.group("characters")),
                "source_file": str(chunks_file),
                "text": chunk_text,
            }
        )

    if not chunks:
        raise ValueError(f"No chunks found in {chunks_file}")

    return chunks


def require_openrouter_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. "
            "Add OPENROUTER_API_KEY=sk-or-your-key to your .env file."
        )

    return api_key


def sanitize_model_name(model: str) -> str:
    return (
        model.replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
        .lower()
    )


def embed_with_openrouter(
    texts: list[str],
    api_key: str,
    model: str,
    batch_size: int,
) -> np.ndarray:
    embeddings: list[list[float]] = []
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = requests.post(
            OPENROUTER_EMBEDDINGS_URL,
            headers=headers,
            json={
                "model": model,
                "input": batch,
                "encoding_format": "float",
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()["data"]
        data.sort(key=lambda item: item["index"])
        embeddings.extend(item["embedding"] for item in data)

    return np.array(embeddings, dtype=np.float32)


def embed_with_huggingface(
    texts: list[str],
    model_name: str,
    batch_size: int,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    passages = [f"passage: {text}" for text in texts]
    embeddings = model.encode(
        passages,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return embeddings.astype(np.float32)


def save_numpy_array(array: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, array)
    print(f"Saved {output_path} with shape {array.shape}")


def save_metadata(
    chunks: list[dict[str, object]],
    output_path: Path,
    generated_files: list[dict[str, object]],
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
    print(f"Saved metadata to {output_path}")


def main() -> None:
    load_dotenv()
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    chunks = parse_chunks(args.chunks_file)
    texts = [str(chunk["text"]) for chunk in chunks]
    generated_files: list[dict[str, object]] = []

    if args.provider in {"openrouter", "both"}:
        api_key = require_openrouter_api_key()
        openrouter_embeddings = embed_with_openrouter(
            texts=texts,
            api_key=api_key,
            model=args.openrouter_model,
            batch_size=args.batch_size,
        )
        openrouter_output = (
            args.output_dir / f"{sanitize_model_name(args.openrouter_model)}.npy"
        )
        save_numpy_array(openrouter_embeddings, openrouter_output)
        generated_files.append(
            {
                "provider": "openrouter",
                "model": args.openrouter_model,
                "path": str(openrouter_output),
                "shape": list(openrouter_embeddings.shape),
            }
        )

    if args.provider in {"hf", "both"}:
        hf_embeddings = embed_with_huggingface(
            texts=texts,
            model_name=args.hf_model,
            batch_size=args.batch_size,
        )
        hf_output = args.output_dir / f"{sanitize_model_name(args.hf_model)}.npy"
        save_numpy_array(hf_embeddings, hf_output)
        generated_files.append(
            {
                "provider": "huggingface",
                "model": args.hf_model,
                "path": str(hf_output),
                "shape": list(hf_embeddings.shape),
            }
        )

    save_metadata(
        chunks=chunks,
        output_path=args.output_dir / "chunks_metadata.json",
        generated_files=generated_files,
    )


if __name__ == "__main__":
    main()
