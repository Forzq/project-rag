from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rag_local.chunk_io import parse_chunks_file
from src.rag_local.config import DEFAULT_HF_MODEL, DEFAULT_RETRIEVAL_MODEL
from src.rag_local.embeddings import (
    HuggingFaceEmbedder,
    OpenRouterEmbedder,
    sanitize_model_name,
    save_embeddings_metadata,
    save_numpy_array,
)


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
        default=DEFAULT_RETRIEVAL_MODEL,
        help="OpenRouter embeddings model.",
    )
    parser.add_argument(
        "--hf-model",
        default=DEFAULT_HF_MODEL,
        help="HuggingFace sentence-transformers model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of chunks sent/encoded in one batch.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    chunks = parse_chunks_file(args.chunks_file)
    texts = [str(chunk["text"]) for chunk in chunks]
    generated_files: list[dict[str, Any]] = []

    if args.provider in {"openrouter", "both"}:
        embedder = OpenRouterEmbedder(
            model=args.openrouter_model,
            batch_size=args.batch_size,
        )
        embeddings = embedder.embed_texts_numpy(texts)
        output_path = args.output_dir / f"{sanitize_model_name(args.openrouter_model)}.npy"
        save_numpy_array(embeddings, output_path)
        print(f"Saved {output_path} with shape {embeddings.shape}")
        generated_files.append(
            {
                "provider": "openrouter",
                "model": args.openrouter_model,
                "path": str(output_path),
                "shape": list(embeddings.shape),
            }
        )

    if args.provider in {"hf", "both"}:
        embedder = HuggingFaceEmbedder(
            model_name=args.hf_model,
            batch_size=args.batch_size,
        )
        embeddings = embedder.embed_texts_numpy(texts)
        output_path = args.output_dir / f"{sanitize_model_name(args.hf_model)}.npy"
        save_numpy_array(embeddings, output_path)
        print(f"Saved {output_path} with shape {embeddings.shape}")
        generated_files.append(
            {
                "provider": "huggingface",
                "model": args.hf_model,
                "path": str(output_path),
                "shape": list(embeddings.shape),
            }
        )

    metadata_path = args.output_dir / "chunks_metadata.json"
    save_embeddings_metadata(chunks, metadata_path, generated_files)
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
