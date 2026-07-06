from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all Markdown chunking scripts.")
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("md_output/my_document.md"),
        help="Path to the input Markdown file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("chunks_output"),
        help="Directory for chunking result files.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Chunk size for fixed and recursive methods.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=120,
        help="Overlap for fixed and recursive methods.",
    )
    parser.add_argument(
        "--semantic-max-chars",
        type=int,
        default=2000,
        help="Maximum chunk size for semantic chunking.",
    )
    parser.add_argument(
        "--semantic-model",
        default="qwen/qwen3-embedding-4b",
        help="OpenRouter embedding model for semantic chunking.",
    )
    parser.add_argument(
        "--break-percentile",
        type=float,
        default=80.0,
        help="Similarity-drop percentile for semantic chunking.",
    )
    parser.add_argument(
        "--semantic-batch-size",
        type=int,
        default=64,
        help="Number of text units sent per semantic embeddings request.",
    )
    return parser.parse_args()


def run_script(script_path: Path, args: list[str]) -> None:
    command = [sys.executable, str(script_path), *args]
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    scripts_dir = Path(__file__).resolve().parent
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_script(
        scripts_dir / "chunk_fixed_size.py",
        [
            str(args.input),
            "-o",
            str(args.output_dir / "fixed_size_chunks.md"),
            "--chunk-size",
            str(args.chunk_size),
            "--overlap",
            str(args.overlap),
        ],
    )
    run_script(
        scripts_dir / "chunk_recursive.py",
        [
            str(args.input),
            "-o",
            str(args.output_dir / "recursive_chunks.md"),
            "--chunk-size",
            str(args.chunk_size),
            "--overlap",
            str(args.overlap),
        ],
    )
    run_script(
        scripts_dir / "chunk_semantic.py",
        [
            str(args.input),
            "-o",
            str(args.output_dir / "semantic_chunks.md"),
            "--max-chars",
            str(args.semantic_max_chars),
            "--model",
            args.semantic_model,
            "--break-percentile",
            str(args.break_percentile),
            "--batch-size",
            str(args.semantic_batch_size),
        ],
    )

    print(f"All chunking results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
