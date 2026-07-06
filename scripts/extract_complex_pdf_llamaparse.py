from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from llama_cloud import LlamaCloud
from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract structured Markdown from a complex PDF with LlamaParse."
    )
    parser.add_argument(
        "pdf",
        type=Path,
        help="Path to the input PDF file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path to the output .md file. If omitted, text is printed to console.",
    )
    parser.add_argument(
        "--tier",
        default="agentic",
        choices=["cost_effective", "agentic", "agentic_plus"],
        help="LlamaParse parsing tier. Default: agentic.",
    )
    return parser.parse_args()


def require_api_key() -> None:
    if not os.getenv("LLAMA_CLOUD_API_KEY"):
        raise RuntimeError(
            "LLAMA_CLOUD_API_KEY is not set. "
            "Create a .env file with LLAMA_CLOUD_API_KEY=llx-your-key"
        )


def validate_pdf_path(pdf_path: Path) -> None:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not pdf_path.is_file():
        raise ValueError(f"PDF path is not a file: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Input file must be a PDF: {pdf_path}")


def extract_markdown_pages(result: object) -> str:
    markdown_result = getattr(result, "markdown", None)
    pages = getattr(markdown_result, "pages", None)

    if not pages:
        raise RuntimeError("LlamaParse response does not contain markdown pages.")

    page_texts: list[str] = []

    for index, page in enumerate(pages, start=1):
        markdown = getattr(page, "markdown", "").strip()
        if markdown:
            page_texts.append(f"<!-- Page {index} -->\n\n{markdown}")

    return "\n\n".join(page_texts)


def parse_pdf_with_llamaparse(pdf_path: Path, tier: str) -> str:
    client = LlamaCloud()

    uploaded_file = client.files.create(
        file=str(pdf_path),
        purpose="parse",
    )

    result = client.parsing.parse(
        file_id=uploaded_file.id,
        tier=tier,
        version="latest",
        output_options={
            "markdown": {
                "tables": {
                    "output_tables_as_markdown": True,
                },
            },
        },
        expand=["markdown"],
    )

    return extract_markdown_pages(result)


def main() -> None:
    load_dotenv()
    args = parse_args()

    try:
        validate_pdf_path(args.pdf)
        require_api_key()
        markdown = parse_pdf_with_llamaparse(args.pdf, args.tier)

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(markdown, encoding="utf-8")
            print(f"Markdown saved to {args.output}")
        else:
            print(markdown)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
