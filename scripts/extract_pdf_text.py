from __future__ import annotations

import argparse
from pathlib import Path

from pypdf import PdfReader


def clean_text(text: str) -> str: # убирает лишние пробелы и пустые строки
    lines = [line.strip() for line in text.splitlines()]
    non_empty_lines = [line for line in lines if line]
    return "\n".join(non_empty_lines)


def extract_text_from_pdf(pdf_path: Path) -> str: # достает текстовый слой
    """Return plain text from all pages of a PDF file."""
    reader = PdfReader(pdf_path)
    pages_text: list[str] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")

        if text:
            pages_text.append(text)
        else:
            pages_text.append(f"[Page {page_number}: no extractable text]")

    return "\n\n".join(pages_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract plain text from a PDF file."
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
        help="Path to the output .txt file. If omitted, text is printed to console.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.pdf.exists():
        raise FileNotFoundError(f"PDF file not found: {args.pdf}")

    if not args.pdf.is_file():
        raise ValueError(f"PDF path is not a file: {args.pdf}")

    text = extract_text_from_pdf(args.pdf)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Text saved to {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
