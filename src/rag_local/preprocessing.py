from __future__ import annotations

import re


PAGE_MARKER_RE = re.compile(r"^\s*<!--\s*Page\s+\d+\s*-->\s*$", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"</?[^>]+>")
BOOK_COVER_RE = re.compile(r"^\s*Book cover of .*$", re.IGNORECASE)
BOOK_COVER_BULLET_RE = re.compile(r"^\s*•\s+.+$")
CHAPTER_TOC_RE = re.compile(r"^\s*\*\s*(?:Глава|Chapter)\s+\d+\s*$", re.IGNORECASE)
AUTHOR_TOC_RE = re.compile(r"^\s*\*\s*(?:Дж\.\s*К\.\s*Ролинг|J\.\s*K\.\s*Rowling)\s*$")


def normalize_blank_lines(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def clean_extracted_markdown(text: str) -> str:
    cleaned_lines: list[str] = []
    seen_heading = False

    for line in text.splitlines():
        if PAGE_MARKER_RE.match(line):
            continue

        line_without_tags = HTML_TAG_RE.sub("", line).strip()

        if not line_without_tags:
            cleaned_lines.append("")
            continue

        if BOOK_COVER_RE.match(line_without_tags):
            continue

        if not seen_heading and BOOK_COVER_BULLET_RE.match(line_without_tags):
            continue

        if AUTHOR_TOC_RE.match(line_without_tags):
            continue

        if CHAPTER_TOC_RE.match(line_without_tags):
            continue

        if line_without_tags.startswith("#"):
            seen_heading = True

        cleaned_lines.append(line_without_tags)

    return normalize_blank_lines("\n".join(cleaned_lines))
