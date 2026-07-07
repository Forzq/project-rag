from __future__ import annotations

import argparse
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


def parse_args() -> argparse.Namespace:
    # Создаем CLI-интерфейс для запуска из терминала.
    parser = argparse.ArgumentParser(
        description="Split Markdown with RecursiveCharacterTextSplitter."
    )

    # Входной Markdown-файл. Если путь не передан, берем my_document.md.
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("md_output/my_document.md"),
        help="Path to the input Markdown file.",
    )

    # Файл, куда будет записан результат.
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("chunks_output/recursive_chunks.md"),
        help="Path to the output chunks Markdown file.",
    )

    # Максимальная длина чанка в символах.
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Maximum chunk size in characters.",
    )

    # Пересечение соседних чанков.
    parser.add_argument(
        "--overlap",
        type=int,
        default=120,
        help="Number of characters repeated between neighboring chunks.",
    )

    # Возвращаем объект args с полями input, output, chunk_size, overlap.
    return parser.parse_args()


def split_recursive(text: str, chunk_size: int, overlap: int) -> list[str]:
    # RecursiveCharacterTextSplitter пытается резать текст "умнее", чем fixed-size.
    # Он сначала пробует крупные разделители, потом более мелкие.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        # Порядок важен: сначала пробуем абзацы, потом строки, предложения,
        # слова и только в крайнем случае отдельные символы.
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # split_text возвращает список готовых чанков.
    return splitter.split_text(text)


def write_chunks(chunks: list[str], output_path: Path, method: str) -> None:
    # Создаем папку для результата.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Начало Markdown-файла с результатами.
    parts = [f"# {method} Chunks", ""]

    # Записываем каждый чанк отдельной секцией.
    for index, chunk in enumerate(chunks, start=1):
        parts.extend(
            [
                f"## Chunk {index}",
                "",
                f"Characters: {len(chunk)}",
                "",
                "````markdown",
                chunk,
                "````",
                "",
            ]
        )

    # Сохраняем все чанки в один Markdown-файл.
    output_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Saved {len(chunks)} recursive chunks to {output_path}")


def main() -> None:
    # Получаем параметры из терминала.
    args = parse_args()

    # Читаем исходный Markdown.
    text = args.input.read_text(encoding="utf-8")

    # Разбиваем текст через LangChain RecursiveCharacterTextSplitter.
    chunks = split_recursive(text, args.chunk_size, args.overlap)

    # Сохраняем результат.
    write_chunks(chunks, args.output, "RecursiveCharacterTextSplitter")


if __name__ == "__main__":
    # Запускаем main только при прямом запуске этого файла.
    main()
