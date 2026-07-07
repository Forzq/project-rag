from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    # argparse отвечает за параметры, которые мы передаем скрипту из терминала.
    parser = argparse.ArgumentParser(description="Split Markdown into fixed-size chunks.")

    # input - путь к Markdown-файлу, который надо разбить на чанки.
    # nargs="?" означает, что аргумент необязательный: если его не передать,
    # будет использован md_output/my_document.md.
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("md_output/my_document.md"),
        help="Path to the input Markdown file.",
    )

    # output - путь к файлу, куда будут сохранены готовые чанки.
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("chunks_output/fixed_size_chunks.md"),
        help="Path to the output chunks Markdown file.",
    )

    # chunk-size - максимальный размер одного чанка в символах.
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Maximum chunk size in characters.",
    )

    # overlap - сколько символов повторять из предыдущего чанка в следующем.
    # Это помогает не потерять контекст на границе чанков.
    parser.add_argument(
        "--overlap",
        type=int,
        default=120,
        help="Number of characters repeated between neighboring chunks.",
    )

    # parse_args читает реальные значения из команды запуска.
    return parser.parse_args()


def split_fixed_size(text: str, chunk_size: int, overlap: int) -> list[str]:
    # Проверяем, что размер чанка задан корректно.
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    # overlap должен быть меньше chunk_size, иначе цикл не сможет двигаться вперед.
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    chunks: list[str] = []

    # start - индекс символа, с которого начинается текущий чанк.
    start = 0

    # Идем по тексту, пока не дошли до конца.
    while start < len(text):
        # end - индекс, где текущий чанк должен закончиться.
        # min нужен, чтобы последний чанк не вышел за конец текста.
        end = min(start + chunk_size, len(text))

        # Берем кусок текста от start до end и убираем лишние пробелы по краям.
        chunks.append(text[start:end].strip())

        # Если дошли до конца текста, выходим из цикла.
        if end == len(text):
            break

        # Следующий чанк начинается не с end, а немного раньше.
        # Так создается overlap: часть прошлого чанка повторится в новом.
        start = end - overlap

    # Возвращаем только непустые чанки.
    return [chunk for chunk in chunks if chunk]


def write_chunks(chunks: list[str], output_path: Path, method: str) -> None:
    # Создаем папку для результата, если ее еще нет.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # parts - список строк, из которых соберется Markdown-файл с чанками.
    parts = [f"# {method} Chunks", ""]

    # enumerate(..., start=1) дает номер чанка: 1, 2, 3...
    for index, chunk in enumerate(chunks, start=1):
        # Каждый чанк записываем как отдельную секцию.
        # Четыре backticks нужны, чтобы внутри чанка могли быть обычные ``` блоки.
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

    # Склеиваем все строки через переносы и сохраняем файл в UTF-8.
    output_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Saved {len(chunks)} fixed-size chunks to {output_path}")


def main() -> None:
    # Читаем параметры запуска.
    args = parse_args()

    # Читаем исходный Markdown-файл.
    text = args.input.read_text(encoding="utf-8")

    # Разбиваем текст простым fixed-size способом.
    chunks = split_fixed_size(text, args.chunk_size, args.overlap)

    # Сохраняем результат.
    write_chunks(chunks, args.output, "Fixed Size")


if __name__ == "__main__":
    # Эта проверка запускает main только при прямом запуске файла.
    main()
