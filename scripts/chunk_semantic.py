from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import requests
from dotenv import load_dotenv


# OpenRouter endpoint для получения embeddings.
# Embeddings - это числовые векторы, которые описывают смысл текста.
OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"


def parse_args() -> argparse.Namespace:
    # Создаем CLI-интерфейс для semantic chunking.
    parser = argparse.ArgumentParser(
        description="Split Markdown into semantic chunks with embeddings."
    )

    # Входной Markdown-файл.
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("md_output/my_document.md"),
        help="Path to the input Markdown file.",
    )

    # Файл, куда сохранить semantic chunks.
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("chunks_output/semantic_chunks.md"),
        help="Path to the output chunks Markdown file.",
    )

    # Максимальный размер semantic chunk в символах.
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2000,
        help="Maximum semantic chunk size in characters.",
    )

    # Модель OpenRouter, которая будет превращать текст в embeddings.
    parser.add_argument(
        "--model",
        default="qwen/qwen3-embedding-4b",
        help="OpenRouter embedding model.",
    )

    # Процентиль для разрыва по смыслу.
    # Чем ниже значение, тем чаще будут появляться новые чанки.
    parser.add_argument(
        "--break-percentile",
        type=float,
        default=80.0,
        help="Similarity-drop percentile used to create semantic breaks.",
    )

    # Сколько текстовых блоков отправлять в OpenRouter за один запрос.
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of text units sent per embeddings request.",
    )

    # Возвращаем объект с параметрами запуска.
    return parser.parse_args()


def require_openrouter_api_key() -> str:
    # Ключ берется из переменной окружения, которую load_dotenv загрузит из .env.
    api_key = os.getenv("OPENROUTER_API_KEY")

    # Без ключа нельзя вызвать OpenRouter embeddings API.
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. "
            "Add OPENROUTER_API_KEY=sk-or-your-key to your .env file."
        )

    return api_key


def split_markdown_units(text: str) -> list[str]:
    # units - маленькие смысловые единицы: абзацы, заголовки, page markers,
    # code blocks. Потом мы будем сравнивать embeddings соседних units.
    units: list[str] = []
    current: list[str] = []
    in_code_block = False

    # Идем по Markdown построчно.
    for line in text.splitlines():
        stripped = line.strip()

        # Если встретили ```, переключаем состояние code block.
        # Это нужно, чтобы не разрезать код или JSON внутри блока.
        if stripped.startswith("```"):
            in_code_block = not in_code_block

        # Заголовок или marker страницы начинает новую смысловую единицу.
        starts_new_unit = (
            not in_code_block
            and current
            and (stripped.startswith("#") or stripped.startswith("<!-- Page "))
        )

        # Если новая единица началась, сохраняем предыдущую.
        if starts_new_unit:
            units.append("\n".join(current).strip())
            current = []

        # Пустая строка вне code block завершает текущий абзац.
        if not in_code_block and not stripped:
            if current:
                units.append("\n".join(current).strip())
                current = []
            continue

        # Добавляем строку в текущую единицу.
        current.append(line)

    # После цикла сохраняем последнюю накопленную единицу.
    if current:
        units.append("\n".join(current).strip())

    # Убираем случайные пустые элементы.
    return [unit for unit in units if unit]


def embed_texts(
    texts: list[str],
    api_key: str,
    model: str,
    batch_size: int,
) -> list[list[float]]:
    # Здесь будут все embeddings, по одному вектору на каждый text unit.
    embeddings: list[list[float]] = []

    # Заголовки HTTP-запроса: авторизация и формат JSON.
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Отправляем units пачками, чтобы не делать запрос на каждый абзац отдельно.
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]

        # POST-запрос к OpenRouter embeddings API.
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

        # Если API вернул ошибку, здесь будет исключение.
        response.raise_for_status()

        # OpenRouter возвращает data со списком embeddings.
        data = response.json()["data"]

        # На всякий случай сортируем по index, чтобы порядок совпадал с input.
        data.sort(key=lambda item: item["index"])

        # Добавляем embeddings текущей пачки в общий список.
        embeddings.extend(item["embedding"] for item in data)

    return embeddings


def cosine_similarity(left: list[float], right: list[float]) -> float:
    # Cosine similarity показывает, насколько два вектора похожи по направлению.
    # Чем ближе к 1, тем ближе смысл текстов.
    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))

    # Защита от деления на ноль.
    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot_product / (left_norm * right_norm)


def percentile(values: list[float], percentile_value: float) -> float:
    # Percentile нужен, чтобы выбрать порог semantic break.
    # Например, 80 означает: берем только верхние 20% самых сильных смысловых разрывов.
    if not values:
        return 1.0

    sorted_values = sorted(values)
    index = round((percentile_value / 100) * (len(sorted_values) - 1))
    return sorted_values[index]


def build_chunks(
    units: list[str],
    distances: list[float],
    threshold: float,
    max_chars: int,
) -> list[str]:
    # distances - это 1 - cosine_similarity для соседних units.
    # Чем больше distance, тем сильнее смысловой разрыв.
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    # Идем по всем units и решаем, добавлять unit в текущий chunk или начать новый.
    for index, unit in enumerate(units):
        extra_length = len(unit) + (2 if current else 0)

        # Новый chunk нужен, если текущий станет слишком большим.
        should_break_by_size = current and current_length + extra_length > max_chars

        # Новый chunk нужен, если между прошлым и текущим unit есть сильный смысловой разрыв.
        should_break_by_meaning = index > 0 and distances[index - 1] >= threshold

        if current and (should_break_by_size or should_break_by_meaning):
            # Закрываем текущий chunk.
            chunks.append("\n\n".join(current))

            # Начинаем новый chunk с текущего unit.
            current = [unit]
            current_length = len(unit)
        else:
            # Продолжаем текущий chunk.
            current.append(unit)
            current_length += extra_length

    # Сохраняем последний chunk.
    if current:
        chunks.append("\n\n".join(current))

    return [chunk for chunk in chunks if chunk]


def split_semantic(
    text: str,
    max_chars: int,
    model: str,
    break_percentile: float,
    batch_size: int,
) -> list[str]:
    # Проверки входных параметров.
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero.")

    if not 0 <= break_percentile <= 100:
        raise ValueError("break_percentile must be between 0 and 100.")

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    # Сначала разбиваем Markdown на маленькие смысловые единицы.
    units = split_markdown_units(text)

    # Если единица всего одна, сравнивать нечего.
    if len(units) <= 1:
        return units

    # Получаем API key и embeddings для всех units.
    api_key = require_openrouter_api_key()
    embeddings = embed_texts(units, api_key, model, batch_size)

    # Считаем distance между каждой соседней парой units.
    distances = [
        1 - cosine_similarity(left, right)
        for left, right in zip(embeddings, embeddings[1:])
    ]

    # Выбираем порог: где distance выше threshold, там считаем смену смысла.
    threshold = percentile(distances, break_percentile)

    # Собираем финальные chunks.
    return build_chunks(units, distances, threshold, max_chars)


def write_chunks(chunks: list[str], output_path: Path, method: str) -> None:
    # Создаем директорию для результата.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # parts - будущий Markdown-файл с чанками.
    parts = [f"# {method} Chunks", ""]

    # Каждый chunk пишем отдельным разделом.
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

    # Сохраняем результат.
    output_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Saved {len(chunks)} semantic chunks to {output_path}")


def main() -> None:
    # Загружаем переменные из .env, включая OPENROUTER_API_KEY.
    load_dotenv()

    # Читаем параметры запуска.
    args = parse_args()

    # Читаем исходный Markdown.
    text = args.input.read_text(encoding="utf-8")

    # Запускаем embedding-based semantic chunking.
    chunks = split_semantic(
        text=text,
        max_chars=args.max_chars,
        model=args.model,
        break_percentile=args.break_percentile,
        batch_size=args.batch_size,
    )

    # Сохраняем результат в Markdown.
    write_chunks(chunks, args.output, "Semantic")


if __name__ == "__main__":
    # Запускаем main только при прямом запуске файла.
    main()
