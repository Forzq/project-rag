from __future__ import annotations  # Включает отложенную обработку type hints, чтобы аннотации не мешали рантайму.

import argparse  # argparse нужен, чтобы скрипт можно было удобно запускать из командной строки.
import os  # os нужен для чтения переменных окружения, например LLAMA_CLOUD_API_KEY.
import sys  # sys нужен для вывода ошибок в stderr и завершения скрипта с кодом ошибки.
from pathlib import Path  # Path удобнее обычных строк для работы с путями к файлам.

from dotenv import load_dotenv  # load_dotenv загружает переменные из .env в окружение процесса.
from llama_cloud import LlamaCloud  # LlamaCloud - клиент LlamaParse/LlamaCloud API.


def parse_args() -> argparse.Namespace:
    # Создаем parser - объект, который знает, какие аргументы принимает скрипт.
    parser = argparse.ArgumentParser(
        # description показывается в help, если запустить скрипт с --help.
        description="Extract structured Markdown from a complex PDF with LlamaParse."
    )
    # Добавляем обязательный позиционный аргумент: путь к PDF.
    parser.add_argument(
        "pdf",  # Имя аргумента в CLI: python script.py input.pdf
        type=Path,  # argparse сразу преобразует строку пути в объект Path.
        help="Path to the input PDF file.",  # Текст подсказки для --help.
    )
    # Добавляем необязательный аргумент для файла результата.
    parser.add_argument(
        "-o",  # Короткая форма аргумента.
        "--output",  # Длинная форма аргумента.
        type=Path,  # Значение тоже превращается в Path.
        help="Path to the output .md file. If omitted, text is printed to console.",
    )
    # Добавляем настройку качества/стоимости LlamaParse.
    parser.add_argument(
        "--tier",  # CLI-флаг: --tier agentic
        default="cost_effective",  # Если tier не указан, используется cost_effective.
        choices=["cost_effective", "agentic", "agentic_plus"],  # Разрешаем только эти значения.
        help="LlamaParse parsing tier. Default: agentic.",
    )
    # Возвращаем объект с полями args.pdf, args.output, args.tier.
    return parser.parse_args()


def require_api_key() -> None:
    # Проверяем, есть ли ключ LlamaCloud в переменных окружения.
    if not os.getenv("LLAMA_CLOUD_API_KEY"):
        # Если ключа нет, останавливаем выполнение с понятной ошибкой.
        raise RuntimeError(
            "LLAMA_CLOUD_API_KEY is not set. "
            "Create a .env file with LLAMA_CLOUD_API_KEY=llx-your-key"
        )


def validate_pdf_path(pdf_path: Path) -> None:
    # Проверяем, существует ли путь к PDF.
    if not pdf_path.exists():
        # Если файла нет, выбрасываем FileNotFoundError.
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Проверяем, что путь указывает именно на файл, а не на папку.
    if not pdf_path.is_file():
        # Если это не файл, сообщаем об ошибке.
        raise ValueError(f"PDF path is not a file: {pdf_path}")

    # Проверяем расширение файла.
    if pdf_path.suffix.lower() != ".pdf":
        # Если расширение не .pdf, скрипт не должен отправлять файл в парсер.
        raise ValueError(f"Input file must be a PDF: {pdf_path}")


def extract_markdown_pages(result: object) -> str:
    # Достаем result.markdown, но безопасно: если поля нет, вернется None.
    markdown_result = getattr(result, "markdown", None)
    # Достаем markdown.pages, где LlamaParse хранит Markdown по страницам.
    pages = getattr(markdown_result, "pages", None)

    # Если страниц нет, значит API вернул неожиданный формат.
    if not pages:
        # Останавливаемся, потому что без pages нечего сохранять.
        raise RuntimeError("LlamaParse response does not contain markdown pages.")

    # Здесь будет список Markdown-текстов по страницам.
    page_texts: list[str] = []

    # Проходим по страницам, начиная нумерацию с 1.
    for index, page in enumerate(pages, start=1):
        # Берем markdown конкретной страницы и убираем пробелы по краям.
        markdown = getattr(page, "markdown", "").strip()
        # Пустые страницы не добавляем в результат.
        if markdown:
            # Добавляем marker страницы, чтобы позже понимать, откуда пришел текст.
            page_texts.append(f"<!-- Page {index} -->\n\n{markdown}")

    # Склеиваем страницы двойным переносом, чтобы Markdown оставался читаемым.
    return "\n\n".join(page_texts)


def parse_pdf_with_llamaparse(pdf_path: Path, tier: str) -> str:
    # Создаем клиент LlamaCloud; он сам берет LLAMA_CLOUD_API_KEY из окружения.
    client = LlamaCloud()

    # Сначала загружаем PDF-файл в LlamaCloud.
    uploaded_file = client.files.create(
        file=str(pdf_path),  # Передаем путь к PDF как строку.
        purpose="parse",  # Указываем, что файл нужен для парсинга.
    )

    # Запускаем парсинг загруженного файла.
    result = client.parsing.parse(
        file_id=uploaded_file.id,  # Используем id файла, который вернул upload.
        tier=tier,  # Передаем выбранный tier парсинга.
        version="latest",  # Используем последнюю доступную версию парсера.
        output_options={  # Настраиваем формат результата.
            "markdown": {  # Просим Markdown-вывод.
                "tables": {  # Настройки таблиц внутри Markdown.
                    "output_tables_as_markdown": True,  # Таблицы сохраняются как Markdown-таблицы.
                },
            },
        },
        expand=["markdown"],  # Просим API сразу вернуть markdown-данные в ответе.
    )

    # Превращаем ответ API в одну Markdown-строку.
    return extract_markdown_pages(result)


def main() -> None:
    # Загружаем .env, чтобы LLAMA_CLOUD_API_KEY был доступен через os.getenv.
    load_dotenv()
    # Читаем аргументы командной строки.
    args = parse_args()

    try:
        # Проверяем, что входной PDF существует и похож на PDF.
        validate_pdf_path(args.pdf)
        # Проверяем наличие API-ключа до обращения к LlamaCloud.
        require_api_key()
        # Отправляем PDF в LlamaParse и получаем Markdown.
        markdown = parse_pdf_with_llamaparse(args.pdf, args.tier)

        # Если пользователь указал --output, сохраняем Markdown в файл.
        if args.output:
            # Создаем папку результата, если ее еще нет.
            args.output.parent.mkdir(parents=True, exist_ok=True)
            # Записываем Markdown в UTF-8.
            args.output.write_text(markdown, encoding="utf-8")
            # Печатаем путь результата, чтобы пользователь видел успешное завершение.
            print(f"Markdown saved to {args.output}")
        # Если --output не указан, выводим Markdown прямо в консоль.
        else:
            # Такой режим удобен для быстрой проверки без сохранения файла.
            print(markdown)
    # Ловим любую ошибку, чтобы красиво вывести ее в stderr.
    except Exception as error:
        # stderr используется для ошибок, чтобы обычный output не смешивался с ними.
        print(f"Error: {error}", file=sys.stderr)
        # Завершаем программу с кодом 1, чтобы терминал/CI видел, что скрипт упал.
        raise SystemExit(1) from error


# Этот блок запускается только при прямом запуске файла, а не при import.
if __name__ == "__main__":
    # Передаем управление основной функции.
    main()
