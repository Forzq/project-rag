from __future__ import annotations  # Делает type hints ленивыми и помогает избежать проблем с forward references.

import argparse  # argparse нужен для CLI-аргументов скрипта.
import json  # json нужен для чтения metadata sidecar и state-файла.
import re  # re нужен для превращения имени файла в безопасный doc_id.
import subprocess  # subprocess запускает другие Python-скрипты пайплайна.
import sys  # sys нужен, чтобы вызвать именно текущий Python из virtualenv.
import time  # time нужен для ожидания между сканированиями папки.
from datetime import datetime  # datetime нужен для timestamps в state-файле.
from pathlib import Path  # Path делает работу с путями удобнее и безопаснее.
from typing import Any  # Any используется для словарей с разными типами значений.


# ROOT_DIR - корень проекта; parents[1] значит: scripts/watch_pdf_input.py -> scripts -> project root.
ROOT_DIR = Path(__file__).resolve().parents[1]
# EXTRACT_SCRIPT - путь к скрипту, который превращает PDF в Markdown через LlamaParse.
EXTRACT_SCRIPT = ROOT_DIR / "scripts" / "extract_complex_pdf_llamaparse.py"
# INGEST_SCRIPT - путь к скрипту, который делает chunking, embeddings и загрузку в ChromaDB.
INGEST_SCRIPT = ROOT_DIR / "scripts" / "ingest_document.py"


def parse_args() -> argparse.Namespace:
    # Создаем CLI parser, чтобы запускать watcher с настройками из командной строки.
    parser = argparse.ArgumentParser(
        # description будет показан при запуске python scripts/watch_pdf_input.py --help.
        description=(
            "Watch pdf_input and automatically run PDF parsing, chunking, "
            "embedding generation, and ChromaDB loading for new PDF files."
        )
    )
    # Папка, которую watcher будет мониторить.
    parser.add_argument(
        "--input-dir",  # CLI-флаг для изменения папки входных PDF.
        type=Path,  # Значение автоматически станет объектом Path.
        default=Path("pdf_input"),  # По умолчанию смотрим папку pdf_input.
        help="Directory to watch for new PDF files.",  # Help-текст.
    )
    # Папка, куда будет сохраняться Markdown после LlamaParse.
    parser.add_argument(
        "--md-output-dir",  # CLI-флаг для Markdown output папки.
        type=Path,  # Значение будет Path.
        default=Path("md_output"),  # По умолчанию сохраняем в md_output.
        help="Directory where parsed Markdown files are saved.",
    )
    # Файл состояния, чтобы не обрабатывать один PDF бесконечно.
    parser.add_argument(
        "--state-file",  # CLI-флаг для пути к state JSON.
        type=Path,  # Значение будет Path.
        default=Path("pdf_input") / ".ingest_state.json",  # State лежит рядом с PDF.
        help="JSON file used to remember already processed PDFs.",
    )
    # Интервал между сканированиями папки.
    parser.add_argument(
        "--scan-interval",  # CLI-флаг для частоты проверки папки.
        type=float,  # Можно указать дробное число секунд.
        default=10.0,  # По умолчанию проверяем каждые 10 секунд.
        help="Seconds between folder scans.",
    )
    # Время стабильности файла перед обработкой.
    parser.add_argument(
        "--stable-seconds",  # CLI-флаг для защиты от чтения недокопированного PDF.
        type=float,  # Можно указать 0.5, 1.0 и т.д.
        default=5.0,  # Ждем 5 секунд без изменения размера/mtime.
        help="How long a PDF size must stay unchanged before processing.",
    )
    # Tier LlamaParse, который влияет на качество/стоимость парсинга.
    parser.add_argument(
        "--tier",  # CLI-флаг для выбора tier.
        default="cost_effective",  # Для watcher выбран более дешевый режим по умолчанию.
        choices=["cost_effective", "agentic", "agentic_plus"],  # Разрешенные tier значения.
        help="LlamaParse parsing tier.",
    )
    # Автор по умолчанию, если рядом с PDF нет JSON с author.
    parser.add_argument(
        "--default-author",  # CLI-флаг для дефолтного автора.
        default="Unknown",  # Если author не задан, будет Unknown.
        help="Author used when sidecar metadata does not define author.",
    )
    # Год по умолчанию, если в metadata JSON нет year.
    parser.add_argument(
        "--default-year",  # CLI-флаг для дефолтного года.
        type=int,  # Год должен быть числом.
        default=datetime.now().year,  # По умолчанию текущий год.
        help="Year used when sidecar metadata does not define year.",
    )
    # Тип документа по умолчанию.
    parser.add_argument(
        "--document-type",  # CLI-флаг для типа документа.
        default="book",  # В нашем проекте чаще всего это book.
        help="Document type used when sidecar metadata does not define document_type.",
    )
    # Имя Chroma collection, если хотим загрузить не в дефолтную.
    parser.add_argument(
        "--collection",  # CLI-флаг для Chroma collection.
        default=None,  # None значит: ingest_document.py использует свой default.
        help="Target Chroma collection. If omitted, ingest_document.py uses its default.",
    )
    parser.add_argument(
        "--chunking-method",
        default="recursive",
        choices=["recursive", "semantic"],
        help="Chunking method passed to ingest_document.py.",
    )
    parser.add_argument(
        "--recursive-chunk-size",
        type=int,
        default=1500,
        help="Recursive chunk size passed to ingest_document.py.",
    )
    parser.add_argument(
        "--recursive-overlap",
        type=int,
        default=200,
        help="Recursive chunk overlap passed to ingest_document.py.",
    )
    # Максимальный размер semantic chunk, передается дальше в ingest_document.py.
    parser.add_argument(
        "--semantic-max-chars",  # CLI-флаг размера чанка.
        type=int,  # Значение должно быть числом.
        default=2000,  # Дефолтный максимум символов.
        help="Maximum semantic chunk size passed to ingest_document.py.",
    )
    # Минимальный размер semantic chunk, передается дальше в ingest_document.py.
    parser.add_argument(
        "--semantic-min-chars",  # CLI-флаг минимального размера.
        type=int,  # Значение должно быть числом.
        default=300,  # Маленькие чанки будут сливаться до этого размера.
        help="Minimum semantic chunk size passed to ingest_document.py.",
    )
    # Percentile для semantic breakpoints.
    parser.add_argument(
        "--break-percentile",  # CLI-флаг чувствительности semantic splitting.
        type=float,  # Значение может быть дробным.
        default=80.0,  # Дефолтный percentile разрыва.
        help="Semantic chunking break percentile passed to ingest_document.py.",
    )
    # Режим одного сканирования.
    parser.add_argument(
        "--once",  # Если указан, watcher сделает один проход и завершится.
        action="store_true",  # Флаг без значения: есть -> True.
        help="Scan once and exit instead of watching forever.",
    )
    # Dry-run режим без API-запросов.
    parser.add_argument(
        "--dry-run",  # Если указан, команды только печатаются.
        action="store_true",  # Флаг без значения.
        help="Print planned commands without calling LlamaParse or OpenRouter.",
    )
    # Режим пропуска уже лежащих PDF.
    parser.add_argument(
        "--skip-existing",  # Полезно при первом запуске watcher в папке с готовыми PDF.
        action="store_true",  # Флаг без значения.
        help="Mark PDFs currently in the folder as done and only process future files.",
    )
    # Режим повторной попытки для файлов со статусом failed.
    parser.add_argument(
        "--retry-failed",  # Если указан, failed PDF снова попадут в обработку.
        action="store_true",  # Флаг без значения.
        help="Retry PDFs that failed during a previous scan.",
    )
    # Возвращаем объект args со всеми настройками.
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    # Если путь абсолютный, возвращаем его как есть; если относительный, считаем его от корня проекта.
    return path if path.is_absolute() else ROOT_DIR / path


def slugify(value: str) -> str:
    # Заменяем все небезопасные символы на underscore, чтобы получить нормальный doc_id.
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip().lower())
    # Схлопываем несколько underscores подряд и убираем underscores по краям.
    slug = re.sub(r"_+", "_", slug).strip("_")
    # Если после очистки ничего не осталось, используем запасной doc_id.
    return slug or "document"


def read_json_file(path: Path) -> dict[str, Any]:
    # Если sidecar JSON не существует, возвращаем пустой словарь.
    if not path.exists():
        return {}

    # Читаем JSON как UTF-8 и парсим в Python-объект.
    data = json.loads(path.read_text(encoding="utf-8"))
    # Проверяем, что JSON содержит объект, а не список/строку/число.
    if not isinstance(data, dict):
        # Metadata должна быть словарем с полями doc_id, title, author и т.д.
        raise ValueError(f"Metadata sidecar must contain a JSON object: {path}")

    # Возвращаем parsed metadata.
    return data


def build_document_metadata(pdf_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    # Пытаемся прочитать JSON с тем же именем, что и PDF: book.pdf -> book.json.
    sidecar_metadata = read_json_file(pdf_path.with_suffix(".json"))
    # doc_id берем из JSON, а если его нет - делаем slug из имени PDF.
    doc_id = str(sidecar_metadata.get("doc_id") or slugify(pdf_path.stem))
    # title берем из JSON, а если его нет - делаем title из имени файла.
    title = str(sidecar_metadata.get("title") or pdf_path.stem.replace("_", " "))
    # author берем из JSON, а если его нет - берем default-author из CLI.
    author = str(sidecar_metadata.get("author") or args.default_author)
    # year берем из JSON, а если его нет - берем default-year из CLI.
    year = int(sidecar_metadata.get("year") or args.default_year)
    # document_type берем из JSON, а если его нет - берем document-type из CLI.
    document_type = str(sidecar_metadata.get("document_type") or args.document_type)

    # Возвращаем metadata в формате, который дальше нужен ingestion-скрипту.
    return {
        "doc_id": doc_id,  # Стабильный id документа.
        "title": title,  # Название документа.
        "author": author,  # Автор документа.
        "year": year,  # Год документа.
        "document_type": document_type,  # Тип документа.
    }


def load_state(state_file: Path) -> dict[str, Any]:
    # Если state-файла еще нет, начинаем с пустой структуры.
    if not state_file.exists():
        return {"records": {}}

    # Читаем state JSON из файла.
    state = json.loads(state_file.read_text(encoding="utf-8"))
    # Если файл поврежден или содержит не объект, начинаем заново.
    if not isinstance(state, dict):
        return {"records": {}}

    # Достаем records - словарь обработанных файлов.
    records = state.get("records")
    # Если records отсутствует или не словарь, исправляем структуру.
    if not isinstance(records, dict):
        state["records"] = {}

    # Возвращаем валидное состояние watcher.
    return state


def save_state(state_file: Path, state: dict[str, Any]) -> None:
    # Создаем папку для state-файла, если ее еще нет.
    state_file.parent.mkdir(parents=True, exist_ok=True)
    # Сначала пишем во временный файл, чтобы не испортить state при внезапном падении.
    temp_file = state_file.with_name(f"{state_file.name}.tmp")
    # Записываем JSON красиво и с сохранением кириллицы.
    temp_file.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),  # ensure_ascii=False оставляет Unicode читаемым.
        encoding="utf-8",  # UTF-8 нужен для русских metadata.
    )
    # Атомарно заменяем старый state новым.
    temp_file.replace(state_file)


def stat_signature(path: Path) -> dict[str, int | None]:
    # Если файла нет, возвращаем пустую подпись.
    if not path.exists():
        return {
            "size": None,  # Размер неизвестен.
            "mtime_ns": None,  # Время изменения неизвестно.
        }

    # Получаем filesystem-stat файла.
    stat = path.stat()
    # Возвращаем минимальную подпись файла: размер и время изменения.
    return {
        "size": stat.st_size,  # Размер файла в байтах.
        "mtime_ns": stat.st_mtime_ns,  # Время изменения в наносекундах.
    }


def build_file_signature(pdf_path: Path) -> dict[str, dict[str, int | None]]:
    # Подпись включает сам PDF и sidecar JSON, потому что изменение metadata тоже важно.
    return {
        "pdf": stat_signature(pdf_path),  # Подпись PDF.
        "metadata": stat_signature(pdf_path.with_suffix(".json")),  # Подпись JSON metadata.
    }


def should_process_pdf(
    pdf_path: Path,
    state: dict[str, Any],
    retry_failed: bool,
) -> bool:
    # Гарантируем, что в state есть records.
    records = state.setdefault("records", {})
    # Ключом записи делаем абсолютный путь к PDF.
    record_key = str(pdf_path.resolve())
    # Достаем прошлую запись по этому PDF, если она есть.
    record = records.get(record_key)
    # Считаем текущую подпись PDF + metadata JSON.
    current_signature = build_file_signature(pdf_path)

    # Если записи нет, файл новый и его нужно обработать.
    if not record:
        return True

    # Если размер или mtime PDF/JSON поменялись, документ нужно обработать заново.
    if record.get("signature") != current_signature:
        return True

    # Если предыдущая обработка упала, решение зависит от флага retry_failed.
    if record.get("status") == "failed":
        return retry_failed

    # Если статус не done, значит обработка не завершена и файл можно взять снова.
    return record.get("status") != "done"


def wait_until_file_is_stable(
    pdf_path: Path,
    stable_seconds: float,
    poll_seconds: float = 1.0,
) -> None:
    # stable_since хранит момент, с которого файл перестал меняться.
    stable_since: float | None = None
    # previous_signature хранит прошлую подпись файла.
    previous_signature: dict[str, dict[str, int | None]] | None = None

    # Крутим цикл, пока файл не будет стабилен stable_seconds секунд.
    while True:
        # Считываем текущую подпись PDF + metadata.
        current_signature = build_file_signature(pdf_path)

        # Если текущая подпись совпадает с прошлой, файл не менялся.
        if current_signature == previous_signature:
            # Если стабильность уже началась и длится достаточно долго, выходим.
            if stable_since is not None and time.monotonic() - stable_since >= stable_seconds:
                return
        # Если подпись изменилась, файл все еще копируется или metadata обновилась.
        else:
            # Обновляем прошлую подпись.
            previous_signature = current_signature
            # Начинаем отсчет стабильности заново.
            stable_since = time.monotonic()

        # Ждем перед следующей проверкой.
        time.sleep(poll_seconds)


def run_command(command: list[str]) -> None:
    # Печатаем команду, чтобы в консоли было видно, какой шаг запускается.
    print("$ " + " ".join(command), flush=True)
    # Запускаем команду из корня проекта; check=True выбросит ошибку при ненулевом exit code.
    subprocess.run(command, cwd=ROOT_DIR, check=True)


def build_extract_command(pdf_path: Path, markdown_path: Path, tier: str) -> list[str]:
    # Возвращаем команду для PDF -> Markdown через LlamaParse.
    return [
        sys.executable,  # Текущий Python, обычно из env/Scripts/python.exe.
        str(EXTRACT_SCRIPT),  # Скрипт парсинга PDF.
        str(pdf_path),  # Входной PDF.
        "--output",  # Флаг output.
        str(markdown_path),  # Куда сохранить Markdown.
        "--tier",  # Флаг tier.
        tier,  # Выбранный tier LlamaParse.
    ]


def build_ingest_command(
    markdown_path: Path,
    metadata: dict[str, Any],
    args: argparse.Namespace,
) -> list[str]:
    # Собираем команду для Markdown -> chunks -> embeddings -> ChromaDB.
    command = [
        sys.executable,  # Текущий Python из virtualenv.
        str(INGEST_SCRIPT),  # Скрипт ingestion.
        str(markdown_path),  # Входной Markdown.
        "--doc-id",  # Флаг doc_id.
        str(metadata["doc_id"]),  # Значение doc_id.
        "--title",  # Флаг title.
        str(metadata["title"]),  # Название документа.
        "--author",  # Флаг author.
        str(metadata["author"]),  # Автор документа.
        "--year",  # Флаг year.
        str(metadata["year"]),  # Год документа.
        "--document-type",  # Флаг document_type.
        str(metadata["document_type"]),  # Тип документа.
        "--chunking-method",
        str(args.chunking_method),
        "--recursive-chunk-size",
        str(args.recursive_chunk_size),
        "--recursive-overlap",
        str(args.recursive_overlap),
        "--semantic-max-chars",  # Флаг максимального размера semantic chunk.
        str(args.semantic_max_chars),  # Значение из CLI.
        "--semantic-min-chars",  # Флаг минимального размера semantic chunk.
        str(args.semantic_min_chars),  # Значение из CLI.
        "--break-percentile",  # Флаг percentile для semantic splitting.
        str(args.break_percentile),  # Значение из CLI.
    ]

    # Если пользователь явно указал collection, передаем его в ingest_document.py.
    if args.collection:
        # Добавляем два элемента: имя флага и значение.
        command.extend(["--collection", str(args.collection)])

    # Возвращаем готовую команду для subprocess.run.
    return command


def mark_state(
    state: dict[str, Any],
    pdf_path: Path,
    status: str,
    metadata: dict[str, Any] | None = None,
    markdown_path: Path | None = None,
    error: str | None = None,
) -> None:
    # Формируем запись о текущем состоянии PDF.
    record: dict[str, Any] = {
        "status": status,  # processing, done или failed.
        "signature": build_file_signature(pdf_path),  # Подпись PDF + JSON metadata.
        "updated_at": datetime.now().isoformat(timespec="seconds"),  # Когда запись обновилась.
    }

    # Если metadata передана, сохраняем ее в state для удобной диагностики.
    if metadata:
        record["metadata"] = metadata

    # Если путь к Markdown известен, тоже сохраняем его.
    if markdown_path:
        record["markdown_path"] = str(markdown_path)

    # Если была ошибка, сохраняем текст ошибки.
    if error:
        record["error"] = error

    # Записываем record по абсолютному пути PDF.
    state.setdefault("records", {})[str(pdf_path.resolve())] = record


def process_pdf(
    pdf_path: Path,
    args: argparse.Namespace,
    state: dict[str, Any],
    state_file: Path,
) -> None:
    # Сообщаем пользователю, какой PDF найден.
    print(f"Detected PDF: {pdf_path}", flush=True)
    # Ждем, пока PDF перестанет изменяться.
    wait_until_file_is_stable(pdf_path, args.stable_seconds)

    # Собираем metadata документа из sidecar JSON или дефолтов.
    metadata = build_document_metadata(pdf_path, args)
    # Определяем путь, куда будет сохранен Markdown.
    markdown_path = resolve_project_path(args.md_output_dir) / f"{metadata['doc_id']}.md"
    # Строим команду PDF -> Markdown.
    extract_command = build_extract_command(pdf_path, markdown_path, args.tier)
    # Строим команду Markdown -> ChromaDB.
    ingest_command = build_ingest_command(markdown_path, metadata, args)

    # Если включен dry-run, ничего не запускаем, только печатаем команды.
    if args.dry_run:
        # Сообщаем, что это безопасный preview.
        print("Dry run only. Planned commands:", flush=True)
        # Показываем команду парсинга.
        print("$ " + " ".join(extract_command), flush=True)
        # Показываем команду ingestion.
        print("$ " + " ".join(ingest_command), flush=True)
        # Выходим без изменения state.
        return

    # Помечаем PDF как processing до запуска тяжелых шагов.
    mark_state(state, pdf_path, "processing", metadata, markdown_path)
    # Сохраняем state сразу, чтобы при падении было видно, где остановились.
    save_state(state_file, state)

    try:
        # Создаем папку для Markdown, если ее нет.
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        # Запускаем LlamaParse extraction.
        run_command(extract_command)
        # Запускаем ingestion в ChromaDB.
        run_command(ingest_command)
    # Ловим любую ошибку из subprocess или filesystem.
    except Exception as error:
        # Помечаем файл как failed.
        mark_state(state, pdf_path, "failed", metadata, markdown_path, str(error))
        # Сохраняем failed status в state.
        save_state(state_file, state)
        # Печатаем ошибку в stderr.
        print(f"Failed to process {pdf_path}: {error}", file=sys.stderr, flush=True)
        # Возвращаемся, чтобы watcher мог продолжить работу с другими PDF.
        return

    # Если оба шага прошли успешно, помечаем PDF как done.
    mark_state(state, pdf_path, "done", metadata, markdown_path)
    # Сохраняем успешный state.
    save_state(state_file, state)
    # Печатаем итоговый doc_id.
    print(f"Finished: doc_id={metadata['doc_id']}", flush=True)


def scan_once(args: argparse.Namespace, state: dict[str, Any], state_file: Path) -> None:
    # Приводим input-dir к абсолютному пути проекта.
    input_dir = resolve_project_path(args.input_dir)
    # Создаем input-dir, если пользователь еще не создал папку.
    input_dir.mkdir(parents=True, exist_ok=True)

    # Находим все PDF в input-dir и сортируем для стабильного порядка обработки.
    pdf_paths = sorted(path for path in input_dir.glob("*.pdf") if path.is_file())
    # Проходим по каждому PDF.
    for pdf_path in pdf_paths:
        # Проверяем, нужно ли обрабатывать этот PDF.
        if should_process_pdf(pdf_path, state, args.retry_failed):
            # Если нужно, запускаем полный pipeline для PDF.
            process_pdf(pdf_path, args, state, state_file)


def mark_existing_pdfs_done(
    args: argparse.Namespace,
    state: dict[str, Any],
    state_file: Path,
) -> None:
    # Получаем абсолютный путь к input-dir.
    input_dir = resolve_project_path(args.input_dir)
    # Создаем папку, если ее нет.
    input_dir.mkdir(parents=True, exist_ok=True)

    # Счетчик файлов, которые мы отметили как уже обработанные.
    marked_count = 0
    # Проходим по всем PDF, которые уже лежат в папке.
    for pdf_path in sorted(path for path in input_dir.glob("*.pdf") if path.is_file()):
        # Если PDF уже есть в state, не трогаем его.
        if str(pdf_path.resolve()) in state.setdefault("records", {}):
            continue

        # Собираем metadata, чтобы state был информативным.
        metadata = build_document_metadata(pdf_path, args)
        # Считаем ожидаемый путь к Markdown.
        markdown_path = resolve_project_path(args.md_output_dir) / f"{metadata['doc_id']}.md"
        # Помечаем файл как done, не запуская pipeline.
        mark_state(state, pdf_path, "done", metadata, markdown_path)
        # Увеличиваем счетчик.
        marked_count += 1

    # Если были новые записи, сохраняем state.
    if marked_count:
        save_state(state_file, state)

    # Печатаем, сколько файлов было пропущено.
    print(f"Skipped existing PDFs: {marked_count}", flush=True)


def main() -> None:
    # Читаем CLI-аргументы.
    args = parse_args()
    # Приводим input-dir к абсолютному пути.
    args.input_dir = resolve_project_path(args.input_dir)
    # Приводим md-output-dir к абсолютному пути.
    args.md_output_dir = resolve_project_path(args.md_output_dir)
    # Приводим state-file к абсолютному пути.
    state_file = resolve_project_path(args.state_file)
    # Загружаем текущее состояние watcher.
    state = load_state(state_file)

    # Печатаем папку, которую watcher будет мониторить.
    print(f"Watching: {args.input_dir}", flush=True)
    # Печатаем путь к state-файлу.
    print(f"State file: {state_file}", flush=True)

    # Если указан --skip-existing, отмечаем текущие PDF как done.
    if args.skip_existing:
        mark_existing_pdfs_done(args, state, state_file)

    # Главный цикл watcher.
    while True:
        # Выполняем одно сканирование папки.
        scan_once(args, state, state_file)

        # Если включен --once, выходим после первого сканирования.
        if args.once:
            return

        # Иначе ждем scan_interval секунд и повторяем.
        time.sleep(args.scan_interval)


# Этот блок выполняется только при прямом запуске файла.
if __name__ == "__main__":
    # Запускаем основной watcher.
    main()
