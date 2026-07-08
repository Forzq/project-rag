from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
EXTRACT_SCRIPT = ROOT_DIR / "scripts" / "extract_complex_pdf_llamaparse.py"
INGEST_SCRIPT = ROOT_DIR / "scripts" / "ingest_document.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Watch pdf_input and automatically run PDF parsing, chunking, "
            "embedding generation, and ChromaDB loading for new PDF files."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("pdf_input"),
        help="Directory to watch for new PDF files.",
    )
    parser.add_argument(
        "--md-output-dir",
        type=Path,
        default=Path("md_output"),
        help="Directory where parsed Markdown files are saved.",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path("pdf_input") / ".ingest_state.json",
        help="JSON file used to remember already processed PDFs.",
    )
    parser.add_argument(
        "--scan-interval",
        type=float,
        default=10.0,
        help="Seconds between folder scans.",
    )
    parser.add_argument(
        "--stable-seconds",
        type=float,
        default=5.0,
        help="How long a PDF size must stay unchanged before processing.",
    )
    parser.add_argument(
        "--tier",
        default="cost_effective",
        choices=["cost_effective", "agentic", "agentic_plus"],
        help="LlamaParse parsing tier.",
    )
    parser.add_argument(
        "--default-author",
        default="Unknown",
        help="Author used when sidecar metadata does not define author.",
    )
    parser.add_argument(
        "--default-year",
        type=int,
        default=datetime.now().year,
        help="Year used when sidecar metadata does not define year.",
    )
    parser.add_argument(
        "--document-type",
        default="book",
        help="Document type used when sidecar metadata does not define document_type.",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Target Chroma collection. If omitted, ingest_document.py uses its default.",
    )
    parser.add_argument(
        "--semantic-max-chars",
        type=int,
        default=2000,
        help="Maximum semantic chunk size passed to ingest_document.py.",
    )
    parser.add_argument(
        "--semantic-min-chars",
        type=int,
        default=300,
        help="Minimum semantic chunk size passed to ingest_document.py.",
    )
    parser.add_argument(
        "--break-percentile",
        type=float,
        default=80.0,
        help="Semantic chunking break percentile passed to ingest_document.py.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Scan once and exit instead of watching forever.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without calling LlamaParse or OpenRouter.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Mark PDFs currently in the folder as done and only process future files.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry PDFs that failed during a previous scan.",
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT_DIR / path


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "document"


def read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Metadata sidecar must contain a JSON object: {path}")

    return data


def build_document_metadata(pdf_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    sidecar_metadata = read_json_file(pdf_path.with_suffix(".json"))
    doc_id = str(sidecar_metadata.get("doc_id") or slugify(pdf_path.stem))
    title = str(sidecar_metadata.get("title") or pdf_path.stem.replace("_", " "))
    author = str(sidecar_metadata.get("author") or args.default_author)
    year = int(sidecar_metadata.get("year") or args.default_year)
    document_type = str(sidecar_metadata.get("document_type") or args.document_type)

    return {
        "doc_id": doc_id,
        "title": title,
        "author": author,
        "year": year,
        "document_type": document_type,
    }


def load_state(state_file: Path) -> dict[str, Any]:
    if not state_file.exists():
        return {"records": {}}

    state = json.loads(state_file.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        return {"records": {}}

    records = state.get("records")
    if not isinstance(records, dict):
        state["records"] = {}

    return state


def save_state(state_file: Path, state: dict[str, Any]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = state_file.with_name(f"{state_file.name}.tmp")
    temp_file.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_file.replace(state_file)


def stat_signature(path: Path) -> dict[str, int | None]:
    if not path.exists():
        return {
            "size": None,
            "mtime_ns": None,
        }

    stat = path.stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def build_file_signature(pdf_path: Path) -> dict[str, dict[str, int | None]]:
    return {
        "pdf": stat_signature(pdf_path),
        "metadata": stat_signature(pdf_path.with_suffix(".json")),
    }


def should_process_pdf(
    pdf_path: Path,
    state: dict[str, Any],
    retry_failed: bool,
) -> bool:
    records = state.setdefault("records", {})
    record_key = str(pdf_path.resolve())
    record = records.get(record_key)
    current_signature = build_file_signature(pdf_path)

    if not record:
        return True

    if record.get("signature") != current_signature:
        return True

    if record.get("status") == "failed":
        return retry_failed

    return record.get("status") != "done"


def wait_until_file_is_stable(
    pdf_path: Path,
    stable_seconds: float,
    poll_seconds: float = 1.0,
) -> None:
    stable_since: float | None = None
    previous_signature: dict[str, dict[str, int | None]] | None = None

    while True:
        current_signature = build_file_signature(pdf_path)

        if current_signature == previous_signature:
            if stable_since is not None and time.monotonic() - stable_since >= stable_seconds:
                return
        else:
            previous_signature = current_signature
            stable_since = time.monotonic()

        time.sleep(poll_seconds)


def run_command(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT_DIR, check=True)


def build_extract_command(pdf_path: Path, markdown_path: Path, tier: str) -> list[str]:
    return [
        sys.executable,
        str(EXTRACT_SCRIPT),
        str(pdf_path),
        "--output",
        str(markdown_path),
        "--tier",
        tier,
    ]


def build_ingest_command(
    markdown_path: Path,
    metadata: dict[str, Any],
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        str(INGEST_SCRIPT),
        str(markdown_path),
        "--doc-id",
        str(metadata["doc_id"]),
        "--title",
        str(metadata["title"]),
        "--author",
        str(metadata["author"]),
        "--year",
        str(metadata["year"]),
        "--document-type",
        str(metadata["document_type"]),
        "--semantic-max-chars",
        str(args.semantic_max_chars),
        "--semantic-min-chars",
        str(args.semantic_min_chars),
        "--break-percentile",
        str(args.break_percentile),
    ]

    if args.collection:
        command.extend(["--collection", str(args.collection)])

    return command


def mark_state(
    state: dict[str, Any],
    pdf_path: Path,
    status: str,
    metadata: dict[str, Any] | None = None,
    markdown_path: Path | None = None,
    error: str | None = None,
) -> None:
    record: dict[str, Any] = {
        "status": status,
        "signature": build_file_signature(pdf_path),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }

    if metadata:
        record["metadata"] = metadata

    if markdown_path:
        record["markdown_path"] = str(markdown_path)

    if error:
        record["error"] = error

    state.setdefault("records", {})[str(pdf_path.resolve())] = record


def process_pdf(
    pdf_path: Path,
    args: argparse.Namespace,
    state: dict[str, Any],
    state_file: Path,
) -> None:
    print(f"Detected PDF: {pdf_path}", flush=True)
    wait_until_file_is_stable(pdf_path, args.stable_seconds)

    metadata = build_document_metadata(pdf_path, args)
    markdown_path = resolve_project_path(args.md_output_dir) / f"{metadata['doc_id']}.md"
    extract_command = build_extract_command(pdf_path, markdown_path, args.tier)
    ingest_command = build_ingest_command(markdown_path, metadata, args)

    if args.dry_run:
        print("Dry run only. Planned commands:", flush=True)
        print("$ " + " ".join(extract_command), flush=True)
        print("$ " + " ".join(ingest_command), flush=True)
        return

    mark_state(state, pdf_path, "processing", metadata, markdown_path)
    save_state(state_file, state)

    try:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        run_command(extract_command)
        run_command(ingest_command)
    except Exception as error:
        mark_state(state, pdf_path, "failed", metadata, markdown_path, str(error))
        save_state(state_file, state)
        print(f"Failed to process {pdf_path}: {error}", file=sys.stderr, flush=True)
        return

    mark_state(state, pdf_path, "done", metadata, markdown_path)
    save_state(state_file, state)
    print(f"Finished: doc_id={metadata['doc_id']}", flush=True)


def scan_once(args: argparse.Namespace, state: dict[str, Any], state_file: Path) -> None:
    input_dir = resolve_project_path(args.input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted(path for path in input_dir.glob("*.pdf") if path.is_file())
    for pdf_path in pdf_paths:
        if should_process_pdf(pdf_path, state, args.retry_failed):
            process_pdf(pdf_path, args, state, state_file)


def mark_existing_pdfs_done(
    args: argparse.Namespace,
    state: dict[str, Any],
    state_file: Path,
) -> None:
    input_dir = resolve_project_path(args.input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    marked_count = 0
    for pdf_path in sorted(path for path in input_dir.glob("*.pdf") if path.is_file()):
        if str(pdf_path.resolve()) in state.setdefault("records", {}):
            continue

        metadata = build_document_metadata(pdf_path, args)
        markdown_path = resolve_project_path(args.md_output_dir) / f"{metadata['doc_id']}.md"
        mark_state(state, pdf_path, "done", metadata, markdown_path)
        marked_count += 1

    if marked_count:
        save_state(state_file, state)

    print(f"Skipped existing PDFs: {marked_count}", flush=True)


def main() -> None:
    args = parse_args()
    args.input_dir = resolve_project_path(args.input_dir)
    args.md_output_dir = resolve_project_path(args.md_output_dir)
    state_file = resolve_project_path(args.state_file)
    state = load_state(state_file)

    print(f"Watching: {args.input_dir}", flush=True)
    print(f"State file: {state_file}", flush=True)

    if args.skip_existing:
        mark_existing_pdfs_done(args, state, state_file)

    while True:
        scan_once(args, state, state_file)

        if args.once:
            return

        time.sleep(args.scan_interval)


if __name__ == "__main__":
    main()
