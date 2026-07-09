from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.evaluate_chunking_methods import judge_answer, summarize_method, write_markdown_report
from src.rag_local.embeddings import sanitize_model_name
from src.rag_local.llm import OpenRouterChatClient


DEFAULT_JUDGE_MODEL = "openai/gpt-4o-mini-2024-07-18"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-judge already generated RAG answers with a different OpenRouter model "
            "without regenerating answers or embeddings."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("chunking_eval_output/chunking_methods_results.json"),
        help="Existing chunking evaluation JSON with generated_answer fields.",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help="OpenRouter chat model used only as LLM judge.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path. Default is derived from judge model name.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        help="Output Markdown report path. Default is derived from judge model name.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        help="Optional subset of methods to judge, for example: semantic fixed.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional number of items per method for a smoke test.",
    )
    return parser.parse_args()


def resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    input_path = ROOT_DIR / args.input if not args.input.is_absolute() else args.input
    suffix = sanitize_model_name(args.judge_model)
    default_json = input_path.with_name(f"{input_path.stem}_judged_{suffix}.json")
    default_markdown = input_path.with_name(f"{input_path.stem}_judged_{suffix}.md")

    json_path = args.output or default_json
    markdown_path = args.markdown_output or default_markdown

    if not json_path.is_absolute():
        json_path = ROOT_DIR / json_path

    if not markdown_path.is_absolute():
        markdown_path = ROOT_DIR / markdown_path

    return json_path, markdown_path


def load_report(input_path: Path) -> dict[str, Any]:
    resolved_input = ROOT_DIR / input_path if not input_path.is_absolute() else input_path
    return json.loads(resolved_input.read_text(encoding="utf-8"))


def selected_methods(report: dict[str, Any], methods: list[str] | None) -> list[str]:
    available_methods = list(report.get("methods", {}).keys())
    if not methods:
        return available_methods

    unknown_methods = sorted(set(methods) - set(available_methods))
    if unknown_methods:
        raise ValueError(f"Unknown methods in report: {', '.join(unknown_methods)}")

    return methods


def get_generated_answer(item: dict[str, Any]) -> str:
    generated_answer = item.get("generated_answer")
    if not isinstance(generated_answer, dict):
        raise ValueError(f"Item {item.get('id')} has no generated_answer object.")

    answer = str(generated_answer.get("answer", "")).strip()
    if not answer:
        raise ValueError(f"Item {item.get('id')} has empty generated answer.")

    return answer


def judge_existing_answers(
    report: dict[str, Any],
    judge_model: str,
    methods: list[str] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    updated_report = deepcopy(report)
    judged_methods = selected_methods(updated_report, methods)
    judge_client = OpenRouterChatClient(judge_model)

    updated_report["answer_model"] = report.get("chat_model")
    updated_report["chat_model"] = report.get("chat_model")
    updated_report["judge_model"] = judge_model
    updated_report["judge_source"] = "existing_generated_answers"
    updated_report["with_llm_judge"] = True

    for method in judged_methods:
        method_report = updated_report["methods"][method]
        items = method_report["items"]
        items_to_judge = items[:limit] if limit else items

        for index, item in enumerate(items_to_judge, start=1):
            answer = get_generated_answer(item)
            print(
                f"Judging {method} {index}/{len(items_to_judge)}: {item['id']}",
                flush=True,
            )
            item["llm_judge"] = judge_answer(
                chat_client=judge_client,
                item=item,
                candidate_answer=answer,
            )
            item["llm_judge_model"] = judge_model

        method_report["items"] = items
        updated_report["summary"][method] = summarize_method(
            items,
            int(method_report["chunk_count"]),
        )

    return updated_report


def main() -> None:
    load_dotenv()
    args = parse_args()
    report = load_report(args.input)
    output_path, markdown_output_path = resolve_output_paths(args)
    updated_report = judge_existing_answers(
        report=report,
        judge_model=args.judge_model,
        methods=args.methods,
        limit=args.limit,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(updated_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown_report(updated_report, markdown_output_path)

    print(f"Saved JSON report to {output_path}")
    print(f"Saved Markdown report to {markdown_output_path}")
    print(json.dumps(updated_report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
