from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from dotenv import load_dotenv
from fastapi import HTTPException

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import app as rag_app  # noqa: E402
from src.rag_local.chunk_quality import get_low_quality_reason  # noqa: E402
from src.rag_local.hybrid_search import tokenize  # noqa: E402


DEFAULT_QUERIES = [
    "Кто такой Гарри Поттер?",
    "Что такое Тайная комната?",
    "Кто такой Сириус Блэк?",
    "Почему Добби предупреждал Гарри?",
    "Что такое Азкабан?",
    "Кто такой Волан-де-Морт?",
    "Кто такая Гермиона?",
    "Что такое Хогвартс?",
    "Кто такой Том Реддл?",
    "Что такое василиск?",
    "Что произошло с дневником Тома Реддла?",
    "Почему Гарри слышал странный голос?",
    "Кто такой Дамблдор?",
    "Что такое квиддич?",
    "Кто такие Дурсли?",
    "Почему Сириус Блэк сбежал из Азкабана?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality and inspect noisy top-k chunks."
    )
    parser.add_argument(
        "--queries-file",
        type=Path,
        help="Optional UTF-8 text file with one query per line.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("retrieval_eval_output"),
        help="Directory for JSON and Markdown reports.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="How many chunks to retrieve for each query.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of queries for a cheaper smoke test.",
    )
    parser.add_argument(
        "--no-vector-compare",
        action="store_true",
        help="Do not run vector top-k comparison.",
    )
    return parser.parse_args()


def model_to_dict(model: Any) -> dict[str, Any]:
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


def load_queries(path: Path | None, limit: int | None) -> list[str]:
    if path is None:
        queries = DEFAULT_QUERIES
    else:
        queries = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

    return queries[:limit] if limit is not None else queries


def install_embedding_cache() -> None:
    original_embed_query = rag_app.embed_query
    cache: dict[str, list[float]] = {}

    def cached_embed_query(query: str) -> list[float]:
        if query not in cache:
            cache[query] = original_embed_query(query)

        return cache[query]

    rag_app.embed_query = cached_embed_query


def query_overlap_count(query: str, text: str) -> int:
    query_tokens = set(tokenize(query))
    text_tokens = set(tokenize(text))
    return len(query_tokens & text_tokens)


def analyze_result(query: str, result: dict[str, Any]) -> dict[str, Any]:
    text = str(result.get("text", ""))
    characters = int(result.get("characters") or len(text))
    low_quality_reason = get_low_quality_reason(text, result)
    overlap = query_overlap_count(query, text)
    suspicious_reasons: list[str] = []

    if low_quality_reason is not None:
        suspicious_reasons.append(low_quality_reason)

    if characters < 250:
        suspicious_reasons.append("very_short")

    if overlap == 0:
        suspicious_reasons.append("no_query_token_overlap")

    if result.get("vector_rank") is None and result.get("bm25_rank") is not None:
        suspicious_reasons.append("bm25_only")

    return {
        **result,
        "characters": characters,
        "low_quality_reason": low_quality_reason,
        "query_token_overlap": overlap,
        "suspicious_reasons": suspicious_reasons,
        "text_preview": " ".join(text.split())[:500],
    }


def run_search(query: str, mode: str, top_k: int) -> tuple[list[dict[str, Any]], str | None]:
    response = rag_app.search_top_k(
        rag_app.SearchRequest(query=query, top_k=top_k, search_mode=mode)
    )
    return [model_to_dict(result) for result in response.results], response.warning


def evaluate_query(
    query: str,
    top_k: int,
    compare_vector: bool,
) -> dict[str, Any]:
    try:
        hybrid_results, warning = run_search(query, "hybrid", top_k)
    except HTTPException as error:
        return {
            "query": query,
            "error": str(error.detail),
            "status_code": error.status_code,
        }

    analyzed_hybrid = [analyze_result(query, result) for result in hybrid_results]
    vector_results: list[dict[str, Any]] = []
    vector_error: str | None = None

    if compare_vector:
        try:
            vector_results, _ = run_search(query, "vector", top_k)
        except HTTPException as error:
            vector_error = str(error.detail)

    vector_ids = {str(result["id"]) for result in vector_results}
    hybrid_ids = {str(result["id"]) for result in analyzed_hybrid}
    introduced_by_hybrid = [
        result for result in analyzed_hybrid if str(result["id"]) not in vector_ids
    ]

    suspicious_counts = Counter(
        reason
        for result in analyzed_hybrid
        for reason in result["suspicious_reasons"]
    )
    character_counts = [result["characters"] for result in analyzed_hybrid]
    low_quality_count = sum(
        1 for result in analyzed_hybrid if result["low_quality_reason"] is not None
    )

    return {
        "query": query,
        "warning": warning,
        "vector_error": vector_error,
        "hybrid_count": len(analyzed_hybrid),
        "vector_count": len(vector_results),
        "hybrid_vector_overlap": len(hybrid_ids & vector_ids),
        "hybrid_not_in_vector_count": len(introduced_by_hybrid),
        "low_quality_count": low_quality_count,
        "very_short_count": suspicious_counts["very_short"],
        "no_query_token_overlap_count": suspicious_counts["no_query_token_overlap"],
        "bm25_only_count": suspicious_counts["bm25_only"],
        "avg_characters": round(mean(character_counts), 2) if character_counts else 0,
        "top_5_suspicious_count": sum(
            1 for result in analyzed_hybrid[:5] if result["suspicious_reasons"]
        ),
        "suspicious_counts": dict(suspicious_counts),
        "results": analyzed_hybrid,
    }


def write_json_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_markdown_report(report: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Retrieval Evaluation Report",
        "",
        f"Top K: {report['top_k']}",
        f"Queries: {report['query_count']}",
        f"Compare vector: {report['compare_vector']}",
        "",
        "## Summary",
        "",
    ]

    for key, value in report["summary"].items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Query Details", ""])

    for item in report["queries"]:
        lines.append(f"### {item['query']}")
        lines.append("")

        if "error" in item:
            lines.append(f"Error: `{item['error']}`")
            lines.append("")
            continue

        lines.extend(
            [
                f"- warning: {item['warning'] or 'none'}",
                f"- hybrid/vector overlap: {item['hybrid_vector_overlap']}/{item['hybrid_count']}",
                f"- hybrid_not_in_vector_count: {item['hybrid_not_in_vector_count']}",
                f"- low_quality_count: {item['low_quality_count']}",
                f"- very_short_count: {item['very_short_count']}",
                f"- no_query_token_overlap_count: {item['no_query_token_overlap_count']}",
                f"- bm25_only_count: {item['bm25_only_count']}",
                f"- avg_characters: {item['avg_characters']}",
                f"- top_5_suspicious_count: {item['top_5_suspicious_count']}",
                "",
                "| rank | id | chars | vector | bm25 | sim | rrf | suspicious | preview |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )

        for result in item["results"]:
            suspicious = ", ".join(result["suspicious_reasons"]) or "-"
            preview = result["text_preview"].replace("|", "\\|")
            lines.append(
                "| "
                f"{result['rank']} | "
                f"{result['id']} | "
                f"{result['characters']} | "
                f"{result.get('vector_rank') or '-'} | "
                f"{result.get('bm25_rank') or '-'} | "
                f"{round(result['similarity'], 4) if result.get('similarity') is not None else '-'} | "
                f"{round(result['rrf_score'], 4) if result.get('rrf_score') is not None else '-'} | "
                f"{suspicious} | "
                f"{preview} |"
            )

        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [item for item in items if "error" not in item]

    if not successful:
        return {"successful_queries": 0}

    return {
        "successful_queries": len(successful),
        "total_low_quality": sum(item["low_quality_count"] for item in successful),
        "total_very_short": sum(item["very_short_count"] for item in successful),
        "total_no_query_token_overlap": sum(
            item["no_query_token_overlap_count"] for item in successful
        ),
        "total_bm25_only": sum(item["bm25_only_count"] for item in successful),
        "avg_hybrid_vector_overlap": round(
            mean(item["hybrid_vector_overlap"] for item in successful), 2
        ),
        "avg_hybrid_not_in_vector": round(
            mean(item["hybrid_not_in_vector_count"] for item in successful), 2
        ),
        "avg_top_5_suspicious": round(
            mean(item["top_5_suspicious_count"] for item in successful), 2
        ),
    }


def main() -> None:
    load_dotenv()
    args = parse_args()
    queries = load_queries(args.queries_file, args.limit)
    compare_vector = not args.no_vector_compare
    args.output_dir.mkdir(parents=True, exist_ok=True)
    install_embedding_cache()

    items = [
        evaluate_query(query, args.top_k, compare_vector)
        for query in queries
    ]
    report = {
        "top_k": args.top_k,
        "query_count": len(queries),
        "compare_vector": compare_vector,
        "summary": build_summary(items),
        "queries": items,
    }

    json_path = args.output_dir / "hybrid_top20_results.json"
    markdown_path = args.output_dir / "hybrid_top20_report.md"
    write_json_report(report, json_path)
    write_markdown_report(report, markdown_path)

    print(f"Saved JSON report to {json_path}")
    print(f"Saved Markdown report to {markdown_path}")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
