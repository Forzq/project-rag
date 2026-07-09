from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rag_local.chunk_quality import filter_quality_chunks
from src.rag_local.chunking import split_fixed_size, split_recursive
from src.rag_local.answer_generation import generate_answer
from src.rag_local.config import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_DIRECT_CONTEXT_TOKEN_LIMIT,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RETRIEVAL_MODEL,
)
from src.rag_local.embeddings import OpenRouterEmbedder, sanitize_model_name
from src.rag_local.hybrid_search import (
    BM25Index,
    StoredChunk,
    reciprocal_rank_fusion,
)
from src.rag_local.llm import OpenRouterChatClient, resolve_chat_model
from src.rag_local.preprocessing import clean_extracted_markdown
from src.rag_local.reranker import CrossEncoderReranker


DOCUMENTS = {
    "harry_potter_1": {
        "path": Path("md_output/Harry_Potter_1.md"),
        "title": "Harry Potter and the Philosopher's Stone",
        "author": "J. K. Rowling",
        "year": 1997,
        "document_type": "book",
    },
    "harry_potter_2": {
        "path": Path("md_output/Harry_Potter_2.md"),
        "title": "Harry Potter and the Chamber of Secrets",
        "author": "J. K. Rowling",
        "year": 1998,
        "document_type": "book",
    },
    "harry_potter_3": {
        "path": Path("md_output/Harry_Potter_3.md"),
        "title": "Harry Potter and the Prisoner of Azkaban",
        "author": "J. K. Rowling",
        "year": 1999,
        "document_type": "book",
    },
}
METHODS = ("semantic", "fixed", "recursive")
VECTOR_RRF_WEIGHT = 0.75
BM25_RRF_WEIGHT = 0.25


@dataclass(frozen=True)
class MethodIndex:
    method: str
    chunks: list[dict[str, Any]]
    embeddings: np.ndarray
    embedding_model: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed, recursive, and semantic chunking on a gold RAG dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("eval_data/gold_rag_dataset.json"),
        help="Gold dataset JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("chunking_eval_output"),
        help="Directory for cached embeddings and reports.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
        help="Chunking methods to evaluate.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k used for retrieval evaluation.",
    )
    parser.add_argument(
        "--fixed-chunk-size",
        type=int,
        default=1500,
        help="Fixed-size chunk length in characters.",
    )
    parser.add_argument(
        "--fixed-overlap",
        type=int,
        default=200,
        help="Fixed-size chunk overlap in characters.",
    )
    parser.add_argument(
        "--recursive-chunk-size",
        type=int,
        default=1500,
        help="Recursive chunk length in characters.",
    )
    parser.add_argument(
        "--recursive-overlap",
        type=int,
        default=200,
        help="Recursive chunk overlap in characters.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_RETRIEVAL_MODEL,
        help="OpenRouter embedding model.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=32,
        help="Batch size for OpenRouter embeddings.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate only first N gold questions.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild cached non-semantic chunk embeddings.",
    )
    parser.add_argument(
        "--with-answers",
        action="store_true",
        help="Generate RAG answers for each question and chunking method.",
    )
    parser.add_argument(
        "--with-llm-judge",
        action="store_true",
        help="Use an LLM judge to compare generated answers with reference answers.",
    )
    parser.add_argument(
        "--answer-top-n",
        type=int,
        default=5,
        help="How many retrieved chunks to pass to the answer generator.",
    )
    parser.add_argument(
        "--with-reranker",
        action="store_true",
        help="Rerank hybrid top-k candidates before answer generation.",
    )
    parser.add_argument(
        "--reranker-model",
        default=DEFAULT_RERANKER_MODEL,
        help="SentenceTransformers CrossEncoder model used for reranking.",
    )
    parser.add_argument(
        "--chat-model",
        default=resolve_chat_model(DEFAULT_CHAT_MODEL),
        help="OpenRouter chat model used for answer generation and judging.",
    )
    return parser.parse_args()


def load_gold_items(dataset_path: Path, limit: int | None) -> list[dict[str, Any]]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    items = list(dataset["items"])
    return items[:limit] if limit else items


def metadata_path_for_semantic(doc_id: str) -> Path:
    return ROOT_DIR / "embeddings_output" / f"{doc_id}_semantic" / "chunks_metadata.json"


def embedding_path_for_semantic(doc_id: str, model: str) -> Path:
    return (
        ROOT_DIR
        / "embeddings_output"
        / f"{doc_id}_semantic"
        / f"{sanitize_model_name(model)}.npy"
    )


def cache_paths(output_dir: Path, method: str, model: str) -> tuple[Path, Path]:
    method_dir = output_dir / method
    return (
        method_dir / "chunks_metadata.json",
        method_dir / f"{sanitize_model_name(model)}.npy",
    )


def enrich_chunk(
    chunk_text: str,
    index: int,
    doc_id: str,
    document_metadata: dict[str, Any],
    method: str,
) -> dict[str, Any]:
    return {
        "id": f"{doc_id}-{method}-chunk-{index + 1}",
        "index": index,
        "chunk_id": index + 1,
        "characters": len(chunk_text),
        "source_file": str(document_metadata["path"]),
        "text": chunk_text,
        "doc_id": doc_id,
        "title": document_metadata["title"],
        "author": document_metadata["author"],
        "year": document_metadata["year"],
        "document_type": document_metadata["document_type"],
        "chunking_method": method,
    }


def generate_chunks_for_method(method: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []

    for doc_id, document_metadata in DOCUMENTS.items():
        source_path = ROOT_DIR / document_metadata["path"]
        text = clean_extracted_markdown(source_path.read_text(encoding="utf-8"))

        if method == "fixed":
            chunk_texts = split_fixed_size(
                text,
                chunk_size=args.fixed_chunk_size,
                overlap=args.fixed_overlap,
            )
        elif method == "recursive":
            chunk_texts = split_recursive(
                text,
                chunk_size=args.recursive_chunk_size,
                overlap=args.recursive_overlap,
            )
        else:
            raise ValueError(f"Cannot generate method with this function: {method}")

        enriched = [
            enrich_chunk(chunk_text, index, doc_id, document_metadata, method)
            for index, chunk_text in enumerate(chunk_texts)
        ]
        kept_chunks, _ = filter_quality_chunks(enriched)
        chunks.extend(kept_chunks)

    return chunks


def load_semantic_index(model: str) -> MethodIndex:
    all_chunks: list[dict[str, Any]] = []
    all_embeddings: list[np.ndarray] = []

    for doc_id in DOCUMENTS:
        metadata_path = metadata_path_for_semantic(doc_id)
        embedding_path = embedding_path_for_semantic(doc_id, model)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        chunks = metadata["chunks"]
        embeddings = np.load(embedding_path)

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Semantic chunks/embeddings mismatch for {doc_id}: "
                f"{len(chunks)} chunks, {len(embeddings)} embeddings."
            )

        for chunk in chunks:
            chunk["id"] = f"{chunk['doc_id']}-semantic-chunk-{chunk['chunk_id']}"
            chunk["chunking_method"] = "semantic"

        all_chunks.extend(chunks)
        all_embeddings.append(embeddings)

    return MethodIndex(
        method="semantic",
        chunks=all_chunks,
        embeddings=np.vstack(all_embeddings).astype(np.float32),
        embedding_model=model,
    )


def save_method_cache(
    chunks: list[dict[str, Any]],
    embeddings: np.ndarray,
    metadata_path: Path,
    embeddings_path: Path,
    method: str,
    model: str,
) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "method": method,
                "embedding_model": model,
                "chunk_count": len(chunks),
                "chunks": chunks,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    np.save(embeddings_path, embeddings.astype(np.float32))


def load_or_build_generated_index(
    method: str,
    args: argparse.Namespace,
    embedder: OpenRouterEmbedder,
) -> MethodIndex:
    metadata_path, embeddings_path = cache_paths(
        ROOT_DIR / args.output_dir,
        method,
        args.embedding_model,
    )

    if not args.force_rebuild and metadata_path.exists() and embeddings_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        embeddings = np.load(embeddings_path)
        chunks = metadata["chunks"]

        if len(chunks) == len(embeddings):
            return MethodIndex(
                method=method,
                chunks=chunks,
                embeddings=embeddings.astype(np.float32),
                embedding_model=args.embedding_model,
            )

    chunks = generate_chunks_for_method(method, args)
    texts = [str(chunk["text"]) for chunk in chunks]
    embeddings = embedder.embed_texts_numpy(texts)
    save_method_cache(
        chunks=chunks,
        embeddings=embeddings,
        metadata_path=metadata_path,
        embeddings_path=embeddings_path,
        method=method,
        model=args.embedding_model,
    )

    return MethodIndex(
        method=method,
        chunks=chunks,
        embeddings=embeddings,
        embedding_model=args.embedding_model,
    )


def load_or_build_query_embeddings(
    items: list[dict[str, Any]],
    args: argparse.Namespace,
    embedder: OpenRouterEmbedder,
) -> dict[str, list[float]]:
    cache_dir = ROOT_DIR / args.output_dir / "query_embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_name = sanitize_model_name(args.embedding_model)
    metadata_path = cache_dir / "queries.json"
    embeddings_path = cache_dir / f"{model_name}.npy"
    queries = [str(item["question"]) for item in items]

    if metadata_path.exists() and embeddings_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        embeddings = np.load(embeddings_path)
        if metadata.get("queries") == queries and len(embeddings) == len(queries):
            return {
                query: embeddings[index].astype(float).tolist()
                for index, query in enumerate(queries)
            }

    embeddings = embedder.embed_texts_numpy(queries)
    metadata_path.write_text(
        json.dumps(
            {
                "embedding_model": args.embedding_model,
                "queries": queries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    np.save(embeddings_path, embeddings.astype(np.float32))

    return {
        query: embeddings[index].astype(float).tolist()
        for index, query in enumerate(queries)
    }


def vector_search(
    method_index: MethodIndex,
    query_embedding: list[float],
    top_k: int,
) -> list[dict[str, Any]]:
    matrix = method_index.embeddings.astype(np.float32)
    query = np.asarray(query_embedding, dtype=np.float32)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    query_norm = np.linalg.norm(query)
    similarities = (matrix @ query) / np.maximum(matrix_norms * query_norm, 1e-12)
    top_indices = np.argsort(-similarities)[:top_k]

    results: list[dict[str, Any]] = []
    for rank, index in enumerate(top_indices, start=1):
        chunk = method_index.chunks[int(index)]
        similarity = float(similarities[int(index)])
        results.append(
            {
                **chunk,
                "rank": rank,
                "similarity": similarity,
                "distance": 1 - similarity,
                "vector_rank": rank,
            }
        )

    return results


def bm25_search(method_index: MethodIndex, query: str, top_k: int) -> list[dict[str, Any]]:
    stored_chunks = [
        StoredChunk(
            id=str(chunk["id"]),
            text=str(chunk["text"]),
            metadata=chunk,
        )
        for chunk in method_index.chunks
    ]
    index = BM25Index(stored_chunks)
    raw_results = index.search(query, top_k)

    return [
        {
            **dict(result["metadata"]),
            "id": str(result["id"]),
            "text": str(result["text"]),
            "rank": rank,
            "bm25_score": float(result["bm25_score"]),
            "bm25_rank": int(result["bm25_rank"]),
            "similarity": None,
            "distance": None,
        }
        for rank, result in enumerate(raw_results, start=1)
    ]


def hybrid_search(
    method_index: MethodIndex,
    query: str,
    query_embedding: list[float],
    top_k: int,
) -> list[dict[str, Any]]:
    candidate_k = min(max(top_k * 5, 50), len(method_index.chunks))
    vector_results = vector_search(method_index, query_embedding, candidate_k)
    bm25_results = bm25_search(method_index, query, candidate_k)
    fused = reciprocal_rank_fusion(
        [vector_results, bm25_results],
        weights=[VECTOR_RRF_WEIGHT, BM25_RRF_WEIGHT],
    )

    for rank, result in enumerate(fused[:top_k], start=1):
        result["rank"] = rank

    return fused[:top_k]


def lower_terms(groups: list[list[str]]) -> list[list[str]]:
    return [[term.lower() for term in group] for group in groups]


def group_is_covered(text: str, group: list[str]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in group)


def context_term_recall(results: list[dict[str, Any]], item: dict[str, Any]) -> float:
    groups = lower_terms(item.get("must_contain_any", []))
    if not groups:
        return 0.0

    context = "\n\n".join(str(result["text"]) for result in results).lower()
    covered = sum(1 for group in groups if any(term in context for term in group))
    return covered / len(groups)


def first_expected_doc_rank(results: list[dict[str, Any]], expected_doc_ids: set[str]) -> int | None:
    for result in results:
        if str(result.get("doc_id")) in expected_doc_ids:
            return int(result["rank"])

    return None


def first_full_term_rank(results: list[dict[str, Any]], item: dict[str, Any]) -> int | None:
    groups = lower_terms(item.get("must_contain_any", []))
    if not groups:
        return None

    for result in results:
        text = str(result["text"]).lower()
        if all(group_is_covered(text, group) for group in groups):
            return int(result["rank"])

    return None


def extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return {
            "score": 0.0,
            "passed": False,
            "reason": f"Judge returned non-JSON response: {text[:300]}",
        }

    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {
            "score": 0.0,
            "passed": False,
            "reason": f"Judge returned invalid JSON: {text[:300]}",
        }

    score = float(data.get("score", 0.0))
    score = max(0.0, min(1.0, score))

    return {
        "score": score,
        "passed": bool(data.get("passed", score >= 0.7)),
        "reason": str(data.get("reason", ""))[:500],
    }


def build_judge_messages(
    question: str,
    reference_answer: str,
    candidate_answer: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Ты оцениваешь качество RAG-ответа. Сравни ответ кандидата с эталоном. "
                "Не требуй дословного совпадения, оцени фактическую эквивалентность. "
                "Игнорируй стиль, если смысл верный. "
                "Верни только JSON без markdown: "
                "{\"score\": 0.0-1.0, \"passed\": true/false, \"reason\": \"short reason in Russian\"}."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Вопрос:\n{question}\n\n"
                f"Эталонный ответ:\n{reference_answer}\n\n"
                f"Ответ кандидата:\n{candidate_answer}\n\n"
                "Оценка JSON:"
            ),
        },
    ]


def judge_answer(
    chat_client: OpenRouterChatClient,
    item: dict[str, Any],
    candidate_answer: str,
) -> dict[str, Any]:
    raw_judgement = chat_client.complete(
        build_judge_messages(
            question=str(item["question"]),
            reference_answer=str(item["reference_answer"]),
            candidate_answer=candidate_answer,
        ),
        temperature=0.0,
        max_tokens=220,
    )
    judgement = extract_json_object(raw_judgement)
    judgement["raw"] = raw_judgement
    return judgement


def generate_rag_answer(
    chat_client: OpenRouterChatClient,
    item: dict[str, Any],
    results: list[dict[str, Any]],
    answer_top_n: int,
) -> dict[str, Any]:
    source_chunks = results[:answer_top_n]
    generated = generate_answer(
        question=str(item["question"]),
        chunks=source_chunks,
        chat_client=chat_client,
        strategy="direct",
        context_token_limit=DEFAULT_DIRECT_CONTEXT_TOKEN_LIMIT,
    )

    return {
        "answer": generated.answer,
        "strategy": generated.strategy,
        "estimated_context_tokens": generated.estimated_context_tokens,
        "source_count": len(source_chunks),
    }


def evaluate_item(
    method_index: MethodIndex,
    item: dict[str, Any],
    query_embedding: list[float],
    top_k: int,
    chat_client: OpenRouterChatClient | None = None,
    answer_top_n: int = 5,
    judge_answers: bool = False,
    reranker: CrossEncoderReranker | None = None,
) -> dict[str, Any]:
    results = hybrid_search(
        method_index=method_index,
        query=str(item["question"]),
        query_embedding=query_embedding,
        top_k=top_k,
    )
    expected_doc_ids = {str(doc_id) for doc_id in item["expected_doc_ids"]}
    top_5 = results[:5]
    doc_rank = first_expected_doc_rank(results, expected_doc_ids)
    full_term_rank = first_full_term_rank(results, item)

    evaluation = {
        "id": item["id"],
        "question": item["question"],
        "reference_answer": item["reference_answer"],
        "expected_doc_ids": sorted(expected_doc_ids),
        "question_type": item.get("question_type"),
        "difficulty": item.get("difficulty"),
        "doc_hit_at_5": first_expected_doc_rank(top_5, expected_doc_ids) is not None,
        "doc_hit_at_20": doc_rank is not None,
        "first_expected_doc_rank": doc_rank,
        "mrr_doc": 0.0 if doc_rank is None else 1 / doc_rank,
        "term_recall_at_5": round(context_term_recall(top_5, item), 4),
        "term_recall_at_20": round(context_term_recall(results, item), 4),
        "full_term_hit_at_20": full_term_rank is not None,
        "first_full_term_rank": full_term_rank,
        "top_results": [
            {
                "rank": result["rank"],
                "id": result["id"],
                "doc_id": result.get("doc_id"),
                "chunk_id": result.get("chunk_id"),
                "similarity": result.get("similarity"),
                "bm25_score": result.get("bm25_score"),
                "rrf_score": result.get("rrf_score"),
                "characters": result.get("characters"),
                "text_preview": " ".join(str(result["text"]).split())[:320],
            }
            for result in results
        ],
    }

    if chat_client is not None:
        try:
            answer_results = results
            if reranker is not None:
                answer_results = reranker.rerank(
                    query=str(item["question"]),
                    candidates=results,
                    top_n=answer_top_n,
                )

            evaluation["answer_context"] = {
                "reranked": reranker is not None,
                "source_ids": [
                    str(result["id"])
                    for result in answer_results[:answer_top_n]
                ],
            }
            answer_data = generate_rag_answer(
                chat_client=chat_client,
                item=item,
                results=answer_results,
                answer_top_n=answer_top_n,
            )
            evaluation["generated_answer"] = answer_data

            if judge_answers:
                evaluation["llm_judge"] = judge_answer(
                    chat_client=chat_client,
                    item=item,
                    candidate_answer=str(answer_data["answer"]),
                )
        except Exception as error:
            evaluation["answer_error"] = str(error)

    return evaluation


def summarize_method(items: list[dict[str, Any]], chunk_count: int) -> dict[str, Any]:
    doc_ranks = [
        item["first_expected_doc_rank"]
        for item in items
        if item["first_expected_doc_rank"] is not None
    ]
    judged_items = [
        item for item in items
        if isinstance(item.get("llm_judge"), dict)
    ]
    answered_items = [
        item for item in items
        if isinstance(item.get("generated_answer"), dict)
    ]

    summary = {
        "chunk_count": chunk_count,
        "questions": len(items),
        "doc_hit_at_5": round(mean(item["doc_hit_at_5"] for item in items), 4),
        "doc_hit_at_20": round(mean(item["doc_hit_at_20"] for item in items), 4),
        "mrr_doc": round(mean(item["mrr_doc"] for item in items), 4),
        "avg_first_expected_doc_rank": round(mean(doc_ranks), 2) if doc_ranks else None,
        "term_recall_at_5": round(mean(item["term_recall_at_5"] for item in items), 4),
        "term_recall_at_20": round(mean(item["term_recall_at_20"] for item in items), 4),
        "full_term_hit_at_20": round(
            mean(item["full_term_hit_at_20"] for item in items),
            4,
        ),
    }

    if answered_items:
        summary["answer_generated_count"] = len(answered_items)

    if judged_items:
        summary["llm_answer_score"] = round(
            mean(float(item["llm_judge"]["score"]) for item in judged_items),
            4,
        )
        summary["llm_answer_pass_rate"] = round(
            mean(bool(item["llm_judge"]["passed"]) for item in judged_items),
            4,
        )

    return summary


def write_markdown_report(report: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Chunking Method Evaluation",
        "",
        f"Dataset: `{report['dataset']}`",
        f"Embedding model: `{report['embedding_model']}`",
        f"Chat model: `{report['chat_model']}`",
        f"With answers: `{report['with_answers']}`",
        f"With LLM judge: `{report['with_llm_judge']}`",
        f"With reranker: `{report['with_reranker']}`",
        f"Reranker model: `{report['reranker_model']}`",
        f"Top K: {report['top_k']}",
        "",
        "## Summary",
        "",
        "| method | chunks | doc@5 | doc@20 | MRR doc | avg doc rank | term@5 | term@20 | full terms@20 | answer score | answer pass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for method, summary in report["summary"].items():
        lines.append(
            "| "
            f"{method} | "
            f"{summary['chunk_count']} | "
            f"{summary['doc_hit_at_5']} | "
            f"{summary['doc_hit_at_20']} | "
            f"{summary['mrr_doc']} | "
            f"{summary['avg_first_expected_doc_rank']} | "
            f"{summary['term_recall_at_5']} | "
            f"{summary['term_recall_at_20']} | "
            f"{summary['full_term_hit_at_20']} | "
            f"{summary.get('llm_answer_score', '-')} | "
            f"{summary.get('llm_answer_pass_rate', '-')} |"
        )

    lines.extend(["", "## Per Question", ""])

    for method, method_report in report["methods"].items():
        lines.extend([f"### {method}", ""])
        lines.extend(
            [
                "| id | doc@5 | rank | term@5 | term@20 | answer score | answer pass | question |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )

        for item in method_report["items"]:
            judge = item.get("llm_judge") or {}
            lines.append(
                "| "
                f"{item['id']} | "
                f"{int(item['doc_hit_at_5'])} | "
                f"{item['first_expected_doc_rank'] or '-'} | "
                f"{item['term_recall_at_5']} | "
                f"{item['term_recall_at_20']} | "
                f"{judge.get('score', '-')} | "
                f"{judge.get('passed', '-')} | "
                f"{str(item['question']).replace('|', '\\|')} |"
            )

        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_method_indexes(
    args: argparse.Namespace,
    embedder: OpenRouterEmbedder,
) -> dict[str, MethodIndex]:
    indexes: dict[str, MethodIndex] = {}

    for method in args.methods:
        if method == "semantic":
            indexes[method] = load_semantic_index(args.embedding_model)
        else:
            indexes[method] = load_or_build_generated_index(method, args, embedder)

    return indexes


def main() -> None:
    load_dotenv()
    args = parse_args()
    output_dir = ROOT_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    items = load_gold_items(ROOT_DIR / args.dataset, args.limit)
    embedder = OpenRouterEmbedder(
        model=args.embedding_model,
        batch_size=args.embedding_batch_size,
    )
    if args.with_llm_judge:
        args.with_answers = True
    chat_client = OpenRouterChatClient(args.chat_model) if args.with_answers else None
    reranker = CrossEncoderReranker(args.reranker_model) if args.with_reranker else None

    method_indexes = build_method_indexes(args, embedder)
    query_embeddings = load_or_build_query_embeddings(items, args, embedder)
    methods_report: dict[str, Any] = {}
    summary: dict[str, Any] = {}

    for method, method_index in method_indexes.items():
        evaluated_items = [
            evaluate_item(
                method_index=method_index,
                item=item,
                query_embedding=query_embeddings[str(item["question"])],
                top_k=args.top_k,
                chat_client=chat_client,
                answer_top_n=args.answer_top_n,
                judge_answers=args.with_llm_judge,
                reranker=reranker,
            )
            for item in items
        ]
        methods_report[method] = {
            "chunk_count": len(method_index.chunks),
            "items": evaluated_items,
        }
        summary[method] = summarize_method(evaluated_items, len(method_index.chunks))

    report = {
        "dataset": str(args.dataset),
        "embedding_model": args.embedding_model,
        "chat_model": args.chat_model if args.with_answers else None,
        "with_answers": args.with_answers,
        "with_llm_judge": args.with_llm_judge,
        "with_reranker": args.with_reranker,
        "reranker_model": args.reranker_model if args.with_reranker else None,
        "answer_top_n": args.answer_top_n if args.with_answers else None,
        "top_k": args.top_k,
        "summary": summary,
        "methods": methods_report,
    }
    json_path = output_dir / "chunking_methods_results.json"
    markdown_path = output_dir / "chunking_methods_report.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown_report(report, markdown_path)

    print(f"Saved JSON report to {json_path}")
    print(f"Saved Markdown report to {markdown_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
