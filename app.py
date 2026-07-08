from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi import Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.rag_local.config import (
    DEFAULT_CHROMA_COLLECTION,
    DEFAULT_CHROMA_DB_PATH,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RETRIEVAL_MODEL,
)
from src.rag_local.embeddings import OpenRouterEmbedder
from src.rag_local.hybrid_search import (
    BM25Index,
    StoredChunk,
    format_chunk_result,
    reciprocal_rank_fusion,
)
from src.rag_local.reranker import CrossEncoderReranker
from src.rag_local.vector_store import ChromaRetriever


INDEX_HTML_PATH = Path("web/index.html")
VECTOR_RRF_WEIGHT = 0.75
BM25_RRF_WEIGHT = 0.25


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=20, ge=1, le=50)
    search_mode: Literal["vector", "bm25", "hybrid"] = "hybrid"
    year: int | None = None
    author: str | None = None
    document_type: str | None = None


class SearchResult(BaseModel):
    rank: int
    id: str
    chunk_id: int | None
    similarity: float | None
    distance: float | None
    bm25_score: float | None = None
    vector_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    original_rank: int | None = None
    characters: int | None
    source_file: str | None
    doc_id: str | None = None
    title: str | None = None
    author: str | None = None
    year: int | None = None
    document_type: str | None = None
    text: str


class SearchResponse(BaseModel):
    query: str
    top_k: int
    search_mode: str
    collection: str
    embedding_model: str
    filters: dict[str, Any]
    warning: str | None = None
    results: list[SearchResult]


class RerankRequest(BaseModel):
    query: str = Field(min_length=1)
    candidates: list[SearchResult] = Field(min_length=1)
    top_n: int = Field(default=5, ge=1, le=20)


class RerankResponse(BaseModel):
    query: str
    top_n: int
    input_count: int
    reranker_model: str
    results: list[SearchResult]


load_dotenv()
app = FastAPI(title="Local RAG Retrieval")


def get_retriever() -> ChromaRetriever:
    try:
        return ChromaRetriever(DEFAULT_CHROMA_DB_PATH, DEFAULT_CHROMA_COLLECTION)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Chroma collection '{DEFAULT_CHROMA_COLLECTION}' was not found. "
                "Load embeddings with scripts/load_chroma.py first."
            ),
        ) from error


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker:
    try:
        return CrossEncoderReranker(DEFAULT_RERANKER_MODEL)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load reranker model '{DEFAULT_RERANKER_MODEL}': {error}",
        ) from error


def embed_query(query: str) -> list[float]:
    try:
        embedder = OpenRouterEmbedder(model=DEFAULT_RETRIEVAL_MODEL, batch_size=1)
        return embedder.embed_query(query)
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    except requests.HTTPError as error:
        response_text = error.response.text if error.response is not None else str(error)
        raise HTTPException(
            status_code=502,
            detail=f"OpenRouter embeddings request failed: {response_text}",
        ) from error
    except requests.RequestException as error:
        raise HTTPException(
            status_code=502,
            detail=f"OpenRouter embeddings request failed: {error}",
        ) from error


def build_where_filter(request: SearchRequest) -> dict[str, Any] | None:
    filters: list[dict[str, Any]] = []

    if request.year is not None:
        filters.append({"year": request.year})

    if request.author:
        filters.append({"author": request.author.strip()})

    if request.document_type:
        filters.append({"document_type": request.document_type.strip()})

    if not filters:
        return None

    if len(filters) == 1:
        return filters[0]

    return {"$and": filters}


def get_bm25_results(
    retriever: ChromaRetriever,
    query: str,
    top_k: int,
    where_filter: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    stored_chunks = [
        StoredChunk(
            id=str(item["id"]),
            text=str(item["text"]),
            metadata=dict(item["metadata"]),
        )
        for item in retriever.get_stored_chunks(where=where_filter)
    ]

    bm25_index = BM25Index(stored_chunks)
    raw_results = bm25_index.search(query, top_k)

    return [
        format_chunk_result(
            rank=index,
            document_id=str(result["id"]),
            text=str(result["text"]),
            metadata=dict(result["metadata"]),
            bm25_score=float(result["bm25_score"]),
            bm25_rank=int(result["bm25_rank"]),
        )
        for index, result in enumerate(raw_results, start=1)
    ]


def get_vector_results(
    retriever: ChromaRetriever,
    query: str,
    top_k: int,
    where_filter: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    query_embedding = embed_query(query)
    raw_results = retriever.search(
        query_embedding=query_embedding,
        top_k=top_k,
        where=where_filter,
    )

    for result in raw_results:
        result["vector_rank"] = result["rank"]

    return raw_results


def get_hybrid_results(
    retriever: ChromaRetriever,
    query: str,
    top_k: int,
    where_filter: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], str | None]:
    stored_count = len(retriever.get_stored_chunks(where=where_filter))
    if stored_count == 0:
        return [], None

    candidate_k = min(max(top_k * 5, 50), stored_count)
    bm25_results = get_bm25_results(
        retriever=retriever,
        query=query,
        top_k=candidate_k,
        where_filter=where_filter,
    )

    try:
        vector_results = get_vector_results(
            retriever=retriever,
            query=query,
            top_k=candidate_k,
            where_filter=where_filter,
        )
    except HTTPException as error:
        if error.status_code != 502:
            raise

        for rank, result in enumerate(bm25_results[:top_k], start=1):
            result["rank"] = rank

        return (
            bm25_results[:top_k],
            "Vector search is unavailable, so Hybrid returned BM25 results only.",
        )

    fused_results = reciprocal_rank_fusion(
        [vector_results, bm25_results],
        weights=[VECTOR_RRF_WEIGHT, BM25_RRF_WEIGHT],
    )

    for rank, result in enumerate(fused_results[:top_k], start=1):
        result["rank"] = rank

    return fused_results[:top_k], None


def search_top_k(request: SearchRequest) -> SearchResponse:
    query = request.query.strip()
    retriever = get_retriever()
    where_filter = build_where_filter(request)
    warning = None

    if request.search_mode == "vector":
        raw_results = get_vector_results(
            retriever=retriever,
            query=query,
            top_k=request.top_k,
            where_filter=where_filter,
        )
    elif request.search_mode == "bm25":
        raw_results = get_bm25_results(
            retriever=retriever,
            query=query,
            top_k=request.top_k,
            where_filter=where_filter,
        )
    else:
        raw_results, warning = get_hybrid_results(
            retriever=retriever,
            query=query,
            top_k=request.top_k,
            where_filter=where_filter,
        )

    return SearchResponse(
        query=query,
        top_k=request.top_k,
        search_mode=request.search_mode,
        collection=DEFAULT_CHROMA_COLLECTION,
        embedding_model=DEFAULT_RETRIEVAL_MODEL,
        filters=where_filter or {},
        warning=warning,
        results=[SearchResult(**result) for result in raw_results],
    )


def rerank_candidates(request: RerankRequest) -> RerankResponse:
    query = request.query.strip()
    candidates = [
        candidate.model_dump() if hasattr(candidate, "model_dump") else candidate.dict()
        for candidate in request.candidates
    ]
    reranker = get_reranker()
    reranked_results = reranker.rerank(
        query=query,
        candidates=candidates,
        top_n=request.top_n,
    )

    return RerankResponse(
        query=query,
        top_n=request.top_n,
        input_count=len(candidates),
        reranker_model=DEFAULT_RERANKER_MODEL,
        results=[SearchResult(**result) for result in reranked_results],
    )


@app.get("/api/health")
def health() -> dict[str, Any]:
    retriever = get_retriever()
    return {
        "status": "ok",
        "collection": DEFAULT_CHROMA_COLLECTION,
        "count": retriever.count(),
        "embedding_model": DEFAULT_RETRIEVAL_MODEL,
    }


@app.post("/api/search")
def search(request: SearchRequest) -> SearchResponse:
    return search_top_k(request)


@app.post("/api/rerank")
def rerank(request: RerankRequest) -> RerankResponse:
    return rerank_candidates(request)


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML_PATH.read_text(encoding="utf-8")
