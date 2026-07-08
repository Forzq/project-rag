from __future__ import annotations

from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.rag_local.config import (
    DEFAULT_CHROMA_COLLECTION,
    DEFAULT_CHROMA_DB_PATH,
    DEFAULT_RETRIEVAL_MODEL,
)
from src.rag_local.embeddings import OpenRouterEmbedder
from src.rag_local.vector_store import ChromaRetriever


INDEX_HTML_PATH = Path("web/index.html")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=20, ge=1, le=50)
    year: int | None = None
    author: str | None = None
    document_type: str | None = None


class SearchResult(BaseModel):
    rank: int
    id: str
    chunk_id: int | None
    similarity: float | None
    distance: float | None
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
    collection: str
    embedding_model: str
    filters: dict[str, Any]
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


def search_top_k(request: SearchRequest) -> SearchResponse:
    query = request.query.strip()
    query_embedding = embed_query(query)
    retriever = get_retriever()
    where_filter = build_where_filter(request)
    raw_results = retriever.search(
        query_embedding=query_embedding,
        top_k=request.top_k,
        where=where_filter,
    )

    return SearchResponse(
        query=query,
        top_k=request.top_k,
        collection=DEFAULT_CHROMA_COLLECTION,
        embedding_model=DEFAULT_RETRIEVAL_MODEL,
        filters=where_filter or {},
        results=[SearchResult(**result) for result in raw_results],
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


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML_PATH.read_text(encoding="utf-8")
