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


class SearchResult(BaseModel):
    rank: int
    id: str
    chunk_id: int | None
    similarity: float | None
    distance: float | None
    characters: int | None
    source_file: str | None
    text: str


class SearchResponse(BaseModel):
    query: str
    top_k: int
    collection: str
    embedding_model: str
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


def search_top_k(query: str, top_k: int) -> SearchResponse:
    query_embedding = embed_query(query)
    retriever = get_retriever()
    raw_results = retriever.search(query_embedding=query_embedding, top_k=top_k)

    return SearchResponse(
        query=query,
        top_k=top_k,
        collection=DEFAULT_CHROMA_COLLECTION,
        embedding_model=DEFAULT_RETRIEVAL_MODEL,
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
    return search_top_k(request.query.strip(), request.top_k)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML_PATH.read_text(encoding="utf-8")

