from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import chromadb
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
DEFAULT_DB_PATH = Path("chroma_db")
DEFAULT_COLLECTION = "harry_potter_openai_1536"
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"


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


def require_openrouter_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENROUTER_API_KEY is not set in .env.",
        )

    return api_key


def embed_query(query: str, api_key: str) -> list[float]:
    response = requests.post(
        OPENROUTER_EMBEDDINGS_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": DEFAULT_EMBEDDING_MODEL,
            "input": [query],
            "encoding_format": "float",
        },
        timeout=60,
    )

    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        raise HTTPException(
            status_code=502,
            detail=f"OpenRouter embeddings request failed: {response.text}",
        ) from error

    data = response.json()["data"]
    data.sort(key=lambda item: item["index"])
    return data[0]["embedding"]


def get_collection():
    client = chromadb.PersistentClient(path=str(DEFAULT_DB_PATH))

    try:
        return client.get_collection(DEFAULT_COLLECTION)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Chroma collection '{DEFAULT_COLLECTION}' was not found. "
                "Load embeddings with scripts/load_chroma.py first."
            ),
        ) from error


def metadata_value(metadata: dict[str, Any], key: str) -> Any:
    value = metadata.get(key)
    return value if value != "" else None


def search_top_k(query: str, top_k: int) -> SearchResponse:
    api_key = require_openrouter_api_key()
    query_embedding = embed_query(query, api_key)
    collection = get_collection()

    raw_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    ids = raw_results.get("ids", [[]])[0]
    documents = raw_results.get("documents", [[]])[0]
    metadatas = raw_results.get("metadatas", [[]])[0]
    distances = raw_results.get("distances", [[]])[0]

    results: list[SearchResult] = []

    for index, document_id in enumerate(ids):
        metadata = metadatas[index] or {}
        distance = float(distances[index]) if distances[index] is not None else None
        similarity = 1 - distance if distance is not None else None

        results.append(
            SearchResult(
                rank=index + 1,
                id=str(document_id),
                chunk_id=metadata_value(metadata, "chunk_id"),
                similarity=similarity,
                distance=distance,
                characters=metadata_value(metadata, "characters"),
                source_file=metadata_value(metadata, "source_file"),
                text=documents[index],
            )
        )

    return SearchResponse(
        query=query,
        top_k=top_k,
        collection=DEFAULT_COLLECTION,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        results=results,
    )


@app.get("/api/health")
def health() -> dict[str, Any]:
    collection = get_collection()
    return {
        "status": "ok",
        "collection": DEFAULT_COLLECTION,
        "count": collection.count(),
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
    }


@app.post("/api/search")
def search(request: SearchRequest) -> SearchResponse:
    return search_top_k(request.query.strip(), request.top_k)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Local RAG Retrieval</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #1e2329;
      --muted: #667085;
      --border: #d9dee7;
      --accent: #2563eb;
      --accent-strong: #1d4ed8;
      --ok: #047857;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
    }

    main {
      width: min(1120px, calc(100% - 32px));
      margin: 0 auto;
      padding: 28px 0 48px;
    }

    header {
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 18px;
    }

    h1 {
      margin: 0;
      font-size: 28px;
      line-height: 1.15;
      letter-spacing: 0;
    }

    .subtitle {
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 14px;
    }

    .status {
      color: var(--muted);
      font-size: 13px;
      text-align: right;
    }

    .search-panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 18px;
    }

    label {
      display: block;
      font-weight: 600;
      font-size: 13px;
      margin-bottom: 6px;
    }

    textarea {
      width: 100%;
      min-height: 96px;
      resize: vertical;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 12px;
      color: var(--text);
      font: inherit;
      background: #fff;
    }

    textarea:focus,
    select:focus {
      outline: 2px solid rgba(37, 99, 235, 0.2);
      border-color: var(--accent);
    }

    .controls {
      display: flex;
      align-items: end;
      gap: 12px;
      margin-top: 12px;
      flex-wrap: wrap;
    }

    .control {
      min-width: 128px;
    }

    select {
      width: 100%;
      height: 40px;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 0 10px;
      background: #fff;
      color: var(--text);
      font: inherit;
    }

    button {
      height: 40px;
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: #fff;
      padding: 0 16px;
      font-weight: 700;
      cursor: pointer;
    }

    button:hover { background: var(--accent-strong); }
    button:disabled { opacity: 0.65; cursor: not-allowed; }

    .summary {
      color: var(--muted);
      font-size: 14px;
      margin: 10px 0 14px;
    }

    .result {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px;
      margin-bottom: 12px;
    }

    .result-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      margin-bottom: 10px;
    }

    .rank {
      font-weight: 800;
      color: var(--text);
    }

    .meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 6px;
      color: var(--muted);
      font-size: 13px;
    }

    .pill {
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 2px 8px;
      background: #fafafa;
    }

    .score {
      color: var(--ok);
      font-weight: 800;
      white-space: nowrap;
    }

    .chunk-text {
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      border-top: 1px solid var(--border);
      padding-top: 10px;
      margin-top: 8px;
      font-size: 14px;
    }

    .error {
      background: #fff1f2;
      border: 1px solid #fecdd3;
      color: #9f1239;
      border-radius: 8px;
      padding: 12px;
      margin-top: 12px;
    }

    @media (max-width: 720px) {
      header {
        display: block;
      }

      .status {
        margin-top: 10px;
        text-align: left;
      }

      .controls {
        align-items: stretch;
      }

      .control,
      button {
        width: 100%;
      }

      .result-head {
        display: block;
      }

      .score {
        margin-top: 8px;
      }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Local RAG Retrieval</h1>
        <p class="subtitle">Cosine search over ChromaDB chunks</p>
      </div>
      <div class="status" id="status">Checking collection...</div>
    </header>

    <section class="search-panel">
      <label for="query">Query</label>
      <textarea id="query" placeholder="Например: Кто такие Дурсли?"></textarea>

      <div class="controls">
        <div class="control">
          <label for="topK">Top K</label>
          <select id="topK">
            <option value="5">5</option>
            <option value="10">10</option>
            <option value="20" selected>20</option>
            <option value="30">30</option>
            <option value="50">50</option>
          </select>
        </div>
        <button id="searchButton">Search</button>
      </div>
      <div id="error"></div>
    </section>

    <section id="results"></section>
  </main>

  <script>
    const statusNode = document.getElementById("status");
    const errorNode = document.getElementById("error");
    const resultsNode = document.getElementById("results");
    const button = document.getElementById("searchButton");
    const queryNode = document.getElementById("query");
    const topKNode = document.getElementById("topK");

    function escapeHtml(value) {
      return value
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function formatScore(value) {
      if (value === null || value === undefined) return "n/a";
      return Number(value).toFixed(4);
    }

    async function loadHealth() {
      try {
        const response = await fetch("/api/health");
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Health check failed");
        statusNode.textContent = `${data.collection} · ${data.count} chunks · ${data.embedding_model}`;
      } catch (error) {
        statusNode.textContent = "Collection unavailable";
      }
    }

    function renderResults(data) {
      const summary = `
        <div class="summary">
          Query: <strong>${escapeHtml(data.query)}</strong><br>
          Collection: ${escapeHtml(data.collection)} · Model: ${escapeHtml(data.embedding_model)} · Results: ${data.results.length}
        </div>
      `;

      const cards = data.results.map((item) => `
        <article class="result">
          <div class="result-head">
            <div>
              <div class="rank">#${item.rank} · ${escapeHtml(item.id)}</div>
              <div class="meta">
                <span class="pill">chunk_id: ${item.chunk_id ?? "n/a"}</span>
                <span class="pill">characters: ${item.characters ?? "n/a"}</span>
                <span class="pill">distance: ${formatScore(item.distance)}</span>
              </div>
            </div>
            <div class="score">similarity ${formatScore(item.similarity)}</div>
          </div>
          <div class="chunk-text">${escapeHtml(item.text)}</div>
        </article>
      `).join("");

      resultsNode.innerHTML = summary + cards;
    }

    async function runSearch() {
      const query = queryNode.value.trim();
      const topK = Number(topKNode.value);

      errorNode.innerHTML = "";
      resultsNode.innerHTML = "";

      if (!query) {
        errorNode.innerHTML = '<div class="error">Enter a query first.</div>';
        return;
      }

      button.disabled = true;
      button.textContent = "Searching...";

      try {
        const response = await fetch("/api/search", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query, top_k: topK }),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Search failed");
        renderResults(data);
      } catch (error) {
        errorNode.innerHTML = `<div class="error">${escapeHtml(error.message)}</div>`;
      } finally {
        button.disabled = false;
        button.textContent = "Search";
      }
    }

    button.addEventListener("click", runSearch);
    queryNode.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        runSearch();
      }
    });

    loadHealth();
  </script>
</body>
</html>
"""
