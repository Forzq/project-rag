from __future__ import annotations

from pathlib import Path


OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_CHROMA_DB_PATH = Path("chroma_db")
DEFAULT_CHROMA_COLLECTION = "harry_potter_openai_1536"

DEFAULT_RETRIEVAL_MODEL = "openai/text-embedding-3-small"
DEFAULT_SEMANTIC_CHUNKING_MODEL = "qwen/qwen3-embedding-4b"
DEFAULT_HF_MODEL = "intfloat/e5-large-v2"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"
DEFAULT_CHAT_MODEL = "google/gemini-2.5-flash-lite"

DEFAULT_ANSWER_RETRIEVAL_TOP_K = 20
DEFAULT_ANSWER_RERANK_TOP_N = 5
DEFAULT_DIRECT_CONTEXT_TOKEN_LIMIT = 12000
