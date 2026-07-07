from __future__ import annotations

from pathlib import Path


OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"

DEFAULT_CHROMA_DB_PATH = Path("chroma_db")
DEFAULT_CHROMA_COLLECTION = "harry_potter_openai_1536"

DEFAULT_RETRIEVAL_MODEL = "openai/text-embedding-3-small"
DEFAULT_SEMANTIC_CHUNKING_MODEL = "qwen/qwen3-embedding-4b"
DEFAULT_HF_MODEL = "intfloat/e5-large-v2"

