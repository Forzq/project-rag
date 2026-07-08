from __future__ import annotations

import os
from typing import Any

import requests

from src.rag_local.config import OPENROUTER_CHAT_URL
from src.rag_local.embeddings import require_env


def resolve_chat_model(default_model: str) -> str:
    return os.getenv("OPENROUTER_CHAT_MODEL") or default_model


class OpenRouterChatClient:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.api_key = api_key or require_env("OPENROUTER_API_KEY")
        self.timeout = timeout

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            OPENROUTER_CHAT_URL,
            headers=headers,
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return str(data["choices"][0]["message"]["content"]).strip()
