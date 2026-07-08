from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rag_local.answer_generation import (  # noqa: E402
    build_direct_messages,
    format_chat_history,
    generate_answer,
    normalize_chat_history,
    rewrite_question,
)


class FakeChatClient:
    model = "fake-chat-model"

    def __init__(self, response: str = "rewritten question") -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> str:
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return self.response


class TestAnswerGenerationMemory(unittest.TestCase):
    def test_normalize_chat_history_keeps_only_valid_recent_messages(self) -> None:
        history = [
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "third"},
        ]

        normalized = normalize_chat_history(history, max_messages=2)

        self.assertEqual(
            [
                {"role": "assistant", "content": "second"},
                {"role": "assistant", "content": "third"},
            ],
            normalized,
        )

    def test_format_chat_history_returns_empty_message_for_no_history(self) -> None:
        self.assertIn("Истории диалога пока нет", format_chat_history([]))

    def test_rewrite_question_skips_llm_when_history_is_empty(self) -> None:
        client = FakeChatClient()

        result = rewrite_question("Who is the author?", [], client)

        self.assertEqual("Who is the author?", result)
        self.assertEqual([], client.calls)

    def test_rewrite_question_uses_history_when_present(self) -> None:
        client = FakeChatClient("Who is the author of Harry Potter?")

        result = rewrite_question(
            "Who is his author?",
            [{"role": "user", "content": "Tell me about Harry Potter."}],
            client,
        )

        self.assertEqual("Who is the author of Harry Potter?", result)
        self.assertEqual(1, len(client.calls))
        self.assertEqual(0.0, client.calls[0]["temperature"])

    def test_direct_prompt_contains_chat_history(self) -> None:
        messages = build_direct_messages(
            question="А кто его автор?",
            context="[1] Some context",
            chat_history=[
                {"role": "user", "content": "Расскажи про Гарри Поттера."},
                {"role": "assistant", "content": "Гарри Поттер - волшебник."},
            ],
        )

        user_prompt = messages[1]["content"]

        self.assertIn("История диалога", user_prompt)
        self.assertIn("Расскажи про Гарри Поттера.", user_prompt)
        self.assertIn("А кто его автор?", user_prompt)

    def test_generate_answer_uses_history_in_direct_strategy(self) -> None:
        client = FakeChatClient("answer")

        generated = generate_answer(
            question="А кто его автор?",
            chunks=[
                {
                    "title": "Harry Potter",
                    "chunk_id": 1,
                    "year": 1997,
                    "text": "J. K. Rowling wrote Harry Potter.",
                }
            ],
            chat_client=client,
            strategy="direct",
            context_token_limit=12000,
            chat_history=[
                {"role": "user", "content": "Tell me about Harry Potter."},
            ],
        )

        self.assertEqual("answer", generated.answer)
        self.assertEqual("direct", generated.strategy)
        self.assertIn("Tell me about Harry Potter.", client.calls[0]["messages"][1]["content"])


if __name__ == "__main__":
    unittest.main()
