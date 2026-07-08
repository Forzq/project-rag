from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol


AnswerStrategy = Literal["auto", "direct", "map_reduce"]


class ChatClient(Protocol):
    model: str

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> str:
        ...


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    strategy: str
    estimated_context_tokens: int
    context_token_limit: int
    map_summaries: list[str]


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 3)


def format_source_label(index: int, chunk: dict[str, Any]) -> str:
    title = chunk.get("title") or chunk.get("doc_id") or "Unknown document"
    chunk_id = chunk.get("chunk_id") or "n/a"
    year = chunk.get("year") or "n/a"
    return f"[{index}] {title}, year {year}, chunk {chunk_id}"


def format_context(chunks: list[dict[str, Any]]) -> str:
    parts: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        parts.append(
            "\n".join(
                [
                    format_source_label(index, chunk),
                    str(chunk["text"]).strip(),
                ]
            )
        )

    return "\n\n---\n\n".join(parts)


def build_direct_messages(question: str, context: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Ты RAG-ассистент. Отвечай только по предоставленному контексту. "
                "Если в контексте нет достаточной информации, прямо скажи об этом. "
                "Не выдумывай факты. Отвечай на русском языке. "
                "В конце кратко укажи использованные источники в формате [1], [2]."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Контекст:\n{context}\n\n"
                f"Вопрос:\n{question}\n\n"
                "Ответ:"
            ),
        },
    ]


def build_map_messages(question: str, source_label: str, chunk_text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Ты извлекаешь факты для RAG-ответа. Используй только данный чанк. "
                "Если чанк не помогает ответить на вопрос, верни ровно: NO_RELEVANT_INFO. "
                "Иначе верни краткие факты на русском языке и сохрани ссылку на источник."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Источник: {source_label}\n\n"
                f"Вопрос:\n{question}\n\n"
                f"Чанк:\n{chunk_text}\n\n"
                "Полезные факты:"
            ),
        },
    ]


def build_reduce_messages(question: str, facts: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Ты RAG-ассистент. Сформулируй финальный ответ только по найденным фактам. "
                "Если фактов недостаточно, скажи, что в контексте нет достаточной информации. "
                "Не добавляй внешние знания. Отвечай на русском языке."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Вопрос:\n{question}\n\n"
                f"Найденные факты:\n{facts}\n\n"
                "Финальный ответ:"
            ),
        },
    ]


def generate_direct_answer(
    question: str,
    chunks: list[dict[str, Any]],
    chat_client: ChatClient,
) -> str:
    return chat_client.complete(
        build_direct_messages(question, format_context(chunks)),
        temperature=0.2,
        max_tokens=800,
    )


def generate_map_reduce_answer(
    question: str,
    chunks: list[dict[str, Any]],
    chat_client: ChatClient,
) -> tuple[str, list[str]]:
    summaries: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        source_label = format_source_label(index, chunk)
        summary = chat_client.complete(
            build_map_messages(question, source_label, str(chunk["text"])),
            temperature=0.0,
            max_tokens=350,
        )

        if summary.strip() != "NO_RELEVANT_INFO":
            summaries.append(summary.strip())

    if not summaries:
        return "В найденном контексте нет достаточной информации для ответа.", []

    answer = chat_client.complete(
        build_reduce_messages(question, "\n\n".join(summaries)),
        temperature=0.2,
        max_tokens=800,
    )
    return answer, summaries


def generate_answer(
    question: str,
    chunks: list[dict[str, Any]],
    chat_client: ChatClient,
    strategy: AnswerStrategy,
    context_token_limit: int,
) -> GeneratedAnswer:
    context = format_context(chunks)
    estimated_context_tokens = estimate_tokens(context + question)
    selected_strategy = strategy

    if strategy == "auto":
        selected_strategy = (
            "direct"
            if estimated_context_tokens <= context_token_limit
            else "map_reduce"
        )

    if selected_strategy == "direct":
        answer = generate_direct_answer(question, chunks, chat_client)
        return GeneratedAnswer(
            answer=answer,
            strategy="direct",
            estimated_context_tokens=estimated_context_tokens,
            context_token_limit=context_token_limit,
            map_summaries=[],
        )

    answer, summaries = generate_map_reduce_answer(question, chunks, chat_client)
    return GeneratedAnswer(
        answer=answer,
        strategy="map_reduce",
        estimated_context_tokens=estimated_context_tokens,
        context_token_limit=context_token_limit,
        map_summaries=summaries,
    )
