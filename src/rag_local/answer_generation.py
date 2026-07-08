from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol


AnswerStrategy = Literal["auto", "direct", "map_reduce"]
ChatHistoryMessage = dict[str, str]
MAX_HISTORY_MESSAGES = 8
MAX_HISTORY_CHARS = 6000


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


def normalize_chat_history(
    chat_history: list[ChatHistoryMessage] | None,
    max_messages: int = MAX_HISTORY_MESSAGES,
    max_chars: int = MAX_HISTORY_CHARS,
) -> list[ChatHistoryMessage]:
    if not chat_history:
        return []

    normalized: list[ChatHistoryMessage] = []
    used_chars = 0

    for message in reversed(chat_history):
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()

        if role not in {"user", "assistant"} or not content:
            continue

        remaining_chars = max_chars - used_chars
        if remaining_chars <= 0:
            break

        normalized.append(
            {
                "role": role,
                "content": content[:remaining_chars],
            }
        )
        used_chars += len(normalized[-1]["content"])

        if len(normalized) >= max_messages:
            break

    normalized.reverse()
    return normalized


def format_chat_history(chat_history: list[ChatHistoryMessage] | None) -> str:
    normalized_history = normalize_chat_history(chat_history)
    if not normalized_history:
        return "Истории диалога пока нет."

    role_labels = {
        "user": "Пользователь",
        "assistant": "Ассистент",
    }
    return "\n\n".join(
        f"{role_labels[message['role']]}:\n{message['content']}"
        for message in normalized_history
    )


def build_question_rewrite_messages(
    question: str,
    chat_history: list[ChatHistoryMessage] | None,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Ты переписываешь уточняющий вопрос пользователя для RAG-поиска. "
                "Используй историю диалога, чтобы раскрыть местоимения и неявные ссылки. "
                "Верни только один самостоятельный поисковый вопрос. "
                "Если вопрос уже самостоятельный, верни его без изменений. "
                "Не отвечай на вопрос и не добавляй пояснения."
            ),
        },
        {
            "role": "user",
            "content": (
                f"История диалога:\n{format_chat_history(chat_history)}\n\n"
                f"Текущий вопрос:\n{question}\n\n"
                "Самостоятельный вопрос для поиска:"
            ),
        },
    ]


def rewrite_question(
    question: str,
    chat_history: list[ChatHistoryMessage] | None,
    chat_client: ChatClient,
) -> str:
    cleaned_question = question.strip()
    if not normalize_chat_history(chat_history):
        return cleaned_question

    rewritten_question = chat_client.complete(
        build_question_rewrite_messages(cleaned_question, chat_history),
        temperature=0.0,
        max_tokens=140,
    ).strip()

    return rewritten_question.strip('"').strip("'").strip() or cleaned_question


def build_direct_messages(
    question: str,
    context: str,
    chat_history: list[ChatHistoryMessage] | None = None,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Ты RAG-ассистент. Отвечай только по предоставленному контексту. "
                "Историю диалога используй только для понимания текущего вопроса и ссылок вроде 'он', 'она', 'этот документ'. "
                "Если в контексте нет достаточной информации, прямо скажи об этом. "
                "Не выдумывай факты. Отвечай на русском языке. "
                "В конце кратко укажи использованные источники в формате [1], [2]."
            ),
        },
        {
            "role": "user",
            "content": (
                f"История диалога:\n{format_chat_history(chat_history)}\n\n"
                f"Контекст:\n{context}\n\n"
                f"Текущий вопрос:\n{question}\n\n"
                "Ответ:"
            ),
        },
    ]


def build_map_messages(
    question: str,
    source_label: str,
    chunk_text: str,
    chat_history: list[ChatHistoryMessage] | None = None,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Ты извлекаешь факты для RAG-ответа. Используй только данный чанк. "
                "Историю диалога используй только для понимания текущего вопроса. "
                "Если чанк не помогает ответить на вопрос, верни ровно: NO_RELEVANT_INFO. "
                "Иначе верни краткие факты на русском языке и сохрани ссылку на источник."
            ),
        },
        {
            "role": "user",
            "content": (
                f"История диалога:\n{format_chat_history(chat_history)}\n\n"
                f"Источник: {source_label}\n\n"
                f"Текущий вопрос:\n{question}\n\n"
                f"Чанк:\n{chunk_text}\n\n"
                "Полезные факты:"
            ),
        },
    ]


def build_reduce_messages(
    question: str,
    facts: str,
    chat_history: list[ChatHistoryMessage] | None = None,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Ты RAG-ассистент. Сформулируй финальный ответ только по найденным фактам. "
                "Историю диалога используй только для понимания текущего вопроса. "
                "Если фактов недостаточно, скажи, что в контексте нет достаточной информации. "
                "Не добавляй внешние знания. Отвечай на русском языке."
            ),
        },
        {
            "role": "user",
            "content": (
                f"История диалога:\n{format_chat_history(chat_history)}\n\n"
                f"Текущий вопрос:\n{question}\n\n"
                f"Найденные факты:\n{facts}\n\n"
                "Финальный ответ:"
            ),
        },
    ]


def generate_direct_answer(
    question: str,
    chunks: list[dict[str, Any]],
    chat_client: ChatClient,
    chat_history: list[ChatHistoryMessage] | None = None,
) -> str:
    return chat_client.complete(
        build_direct_messages(question, format_context(chunks), chat_history),
        temperature=0.2,
        max_tokens=800,
    )


def generate_map_reduce_answer(
    question: str,
    chunks: list[dict[str, Any]],
    chat_client: ChatClient,
    chat_history: list[ChatHistoryMessage] | None = None,
) -> tuple[str, list[str]]:
    summaries: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        source_label = format_source_label(index, chunk)
        summary = chat_client.complete(
            build_map_messages(question, source_label, str(chunk["text"]), chat_history),
            temperature=0.0,
            max_tokens=350,
        )

        if summary.strip().upper() != "NO_RELEVANT_INFO":
            summaries.append(summary.strip())

    if not summaries:
        return "В найденном контексте нет достаточной информации для ответа.", []

    answer = chat_client.complete(
        build_reduce_messages(question, "\n\n".join(summaries), chat_history),
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
    chat_history: list[ChatHistoryMessage] | None = None,
) -> GeneratedAnswer:
    context = format_context(chunks)
    estimated_context_tokens = estimate_tokens(
        context + question + format_chat_history(chat_history)
    )
    selected_strategy = strategy

    if strategy == "auto":
        selected_strategy = (
            "direct"
            if estimated_context_tokens <= context_token_limit
            else "map_reduce"
        )

    if selected_strategy == "direct":
        answer = generate_direct_answer(question, chunks, chat_client, chat_history)
        return GeneratedAnswer(
            answer=answer,
            strategy="direct",
            estimated_context_tokens=estimated_context_tokens,
            context_token_limit=context_token_limit,
            map_summaries=[],
        )

    answer, summaries = generate_map_reduce_answer(
        question,
        chunks,
        chat_client,
        chat_history,
    )
    return GeneratedAnswer(
        answer=answer,
        strategy="map_reduce",
        estimated_context_tokens=estimated_context_tokens,
        context_token_limit=context_token_limit,
        map_summaries=summaries,
    )
