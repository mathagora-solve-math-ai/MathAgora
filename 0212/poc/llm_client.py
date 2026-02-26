"""Thin wrapper around OpenRouter / OpenAI API using the openai SDK."""

import os
from typing import Any
from openai import OpenAI
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL

OPENAI_API_KEY = os.environ.get(
    "OPENAI_API_KEY",
    # 편하게 하려면 OpenAI api key 하드코딩
)

# Models that should go through the OpenAI API directly (BYOK on OpenRouter)
OPENAI_DIRECT_PREFIXES = ("openai/",)


def _get_openrouter_client() -> OpenAI:
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)


def _get_openai_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)


def _normalize_content_item_for_chat(item: Any) -> Any:
    """Normalize multimodal content item to OpenAI chat-completions compatible format."""
    if not isinstance(item, dict):
        return item

    item_type = item.get("type")
    if item_type == "image":
        source = item.get("source", {}) if isinstance(item.get("source"), dict) else {}
        if source.get("type") == "base64":
            media_type = source.get("media_type", "image/png")
            data = source.get("data", "")
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{data}",
                },
            }
    return item


def _normalize_messages_for_chat(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = [_normalize_content_item_for_chat(item) for item in content]
        normalized.append({"role": role, "content": content})
    return normalized


def call_llm(
    prompt: str,
    model: str,
    temperature: float = 1.0,
    max_tokens: int = 4096,
    system_prompt: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    response_format: dict[str, Any] | None = None,
) -> str:
    """Call an LLM and return the text response.

    Routes openai/* models directly to the OpenAI API;
    everything else goes through OpenRouter.
    """
    chat_messages: list[dict[str, Any]] = []
    if system_prompt:
        chat_messages.append({"role": "system", "content": system_prompt})
    if messages is not None:
        chat_messages.extend(messages)
    else:
        chat_messages.append({"role": "user", "content": prompt})
    chat_messages = _normalize_messages_for_chat(chat_messages)

    if model.startswith(OPENAI_DIRECT_PREFIXES):
        client = _get_openai_client()
        native_model = model.split("/", 1)[1]
        create_kwargs: dict[str, Any] = {
            "model": native_model,
            "messages": chat_messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
        if response_format is not None:
            create_kwargs["response_format"] = response_format
        resp = client.chat.completions.create(**create_kwargs)
    else:
        client = _get_openrouter_client()
        create_kwargs = {
            "model": model,
            "messages": chat_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            create_kwargs["response_format"] = response_format
        resp = client.chat.completions.create(**create_kwargs)
    choices = getattr(resp, "choices", None)
    if not choices or choices[0] is None:
        raise RuntimeError(f"LLM response has no choices (model={model})")

    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None) if message is not None else None
    if content is None:
        raise RuntimeError(f"LLM response content is None (model={model})")

    if isinstance(content, list):
        text_chunks = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_chunks.append(str(item.get("text", "")))
        content = "".join(text_chunks)

    content = str(content).strip()
    if not content:
        raise RuntimeError(f"LLM response content is empty (model={model})")
    return content
