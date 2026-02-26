# -*- coding: utf-8 -*-
"""
Solve stream: NDJSON stream of model events/text.
Supports 5 models and 3 modalities (text, image, image+text).

Prompt / schema synced from run_backend_solve_parallel_5y.py:
- Single English prompt (V4_PROMPT_SIMPLE) — model self-detects MCQ vs short-answer
- Unified schema with no prob_type dependency
- Temperature policy: GPT-5-Codex and GPT-5 do NOT receive explicit temperature
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, AsyncIterator

from openai import OpenAI

WORKSPACE = Path(__file__).resolve().parent.parent
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

OPENAI_DIRECT_PREFIXES = ("openai/",)

# Models that must NOT receive an explicit temperature parameter
NO_TEMPERATURE_MODELS = {"openai/gpt-5-codex", "openai/gpt-5"}

# Per-model maximum output tokens — set to provider/API ceiling to avoid mid-response truncation.
# Sources: batch_solve.py notes + OpenRouter model pages.
MODEL_MAX_OUTPUT_TOKENS: dict[str, int] = {
    "openai/gpt-5-codex":          32768,  # Responses API ceiling
    "openai/gpt-5":                32768,  # Chat Completions ceiling
    "anthropic/claude-opus-4.5":   32000,  # Anthropic / OpenRouter ceiling
    "google/gemini-3-pro-preview": 16384,  # OpenRouter Gemini ceiling
    "x-ai/grok-4-fast":            32768,  # OpenRouter Grok ceiling
}
_DEFAULT_MAX_OUTPUT_TOKENS = 8192  # safe fallback for unlisted models

# ---------------------------------------------------------------------------
# Prompt — mirrors run_backend_solve_parallel_5y.py V4_PROMPT_SIMPLE
# Self-contained: model infers MCQ vs short-answer from problem content.
# ---------------------------------------------------------------------------
V4_PROMPT_SIMPLE = """\
Solve the following math problem step by step.

**Important Guidelines**:
1. Split into a new step only when the approach or strategy meaningfully changes — avoid unnecessary fragmentation.
2. Each step must have a clear, distinct purpose.
3. Do not over-split (e.g., merge "expand parentheses" + "collect like terms" into one "simplify expression" step).

**Output format**:
Output exactly one JSON object. (No code blocks or extra text.)

{
  "model_name": "<model name>",
  "steps": [
    {
      "step_idx": 0,
      "title": "Step title (concise)",
      "content": "Detailed solution process for this step"
    },
    {
      "step_idx": 1,
      "title": "...",
      "content": "..."
    }
  ],
  "final_answer": 0
}

**step_idx** starts from 0 and increments sequentially.
**title** must be concise, within 10 words.
**content** must contain the concrete solution process for that step.
**final_answer rules**:
- If the problem has multiple-choice options (①②③④⑤ or options 1–5), it is multiple-choice: enter the option number as an integer (1–5).
- If there are no options and a specific value must be computed, it is short-answer: enter an integer between 0 and 999.
"""

# ---------------------------------------------------------------------------
# Schemas — mirrors run_backend_solve_parallel_5y.py
# No prob_type dependency; no min/max on final_answer.
# Claude schema kept separate (provider compatibility, same structure).
# ---------------------------------------------------------------------------
SOLUTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "model_name": {"type": "string", "description": "Model name"},
        "steps": {
            "type": "array",
            "description": "Array of solution steps",
            "items": {
                "type": "object",
                "properties": {
                    "step_idx": {"type": "integer", "description": "Step index (starting from 0)"},
                    "title": {"type": "string", "description": "Step title (concise, within 10 words)"},
                    "content": {"type": "string", "description": "Step content (detailed solution process)"},
                },
                "required": ["step_idx", "title", "content"],
                "additionalProperties": False,
            },
        },
        "final_answer": {
            "type": "integer",
            "description": "Final answer: option number (1–5) for multiple-choice, integer 0–999 for short-answer",
        },
    },
    "required": ["model_name", "steps", "final_answer"],
    "additionalProperties": False,
}

CLAUDE_SOLUTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "model_name": {"type": "string", "description": "Model name"},
        "steps": {
            "type": "array",
            "description": "Array of solution steps",
            "items": {
                "type": "object",
                "properties": {
                    "step_idx": {"type": "integer", "description": "Step index (starting from 0)"},
                    "title": {"type": "string", "description": "Step title (concise, within 10 words)"},
                    "content": {"type": "string", "description": "Step content (detailed solution process)"},
                },
                "required": ["step_idx", "title", "content"],
                "additionalProperties": False,
            },
        },
        "final_answer": {
            "type": "integer",
            "description": "Final answer: option number (1–5) for multiple-choice, integer 0–999 for short-answer",
        },
    },
    "required": ["model_name", "steps", "final_answer"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
_DATA_URL_RE = re.compile(r"^data:(?P<media>[^;]+);base64,(?P<data>.+)$", re.DOTALL)
_OPENAI_CLIENT: OpenAI | None = None
_OPENROUTER_CLIENT: OpenAI | None = None

logger = logging.getLogger(__name__)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _chunk_text(text: str, chunk_size: int = 120) -> list[str]:
    if not text:
        return []
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def _decode_data_url(data_url: str) -> tuple[str, str] | None:
    match = _DATA_URL_RE.match(data_url.strip())
    if not match:
        return None
    return match.group("data"), match.group("media")


def _schema_for_model(model_id: str) -> dict[str, Any]:
    return CLAUDE_SOLUTION_SCHEMA if model_id.startswith("anthropic/") else SOLUTION_SCHEMA


def _temperature_for_model(model_id: str) -> float | None:
    """Return None for models that must not receive an explicit temperature."""
    if model_id in NO_TEMPERATURE_MODELS:
        return None
    return 1.0


def _normalize_content_item_for_chat(item: Any) -> Any:
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
                "image_url": {"url": f"data:{media_type};base64,{data}"},
            }
    if item_type == "text":
        return str(item.get("text", ""))
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


def _normalize_content_item_for_responses(item: Any) -> Any:
    if not isinstance(item, dict):
        return {"type": "input_text", "text": str(item)}
    item_type = item.get("type")
    if item_type == "text":
        return {"type": "input_text", "text": str(item.get("text", ""))}
    if item_type == "image":
        source = item.get("source", {}) if isinstance(item.get("source"), dict) else {}
        if source.get("type") == "base64":
            media_type = source.get("media_type", "image/png")
            data = source.get("data", "")
            return {"type": "input_image", "image_url": f"data:{media_type};base64,{data}"}
    if item_type == "image_url":
        image_url = item.get("image_url")
        if isinstance(image_url, dict):
            image_url = image_url.get("url")
        if isinstance(image_url, str) and image_url:
            return {"type": "input_image", "image_url": image_url}
    return {"type": "input_text", "text": json.dumps(item, ensure_ascii=False)}


def _normalize_messages_for_responses(
    prompt: str,
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "input_text", "text": prompt}]},
    ]
    for msg in messages:
        role = str(msg.get("role", "user"))
        content = msg.get("content", "")
        if isinstance(content, list):
            content_items = [_normalize_content_item_for_responses(item) for item in content]
        elif isinstance(content, str):
            content_items = [{"type": "input_text", "text": content}]
        else:
            content_items = [{"type": "input_text", "text": str(content)}]
        normalized.append({"role": role, "content": content_items})
    return normalized


def _extract_text_from_responses_response(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    chunks: list[str] = []
    outputs = getattr(response, "output", None) or []
    for output_item in outputs:
        out_type = getattr(output_item, "type", None)
        if out_type != "message":
            continue
        contents = getattr(output_item, "content", None) or []
        for content_item in contents:
            content_type = getattr(content_item, "type", None)
            text = getattr(content_item, "text", None)
            if content_type in ("output_text", "text") and isinstance(text, str):
                chunks.append(text)
    text = "".join(chunks).strip()
    if not text:
        raise RuntimeError("Responses API returned empty text output")
    return text


def _get_openai_client() -> OpenAI:
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing")
        _OPENAI_CLIENT = OpenAI(api_key=api_key)
    return _OPENAI_CLIENT


def _get_openrouter_client() -> OpenAI:
    global _OPENROUTER_CLIENT
    if _OPENROUTER_CLIENT is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is missing")
        _OPENROUTER_CLIENT = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    return _OPENROUTER_CLIENT


def _build_prompt_and_messages(
    problem_text: str | None,
    crop_image_data_url: str | None,
    modality: str | None,
) -> tuple[str, list[dict[str, Any]]]:
    """Build (prompt, user_messages) from modality (text | image | image+text).

    Returns the fixed V4_PROMPT_SIMPLE and the appropriately structured user
    message list.  No prob_type needed — the model infers MCQ vs short-answer
    from the problem content itself.
    """
    modality = (modality or "image+text").strip().lower()
    problem_text = (problem_text or "").strip()

    image_payload: tuple[str, str] | None = None
    if crop_image_data_url and crop_image_data_url.startswith("data:"):
        image_payload = _decode_data_url(crop_image_data_url)

    if modality == "text":
        if not problem_text:
            raise RuntimeError("Text modality requires problemText.")
        messages: list[dict[str, Any]] = [{"role": "user", "content": problem_text}]
    elif modality == "image":
        if not image_payload:
            raise RuntimeError("Image modality requires cropImageDataUrl (data URL).")
        image_b64, media_type = image_payload
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": image_b64},
                    }
                ],
            }
        ]
    else:
        # image+text: require at least one
        if not image_payload and not problem_text:
            raise RuntimeError("image+text modality requires cropImageDataUrl and/or problemText.")
        parts: list[Any] = []
        if image_payload:
            image_b64, media_type = image_payload
            parts.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": image_b64},
                }
            )
        if problem_text:
            parts.append({"type": "text", "text": problem_text})
        messages = [{"role": "user", "content": parts}]

    return V4_PROMPT_SIMPLE, messages


def _max_output_tokens(model_id: str) -> int:
    return MODEL_MAX_OUTPUT_TOKENS.get(model_id, _DEFAULT_MAX_OUTPUT_TOKENS)


def _call_llm_once(
    model_id: str,
    prompt: str,
    messages: list[dict[str, Any]],
) -> str:
    schema = _schema_for_model(model_id)
    temp = _temperature_for_model(model_id)
    max_tok = _max_output_tokens(model_id)

    if model_id.startswith(OPENAI_DIRECT_PREFIXES):
        client = _get_openai_client()
        native_model = model_id.split("/", 1)[1]

        if native_model == "gpt-5-codex":
            responses_input = _normalize_messages_for_responses(prompt=prompt, messages=messages)
            response = client.responses.create(
                model=native_model,
                input=responses_input,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "model_solution",
                        "strict": True,
                        "schema": schema,
                    }
                },
                max_output_tokens=max_tok,
            )
            content = _extract_text_from_responses_response(response)
            if not content:
                raise RuntimeError(f"LLM response content is empty (model={model_id})")
            return content

        # Other OpenAI models (e.g. gpt-5)
        chat_messages = [{"role": "system", "content": prompt}, *messages]
        chat_messages = _normalize_messages_for_chat(chat_messages)
        kwargs: dict[str, Any] = dict(
            model=native_model,
            messages=chat_messages,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "model_solution", "strict": True, "schema": schema},
            },
            max_completion_tokens=max_tok,
        )
        if temp is not None:
            kwargs["temperature"] = temp
        response = client.chat.completions.create(**kwargs)
    else:
        chat_messages = [{"role": "system", "content": prompt}, *messages]
        chat_messages = _normalize_messages_for_chat(chat_messages)
        client = _get_openrouter_client()
        kwargs = dict(
            model=model_id,
            messages=chat_messages,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "model_solution", "strict": True, "schema": schema},
            },
            max_tokens=max_tok,
        )
        if temp is not None:
            kwargs["temperature"] = temp
        response = client.chat.completions.create(**kwargs)

    choices = getattr(response, "choices", None)
    if not choices or choices[0] is None:
        raise RuntimeError(f"LLM response has no choices (model={model_id})")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None) if message is not None else None
    if content is None:
        raise RuntimeError(f"LLM response content is None (model={model_id})")
    if isinstance(content, list):
        text_chunks = [
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        content = "".join(text_chunks)
    content = str(content).strip()
    if not content:
        raise RuntimeError(f"LLM response content is empty (model={model_id})")
    return content


def _call_llm_with_retry(
    model_id: str,
    prompt: str,
    messages: list[dict[str, Any]],
    max_retries: int = 3,
    base_backoff_s: float = 2.0,
) -> str:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return _call_llm_once(model_id=model_id, prompt=prompt, messages=messages)
        except Exception as exc:
            last_error = exc
            msg = str(exc)
            if "API_KEY is missing" in msg:
                logger.error("LLM configuration error (model=%s): %s", model_id, msg)
                break
            logger.warning(
                "LLM call failed (attempt=%s/%s, model=%s): %s",
                attempt, max_retries, model_id, exc,
            )
            if attempt < max_retries:
                time.sleep(base_backoff_s * (2 ** (attempt - 1)))
    raise RuntimeError(f"LLM call failed (model={model_id}): {last_error}")


def _required_api_key_for_model(model_id: str) -> str:
    if model_id.startswith(OPENAI_DIRECT_PREFIXES):
        return "OPENAI_API_KEY"
    return "OPENROUTER_API_KEY"


def _collect_missing_api_keys(models: list[dict[str, Any]]) -> dict[str, str]:
    missing: dict[str, str] = {}
    for model in models:
        model_id = model.get("modelId") or model.get("model_id") or ""
        if not model_id:
            continue
        key_name = _required_api_key_for_model(model_id)
        if not os.environ.get(key_name):
            missing[model_id] = key_name
    return missing


async def stream_solve_ndjson(
    *,
    problem_id: str,
    problem_label: str,
    crop_image_data_url: str | None = None,
    problem_text: str | None = None,
    modality: str | None = None,
    models: list[dict[str, Any]] | None = None,
) -> AsyncIterator[str]:
    """Yield NDJSON lines (one JSON object per line) for the solve stream."""
    model_list = models or []
    logger.info(
        "Solve stream request: problem_id=%s label=%s modality=%s models=%s",
        problem_id,
        problem_label,
        modality,
        [m.get("modelId") or m.get("model_id") for m in model_list],
    )

    missing_keys = _collect_missing_api_keys(model_list)
    runnable_models = [
        m for m in model_list
        if (m.get("modelId") or m.get("model_id")) not in missing_keys
    ]

    if missing_keys:
        logger.warning(
            "Solve stream: missing API keys for %s",
            ", ".join(f"{mid}:{key}" for mid, key in missing_keys.items()),
        )
        for model in model_list:
            model_id = model.get("modelId") or model.get("model_id")
            if not model_id or model_id not in missing_keys:
                continue
            yield json.dumps(
                {
                    "modelId": model_id,
                    "kind": "event",
                    "event": "error",
                    "errorMessage": f"{missing_keys[model_id]} is missing",
                    "timestampMs": _now_ms(),
                },
                ensure_ascii=False,
            ) + "\n"

    if not runnable_models:
        logger.error("Solve stream aborted: no runnable models")
        return

    try:
        prompt, messages = _build_prompt_and_messages(
            problem_text=problem_text,
            crop_image_data_url=crop_image_data_url,
            modality=modality,
        )
        logger.info(
            "Solve prompt/messages prepared (problem_id=%s, modality=%s)",
            problem_id,
            modality,
        )
    except Exception as exc:
        logger.exception("Failed to build prompt/messages: %s", exc)
        error_message = str(exc)
        for model in runnable_models:
            mid = model.get("modelId") or model.get("model_id")
            if not mid:
                continue
            yield json.dumps(
                {
                    "modelId": mid,
                    "kind": "event",
                    "event": "error",
                    "errorMessage": error_message,
                    "timestampMs": _now_ms(),
                },
                ensure_ascii=False,
            ) + "\n"
        return

    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    async def run_one_model(model: dict[str, Any]) -> None:
        model_id = model.get("modelId") or model.get("model_id") or "unknown"
        logger.info("Solve model start: %s", model_id)
        await queue.put(
            {"modelId": model_id, "kind": "event", "event": "start", "timestampMs": _now_ms()}
        )
        try:
            text = await loop.run_in_executor(
                None,
                _call_llm_with_retry,
                model_id,
                prompt,
                messages,
            )
            for piece in _chunk_text(text):
                await queue.put(
                    {"modelId": model_id, "kind": "text", "text": piece, "timestampMs": _now_ms()}
                )
            await queue.put(
                {"modelId": model_id, "kind": "event", "event": "done", "timestampMs": _now_ms()}
            )
            logger.info("Solve model done: %s", model_id)
        except Exception as exc:
            logger.exception("Solve model error (%s): %s", model_id, exc)
            await queue.put(
                {
                    "modelId": model_id,
                    "kind": "event",
                    "event": "error",
                    "errorMessage": str(exc),
                    "timestampMs": _now_ms(),
                }
            )
        finally:
            await queue.put(None)

    tasks = [asyncio.create_task(run_one_model(model)) for model in runnable_models]
    remaining = len(tasks)

    try:
        while remaining > 0:
            item = await queue.get()
            if item is None:
                remaining -= 1
                continue
            yield json.dumps(item, ensure_ascii=False) + "\n"
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
