# -*- coding: utf-8 -*-
"""
Solve stream: NDJSON stream of model events/text.
Supports 5 models and 3 modalities (text, image, image+text).
"""
from __future__ import annotations

import asyncio
import copy
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

V4_PROMPT_TEMPLATE = """\
다음 수학 문제를 단계별로 풀어주세요.

**중요 지침**:
1. 단계(STEP)는 풀이의 흐름(전략/접근)이 바뀌는 지점마다 구분하되, **의미 없는 세분화는 피하고 필요한 만큼만** 나누세요
2. 각 단계는 명확한 목적을 가져야 합니다
3. 불필요하게 세분화하지 마세요 (예: "식 정리" → "괄호 풀기" → "동류항 정리" 대신 "식 정리"로 통합)

정답 출력 규칙:
{answer_format_rule}

**출력 형식**:
반드시 아래 JSON 객체 하나만 출력하세요. (코드블록/추가 텍스트 금지)

{{
  "model_name": "모델 이름",
  "steps": [
    {{
      "step_idx": 0,
      "title": "단계 제목 (간결하게)",
      "content": "상세한 풀이 내용"
    }},
    {{
      "step_idx": 1,
      "title": "...",
      "content": "..."
    }}
  ],
  "final_answer": {final_answer_hint}
}}

**step_idx는 0부터 시작**하며, 순차적으로 증가합니다.
**title은 10단어 이내**로 간결하게 작성합니다 (반드시 영어로 출력).
**content는 해당 단계의 구체적인 풀이 과정**을 포함합니다 (반드시 영어로 출력).
**final_answer는 {final_answer_desc}**
**model_name은 영어로** 출력하세요.
"""

DEFAULT_ANSWER_RULE = "정답은 반드시 정수 하나만 출력하세요. 불필요한 텍스트를 포함하지 마세요."
MCQ_ANSWER_RULE = (
    "정답은 반드시 5지선다 번호로 출력하세요. "
    "원문자(①~⑤), 문자, 단위, 추가 설명을 쓰지 마세요."
)
SHORT_ANSWER_RULE = (
    "정답은 반드시 0~999 사이의 정수 하나만 출력하세요. "
    "단위, 추가 설명, 선행 0을 쓰지 마세요."
)

FINAL_ANSWER_HINTS = {
    "5지선다형": {
        "hint": "1",
        "desc": "객관식 선택지 번호 정수 (1~5 중 하나)",
    },
    "단답형": {
        "hint": "0",
        "desc": "단답형 정답 정수 (0~999 사이)",
    },
}
DEFAULT_FINAL_ANSWER_HINT = "0"
DEFAULT_FINAL_ANSWER_DESC = "정답 정수"

_DATA_URL_RE = re.compile(r"^data:(?P<media>[^;]+);base64,(?P<data>.+)$", re.DOTALL)
_OPENAI_CLIENT: OpenAI | None = None
_OPENROUTER_CLIENT: OpenAI | None = None

logger = logging.getLogger(__name__)


def _problem_type_meta(prob_type_kr: str) -> tuple[str, str, str]:
    prob_type = (prob_type_kr or "").strip()
    if "선다" in prob_type:
        answer_rule = MCQ_ANSWER_RULE
    elif prob_type:
        answer_rule = SHORT_ANSWER_RULE
    else:
        answer_rule = DEFAULT_ANSWER_RULE

    answer_info = FINAL_ANSWER_HINTS.get(prob_type)
    if answer_info:
        final_answer_hint = answer_info["hint"]
        final_answer_desc = answer_info["desc"]
    else:
        final_answer_hint = DEFAULT_FINAL_ANSWER_HINT
        final_answer_desc = DEFAULT_FINAL_ANSWER_DESC
    return answer_rule, final_answer_hint, final_answer_desc


def make_solution_schema(prob_type: str) -> dict[str, Any]:
    if prob_type == "5지선다형":
        final_answer_schema: dict[str, Any] = {
            "type": "integer",
            "description": "최종 답 (객관식 선택지 번호: 1~5)",
            "minimum": 1,
            "maximum": 5,
        }
    else:
        final_answer_schema = {
            "type": "integer",
            "description": "최종 답 (단답형 정수: 0~999)",
            "minimum": 0,
            "maximum": 999,
        }

    return {
        "type": "object",
        "properties": {
            "model_name": {
                "type": "string",
                "description": "모델 이름",
            },
            "steps": {
                "type": "array",
                "description": "풀이 단계 배열",
                "items": {
                    "type": "object",
                    "properties": {
                        "step_idx": {
                            "type": "integer",
                            "description": "단계 번호 (0부터 시작)",
                        },
                        "title": {
                            "type": "string",
                            "description": "단계 제목 (간결하게, 10단어 이내)",
                        },
                        "content": {
                            "type": "string",
                            "description": "단계 내용 (상세한 풀이)",
                        },
                    },
                    "required": ["step_idx", "title", "content"],
                    "additionalProperties": False,
                },
            },
            "final_answer": final_answer_schema,
        },
        "required": ["model_name", "steps", "final_answer"],
        "additionalProperties": False,
    }


def _strip_integer_bounds_for_claude(node: Any) -> Any:
    if isinstance(node, dict):
        node_type = node.get("type")
        if node_type == "integer":
            node.pop("minimum", None)
            node.pop("maximum", None)
        for value in node.values():
            _strip_integer_bounds_for_claude(value)
    elif isinstance(node, list):
        for item in node:
            _strip_integer_bounds_for_claude(item)
    return node


def make_claude_solution_schema(prob_type: str) -> dict[str, Any]:
    schema = copy.deepcopy(make_solution_schema(prob_type))
    return _strip_integer_bounds_for_claude(schema)


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


def _build_prompt_and_messages(
    problem_text: str | None,
    crop_image_data_url: str | None,
    modality: str | None,
) -> tuple[str, list[dict[str, Any]], str]:
    """Build system prompt and user messages from modality (text, image, image+text)."""
    modality = (modality or "image+text").strip().lower()
    problem_text = (problem_text or "").strip()
    image_payload: tuple[str, str] | None = None
    if crop_image_data_url and crop_image_data_url.startswith("data:"):
        image_payload = _decode_data_url(crop_image_data_url)

    prob_type_kr = "5지선다형"
    answer_rule, final_answer_hint, final_answer_desc = _problem_type_meta(prob_type_kr)
    prompt = V4_PROMPT_TEMPLATE.format(
        answer_format_rule=answer_rule,
        final_answer_hint=final_answer_hint,
        final_answer_desc=final_answer_desc,
    )

    content: Any
    if modality == "text":
        if not problem_text:
            raise RuntimeError("Text modality requires problemText.")
        content = problem_text
        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
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
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    }
                ],
            }
        ]
    else:
        # image+text: prefer both; require at least one
        if not image_payload and not problem_text:
            raise RuntimeError("image+text modality requires cropImageDataUrl and/or problemText.")
        parts: list[Any] = []
        if image_payload:
            image_b64, media_type = image_payload
            parts.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_b64,
                    },
                }
            )
        if problem_text:
            parts.append({"type": "text", "text": problem_text})
        messages = [{"role": "user", "content": parts}]
    return prompt, messages, prob_type_kr


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


def _call_llm_once(
    model_id: str,
    prompt: str,
    messages: list[dict[str, Any]],
    prob_type: str,
) -> str:
    if model_id.startswith("anthropic/"):
        schema = make_claude_solution_schema(prob_type)
    else:
        schema = make_solution_schema(prob_type)

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
                max_output_tokens=4096,
            )
            content = _extract_text_from_responses_response(response)
            if not content:
                raise RuntimeError(f"LLM response content is empty (model={model_id})")
            return content
        chat_messages = [{"role": "system", "content": prompt}, *messages]
        chat_messages = _normalize_messages_for_chat(chat_messages)
        response = client.chat.completions.create(
            model=native_model,
            messages=chat_messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "model_solution",
                    "strict": True,
                    "schema": schema,
                },
            },
            temperature=1.0,
            max_completion_tokens=4096,
        )
    else:
        chat_messages = [{"role": "system", "content": prompt}, *messages]
        chat_messages = _normalize_messages_for_chat(chat_messages)
        client = _get_openrouter_client()
        response = client.chat.completions.create(
            model=model_id,
            messages=chat_messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "model_solution",
                    "strict": True,
                    "schema": schema,
                },
            },
            temperature=1.0,
            max_tokens=4096,
        )

    choices = getattr(response, "choices", None)
    if not choices or choices[0] is None:
        raise RuntimeError(f"LLM response has no choices (model={model_id})")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None) if message is not None else None
    if content is None:
        raise RuntimeError(f"LLM response content is None (model={model_id})")
    if isinstance(content, list):
        text_chunks = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_chunks.append(str(item.get("text", "")))
        content = "".join(text_chunks)
    content = str(content).strip()
    if not content:
        raise RuntimeError(f"LLM response content is empty (model={model_id})")
    return content


def _call_llm_with_retry(
    model_id: str,
    prompt: str,
    messages: list[dict[str, Any]],
    prob_type: str,
    max_retries: int = 3,
    base_backoff_s: float = 2.0,
) -> str:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return _call_llm_once(
                model_id=model_id,
                prompt=prompt,
                messages=messages,
                prob_type=prob_type,
            )
        except Exception as exc:
            last_error = exc
            msg = str(exc)
            if "API_KEY is missing" in msg or "OPENAI_API_KEY is missing" in msg or "OPENROUTER_API_KEY is missing" in msg:
                logger.error("LLM configuration error (model=%s): %s", model_id, msg)
                break
            logger.warning(
                "LLM call failed (attempt=%s/%s, model=%s): %s",
                attempt,
                max_retries,
                model_id,
                exc,
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
        prompt, messages, prob_type = _build_prompt_and_messages(
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
            {
                "modelId": model_id,
                "kind": "event",
                "event": "start",
                "timestampMs": _now_ms(),
            }
        )
        try:
            text = await loop.run_in_executor(
                None,
                _call_llm_with_retry,
                model_id,
                prompt,
                messages,
                prob_type,
            )
            for piece in _chunk_text(text):
                await queue.put(
                    {
                        "modelId": model_id,
                        "kind": "text",
                        "text": piece,
                        "timestampMs": _now_ms(),
                    }
                )
            await queue.put(
                {
                    "modelId": model_id,
                    "kind": "event",
                    "event": "done",
                    "timestampMs": _now_ms(),
                }
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
