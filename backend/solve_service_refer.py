from __future__ import annotations

import asyncio
import base64
import copy
import csv
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, AsyncIterator

from openai import OpenAI

WORKSPACE = Path(__file__).resolve().parent.parent
FRONTEND_PUBLIC_DIR = WORKSPACE / "frontend" / "public"
DATA_ROOT = WORKSPACE / "data"
DEMO_PARSING_TSV = DATA_ROOT / "demo_parsing" / "llm_input.tsv"
DEFAULT_DATA_FILE = os.environ.get("MATH_DATA_FILE", "2026_math_odd.tsv")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

OPENAI_DIRECT_PREFIXES = ("openai/",)
SOLVE_MODEL_SPECS: list[dict[str, str]] = [
    {"modelId": "openai/gpt-5-codex", "displayName": "GPT-5-Codex"},
    {"modelId": "anthropic/claude-opus-4.5", "displayName": "Claude Opus 4.5"},
    {"modelId": "google/gemini-3-pro-preview", "displayName": "Gemini 3 Pro"},
]

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
**title은 10단어 이내**로 간결하게 작성합니다.
**content는 해당 단계의 구체적인 풀이 과정**을 포함합니다.
**final_answer는 {final_answer_desc}**
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
_PROBLEM_CACHE: dict[str, dict[str, Any]] | None = None
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


def _encode_local_image(image_path: Path) -> tuple[str, str]:
    raw = image_path.read_bytes()
    encoded = base64.standard_b64encode(raw).decode("utf-8")
    media_by_suffix = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    media = media_by_suffix.get(image_path.suffix.lower(), "image/png")
    return encoded, media


def _infer_delimiter(file_path: Path) -> str:
    with file_path.open("r", encoding="utf-8") as f:
        first_line = f.readline()
    return "\t" if first_line.count("\t") > first_line.count(",") else ","


def _resolve_data_asset_path(path_str: str) -> str:
    rel = (path_str or "").strip()
    if not rel:
        return ""
    cleaned = rel.lstrip("/").replace("\\", "/")
    if cleaned.startswith("data/"):
        cleaned = cleaned[len("data/") :]
    return str((DATA_ROOT / cleaned).resolve())


def _load_problem_cache() -> dict[str, dict[str, Any]]:
    global _PROBLEM_CACHE
    if _PROBLEM_CACHE is not None:
        return _PROBLEM_CACHE

    csv_path = DATA_ROOT / DEFAULT_DATA_FILE
    if not csv_path.exists():
        logger.warning("Problem data file not found: %s", csv_path)
        _PROBLEM_CACHE = {}
        return _PROBLEM_CACHE

    delimiter = _infer_delimiter(csv_path)
    out: dict[str, dict[str, Any]] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            prob_id = (row.get("prob_id") or "").strip()
            if not prob_id:
                continue
            prob_type = (row.get("prob_type") or "").strip()
            answer_rule, final_answer_hint, final_answer_desc = _problem_type_meta(prob_type)
            out[prob_id] = {
                "text": (row.get("prob_desc") or "").strip(),
                "prob_img_abs_path": _resolve_data_asset_path(row.get("prob_img_path") or ""),
                "prob_type": prob_type,
                "answer_format_rule": answer_rule,
                "final_answer_hint": final_answer_hint,
                "final_answer_desc": final_answer_desc,
            }

    _PROBLEM_CACHE = out
    return _PROBLEM_CACHE


def _get_demo_parsing_problems() -> dict[str, dict[str, Any]]:
    if not DEMO_PARSING_TSV.exists():
        return {}
    problems: dict[str, dict[str, Any]] = {}
    try:
        with DEMO_PARSING_TSV.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                prob_id = (row.get("prob_id") or "").strip()
                if not prob_id:
                    continue
                prob_img_path = (row.get("prob_img_path") or "").strip()
                prob_type = (row.get("prob_type") or "5지선다형").strip()
                answer_rule, final_answer_hint, final_answer_desc = _problem_type_meta(prob_type)
                problems[prob_id] = {
                    "text": (row.get("prob_desc") or "").strip(),
                    "prob_img_abs_path": _resolve_data_asset_path(prob_img_path),
                    "prob_type": prob_type,
                    "answer_format_rule": answer_rule,
                    "final_answer_hint": final_answer_hint,
                    "final_answer_desc": final_answer_desc,
                }
    except Exception:
        logger.exception("Failed to parse demo_parsing TSV: %s", DEMO_PARSING_TSV)
        return {}
    return problems


def _resolve_crop_image(crop_image_url: str | None) -> tuple[str, str] | None:
    if not crop_image_url:
        return None

    if crop_image_url.startswith("data:"):
        return _decode_data_url(crop_image_url)

    if crop_image_url.startswith("/data/"):
        local_path = FRONTEND_PUBLIC_DIR / crop_image_url.lstrip("/")
        if local_path.exists():
            return _encode_local_image(local_path)

    return None


def _build_prompt_and_messages(
    problem_id: str,
    problem_label: str,
    crop_image_url: str | None
) -> tuple[str, list[dict[str, Any]], str]:
    demo_problems = _get_demo_parsing_problems()
    # all_problems = _load_problem_cache()
    problem = demo_problems.get(problem_id)# or all_problems.get(problem_id, {})

    answer_rule = str(problem.get("answer_format_rule") or DEFAULT_ANSWER_RULE)
    prob_type_kr = str(problem.get("prob_type") or "").strip()
    final_answer_hint = str(problem.get("final_answer_hint") or DEFAULT_FINAL_ANSWER_HINT)
    final_answer_desc = str(problem.get("final_answer_desc") or DEFAULT_FINAL_ANSWER_DESC)

    image_payload: tuple[str, str] | None = None
    problem_image_path = str(problem.get("prob_img_abs_path") or "").strip()

    if problem_image_path and os.path.exists(problem_image_path):
        image_payload = _encode_local_image(Path(problem_image_path))
    else:
        image_payload = _resolve_crop_image(crop_image_url)

    if not image_payload:
        raise RuntimeError("Problem image payload is required. Text-only fallback is disabled.")

    prompt = V4_PROMPT_TEMPLATE.format(
        answer_format_rule=answer_rule,
        final_answer_hint=final_answer_hint,
        final_answer_desc=final_answer_desc,
    )

    image_b64, media_type = image_payload
    messages: list[dict[str, Any]] = [
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
            return {
                "type": "input_image",
                "image_url": f"data:{media_type};base64,{data}",
            }
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
        {
            "role": "system",
            "content": [{"type": "input_text", "text": prompt}],
        }
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
            temperature=0.7,
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
            temperature=0.7,
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
        except Exception as exc:  # pragma: no cover - network/API errors
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


def _collect_missing_api_keys(models: list[dict[str, str]]) -> dict[str, str]:
    missing: dict[str, str] = {}
    for model in models:
        model_id = model["modelId"]
        key_name = _required_api_key_for_model(model_id)
        if not os.environ.get(key_name):
            missing[model_id] = key_name
    return missing


async def stream_solve_ndjson(
    problem_id: str,
    problem_label: str,
    crop_image_url: str | None = None,
    extracted_text: str | None = None,
) -> AsyncIterator[str]:
    # Kept for backward compatibility with callers; text fallback is intentionally disabled.
    _ = extracted_text
    logger.info("Solve stream request: problem_id=%s label=%s", problem_id, problem_label)
    missing_keys = _collect_missing_api_keys(SOLVE_MODEL_SPECS)
    runnable_models = [m for m in SOLVE_MODEL_SPECS if m["modelId"] not in missing_keys]

    if missing_keys:
        logger.error(
            "Solve stream cannot call some models due to missing keys: %s",
            ", ".join(f"{mid}:{key}" for mid, key in missing_keys.items()),
        )
        for model in SOLVE_MODEL_SPECS:
            model_id = model["modelId"]
            key_name = missing_keys.get(model_id)
            if not key_name:
                continue
            yield json.dumps(
                {
                    "modelId": model_id,
                    "kind": "event",
                    "event": "error",
                    "errorMessage": f"{key_name} is missing",
                    "timestampMs": _now_ms(),
                },
                ensure_ascii=False,
            ) + "\n"

    if not runnable_models:
        logger.error("Solve stream aborted: no runnable models")
        return

    try:
        prompt, messages, prob_type = _build_prompt_and_messages(
            problem_id=problem_id,
            problem_label=problem_label,
            crop_image_url=crop_image_url
        )
        logger.info(
            "Solve prompt/messages prepared (problem_id=%s, has_image=%s)",
            problem_id,
            any(
                isinstance(msg.get("content"), list)
                and any(isinstance(item, dict) and item.get("type") == "image" for item in msg["content"])
                for msg in messages
            ),
        )
    except Exception as exc:
        logger.exception("Failed to build prompt/messages: %s", exc)
        error_message = str(exc)
        for model in runnable_models:
            yield json.dumps(
                {
                    "modelId": model["modelId"],
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

    async def run_one_model(model: dict[str, str]) -> None:
        model_id = model["modelId"]
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
        except Exception as exc:  # pragma: no cover - network/API errors
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
