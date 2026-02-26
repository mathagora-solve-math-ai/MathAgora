#!/usr/bin/env python3
"""
Backend-style parallel solve experiment runner (5-year TSVs).

This script intentionally mirrors /workspace/backend/solve_service.py behavior:
- Fixed 5-model set
- V4_PROMPT_SIMPLE
- Two schemas (general / Claude-specific)
- gpt-5-codex via responses.create (json_schema in text.format)
- Others via chat.completions.create (response_format=json_schema)
- Temperature policy:
  - openai/gpt-5-codex, openai/gpt-5: do not send temperature
  - others: temperature=1.0

Default workload:
- years: 2022~2026
- runs per model: 3
- concurrency: 16
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

csv.field_size_limit(10_000_000)


# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).resolve().parents[2]
DATA_ROOT = WORKSPACE / "data"
OUT_DIR_DEFAULT = Path(__file__).resolve().parent / "results"
DEFAULT_MAX_OUTPUT_TOKENS = 8192

OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENAI_DIRECT_PREFIXES = ("openai/",)

MODEL_SPECS: list[dict[str, str]] = [
    {"modelId": "openai/gpt-5-codex", "displayName": "GPT-5-Codex"},
    {"modelId": "openai/gpt-5", "displayName": "GPT-5"},
    {"modelId": "anthropic/claude-opus-4.5", "displayName": "Claude Opus 4.5"},
    {"modelId": "google/gemini-3-pro-preview", "displayName": "Gemini 3 Pro"},
    {"modelId": "x-ai/grok-4-fast", "displayName": "Grok 4 Fast"},
]

NO_TEMPERATURE_MODELS = {"openai/gpt-5-codex", "openai/gpt-5"}

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

# Kept separate to mirror backend practice (provider compatibility).
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

FLAT_HEADER = [
    "recorded_at_utc",
    "year",
    "source_file",
    "prob_id",
    "prob_type",
    "model",
    "model_id",
    "run_index",
    "temperature_sent",
    "step_count",
    "parse_ok",
    "model_final_answer",
    "gold_answer",
    "gold_choice",
    "model_choice",
    "correct",
    "correct_recheck",
    "elapsed_time_seconds",
    "error",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProblemItem:
    year: str
    source_file: str
    prob_id: str
    prob_type: str
    prob_desc: str
    answer: str
    prob_img_abs_path: str


@dataclass(frozen=True)
class TaskItem:
    problem: ProblemItem
    model_key: str
    model_id: str
    run_index: int


@dataclass(frozen=True)
class LLMCallResult:
    text: str
    attempts: int
    retry_count: int
    last_attempt_elapsed_s: float


class LLMCallFailedError(RuntimeError):
    def __init__(self, message: str, *, attempts: int, last_attempt_elapsed_s: float):
        super().__init__(message)
        self.attempts = attempts
        self.retry_count = max(0, attempts - 1)
        self.last_attempt_elapsed_s = last_attempt_elapsed_s


@dataclass
class StatsBucket:
    n: int = 0
    parse_ok: int = 0
    correct: int = 0
    correct_recheck: int = 0
    errors: int = 0
    elapsed_values: list[float] = field(default_factory=list)
    step_count_values: list[int] = field(default_factory=list)

    def add(
        self,
        *,
        parse_ok: bool,
        correct: bool,
        correct_recheck: bool,
        has_error: bool,
        elapsed: float | None,
        step_count: int | None,
    ) -> None:
        self.n += 1
        self.parse_ok += 1 if parse_ok else 0
        self.correct += 1 if correct else 0
        self.correct_recheck += 1 if correct_recheck else 0
        self.errors += 1 if has_error else 0
        if elapsed is not None:
            self.elapsed_values.append(elapsed)
        if step_count is not None:
            self.step_count_values.append(step_count)

    def as_summary(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "parse_ok": self.parse_ok,
            "correct": self.correct,
            "correct_recheck": self.correct_recheck,
            "errors": self.errors,
            "parse_ok_rate": (self.parse_ok / self.n) if self.n else 0.0,
            "correct_rate": (self.correct / self.n) if self.n else 0.0,
            "correct_recheck_rate": (self.correct_recheck / self.n) if self.n else 0.0,
            "error_rate": (self.errors / self.n) if self.n else 0.0,
            "latency_seconds": numeric_stats(self.elapsed_values),
            "step_count_stats": numeric_stats([float(v) for v in self.step_count_values]),
        }


# ---------------------------------------------------------------------------
# Caches / clients
# ---------------------------------------------------------------------------
_OPENAI_CLIENT: OpenAI | None = None
_OPENROUTER_CLIENT: OpenAI | None = None
_IMAGE_CACHE: dict[str, tuple[str, str]] = {}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "t", "y", "yes"}


def parse_float(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(str(value).strip())
    except Exception:
        return None


def parse_int(value: Any) -> int | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(str(value).strip())
    except Exception:
        return None


def format_seconds_for_log(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def numeric_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    vals = sorted(values)
    n = len(vals)
    mean_val = sum(vals) / n

    def percentile(p: float) -> float:
        if n == 1:
            return vals[0]
        idx = (n - 1) * p
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return vals[lo]
        frac = idx - lo
        return vals[lo] + (vals[hi] - vals[lo]) * frac

    return {
        "count": n,
        "mean": round(mean_val, 6),
        "median": round(percentile(0.5), 6),
        "p95": round(percentile(0.95), 6),
        "min": round(vals[0], 6),
        "max": round(vals[-1], 6),
    }


class ProgressReporter:
    def __init__(self, total: int, *, use_tqdm: bool, start_ts: float):
        self.total = max(1, total)
        self.use_tqdm = use_tqdm
        self.start_ts = start_ts
        self.completed = 0
        self.success = 0
        self.error = 0
        self.log_every_n = max(10, min(100, total // 50 if total >= 50 else 10))
        self.last_log_ts = start_ts
        self._progress = tqdm(total=total, desc="LLM calls", smoothing=0.1) if use_tqdm else None

    def _line_log(self, force: bool = False) -> None:
        now = time.time()
        if not force and self.completed > 0:
            if self.completed % self.log_every_n != 0 and (now - self.last_log_ts) < 20:
                return
        elapsed = now - self.start_ts
        rate = (self.completed / elapsed) if elapsed > 0 else 0.0
        remaining = max(0, self.total - self.completed)
        eta_s = (remaining / rate) if rate > 0 else float("inf")
        print(
            f"[progress] {self.completed}/{self.total} "
            f"({(self.completed / self.total * 100):.1f}%) "
            f"ok={self.success} err={self.error} "
            f"rate={rate:.2f} calls/s eta={format_seconds_for_log(eta_s)}",
            flush=True,
        )
        self.last_log_ts = now

    def update(self, *, success: bool) -> None:
        self.completed += 1
        if success:
            self.success += 1
        else:
            self.error += 1
        if self._progress is not None:
            self._progress.update(1)
        else:
            self._line_log(force=False)

    def close(self) -> None:
        if self._progress is not None:
            self._progress.close()
        else:
            self._line_log(force=True)


def infer_delimiter(file_path: Path) -> str:
    with file_path.open("r", encoding="utf-8") as f:
        first_line = f.readline()
    return "\t" if first_line.count("\t") > first_line.count(",") else ","


def resolve_data_asset_path(path_str: str) -> str:
    rel = (path_str or "").strip()
    if not rel:
        return ""
    cleaned = rel.lstrip("/").replace("\\", "/")
    if cleaned.startswith("data/"):
        cleaned = cleaned[len("data/") :]
    return str((DATA_ROOT / cleaned).resolve())


def encode_image_base64(image_path: str) -> tuple[str, str]:
    with open(image_path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode("utf-8")
    suffix = Path(image_path).suffix.lower()
    media_type_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(suffix, "image/png")
    return b64, media_type


def get_cached_image(image_path: str) -> tuple[str, str]:
    cached = _IMAGE_CACHE.get(image_path)
    if cached is not None:
        return cached
    encoded = encode_image_base64(image_path)
    _IMAGE_CACHE[image_path] = encoded
    return encoded


def get_openai_client() -> OpenAI:
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing")
        _OPENAI_CLIENT = OpenAI(api_key=api_key)
    return _OPENAI_CLIENT


def get_openrouter_client() -> OpenAI:
    global _OPENROUTER_CLIENT
    if _OPENROUTER_CLIENT is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is missing")
        _OPENROUTER_CLIENT = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    return _OPENROUTER_CLIENT


# ---------------------------------------------------------------------------
# Message normalization (backend-style)
# ---------------------------------------------------------------------------
def normalize_content_item_for_chat(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    if item.get("type") == "image":
        source = item.get("source", {}) if isinstance(item.get("source"), dict) else {}
        if source.get("type") == "base64":
            media_type = source.get("media_type", "image/png")
            data = source.get("data", "")
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{data}"},
            }
    return item


def normalize_messages_for_chat(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = [normalize_content_item_for_chat(item) for item in content]
        normalized.append({"role": role, "content": content})
    return normalized


def normalize_content_item_for_responses(item: Any) -> Any:
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


def normalize_messages_for_responses(prompt: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "input_text", "text": prompt}]}
    ]
    for msg in messages:
        role = str(msg.get("role", "user"))
        content = msg.get("content", "")
        if isinstance(content, list):
            content_items = [normalize_content_item_for_responses(item) for item in content]
        elif isinstance(content, str):
            content_items = [{"type": "input_text", "text": content}]
        else:
            content_items = [{"type": "input_text", "text": str(content)}]
        normalized.append({"role": role, "content": content_items})
    return normalized


def extract_text_from_responses_response(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    chunks: list[str] = []
    outputs = getattr(response, "output", None) or []
    for output_item in outputs:
        if getattr(output_item, "type", None) != "message":
            continue
        for content_item in getattr(output_item, "content", None) or []:
            if getattr(content_item, "type", None) in ("output_text", "text"):
                text = getattr(content_item, "text", None)
                if isinstance(text, str):
                    chunks.append(text)
    text = "".join(chunks).strip()
    if not text:
        raise RuntimeError("Responses API returned empty text output")
    return text


# ---------------------------------------------------------------------------
# Parsing / scoring helpers
# ---------------------------------------------------------------------------
def parse_model_solution_json(response_text: str) -> dict | None:
    try:
        payload = json.loads(response_text)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    raw_steps = payload.get("steps", [])
    if not isinstance(raw_steps, list):
        return None
    steps = [s for s in raw_steps if isinstance(s, dict)]
    titles = [str(s.get("title", "")).strip() for s in steps if str(s.get("title", "")).strip()]
    return {
        "model_name": str(payload.get("model_name", "")).strip(),
        "steps": steps,
        "step_count": len(steps),
        "titles": titles,
        "final_answer": str(payload.get("final_answer", "")).strip(),
    }


_CIRCLED_NUM_MAP = str.maketrans({"①": "1", "②": "2", "③": "3", "④": "4", "⑤": "5"})


def normalize_for_match(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]+", "", text).lower()


def extract_choice_label(text: str) -> str:
    if not text:
        return ""
    s = text.translate(_CIRCLED_NUM_MAP)
    explicit_patterns = [
        r"(?:정답|답)\s*[:：]?\s*(?:은|는)?\s*([1-5A-Ea-e])\s*(?:번|choice|option)?",
        r"(?:final\s*answer)\s*[:：]?\s*([1-5A-Ea-e])\b",
        r"\b([1-5A-Ea-e])\s*(?:번|choice|option)\b",
    ]
    for pat in explicit_patterns:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
    short = s.strip()
    if len(short) <= 6:
        m = re.fullmatch(r"[ \t()]*([1-5A-Ea-e])[ \t()]*", short)
        if m:
            return m.group(1).upper()
    return ""


def is_correct_answer(pred_answer: str, gold_answer: str) -> bool:
    pred_norm = normalize_for_match(pred_answer)
    gold_norm = normalize_for_match(gold_answer)
    if not pred_norm or not gold_norm:
        return False
    if pred_norm == gold_norm:
        return True
    if gold_norm in pred_norm:
        return True
    gold_nums = re.findall(r"-?\d+(?:\.\d+)?", gold_answer)
    pred_nums = re.findall(r"-?\d+(?:\.\d+)?", pred_answer)
    if gold_nums and pred_nums and set(gold_nums).issubset(set(pred_nums)):
        return True
    return False


def is_correct_answer_recheck(pred_answer: str, gold_answer: str) -> bool:
    gold_choice = extract_choice_label(gold_answer)
    pred_choice = extract_choice_label(pred_answer)
    if gold_choice and pred_choice and gold_choice == pred_choice:
        return True
    return is_correct_answer(pred_answer, gold_answer)


def make_response_format(schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {"name": "model_solution", "strict": True, "schema": schema},
    }


def make_chat_messages_with_system(prompt: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return normalize_messages_for_chat([{"role": "system", "content": prompt}, *messages])


def extract_text_from_chat_response(response: Any, model_id: str) -> str:
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


def call_chat_json_schema(
    *,
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    schema: dict[str, Any],
    max_tokens_field: str,
    max_tokens: int | None,
    temperature: float | None,
) -> str:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": make_response_format(schema),
    }
    if max_tokens is not None:
        kwargs[max_tokens_field] = max_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature
    response = client.chat.completions.create(**kwargs)
    return extract_text_from_chat_response(response=response, model_id=model)


# ---------------------------------------------------------------------------
# Core experiment logic
# ---------------------------------------------------------------------------
def build_user_messages(problem: ProblemItem) -> list[dict[str, Any]]:
    image_path = problem.prob_img_abs_path
    if not image_path or not os.path.exists(image_path):
        raise FileNotFoundError(f"Problem image not found: {image_path}")
    image_b64, media_type = get_cached_image(image_path)
    return [
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


def schema_for_model(model_id: str) -> dict[str, Any]:
    return CLAUDE_SOLUTION_SCHEMA if model_id.startswith("anthropic/") else SOLUTION_SCHEMA


def temperature_for_model(model_id: str) -> float | None:
    if model_id in NO_TEMPERATURE_MODELS:
        return None
    return 1.0


def call_llm_once(
    model_id: str,
    prompt: str,
    messages: list[dict[str, Any]],
    max_output_tokens: int | None,
) -> str:
    schema = schema_for_model(model_id)
    temp = temperature_for_model(model_id)

    if model_id.startswith(OPENAI_DIRECT_PREFIXES):
        client = get_openai_client()
        native_model = model_id.split("/", 1)[1]

        if native_model == "gpt-5-codex":
            responses_input = normalize_messages_for_responses(prompt=prompt, messages=messages)
            kwargs: dict[str, Any] = {
                "model": native_model,
                "input": responses_input,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "model_solution",
                        "strict": True,
                        "schema": schema,
                    }
                },
            }
            if max_output_tokens is not None:
                kwargs["max_output_tokens"] = max_output_tokens
            # Keep codex at provider default temperature (backend-style policy).
            response = client.responses.create(**kwargs)
            content = extract_text_from_responses_response(response)
            if not content:
                raise RuntimeError(f"LLM response content is empty (model={model_id})")
            return content

        chat_messages = make_chat_messages_with_system(prompt=prompt, messages=messages)
        return call_chat_json_schema(
            client=client,
            model=native_model,
            messages=chat_messages,
            schema=schema,
            max_tokens_field="max_completion_tokens",
            max_tokens=max_output_tokens,
            temperature=temp,
        )

    chat_messages = make_chat_messages_with_system(prompt=prompt, messages=messages)
    client = get_openrouter_client()
    return call_chat_json_schema(
        client=client,
        model=model_id,
        messages=chat_messages,
        schema=schema,
        max_tokens_field="max_tokens",
        max_tokens=max_output_tokens,
        temperature=temp,
    )


def call_llm_with_retry(
    model_id: str,
    prompt: str,
    messages: list[dict[str, Any]],
    max_output_tokens: int | None,
    max_attempts: int = 2,  # 1 retry after first failure
    base_backoff_s: float = 2.0,
) -> LLMCallResult:
    last_error: Exception | None = None
    last_attempt_elapsed_s = 0.0
    for attempt in range(1, max_attempts + 1):
        t0 = time.time()
        try:
            text = call_llm_once(
                model_id=model_id,
                prompt=prompt,
                messages=messages,
                max_output_tokens=max_output_tokens,
            )
            elapsed = time.time() - t0
            return LLMCallResult(
                text=text,
                attempts=attempt,
                retry_count=max(0, attempt - 1),
                last_attempt_elapsed_s=elapsed,
            )
        except Exception as exc:  # pragma: no cover
            last_error = exc
            last_attempt_elapsed_s = time.time() - t0
            if attempt < max_attempts:
                time.sleep(base_backoff_s * (2 ** (attempt - 1)))
    raise LLMCallFailedError(
        f"LLM call failed after retries (model={model_id}): {last_error}",
        attempts=max_attempts,
        last_attempt_elapsed_s=last_attempt_elapsed_s,
    ) from last_error


def run_one_task(task: TaskItem, max_output_tokens: int) -> dict[str, Any]:
    response_text = ""
    error_message = ""
    parse_ok = False
    step_count = 0
    final_answer = ""
    titles: list[str] = []
    elapsed_s = 0.0

    try:
        messages = build_user_messages(task.problem)
        call_result = call_llm_with_retry(
            model_id=task.model_id,
            prompt=V4_PROMPT_SIMPLE,
            messages=messages,
            max_output_tokens=max_output_tokens,
        )
        response_text = call_result.text
        # Record only the final attempt latency (not cumulative retries).
        elapsed_s = call_result.last_attempt_elapsed_s
        parsed = parse_model_solution_json(response_text)
        if parsed is not None:
            step_count = int(parsed["step_count"])
            final_answer = str(parsed["final_answer"]).strip()
            titles = list(parsed["titles"])
            parse_ok = bool(step_count > 0 and final_answer)
    except LLMCallFailedError as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        # On failure, record the last failed attempt latency.
        elapsed_s = exc.last_attempt_elapsed_s
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        elapsed_s = 0.0
    gold_answer = str(task.problem.answer).strip()
    gold_choice = extract_choice_label(gold_answer)
    model_choice = extract_choice_label(final_answer)
    correct = is_correct_answer(final_answer, gold_answer) if final_answer else False
    correct_recheck = is_correct_answer_recheck(final_answer, gold_answer) if final_answer else False
    temp = temperature_for_model(task.model_id)

    return {
        "recorded_at_utc": utc_now_iso(),
        "year": task.problem.year,
        "source_file": task.problem.source_file,
        "prob_id": task.problem.prob_id,
        "prob_type": task.problem.prob_type,
        "model": task.model_key,
        "model_id": task.model_id,
        "run_index": task.run_index,
        "temperature_sent": "" if temp is None else f"{temp:.1f}",
        "step_count": step_count,
        "parse_ok": parse_ok,
        "model_final_answer": final_answer,
        "gold_answer": gold_answer,
        "gold_choice": gold_choice,
        "model_choice": model_choice,
        "correct": correct,
        "correct_recheck": correct_recheck,
        "elapsed_time_seconds": round(elapsed_s, 6),
        "error": error_message,
        # raw-only extras
        "titles": titles,
        "response_text": response_text,
    }


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_existing_keys(flat_csv_path: Path) -> set[tuple[str, str, str, int]]:
    keys: set[tuple[str, str, str, int]] = set()
    if not flat_csv_path.exists():
        return keys
    with flat_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            year = str(row.get("year", "")).strip()
            prob_id = str(row.get("prob_id", "")).strip()
            model_id = str(row.get("model_id", "")).strip()
            run_idx_raw = row.get("run_index", "")
            try:
                run_idx = int(str(run_idx_raw).strip())
            except Exception:
                continue
            if year and prob_id and model_id and run_idx > 0:
                keys.add((year, prob_id, model_id, run_idx))
    return keys


def append_flat_row(flat_csv_path: Path, row: dict[str, Any]) -> None:
    ensure_parent(flat_csv_path)
    write_header = (not flat_csv_path.exists()) or flat_csv_path.stat().st_size == 0
    with flat_csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FLAT_HEADER)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in FLAT_HEADER})


def append_raw_record(raw_jsonl_path: Path, row: dict[str, Any]) -> None:
    ensure_parent(raw_jsonl_path)
    payload = {
        "recorded_at_utc": row.get("recorded_at_utc"),
        "year": row.get("year"),
        "source_file": row.get("source_file"),
        "prob_id": row.get("prob_id"),
        "prob_type": row.get("prob_type"),
        "model": row.get("model"),
        "model_id": row.get("model_id"),
        "run_index": row.get("run_index"),
        "temperature_sent": row.get("temperature_sent"),
        "elapsed_time_seconds": row.get("elapsed_time_seconds"),
        "step_count": row.get("step_count"),
        "parse_ok": row.get("parse_ok"),
        "correct": row.get("correct"),
        "correct_recheck": row.get("correct_recheck"),
        "model_final_answer": row.get("model_final_answer"),
        "gold_answer": row.get("gold_answer"),
        "gold_choice": row.get("gold_choice"),
        "model_choice": row.get("model_choice"),
        "titles": row.get("titles", []),
        "error": row.get("error", ""),
        "response_text": row.get("response_text", ""),
    }
    with raw_jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def task_key(year: str, prob_id: str, model_id: str, run_index: int) -> tuple[str, str, str, int]:
    return (str(year).strip(), str(prob_id).strip(), str(model_id).strip(), int(run_index))


def backup_if_exists(path: Path, stamp: str) -> Path | None:
    if not path.exists():
        return None
    backup_path = path.with_name(f"{path.name}.bak_{stamp}")
    shutil.copy2(path, backup_path)
    return backup_path


def prune_failed_records(flat_csv_path: Path, raw_jsonl_path: Path) -> tuple[int, Path | None, Path | None]:
    """Remove rows with non-empty error from flat/raw outputs in-place.

    Returns:
        (num_removed, flat_backup_path, raw_backup_path)
    """
    if not flat_csv_path.exists():
        return 0, None, None

    with flat_csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    failed_keys: set[tuple[str, str, str, int]] = set()
    kept_rows: list[dict[str, str]] = []
    for row in rows:
        has_error = bool(str(row.get("error", "")).strip())
        run_idx = parse_int(row.get("run_index")) or 0
        key = task_key(
            row.get("year", ""),
            row.get("prob_id", ""),
            row.get("model_id", ""),
            run_idx,
        )
        if has_error:
            failed_keys.add(key)
        else:
            kept_rows.append(row)

    if not failed_keys:
        return 0, None, None

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    flat_backup = backup_if_exists(flat_csv_path, stamp)
    raw_backup = backup_if_exists(raw_jsonl_path, stamp)

    with flat_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FLAT_HEADER)
        writer.writeheader()
        for row in kept_rows:
            writer.writerow({k: row.get(k, "") for k in FLAT_HEADER})

    if raw_jsonl_path.exists():
        kept_raw_lines: list[str] = []
        with raw_jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line_strip = line.strip()
                if not line_strip:
                    continue
                try:
                    obj = json.loads(line_strip)
                    run_idx = parse_int(obj.get("run_index")) or 0
                    key = task_key(
                        obj.get("year", ""),
                        obj.get("prob_id", ""),
                        obj.get("model_id", ""),
                        run_idx,
                    )
                except Exception:
                    # Keep malformed lines untouched.
                    kept_raw_lines.append(line)
                    continue
                if key not in failed_keys:
                    kept_raw_lines.append(line)
        with raw_jsonl_path.open("w", encoding="utf-8") as f:
            f.writelines(kept_raw_lines)

    return len(failed_keys), flat_backup, raw_backup


def summarize_flat(flat_csv_path: Path) -> dict[str, Any]:
    if not flat_csv_path.exists():
        return {
            "n_rows": 0,
            "by_model": {},
            "by_year": {},
        }
    with flat_csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    by_model: dict[str, StatsBucket] = {}
    by_year: dict[str, StatsBucket] = {}
    parse_ok_total = 0
    correct_total = 0
    correct_recheck_total = 0
    err_total = 0
    elapsed_all: list[float] = []
    step_count_all: list[int] = []

    for row in rows:
        model = str(row.get("model_id", "")).strip()
        year = str(row.get("year", "")).strip()
        parse_ok = parse_bool(row.get("parse_ok"))
        correct = parse_bool(row.get("correct"))
        correct_recheck = parse_bool(row.get("correct_recheck"))
        has_error = bool(str(row.get("error", "")).strip())
        elapsed = parse_float(row.get("elapsed_time_seconds"))
        step_count = parse_int(row.get("step_count"))
        if parse_ok:
            parse_ok_total += 1
        if correct:
            correct_total += 1
        if correct_recheck:
            correct_recheck_total += 1
        if has_error:
            err_total += 1

        m = by_model.setdefault(model, StatsBucket())
        m.add(
            parse_ok=parse_ok,
            correct=correct,
            correct_recheck=correct_recheck,
            has_error=has_error,
            elapsed=elapsed,
            step_count=step_count,
        )
        if elapsed is not None:
            elapsed_all.append(elapsed)
        if step_count is not None:
            step_count_all.append(step_count)

        y = by_year.setdefault(year, StatsBucket())
        y.add(
            parse_ok=parse_ok,
            correct=correct,
            correct_recheck=correct_recheck,
            has_error=has_error,
            elapsed=elapsed,
            step_count=step_count,
        )

    n_rows = len(rows)
    return {
        "n_rows": n_rows,
        "parse_ok_rate": (parse_ok_total / n_rows) if n_rows else 0.0,
        "correct_rate": (correct_total / n_rows) if n_rows else 0.0,
        "correct_recheck_rate": (correct_recheck_total / n_rows) if n_rows else 0.0,
        "error_rate": (err_total / n_rows) if n_rows else 0.0,
        "latency_seconds": numeric_stats(elapsed_all),
        "step_count_stats": numeric_stats([float(v) for v in step_count_all]),
        "by_model": {k: v.as_summary() for k, v in by_model.items()},
        "by_year": {k: v.as_summary() for k, v in by_year.items()},
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_problems_from_year_tsv(year: str, limit_per_year: int = 0) -> list[ProblemItem]:
    tsv_path = DATA_ROOT / f"{year}_math_odd.tsv"
    if not tsv_path.exists():
        raise FileNotFoundError(f"Missing TSV: {tsv_path}")
    delimiter = infer_delimiter(tsv_path)
    items: list[ProblemItem] = []
    with tsv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            prob_id = (row.get("prob_id") or "").strip()
            if not prob_id:
                continue
            prob_img_abs = resolve_data_asset_path(row.get("prob_img_path") or "")
            items.append(
                ProblemItem(
                    year=year,
                    source_file=tsv_path.name,
                    prob_id=prob_id,
                    prob_type=(row.get("prob_type") or "").strip(),
                    prob_desc=(row.get("prob_desc") or "").strip(),
                    answer=(row.get("answer") or "").strip(),
                    prob_img_abs_path=prob_img_abs,
                )
            )
            if limit_per_year > 0 and len(items) >= limit_per_year:
                break
    return items


def build_tasks(
    problems: list[ProblemItem],
    runs: int,
    existing_keys: set[tuple[str, str, str, int]],
) -> tuple[list[TaskItem], int]:
    tasks: list[TaskItem] = []
    skipped = 0
    for p in problems:
        for spec in MODEL_SPECS:
            model_id = spec["modelId"]
            model_key = spec["displayName"]
            for run_idx in range(1, runs + 1):
                key = (p.year, p.prob_id, model_id, run_idx)
                if key in existing_keys:
                    skipped += 1
                    continue
                tasks.append(
                    TaskItem(
                        problem=p,
                        model_key=model_key,
                        model_id=model_id,
                        run_index=run_idx,
                    )
                )
    return tasks, skipped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run backend-style 5-year parallel solve experiment.")
    parser.add_argument("--years", type=str, default="2022,2023,2024,2025,2026")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--limit-problems-per-year", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=str(OUT_DIR_DEFAULT))
    parser.add_argument("--output-prefix", type=str, default="backend_solve_parallel_5y")
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="Max output tokens. Set 0 to omit token limit parameter (provider/model default applies).",
    )
    parser.add_argument(
        "--prune-failed-before-run",
        action="store_true",
        help="Remove failed rows (error != '') from flat/raw outputs before scheduling tasks.",
    )
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    years = [y.strip() for y in args.years.split(",") if y.strip()]
    runs = max(1, int(args.runs))
    concurrency = max(1, int(args.concurrency))
    limit_per_year = max(0, int(args.limit_problems_per_year))
    raw_max_output_tokens = int(args.max_output_tokens)
    max_output_tokens = raw_max_output_tokens if raw_max_output_tokens > 0 else None
    resume = not args.no_resume
    prune_failed_before_run = bool(args.prune_failed_before_run)

    if prune_failed_before_run and not resume:
        raise ValueError("--prune-failed-before-run requires resume mode (do not use --no-resume).")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flat_csv_path = out_dir / f"{args.output_prefix}_flat.csv"
    raw_jsonl_path = out_dir / f"{args.output_prefix}_raw.jsonl"
    summary_json_path = out_dir / f"{args.output_prefix}_summary.json"

    print("=== Backend-style parallel solve experiment ===")
    print(f"Years: {years}")
    print(f"Runs per model: {runs}")
    print(f"Concurrency: {concurrency}")
    print(f"Limit problems per year: {limit_per_year if limit_per_year else 'ALL'}")
    print(f"Max output tokens: {max_output_tokens if max_output_tokens is not None else 'UNSET(provider default)'}")
    print(f"Resume: {resume}")
    print(f"Prune failed before run: {prune_failed_before_run}")
    print(f"Output dir: {out_dir}")
    print(f"Models: {[m['modelId'] for m in MODEL_SPECS]}")
    print()

    # API key pre-check
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is missing")

    # Load problems
    all_problems: list[ProblemItem] = []
    for year in years:
        year_items = load_problems_from_year_tsv(year, limit_per_year)
        print(f"{year}: loaded {len(year_items)} problems from {year}_math_odd.tsv")
        all_problems.extend(year_items)
    print(f"Total problems: {len(all_problems)}")

    pruned_failed_count = 0
    flat_backup_path: Path | None = None
    raw_backup_path: Path | None = None
    if prune_failed_before_run:
        pruned_failed_count, flat_backup_path, raw_backup_path = prune_failed_records(
            flat_csv_path=flat_csv_path,
            raw_jsonl_path=raw_jsonl_path,
        )
        print(f"Pruned failed rows before run: {pruned_failed_count}")
        if flat_backup_path is not None:
            print(f"Flat backup: {flat_backup_path}")
        if raw_backup_path is not None:
            print(f"Raw backup: {raw_backup_path}")

    existing_keys = load_existing_keys(flat_csv_path) if resume else set()
    tasks, skipped = build_tasks(all_problems, runs, existing_keys)
    expected_total = len(all_problems) * len(MODEL_SPECS) * runs
    print(f"Expected calls (full): {expected_total}")
    print(f"Skipped by resume: {skipped}")
    print(f"Pending calls: {len(tasks)}")
    if not tasks:
        print("Nothing to run. Writing summary from existing flat CSV.")
        summary = summarize_flat(flat_csv_path)
        summary["meta"] = {
            "recorded_at_utc": utc_now_iso(),
            "years": years,
            "runs": runs,
            "concurrency": concurrency,
            "max_output_tokens": max_output_tokens,
            "expected_calls_full": expected_total,
            "pending_calls": 0,
            "skipped_by_resume": skipped,
            "prune_failed_before_run": prune_failed_before_run,
            "pruned_failed_count": pruned_failed_count,
            "flat_backup_path": str(flat_backup_path) if flat_backup_path else "",
            "raw_backup_path": str(raw_backup_path) if raw_backup_path else "",
            "models": MODEL_SPECS,
            "resume": resume,
        }
        with summary_json_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Summary saved: {summary_json_path}")
        return

    start = time.time()
    use_tqdm = bool(tqdm is not None and sys.stderr.isatty())
    print(f"Progress logger: {'tqdm (interactive)' if use_tqdm else 'line logs with ETA (nohup-friendly)'}")
    progress = ProgressReporter(total=len(tasks), use_tqdm=use_tqdm, start_ts=start)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_map = {executor.submit(run_one_task, task, max_output_tokens): task for task in tasks}
        for future in as_completed(future_map):
            row = future.result()
            append_flat_row(flat_csv_path, row)
            append_raw_record(raw_jsonl_path, row)
            progress.update(success=not bool(row.get("error")))
    progress.close()

    elapsed = time.time() - start
    summary = summarize_flat(flat_csv_path)
    summary["meta"] = {
        "recorded_at_utc": utc_now_iso(),
        "years": years,
        "runs": runs,
        "concurrency": concurrency,
        "max_output_tokens": max_output_tokens,
        "expected_calls_full": expected_total,
        "pending_calls": len(tasks),
        "skipped_by_resume": skipped,
        "prune_failed_before_run": prune_failed_before_run,
        "pruned_failed_count": pruned_failed_count,
        "flat_backup_path": str(flat_backup_path) if flat_backup_path else "",
        "raw_backup_path": str(raw_backup_path) if raw_backup_path else "",
        "success_calls_this_run": progress.success,
        "error_calls_this_run": progress.error,
        "elapsed_seconds_this_run": round(elapsed, 3),
        "models": MODEL_SPECS,
        "resume": resume,
        "temperature_policy": {
            "no_temperature_models": sorted(NO_TEMPERATURE_MODELS),
            "others_temperature": 1.0,
        },
        "retry_policy": {
            "max_attempts": 2,
            "max_retries": 1,
            "retry_on_exception_only": True,
            "elapsed_time_seconds_definition": "final_attempt_only",
        },
    }
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print()
    print("=== Done ===")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Success (this run): {progress.success}")
    print(f"Error (this run): {progress.error}")
    print(f"Flat CSV: {flat_csv_path}")
    print(f"Raw JSONL: {raw_jsonl_path}")
    print(f"Summary JSON: {summary_json_path}")


if __name__ == "__main__":
    main()
