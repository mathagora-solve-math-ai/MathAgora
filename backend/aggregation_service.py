from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
AGGREGATION_MODEL = "google/gemini-3.1-pro-preview"

AGGREGATION_PROMPT_IMAGE = """\
You are an expert math solver. The problem image is shown above.
Multiple AI models have already attempted this problem — study their solutions, \
then write your OWN complete solution from scratch by looking at the problem directly.
Do NOT copy steps verbatim — reason through it yourself using the image.

## All Model Solutions
{all_solutions}

## Task
1. Read the problem from the image carefully.
2. Study the reference solutions to understand where models agreed or diverged.
3. Identify the most mathematically sound approach.
4. Write your own original step-by-step solution independently.

## Output Format (JSON only, no markdown fences)
{{
  "steps": [
    {{"step_idx": 0, "title": "<concise title>", "content": "<your own solution content>"}},
    ...
  ],
  "final_answer": "<answer>",
  "rationale": "<brief note on which parts of the reference you found reliable or unreliable>",
  "confidence": "high|medium|low"
}}

**final_answer rules**:
- CSAT multiple-choice (options ①②③④⑤ or 1–5): enter the option number as a string ("1"–"5").
- SAT multiple-choice (options A/B/C/D): enter the option letter as a string ("A", "B", "C", or "D"). Do NOT convert to numbers.
- Short-answer / grid-in (no options given): enter the exact numerical answer as a string — may be an integer (e.g. "42"), a decimal (e.g. "0.2"), or a fraction (e.g. "1/3"). Do NOT round or truncate.
"""

AGGREGATION_PROMPT_TEXT = """\
You are an expert math solver. Multiple AI models have attempted the same problem.
Study all the solutions provided, then write your OWN complete solution from scratch.
Do NOT copy steps verbatim — reason through it yourself.

## Problem
{problem}

## All Model Solutions
{all_solutions}

## Task
1. Study the reference solutions to understand where models agreed or diverged.
2. Identify the most mathematically sound approach.
3. Write your own original step-by-step solution independently.

## Output Format (JSON only, no markdown fences)
{{
  "steps": [
    {{"step_idx": 0, "title": "<concise title>", "content": "<your own solution content>"}},
    ...
  ],
  "final_answer": "<answer>",
  "rationale": "<brief note on which parts of the reference you found reliable or unreliable>",
  "confidence": "high|medium|low"
}}

**final_answer rules**:
- CSAT multiple-choice (options ①②③④⑤ or 1–5): enter the option number as a string ("1"–"5").
- SAT multiple-choice (options A/B/C/D): enter the option letter as a string ("A", "B", "C", or "D"). Do NOT convert to numbers.
- Short-answer / grid-in (no options given): enter the exact numerical answer as a string — may be an integer (e.g. "42"), a decimal (e.g. "0.2"), or a fraction (e.g. "1/3"). Do NOT round or truncate.
"""


def _get_openrouter_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing")
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


def _steps_to_text(steps: list[dict[str, Any]]) -> str:
    parts = []
    for s in steps:
        title = s.get("title", f"Step {s.get('step_idx', '?')}")
        content = s.get("content", "")
        idx = s.get("step_idx", "?")
        parts.append(f"Step {idx}: {title}\n{content}")
    return "\n\n".join(parts)


def _fix_json_escapes(text: str) -> str:
    """Double bare backslashes (e.g. LaTeX) that are invalid in JSON."""
    VALID = set('"\\\/bfnrtu')
    result: list[str] = []
    i = 0
    while i < len(text):
        c = text[i]
        if c == '\\' and i + 1 < len(text):
            nxt = text[i + 1]
            if nxt in VALID:
                result.append(c); result.append(nxt); i += 2
            else:
                result.append('\\'); result.append('\\'); i += 1
        else:
            result.append(c); i += 1
    return ''.join(result)


def _fix_trailing_commas(text: str) -> str:
    """Remove trailing commas before } or ] (common Gemini quirk)."""
    return re.sub(r',(\s*[}\]])', r'\1', text)


def _extract_json(text: str) -> dict[str, Any]:
    # Strip markdown fences Gemini sometimes adds despite being told not to
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    def _to_dict(r: Any) -> dict | None:
        if isinstance(r, dict):
            return r
        if isinstance(r, list) and r and isinstance(r[0], dict):
            return r[0]
        return None

    def _try(s: str) -> dict | None:
        for attempt in (s, _fix_trailing_commas(s), _fix_json_escapes(s),
                        _fix_trailing_commas(_fix_json_escapes(s))):
            try:
                return _to_dict(json.loads(attempt))
            except (json.JSONDecodeError, ValueError):
                pass
        return None

    # Pass 1: try the whole text
    r = _try(text)
    if r is not None:
        return r

    # Pass 2: extract outermost {...} block (handles leading/trailing prose)
    m = re.search(r'\{[\s\S]*\}', text)
    if m:
        r = _try(m.group())
        if r is not None:
            return r

    # Pass 3: recover whatever we can via regex (truncated / badly malformed output)
    m_ans = re.search(r'"final_answer"\s*:\s*"([^"]*)"', text)
    final_answer = m_ans.group(1) if m_ans else ""

    steps: list[dict[str, Any]] = []
    for sm in re.finditer(
        r'"step_idx"\s*:\s*(\d+)[^}]*?"title"\s*:\s*"((?:[^"\\]|\\.)*)"'
        r'[^}]*?"content"\s*:\s*"((?:[^"\\]|\\.)*)"',
        text, re.DOTALL,
    ):
        steps.append({"step_idx": int(sm.group(1)), "title": sm.group(2), "content": sm.group(3)})

    logger.warning(
        "JSON parse failed; recovered final_answer=%r steps=%d. Response head: %s",
        final_answer, len(steps), text[:300],
    )
    return {
        "steps": steps,
        "final_answer": final_answer,
        "rationale": "recovered from malformed LLM output",
        "confidence": "low",
    }


def generate_aggregation(
    problem_text: str,
    solutions: list[dict[str, Any]],
    aggregation_model: str = AGGREGATION_MODEL,
    problem_image_b64: str | None = None,
    problem_image_media_type: str = "image/png",
) -> dict[str, Any]:
    """Synthesize the best answer from multiple model solutions.

    Args:
        problem_text: problem description (used when no image; ignored if image provided)
        solutions: list of {model_name: str, steps: [{step_idx, title, content}]}
        aggregation_model: LLM model ID to use (via OpenRouter)
        problem_image_b64: base64-encoded problem image (enables image-aware aggregation)
        problem_image_media_type: MIME type of the image (default: image/png)

    Returns:
        dict with keys: steps, final_answer, rationale, confidence
    """
    all_solutions_parts = []
    for sol in solutions:
        model_name = sol["model_name"]
        steps_text = _steps_to_text(sol.get("steps", []))
        final = sol.get("final_answer")
        answer_note = f" (answer: {final})" if final is not None else ""
        all_solutions_parts.append(f"### {model_name}{answer_note}\n{steps_text}")
    all_solutions_text = "\n\n".join(all_solutions_parts)

    client = _get_openrouter_client()

    if problem_image_b64:
        # Multimodal: image first, then solutions text
        prompt_text = AGGREGATION_PROMPT_IMAGE.format(all_solutions=all_solutions_text)
        content: list[dict[str, Any]] = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{problem_image_media_type};base64,{problem_image_b64}"
                },
            },
            {"type": "text", "text": prompt_text},
        ]
        messages = [{"role": "user", "content": content}]
    else:
        prompt_text = AGGREGATION_PROMPT_TEXT.format(
            problem=problem_text or "(no problem text available)",
            all_solutions=all_solutions_text,
        )
        messages = [{"role": "user", "content": prompt_text}]

    create_kwargs: dict[str, Any] = dict(
        model=aggregation_model,
        messages=messages,
        temperature=1.0,
        max_tokens=16384,
        response_format={"type": "json_object"},
    )
    try:
        resp = client.chat.completions.create(**create_kwargs)
    except Exception as e:
        # Some models via OpenRouter reject response_format; retry without it
        if "response_format" in str(e).lower() or "json_object" in str(e).lower():
            logger.warning("response_format not supported; retrying without it: %s", e)
            create_kwargs.pop("response_format")
            resp = client.chat.completions.create(**create_kwargs)
        else:
            raise
    response_text = resp.choices[0].message.content or ""

    result = _extract_json(response_text)
    if resp.usage:
        result["_usage"] = {
            "input_tokens":  resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
        }
    return result
