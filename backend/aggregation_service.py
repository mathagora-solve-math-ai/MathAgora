from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
AGGREGATION_MODEL = "anthropic/claude-opus-4.5"

AGGREGATION_PROMPT = """\
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
  "final_answer": <integer>,
  "rationale": "<brief note on which parts of the reference you found reliable or unreliable>",
  "confidence": "high|medium|low"
}}

**final_answer rules**:
- If the problem has multiple-choice options (①②③④⑤ or options 1–5), it is multiple-choice: enter the option number as an integer (1–5).
- If there are no options and a specific value must be computed, it is short-answer: enter an integer between 0 and 999.
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


def _extract_json(text: str) -> dict[str, Any]:
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*", "", text)

    def _try(s: str) -> dict | None:
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(_fix_json_escapes(s))
        except json.JSONDecodeError:
            return None

    r = _try(text.strip())
    if r is not None:
        return r
    m = re.search(r'\{[\s\S]*\}', text)
    if m:
        r = _try(m.group())
        if r is not None:
            return r
    raise ValueError(f"LLM did not return valid JSON:\n{text[:400]}")


def generate_aggregation(
    problem_text: str,
    solutions: list[dict[str, Any]],
    aggregation_model: str = AGGREGATION_MODEL,
) -> dict[str, Any]:
    """Synthesize the best answer from multiple model solutions.

    Args:
        problem_text: problem description (may be empty if image-only)
        solutions: list of {model_name: str, steps: [{step_idx, title, content}]}
        aggregation_model: LLM model ID to use (via OpenRouter)

    Returns:
        dict with keys: steps, final_answer, rationale, confidence
    """
    all_solutions_parts = []
    for sol in solutions:
        model_name = sol["model_name"]
        steps_text = _steps_to_text(sol.get("steps", []))
        # Include the model's final answer if available
        final = sol.get("final_answer")
        answer_note = f" (answer: {final})" if final is not None else ""
        all_solutions_parts.append(f"### {model_name}{answer_note}\n{steps_text}")
    all_solutions_text = "\n\n".join(all_solutions_parts)

    prompt = AGGREGATION_PROMPT.format(
        problem=problem_text or "(image-only — no text available)",
        all_solutions=all_solutions_text,
    )

    client = _get_openrouter_client()
    resp = client.chat.completions.create(
        model=aggregation_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
        # max_tokens=6000,
        response_format={"type": "json_object"},
    )
    response_text = resp.choices[0].message.content or ""

    return _extract_json(response_text)
