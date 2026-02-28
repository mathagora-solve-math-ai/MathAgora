from __future__ import annotations

import json
import logging
import os
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
FLOWMAP_GENERATOR_MODEL = "anthropic/claude-opus-4.5"

GENERATOR_PROMPT = """\
You are an expert at analyzing step-by-step math solutions from multiple LLMs and generating a Step Network.

Given the problem and each model's step-by-step solution, do the following:

1. **Group similar solution steps**: Even if steps come from different models, place them in the same group if they perform the same solution stage (e.g., "Compute the derivative").
2. **Name each group**: Give each group a concise, clear label in English (e.g., "Compute derivative", "Find critical points", "Classify extrema").
3. **Order the groups**: Assign group order according to the logical flow of the solution.

# Input

## Problem
{problem}

## Model Solutions

{solutions}

# Output Format

Output as JSON:

```json
{{
  "groups": [
    {{
      "group_id": 0,
      "group_name": "Group name (concise)",
      "steps": [
        {{"model": "model_name", "step_idx": original_index}}
      ]
    }}
  ]
}}
```

**Important:**
- Each step belongs to exactly one group.
- group_id starts from 0 and increases in solution order.
- group_name must be in English.
- The steps array records only the model name and step_idx for steps in that group (omit title/content).
- Every step from every model must be included in some group.

Output JSON only — no additional explanation.
"""


def _get_openrouter_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing")
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


def generate_flow_map_json(
    problem_text: str,
    solutions: list[dict[str, Any]],
    generator_model: str = FLOWMAP_GENERATOR_MODEL,
) -> dict[str, Any]:
    """Generate flow map from multiple model solutions.

    Args:
        problem_text: problem description (may be empty if image-only)
        solutions: list of {model_name: str, steps: [{step_idx, title, content}]}
        generator_model: LLM model ID to use for generation (via OpenRouter)

    Returns:
        flowmap dict: {groups: [...], flows: [...]}
    """
    solutions_text = ""
    for sol in solutions:
        solutions_text += f"\n### {sol['model_name']}\n"
        for step in sol["steps"]:
            solutions_text += f"Step {step['step_idx']}: {step['title']}\n"
            solutions_text += f"{step['content']}\n\n"

    prompt = GENERATOR_PROMPT.format(
        problem=problem_text or "(이미지로 제공됨 - 텍스트 없음)",
        solutions=solutions_text,
    )

    client = _get_openrouter_client()
    resp = client.chat.completions.create(
        model=generator_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=4096,
    )
    response_text = resp.choices[0].message.content or ""

    json_start = response_text.find("{")
    json_end = response_text.rfind("}") + 1
    if json_start == -1 or json_end == 0:
        raise RuntimeError("LLM did not return valid JSON for flow map")
    grouping = json.loads(response_text[json_start:json_end])

    # Build lookup: (model_name, step_idx) → step dict
    step_lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for sol in solutions:
        for step in sol["steps"]:
            step_lookup[(sol["model_name"], step["step_idx"])] = step

    groups = []
    for g in grouping.get("groups", []):
        group_steps = []
        for step_ref in g.get("steps", []):
            key = (step_ref["model"], step_ref["step_idx"])
            original = step_lookup.get(key)
            if original is None:
                logger.warning("Flow map: step not found in lookup: %s", key)
                continue
            group_steps.append(
                {
                    "model": step_ref["model"],
                    "step_idx": step_ref["step_idx"],
                    "title": original["title"],
                    "content": original["content"],
                }
            )
        groups.append(
            {
                "group_id": g["group_id"],
                "group_name": g["group_name"],
                "steps": group_steps,
            }
        )

    # Sequential flow connections within each model
    flows = []
    for sol in solutions:
        steps = sol["steps"]
        for i in range(len(steps) - 1):
            flows.append(
                {
                    "model": sol["model_name"],
                    "from_step": steps[i]["step_idx"],
                    "to_step": steps[i + 1]["step_idx"],
                }
            )

    return {"groups": groups, "flows": flows}
