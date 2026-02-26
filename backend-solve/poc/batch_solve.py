#!/usr/bin/env python3
"""
Batch solve: 5 production models × 46 problems (2024 수능 수학 홀수)

Common 22 problems:
  - claude-opus-4.5, gemini-3-pro  → copy existing v4_all solutions
  - gpt-5-codex, gpt-5, grok-4-fast → generate fresh

Selective 24 problems (확률, 미적분, 기하):
  - all 5 models → generate fresh

Output: flowmap/outputs/v5_full/steps_{prob_id}.json

Resume-safe: skips any model already present in the output file.

Usage:
    cd /workspace/backend-solve/poc
    python batch_solve.py                          # all 46 problems
    python batch_solve.py --prob 2024_odd_common_7 # single problem
    python batch_solve.py --models gpt-5-codex grok-4-fast
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

POC_DIR     = Path(__file__).resolve().parent
V4_STEP_DIR = POC_DIR.parent / "flowmap" / "outputs" / "v4_all"
V5_STEP_DIR = POC_DIR.parent / "flowmap" / "outputs" / "v5_full"
DATA_CSV    = Path("/workspace/data/2024_math_odd.csv")

V5_STEP_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(POC_DIR))

from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL  # noqa: E402

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

from openai import OpenAI  # noqa: E402

# ── production models ─────────────────────────────────────────────────────────
PROD_MODELS: list[dict[str, str]] = [
    {"name": "gpt-5-codex",     "id": "openai/gpt-5-codex"},
    {"name": "gpt-5",           "id": "openai/gpt-5"},
    {"name": "claude-opus-4.5", "id": "anthropic/claude-opus-4.5"},
    {"name": "gemini-3-pro",    "id": "google/gemini-3-pro-preview"},
    {"name": "grok-4-fast",     "id": "x-ai/grok-4-fast"},
]

# Models that already have v4 data for common problems
V4_MODEL_NAMES = {"claude-opus-4.5", "gemini-3-pro"}

# OpenAI models that must NOT receive explicit temperature param
OPENAI_NO_TEMP = {"gpt-5", "gpt-5-codex"}

# OpenAI models that use the Responses API (not Chat Completions)
OPENAI_RESPONSES_API = {"gpt-5-codex"}

# Per-model max output tokens (None = use default 4096)
MODEL_MAX_TOKENS: dict[str, int] = {
    "gpt-5-codex": 32768,   # Responses API max
    "gpt-5":       32768,   # Chat Completions max
    "gemini-3-pro": 16384,  # OpenRouter Gemini max
}

# ── solve prompt (same as backend V4_PROMPT_SIMPLE) ───────────────────────────
SOLVE_PROMPT = """\
Solve the following math problem step by step.

**Important Guidelines**:
1. Split into a new step only when the approach or strategy meaningfully changes.
2. Each step must have a clear, distinct purpose.
3. Do not over-split (merge minor sub-calculations into one step).

**Output format** (JSON only, no markdown fences):
{{
  "model_name": "<model name>",
  "steps": [
    {{"step_idx": 0, "title": "Step title (concise, ≤10 words)", "content": "Detailed solution process"}},
    ...
  ],
  "final_answer": 0
}}

**final_answer rules**:
- Multiple choice (①②③④⑤): enter option number as integer 1–5.
- Short answer (no options): enter integer 0–999.

**Problem**:
{problem}
"""


# ── JSON extraction (standalone copy from feedback_poc) ──────────────────────

def _fix_json_escapes(text: str) -> str:
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


def extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*", "", text)

    def _try(s: str) -> dict | None:
        for candidate in (s, _fix_json_escapes(s)):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
        return None

    r = _try(text.strip())
    if r is not None:
        return r
    m = re.search(r'\{[\s\S]*\}', text)
    if m:
        r = _try(m.group())
        if r is not None:
            return r
    raise ValueError(f"JSON parse failed:\n{text[:300]}")


# ── API callers ───────────────────────────────────────────────────────────────

def _openai_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)


def _openrouter_client() -> OpenAI:
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)


def call_solve(model_id: str, problem_text: str, model_name: str = "") -> str:
    """Call an LLM with the solve prompt and return raw text."""
    prompt = SOLVE_PROMPT.format(problem=problem_text)
    max_tok = MODEL_MAX_TOKENS.get(model_name, 4096)

    if model_id.startswith("openai/"):
        native = model_id.split("/", 1)[1]
        client = _openai_client()

        if native in OPENAI_RESPONSES_API:
            # gpt-5-codex requires the Responses API
            resp = client.responses.create(
                model=native,
                input=[{"role": "user", "content": prompt}],
                max_output_tokens=max_tok,
                text={"format": {"type": "json_object"}},
            )
            return resp.output_text or ""
        else:
            # Standard Chat Completions (omit temperature for gpt-5 family)
            kwargs: dict[str, Any] = dict(
                model=native,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tok,
                response_format={"type": "json_object"},
            )
            if native not in OPENAI_NO_TEMP:
                kwargs["temperature"] = 1.0
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
    else:
        client = _openrouter_client()
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
            max_tokens=max_tok,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""


# ── data loading ─────────────────────────────────────────────────────────────

def load_csv() -> dict[str, dict]:
    csv.field_size_limit(10_000_000)
    probs: dict[str, dict] = {}
    with open(DATA_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            probs[row["prob_id"]] = dict(row)
    return probs


def load_v4_solutions(prob_id: str) -> dict[str, list[dict]]:
    path = V4_STEP_DIR / f"steps_{prob_id}.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("solutions", {})


# ── solve one model on one problem ───────────────────────────────────────────

def solve_one(prob_id: str, problem_text: str,
              model_name: str, model_id: str, name_for_tokens: str = "") -> list[dict]:
    """Return steps as [{title, body}]."""
    raw    = call_solve(model_id, problem_text, name_for_tokens)
    parsed = extract_json(raw)
    steps  = []
    for s in parsed.get("steps", []):
        steps.append({
            "title": s.get("title", ""),
            "body":  s.get("content", s.get("body", "")),
        })
    # Append final answer as last step if not already present
    fa = parsed.get("final_answer")
    if fa is not None:
        last = (steps[-1]["title"] if steps else "").lower()
        if "final" not in last and "answer" not in last:
            steps.append({"title": "Final Answer", "body": str(fa)})
    return steps


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prob",   nargs="*", help="Specific prob_ids to run")
    parser.add_argument("--models", nargs="*", help="Specific model names to run")
    args = parser.parse_args()

    csv_data  = load_csv()
    prob_ids  = args.prob if args.prob else sorted(csv_data.keys())
    model_filter = set(args.models) if args.models else None

    total_probs  = len(prob_ids)
    total_models = len(PROD_MODELS)
    print(f"{'='*68}")
    print(f"Batch Solve: {total_probs} problems × {total_models} models")
    print(f"Output dir: {V5_STEP_DIR}")
    print(f"{'='*68}")

    for pi, prob_id in enumerate(prob_ids, 1):
        if prob_id not in csv_data:
            print(f"[{pi}/{total_probs}] SKIP {prob_id} (not in CSV)")
            continue

        row          = csv_data[prob_id]
        problem_text = row["prob_desc"]
        is_common    = "_common_" in prob_id
        out_path     = V5_STEP_DIR / f"steps_{prob_id}.json"

        # Load existing output (resume support)
        if out_path.exists():
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            solutions: dict[str, list[dict]] = existing.get("solutions", {})
        else:
            # Seed with v4 data for common problems
            solutions = load_v4_solutions(prob_id) if is_common else {}

        print(f"\n[{pi}/{total_probs}] {prob_id}")

        changed = False
        for m in PROD_MODELS:
            m_name = m["name"]
            m_id   = m["id"]

            if model_filter and m_name not in model_filter:
                continue

            if m_name in solutions and len(solutions[m_name]) > 0:
                print(f"  [{m_name}] already done — skip")
                continue

            # For common problems: copy v4 solutions for overlapping models
            # Only copy if v4 actually has steps (skip empty v4 entries)
            if is_common and m_name in V4_MODEL_NAMES:
                v4_sols = load_v4_solutions(prob_id)
                if m_name in v4_sols and len(v4_sols[m_name]) > 0:
                    solutions[m_name] = v4_sols[m_name]
                    print(f"  [{m_name}] copied from v4 ({len(v4_sols[m_name])} steps)")
                    changed = True
                    continue

            # Generate new solution
            print(f"  [{m_name}] solving ...", end="", flush=True)
            t0 = time.time()
            try:
                steps = solve_one(prob_id, problem_text, m_name, m_id, m_name)
                solutions[m_name] = steps
                elapsed = time.time() - t0
                print(f" done ({elapsed:.1f}s, {len(steps)} steps)")
                changed = True
            except Exception as e:
                elapsed = time.time() - t0
                print(f" ERROR ({elapsed:.1f}s): {e}")
                solutions[m_name] = []
                changed = True

            # Save after each model (resume-safe)
            out_path.write_text(json.dumps({
                "problem": {
                    "prob_id":    prob_id,
                    "prob_area":  row.get("prob_area", ""),
                    "prob_point": row.get("prob_point", ""),
                    "prob_desc":  problem_text,
                    "answer":     row.get("answer", ""),
                },
                "solutions": solutions,
            }, ensure_ascii=False, indent=2), encoding="utf-8")

            time.sleep(0.3)  # small rate-limit buffer

        if not changed:
            print(f"  all models already done")

    print(f"\n{'='*68}")
    print("Batch solve complete.")
    print(f"Output: {V5_STEP_DIR}")


if __name__ == "__main__":
    main()
