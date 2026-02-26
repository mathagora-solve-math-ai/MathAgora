#!/usr/bin/env python3
"""
POC: Self-feedback, Cross-feedback, Flowmap Aggregation
for LLM Math Problem Solving.

세 가지 방식으로 성능 변화를 테스트:
  1. Self-feedback   : 각 모델이 자기 자신의 풀이를 검증/수정
  2. Cross-feedback  : 각 모델이 다른 모델들의 풀이를 보고 자기 답안 수정
  3. Aggregation     : Aligned flowmap 보고 최적 답안 합성 (grok-4 사용)

기존 steps_*.json / flowmap_*.json 재사용 (새 초기 풀이 불필요).

Usage:
    cd /workspace/backend-solve/poc
    python feedback_poc.py                  # prob 3개 모두
    python feedback_poc.py --prob 2024_odd_common_7
    python feedback_poc.py --methods self cross agg
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

# ── paths ────────────────────────────────────────────────────────────────────
POC_DIR      = Path(__file__).resolve().parent
STEP_DIR_V4  = POC_DIR.parent / "flowmap" / "outputs" / "v4_all"   # 3-model baseline
STEP_DIR     = POC_DIR.parent / "flowmap" / "outputs" / "v5_full"  # 5-model full run
DATA_CSV     = Path("/workspace/data/2024_math_odd.csv")
OUT_DIR      = POC_DIR / "results" / "feedback"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(POC_DIR))
from llm_client import call_llm  # noqa: E402

# ── problem config ────────────────────────────────────────────────────────────
# 2024 수능 수학 홀수 단답형 문제 목록 (나머지는 choice)
_SHORT_ANSWER_PROB_IDS: set[str] = {
    "2024_odd_common_10", "2024_odd_common_17", "2024_odd_common_18",
    "2024_odd_common_19", "2024_odd_common_20", "2024_odd_common_21",
    "2024_odd_common_22",
    "2024_odd_probalitity_29", "2024_odd_probalitity_30",
    "2024_odd_calculus_29",    "2024_odd_calculus_30",
    "2024_odd_geometry_29",    "2024_odd_geometry_30",
}


def _build_problems_meta() -> dict[str, dict]:
    """Load all 46 problems from CSV and classify answer type."""
    csv.field_size_limit(10_000_000)
    meta: dict[str, dict] = {}
    with open(DATA_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row["prob_id"]
            meta[pid] = {
                "answer":      row["answer"],
                "answer_type": "short" if pid in _SHORT_ANSWER_PROB_IDS else "choice",
            }
    return meta


PROBLEMS_META: dict[str, dict] = _build_problems_meta()

# ── reviewer / synthesizer ───────────────────────────────────────────────────
#   aggregation synthesizer: Claude Opus 4.5
#   → baseline과 직접 비교 가능 ("혼자 풀기" vs "다른 풀이 보고 재구성")
REVIEWER_MODEL  = "anthropic/claude-opus-4.5"
SYNTH_MODEL     = "anthropic/claude-opus-4.5"

# ── answer type hint for prompts ─────────────────────────────────────────────
ANSWER_FORMAT = {
    "choice": "integer 1–5 (choice number: ①=1 ②=2 ③=3 ④=4 ⑤=5)",
    "short":  "integer 0–999 (actual computed value)",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_problem_text(prob_id: str) -> str:
    csv.field_size_limit(10_000_000)
    with open(DATA_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["prob_id"] == prob_id:
                return row["prob_desc"]
    raise ValueError(f"prob_id {prob_id!r} not found in CSV")


def load_steps(prob_id: str) -> dict[str, list[dict]]:
    """Load solutions dict from v5_full (falls back to v4_all)."""
    for d in (STEP_DIR, STEP_DIR_V4):
        path = d / f"steps_{prob_id}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data["solutions"]
    raise FileNotFoundError(f"No steps file for {prob_id}")


def load_flowmap(prob_id: str) -> dict | None:
    """Load flowmap if available (v4_all only); returns None if missing."""
    path = STEP_DIR_V4 / f"flowmap_{prob_id}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────

def _steps_to_text(steps: list[dict]) -> str:
    lines = []
    for i, s in enumerate(steps):
        title = s.get("title") or s.get("step_title") or f"Step {i+1}"
        body  = s.get("body") or s.get("content") or ""
        lines.append(f"[Step {i+1}: {title}]\n{body}")
    return "\n\n".join(lines)


def _flowmap_to_text(flowmap: dict) -> str:
    lines = []
    for g in flowmap.get("groups", []):
        gname = g.get("group_name", f"Group {g['group_id']}")
        lines.append(f'Group "{gname}":')
        for s in g.get("steps", []):
            lines.append(f'  [{s["model"]}] Step {s["step_idx"]+1}: {s["title"]}')
            body = s.get("content", "")
            if body:
                lines.append(f'    {body[:200]}{"..." if len(body) > 200 else ""}')
    return "\n".join(lines)


SELF_FEEDBACK_PROMPT = """\
You are reviewing your own step-by-step math solution.

## Problem
{problem}

## Your Previous Solution
{own_steps}

## Task
1. Verify each step carefully for mathematical errors.
2. If you find an error, correct the step(s) and update the final answer.
3. If everything is correct, keep the steps unchanged.

## Output Format (JSON only, no markdown fences)
{{
  "steps": [
    {{"step_idx": 0, "title": "<concise title>", "content": "<solution content>"}},
    ...
  ],
  "final_answer": <{answer_format}>,
  "feedback": "<what you verified and what if anything you changed>",
  "revised": <true if any step or answer changed, false otherwise>
}}
"""

CROSS_FEEDBACK_PROMPT = """\
You are reviewing your solution after seeing other models' solutions to the same problem.

## Problem
{problem}

## Your Own Solution
{own_steps}

## Other Models' Solutions
{other_solutions}

## Task
1. Compare your approach with the other models'.
2. If you spot an error in your solution that others avoided, correct it.
3. If you remain confident in your approach, keep it unchanged.

## Output Format (JSON only, no markdown fences)
{{
  "steps": [
    {{"step_idx": 0, "title": "<concise title>", "content": "<solution content>"}},
    ...
  ],
  "final_answer": <{answer_format}>,
  "feedback": "<comparison notes and what if anything you changed>",
  "revised": <true if changed, false otherwise>
}}
"""

AGGREGATION_PROMPT = """\
You are an expert math solver. You have been given reference material: \
step-by-step solutions from multiple AI models that attempted the same problem.
Use this material to deeply understand the problem, then write your OWN \
complete solution from scratch.

Do NOT copy or paraphrase steps from the reference solutions.
Write each step in your own words, with your own reasoning and calculations.

## Problem
{problem}
{flowmap_section}
## Reference: All Model Solutions
{all_solutions}

## Your Task
1. Study the references to understand where models agreed or diverged.
2. Identify the most mathematically sound approach.
3. Write your own original step-by-step solution independently.

## Output Format (JSON only, no markdown fences)
{{
  "steps": [
    {{"step_idx": 0, "title": "<concise title>", "content": "<your own solution content>"}},
    ...
  ],
  "final_answer": <{answer_format}>,
  "rationale": "<brief note on which parts of the reference you found reliable/unreliable>",
  "confidence": "high|medium|low"
}}
"""

_FLOWMAP_SECTION = """\
## Reference: Aligned Solution Flow Map
(Shows which reasoning steps different models took and how they align)
{flowmap}

"""


# ─────────────────────────────────────────────────────────────────────────────
# JSON extraction
# ─────────────────────────────────────────────────────────────────────────────

def _fix_json_escapes(text: str) -> str:
    """Fix bare LaTeX backslash escapes (e.g. \\( \\) \\frac) that are invalid JSON.

    Uses a character-by-character walk so that already-valid JSON escape
    sequences (\\\\, \\n, \\t, \\uXXXX, etc.) are left untouched.
    """
    VALID_ESCAPES = set('"\\\/bfnrtu')
    result: list[str] = []
    i = 0
    while i < len(text):
        c = text[i]
        if c == '\\' and i + 1 < len(text):
            nxt = text[i + 1]
            if nxt in VALID_ESCAPES:
                # Already a valid JSON escape – consume both chars unchanged
                result.append(c)
                result.append(nxt)
                i += 2
            else:
                # Bare backslash before an invalid escape char → double it
                result.append('\\')
                result.append('\\')
                i += 1  # leave nxt to be processed on the next iteration
        else:
            result.append(c)
            i += 1
    return ''.join(result)


def extract_json(text: str) -> dict:
    """Extract JSON object from LLM response (handles markdown fences)."""
    # Strip markdown fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*", "", text)

    def _try_loads(s: str) -> dict | None:
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        # Retry with escaped backslashes (LaTeX in JSON)
        try:
            return json.loads(_fix_json_escapes(s))
        except json.JSONDecodeError:
            return None

    # Try full text first
    result = _try_loads(text.strip())
    if result is not None:
        return result

    # Find first {...} block
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        result = _try_loads(match.group())
        if result is not None:
            return result

    raise ValueError(f"Could not extract JSON from response:\n{text[:400]}")


def extract_final_answer(result: dict) -> str | None:
    """Safely get final_answer as string."""
    val = result.get("final_answer")
    if val is None:
        return None
    return str(val).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Baseline: extract original answers from steps JSON
# ─────────────────────────────────────────────────────────────────────────────

CHOICE_MAP = {"①": "1", "②": "2", "③": "3", "④": "4", "⑤": "5"}

def parse_baseline_answer(steps: list[dict], answer_type: str) -> str | None:
    """Extract final answer from the 'Final Answer' step body."""
    for s in reversed(steps):
        title = (s.get("title") or "").lower()
        body  = (s.get("body") or s.get("content") or "")
        if "final" in title or "answer" in title or "정답" in title:
            # Normalize choice symbols
            for sym, num in CHOICE_MAP.items():
                body = body.replace(sym, f" {num} ")
            nums = re.findall(r'\b(\d+)\b', body)
            if answer_type == "choice":
                # Want 1-5
                choices = [n for n in nums if 1 <= int(n) <= 5]
                if choices:
                    return choices[0]
                # Fallback: any number
                if nums:
                    return nums[0]
            else:
                # short answer: likely largest or last number
                if nums:
                    return nums[-1]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Method 1: Self-feedback
# ─────────────────────────────────────────────────────────────────────────────

def run_self_feedback(
    prob_id: str,
    problem_text: str,
    solutions: dict[str, list[dict]],
    answer_type: str,
) -> dict:
    """Each model reviews its own solution."""
    print(f"\n  [Self-feedback] {prob_id}")
    results: dict[str, dict] = {}

    for model_name, steps in solutions.items():
        # map display name → API model ID
        model_id = _resolve_model_id(model_name)
        own_text = _steps_to_text(steps)
        fmt = ANSWER_FORMAT[answer_type]

        prompt = SELF_FEEDBACK_PROMPT.format(
            problem=problem_text,
            own_steps=own_text,
            answer_format=fmt,
        )

        print(f"    → {model_name} reviewing own solution ... ", end="", flush=True)
        t0 = time.time()
        try:
            raw = call_llm(prompt, model_id, temperature=1.0, max_tokens=4096, json_mode=True)
            result = extract_json(raw)
        except Exception as e:
            print(f"ERROR: {e}")
            result = {"error": str(e), "final_answer": None, "revised": False}
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s) → answer={result.get('final_answer')} revised={result.get('revised')}")
        results[model_name] = {**result, "_model_id": model_id, "_elapsed_s": elapsed}

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Method 2: Cross-feedback
# ─────────────────────────────────────────────────────────────────────────────

def run_cross_feedback(
    prob_id: str,
    problem_text: str,
    solutions: dict[str, list[dict]],
    answer_type: str,
) -> dict:
    """Each model sees other models' solutions, then revises its own."""
    print(f"\n  [Cross-feedback] {prob_id}")
    results: dict[str, dict] = {}
    fmt = ANSWER_FORMAT[answer_type]
    model_names = list(solutions.keys())

    for target_name in model_names:
        target_id = _resolve_model_id(target_name)
        own_steps = solutions[target_name]
        own_text  = _steps_to_text(own_steps)

        # Other models' solutions
        other_parts = []
        for other_name, other_steps in solutions.items():
            if other_name == target_name:
                continue
            other_parts.append(f"### {other_name}\n{_steps_to_text(other_steps)}")
        other_text = "\n\n".join(other_parts)

        prompt = CROSS_FEEDBACK_PROMPT.format(
            problem=problem_text,
            own_steps=own_text,
            other_solutions=other_text,
            answer_format=fmt,
        )

        print(f"    → {target_name} reviewing with cross-feedback ... ", end="", flush=True)
        t0 = time.time()
        try:
            raw = call_llm(prompt, target_id, temperature=1.0, max_tokens=4096, json_mode=True)
            result = extract_json(raw)
        except Exception as e:
            print(f"ERROR: {e}")
            result = {"error": str(e), "final_answer": None, "revised": False}
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s) → answer={result.get('final_answer')} revised={result.get('revised')}")
        results[target_name] = {**result, "_model_id": target_id, "_elapsed_s": elapsed}

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Method 3: Flowmap Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def run_aggregation(
    prob_id: str,
    problem_text: str,
    solutions: dict[str, list[dict]],
    flowmap: dict | None,
    answer_type: str,
) -> dict:
    """Synthesizer sees all solutions (+ optional flowmap) → reconstructs answer."""
    print(f"\n  [Aggregation] {prob_id}")
    fmt = ANSWER_FORMAT[answer_type]

    flowmap_section = (
        _FLOWMAP_SECTION.format(flowmap=_flowmap_to_text(flowmap))
        if flowmap else ""
    )
    all_solutions_parts = []
    for name, steps in solutions.items():
        all_solutions_parts.append(f"### {name}\n{_steps_to_text(steps)}")
    all_solutions_text = "\n\n".join(all_solutions_parts)

    prompt = AGGREGATION_PROMPT.format(
        problem=problem_text,
        flowmap_section=flowmap_section,
        all_solutions=all_solutions_text,
        answer_format=fmt,
    )

    print(f"    → {SYNTH_MODEL} synthesizing ... ", end="", flush=True)
    t0 = time.time()
    try:
        raw = call_llm(prompt, SYNTH_MODEL, temperature=1.0, max_tokens=6000, json_mode=True)
        result = extract_json(raw)
    except Exception as e:
        print(f"ERROR: {e}")
        result = {"error": str(e), "final_answer": None, "confidence": "low"}
    elapsed = time.time() - t0
    print(f"done ({elapsed:.1f}s) → answer={result.get('final_answer')} confidence={result.get('confidence')}")
    return {**result, "_model_id": SYNTH_MODEL, "_elapsed_s": elapsed}


# ─────────────────────────────────────────────────────────────────────────────
# Model ID resolution
# ─────────────────────────────────────────────────────────────────────────────

# Display name → OpenRouter/OpenAI model ID
_MODEL_ID_MAP: dict[str, str] = {
    # legacy v4 models
    "gpt-5.2":           "openai/gpt-5.2",
    "grok-4":            "x-ai/grok-4",
    "minimax-m2.1":      "minimax/minimax-m2.1",
    # 5 production models (v5)
    "gpt-5-codex":       "openai/gpt-5-codex",
    "gpt-5":             "openai/gpt-5",
    "claude-opus-4.5":   "anthropic/claude-opus-4.5",
    "gemini-3-pro":      "google/gemini-3-pro-preview",
    "grok-4-fast":       "x-ai/grok-4-fast",
}

def _resolve_model_id(display_name: str) -> str:
    return _MODEL_ID_MAP.get(display_name, display_name)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def is_correct(predicted: str | None, ground_truth: str) -> bool:
    if predicted is None:
        return False
    return str(predicted).strip() == str(ground_truth).strip()


def score_method(answers: dict[str, str | None], ground_truth: str) -> dict:
    """Per-model correctness for self/cross methods."""
    return {
        model: {
            "answer": ans,
            "correct": is_correct(ans, ground_truth),
        }
        for model, ans in answers.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pretty print summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(all_results: dict):
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    for prob_id, res in all_results.items():
        gt = res["ground_truth"]
        print(f"\n  Problem: {prob_id}  (ground truth={gt})")
        print(f"  {'Method':<20} {'Model':<22} {'Answer':<10} {'Correct'}")
        print(f"  {'-'*68}")

        # Baseline
        for model, info in res["baseline"].items():
            ans  = info["answer"]
            corr = "✓" if is_correct(ans, gt) else "✗"
            print(f"  {'baseline':<20} {model:<22} {str(ans):<10} {corr}")

        # Self-feedback
        for model, info in res.get("self_feedback", {}).items():
            ans  = extract_final_answer(info)
            corr = "✓" if is_correct(ans, gt) else "✗"
            rev  = " (revised)" if info.get("revised") else ""
            print(f"  {'self-feedback':<20} {model:<22} {str(ans):<10} {corr}{rev}")

        # Cross-feedback
        for model, info in res.get("cross_feedback", {}).items():
            ans  = extract_final_answer(info)
            corr = "✓" if is_correct(ans, gt) else "✗"
            rev  = " (revised)" if info.get("revised") else ""
            print(f"  {'cross-feedback':<20} {model:<22} {str(ans):<10} {corr}{rev}")

        # Aggregation
        agg = res.get("aggregation", {})
        if agg:
            ans  = extract_final_answer(agg)
            corr = "✓" if is_correct(ans, gt) else "✗"
            conf = agg.get("confidence", "?")
            print(f"  {'aggregation':<20} {SYNTH_MODEL:<22} {str(ans):<10} {corr} (conf={conf})")

    print()

    # Accuracy per method (across all problems/models)
    print("  Accuracy by method (across all problems & models):")
    method_stats: dict[str, list[bool]] = {}

    for prob_id, res in all_results.items():
        gt = res["ground_truth"]

        for model, info in res.get("baseline", {}).items():
            method_stats.setdefault("baseline", []).append(is_correct(info["answer"], gt))

        for model, info in res.get("self_feedback", {}).items():
            method_stats.setdefault("self-feedback", []).append(
                is_correct(extract_final_answer(info), gt))

        for model, info in res.get("cross_feedback", {}).items():
            method_stats.setdefault("cross-feedback", []).append(
                is_correct(extract_final_answer(info), gt))

        agg = res.get("aggregation", {})
        if agg and "error" not in agg:
            method_stats.setdefault("aggregation", []).append(
                is_correct(extract_final_answer(agg), gt))

    for method, bools in method_stats.items():
        n   = len(bools)
        ok  = sum(bools)
        print(f"    {method:<20} {ok}/{n}  ({100*ok/n:.0f}%)")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prob", nargs="*",
                        help="Problem IDs to run (default: all 3)")
    parser.add_argument("--methods", nargs="*",
                        default=["self", "cross", "agg"],
                        help="Methods to run: self cross agg (default: all)")
    args = parser.parse_args()

    prob_ids = args.prob if args.prob else list(PROBLEMS_META.keys())
    methods  = set(args.methods)

    print("=" * 72)
    print("Feedback POC  (v5_full: 5 models × 46 problems)")
    print(f"  Problems : {len(prob_ids)} problems")
    print(f"  Methods  : {sorted(methods)}")
    print(f"  Synth    : {SYNTH_MODEL}")
    print("=" * 72)

    all_results: dict[str, dict] = {}

    for prob_id in prob_ids:
        meta        = PROBLEMS_META[prob_id]
        ground_truth = meta["answer"]
        answer_type  = meta["answer_type"]

        # Resume: load existing result and skip
        out_path = OUT_DIR / f"result_{prob_id}.json"
        if out_path.exists():
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            all_results[prob_id] = existing
            print(f"\n  SKIP (already done) {prob_id}")
            continue

        print(f"\n{'─'*72}")
        print(f"Problem: {prob_id}  answer={ground_truth} ({answer_type})")
        print(f"{'─'*72}")

        try:
            solutions = load_steps(prob_id)
        except FileNotFoundError:
            print(f"  SKIP — no steps file yet (run batch_solve.py first)")
            continue
        problem_text = load_problem_text(prob_id)
        flowmap      = load_flowmap(prob_id)  # None if not available

        model_names  = list(solutions.keys())
        has_flowmap  = flowmap is not None
        print(f"  Models in steps: {model_names}  flowmap={'yes' if has_flowmap else 'no'}")

        # Baseline: original answers from steps JSON
        baseline: dict[str, dict] = {}
        for model, steps in solutions.items():
            ans = parse_baseline_answer(steps, answer_type)
            baseline[model] = {"answer": ans, "correct": is_correct(ans, ground_truth)}
            print(f"  Baseline [{model}] → {ans}  {'✓' if is_correct(ans, ground_truth) else '✗'}")

        res: dict = {
            "problem_id":   prob_id,
            "ground_truth": ground_truth,
            "answer_type":  answer_type,
            "baseline":     baseline,
        }

        if "self" in methods:
            res["self_feedback"] = run_self_feedback(
                prob_id, problem_text, solutions, answer_type)

        if "cross" in methods:
            res["cross_feedback"] = run_cross_feedback(
                prob_id, problem_text, solutions, answer_type)

        if "agg" in methods:
            res["aggregation"] = run_aggregation(
                prob_id, problem_text, solutions, flowmap, answer_type)

        all_results[prob_id] = res

        # Save per-problem JSON
        out_path = OUT_DIR / f"result_{prob_id}.json"
        out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2))
        print(f"\n  Saved → {out_path}")

    # Save full results
    full_path = OUT_DIR / "all_results.json"
    full_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))

    print_summary(all_results)
    print(f"Full results → {full_path}")


if __name__ == "__main__":
    main()
