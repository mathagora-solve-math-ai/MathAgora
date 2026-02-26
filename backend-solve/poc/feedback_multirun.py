#!/usr/bin/env python3
"""
Multi-run Feedback POC
======================
이미지 입력 기반 multi-run 풀이 결과(JSONL)를 사용하여
두 가지 피드백 파이프라인을 비교:

  1. Self-feedback   → Aggregation
     각 모델이 자기 자신의 3번 시도를 검토 → 모델별 최종답 → Claude가 합성
  2. Cross-feedback  → Aggregation
     각 모델이 다른 모델들의 best-run을 참고 → 모델별 최종답 → Claude가 합성

Usage:
    cd /workspace/backend-solve/poc
    python feedback_multirun.py                       # all 46 problems, both methods
    python feedback_multirun.py --method self         # self only
    python feedback_multirun.py --method cross        # cross only
    python feedback_multirun.py --prob 2024_odd_calculus_23
    python feedback_multirun.py --resume              # skip already-saved results
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

# ── paths ──────────────────────────────────────────────────────────────────────
POC_DIR   = Path(__file__).resolve().parent
DATA_CSV  = Path("/workspace/data/2024_math_odd.csv")
JSONL_SRC = Path("/workspace/0212/poc/output/backend_solve_parallel_5y_raw.jsonl")
OUT_DIR   = POC_DIR / "results" / "feedback_multirun"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(POC_DIR))
from llm_client import call_llm  # noqa: E402

# ── constants ──────────────────────────────────────────────────────────────────
SYNTH_MODEL = "anthropic/claude-opus-4.5"

ANSWER_FORMAT = {
    "5지선다형": "integer 1–5 (choice number: ①=1 ②=2 ③=3 ④=4 ⑤=5)",
    "단답형":    "integer 0–999 (actual computed value)",
}

# display-name → model_id used for API calls
_MODEL_ID_MAP: dict[str, str] = {
    "GPT-5-Codex":   "openai/gpt-5-codex",
    "GPT-5":         "openai/gpt-5",
    "Claude Opus 4.5": "anthropic/claude-opus-4.5",
    "Gemini 3 Pro":  "google/gemini-3-pro-preview",
    "Grok 4 Fast":   "x-ai/grok-4-fast",
}

# ── data loading ───────────────────────────────────────────────────────────────

def load_problem_meta() -> dict[str, dict]:
    """prob_id → {answer, prob_type, prob_desc}

    CSV uses a legacy typo 'probalitity'; JSONL uses the correct 'probability'.
    Both keys are registered so either spelling works.
    """
    csv.field_size_limit(10_000_000)
    meta: dict[str, dict] = {}
    with open(DATA_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row["prob_id"]
            entry = {
                "answer":    row["answer"],
                "prob_type": row["prob_type"],
                "prob_desc": row["prob_desc"],
            }
            meta[pid] = entry
            # Also register corrected spelling so JSONL prob_ids resolve correctly
            corrected = pid.replace("probalitity", "probability")
            if corrected != pid:
                meta[corrected] = entry
    return meta


def load_multirun_data(year: str = "2024") -> dict[str, dict[str, list[dict]]]:
    """Load JSONL → {prob_id: {model_display: [run1, run2, run3]}} for parse_ok=True runs."""
    all_records: dict[str, dict[str, list[dict]]] = {}

    with open(JSONL_SRC, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("year") != year:
                continue

            pid  = rec["prob_id"]
            mdl  = rec["model"]   # display name
            ri   = rec["run_index"]

            # Parse response_text
            if not rec.get("parse_ok"):
                continue  # skip broken JSON runs
            try:
                rt = json.loads(rec["response_text"])
            except Exception:
                continue

            steps = rt.get("steps", [])
            if not steps:
                continue  # no usable content

            entry = {
                "run_index":   ri,
                "steps":       steps,
                "final_answer": rt.get("final_answer"),
                "model_id":    rec["model_id"],
            }

            all_records.setdefault(pid, {}).setdefault(mdl, [])
            all_records[pid][mdl].append(entry)

    # Sort runs by run_index
    for pid in all_records:
        for mdl in all_records[pid]:
            all_records[pid][mdl].sort(key=lambda x: x["run_index"])

    return all_records


# ── prompt helpers ──────────────────────────────────────────────────────────────

def _steps_to_text(steps: list[dict]) -> str:
    parts = []
    for i, s in enumerate(steps):
        title = s.get("title") or f"Step {i+1}"
        body  = s.get("content") or s.get("body") or ""
        parts.append(f"[Step {i+1}: {title}]\n{body}")
    return "\n\n".join(parts)


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
    raise ValueError(f"Could not extract JSON from response:\n{text[:400]}")


# ── prompts ────────────────────────────────────────────────────────────────────

SELF_MULTIRUN_PROMPT = """\
You are reviewing your own multiple attempts at the same math problem.

## Problem
{problem}

## Your Previous Attempts
{attempts_text}

## Task
1. Compare all your attempts carefully for mathematical correctness.
2. Identify the most sound approach and correct any errors you find.
3. Produce your single best final answer.

## Output Format (JSON only, no markdown fences)
{{
  "steps": [
    {{"step_idx": 0, "title": "<concise title>", "content": "<solution content>"}},
    ...
  ],
  "final_answer": <{answer_format}>,
  "feedback": "<which attempt(s) you relied on and what you corrected>",
  "revised": <true if you changed the answer from attempt 1, false otherwise>
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
You are an expert math solver. Multiple AI models have attempted the same problem.
Study all the solutions provided, then write your OWN complete solution from scratch.
Do NOT copy steps verbatim — reason through it yourself.

## Problem
{problem}

## All Model Solutions
{all_solutions}

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


# ── method implementations ──────────────────────────────────────────────────────

def run_self_feedback(
    prob_id: str,
    problem_text: str,
    model_runs: dict[str, list[dict]],  # {display_name: [run1, run2, ...]}
    prob_type: str,
) -> dict[str, dict]:
    """Each model reviews its own 3 (or fewer) runs → 1 revised answer."""
    print(f"\n  [Self-feedback] {prob_id}")
    results: dict[str, dict] = {}
    fmt = ANSWER_FORMAT.get(prob_type, "integer")

    for disp_name, runs in model_runs.items():
        model_id = _MODEL_ID_MAP.get(disp_name, disp_name)

        # Build attempts text
        attempts_parts = []
        for run in runs:
            header = f"### Attempt {run['run_index']} (answer: {run['final_answer']})"
            attempts_parts.append(f"{header}\n{_steps_to_text(run['steps'])}")
        attempts_text = "\n\n".join(attempts_parts)

        prompt = SELF_MULTIRUN_PROMPT.format(
            problem=problem_text,
            attempts_text=attempts_text,
            answer_format=fmt,
        )

        print(f"    → {disp_name} reviewing {len(runs)} own run(s) ... ", end="", flush=True)
        t0 = time.time()
        try:
            raw    = call_llm(prompt, model_id, temperature=1.0, max_tokens=4096, json_mode=True)
            result = extract_json(raw)
        except Exception as e:
            print(f"ERROR: {e}")
            result = {"error": str(e), "final_answer": None, "revised": False}
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s) → answer={result.get('final_answer')} revised={result.get('revised')}")
        results[disp_name] = {**result, "_model_id": model_id, "_elapsed_s": elapsed}

    return results


def _best_run(runs: list[dict]) -> dict | None:
    """Return first valid run (already sorted by run_index)."""
    return runs[0] if runs else None


def run_cross_feedback(
    prob_id: str,
    problem_text: str,
    model_runs: dict[str, list[dict]],
    prob_type: str,
) -> dict[str, dict]:
    """Each model sees its own best run + all other models' best runs → revised answer."""
    print(f"\n  [Cross-feedback] {prob_id}")
    results: dict[str, dict] = {}
    fmt = ANSWER_FORMAT.get(prob_type, "integer")

    # Precompute best run for each model
    best_runs = {name: _best_run(runs) for name, runs in model_runs.items()}

    for target_name, target_runs in model_runs.items():
        model_id = _MODEL_ID_MAP.get(target_name, target_name)
        own_run  = _best_run(target_runs)
        if own_run is None:
            continue
        own_text = _steps_to_text(own_run["steps"])

        # Other models' best run
        other_parts = []
        for other_name, other_run in best_runs.items():
            if other_name == target_name or other_run is None:
                continue
            other_parts.append(f"### {other_name} (answer: {other_run['final_answer']})\n{_steps_to_text(other_run['steps'])}")
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
            raw    = call_llm(prompt, model_id, temperature=1.0, max_tokens=4096, json_mode=True)
            result = extract_json(raw)
        except Exception as e:
            print(f"ERROR: {e}")
            result = {"error": str(e), "final_answer": None, "revised": False}
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s) → answer={result.get('final_answer')} revised={result.get('revised')}")
        results[target_name] = {**result, "_model_id": model_id, "_elapsed_s": elapsed}

    return results


def run_aggregation(
    prob_id: str,
    problem_text: str,
    feedback_results: dict[str, dict],  # {model_name: feedback_dict}
    prob_type: str,
    method_label: str,
) -> dict:
    """Synthesizer reads all post-feedback solutions → final answer."""
    print(f"\n  [Aggregation/{method_label}] {prob_id}")
    fmt = ANSWER_FORMAT.get(prob_type, "integer")

    parts = []
    for name, res in feedback_results.items():
        steps = res.get("steps", [])
        ans   = res.get("final_answer")
        if steps:
            parts.append(f"### {name} (answer: {ans})\n{_steps_to_text(steps)}")
        else:
            parts.append(f"### {name}\n(no steps available, answer: {ans})")
    all_solutions_text = "\n\n".join(parts)

    prompt = AGGREGATION_PROMPT.format(
        problem=problem_text,
        all_solutions=all_solutions_text,
        answer_format=fmt,
    )

    print(f"    → {SYNTH_MODEL} synthesizing ... ", end="", flush=True)
    t0 = time.time()
    try:
        raw    = call_llm(prompt, SYNTH_MODEL, temperature=1.0, max_tokens=6000, json_mode=True)
        result = extract_json(raw)
    except Exception as e:
        print(f"ERROR: {e}")
        result = {"error": str(e), "final_answer": None, "confidence": "low"}
    elapsed = time.time() - t0
    print(f"done ({elapsed:.1f}s) → answer={result.get('final_answer')} confidence={result.get('confidence')}")
    return {**result, "_model_id": SYNTH_MODEL, "_elapsed_s": elapsed}


# ── baseline extraction ─────────────────────────────────────────────────────────

def get_baseline(model_runs: dict[str, list[dict]], ground_truth: str) -> dict[str, dict]:
    """Majority-vote over runs for each model as baseline."""
    from collections import Counter
    baseline: dict[str, dict] = {}
    for name, runs in model_runs.items():
        answers = [str(r["final_answer"]) for r in runs if r.get("final_answer") is not None]
        if not answers:
            voted = None
        else:
            voted = Counter(answers).most_common(1)[0][0]
        baseline[name] = {
            "answer":   voted,
            "correct":  str(voted) == str(ground_truth) if voted else False,
            "_runs":    [{"run_index": r["run_index"], "answer": r["final_answer"]} for r in runs],
        }
    return baseline


# ── main loop ──────────────────────────────────────────────────────────────────

def process_problem(
    prob_id: str,
    meta: dict,
    model_runs: dict[str, list[dict]],
    methods: list[str],
    resume: bool,
) -> dict:
    out_path = OUT_DIR / f"result_{prob_id}.json"

    # Load existing partial result if resuming
    existing: dict = {}
    if resume and out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        # Check if fully done
        done = True
        for m in methods:
            if m not in existing or f"{m}_aggregation" not in existing:
                done = False; break
        if done:
            print(f"[SKIP] {prob_id} (already complete)")
            return existing

    gt        = meta["answer"]
    prob_type = meta["prob_type"]
    prob_text = meta["prob_desc"]

    print(f"\n{'─'*72}")
    print(f"Problem: {prob_id}  answer={gt} ({prob_type})")
    print(f"  Models in data: {list(model_runs.keys())}")

    # Baseline (always compute)
    baseline = get_baseline(model_runs, gt)
    print("  Baseline:", {k: v["answer"] for k, v in baseline.items()})

    result: dict = {
        **existing,
        "problem_id":   prob_id,
        "ground_truth": gt,
        "prob_type":    prob_type,
        "baseline":     baseline,
    }

    # ── Self-feedback → Aggregation ────────────────────────────────────────────
    if "self" in methods:
        if "self" not in existing:
            sf = run_self_feedback(prob_id, prob_text, model_runs, prob_type)
            result["self"] = sf
        else:
            sf = existing["self"]
            print(f"  [Self-feedback] {prob_id} (cached)")

        if "self_aggregation" not in existing:
            sf_agg = run_aggregation(prob_id, prob_text, sf, prob_type, "self")
            result["self_aggregation"] = sf_agg
        else:
            result["self_aggregation"] = existing["self_aggregation"]

    # ── Cross-feedback → Aggregation ───────────────────────────────────────────
    if "cross" in methods:
        if "cross" not in existing:
            cf = run_cross_feedback(prob_id, prob_text, model_runs, prob_type)
            result["cross"] = cf
        else:
            cf = existing["cross"]
            print(f"  [Cross-feedback] {prob_id} (cached)")

        if "cross_aggregation" not in existing:
            cf_agg = run_aggregation(prob_id, prob_text, cf, prob_type, "cross")
            result["cross_aggregation"] = cf_agg
        else:
            result["cross_aggregation"] = existing["cross_aggregation"]

    # Save
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Saved → {out_path}")
    return result


# ── analysis ───────────────────────────────────────────────────────────────────

def analyze_results(results: list[dict]):
    def eq(a, b):
        return str(a).strip() == str(b).strip() if a is not None and b is not None else False

    methods_stats: dict[str, dict] = {}

    for r in results:
        gt = r["ground_truth"]

        # Baseline (majority-vote per model)
        for name, bdata in r.get("baseline", {}).items():
            methods_stats.setdefault("baseline", {"c": 0, "t": 0, "null": 0})
            methods_stats["baseline"]["t"] += 1
            if bdata["answer"] is None:
                methods_stats["baseline"]["null"] += 1
            elif eq(bdata["answer"], gt):
                methods_stats["baseline"]["c"] += 1

        # Self-feedback per model
        for name, sdata in r.get("self", {}).items():
            ans = sdata.get("final_answer")
            methods_stats.setdefault("self-feedback", {"c": 0, "t": 0, "null": 0})
            methods_stats["self-feedback"]["t"] += 1
            if ans is None:
                methods_stats["self-feedback"]["null"] += 1
            elif eq(ans, gt):
                methods_stats["self-feedback"]["c"] += 1

        # Self aggregation
        sa = r.get("self_aggregation", {})
        if sa:
            ans = sa.get("final_answer")
            methods_stats.setdefault("self→agg", {"c": 0, "t": 0, "null": 0})
            methods_stats["self→agg"]["t"] += 1
            if ans is None:
                methods_stats["self→agg"]["null"] += 1
            elif eq(ans, gt):
                methods_stats["self→agg"]["c"] += 1

        # Cross-feedback per model
        for name, cdata in r.get("cross", {}).items():
            ans = cdata.get("final_answer")
            methods_stats.setdefault("cross-feedback", {"c": 0, "t": 0, "null": 0})
            methods_stats["cross-feedback"]["t"] += 1
            if ans is None:
                methods_stats["cross-feedback"]["null"] += 1
            elif eq(ans, gt):
                methods_stats["cross-feedback"]["c"] += 1

        # Cross aggregation
        ca = r.get("cross_aggregation", {})
        if ca:
            ans = ca.get("final_answer")
            methods_stats.setdefault("cross→agg", {"c": 0, "t": 0, "null": 0})
            methods_stats["cross→agg"]["t"] += 1
            if ans is None:
                methods_stats["cross→agg"]["null"] += 1
            elif eq(ans, gt):
                methods_stats["cross→agg"]["c"] += 1

    print("\n" + "=" * 65)
    print(f"RESULTS  ({len(results)} problems)")
    print("=" * 65)
    print(f"{'Method':<20} {'Correct':>8} {'Total':>7} {'Null':>6} {'Acc%':>8}")
    print("-" * 65)
    for method, s in methods_stats.items():
        acc = s["c"] / s["t"] * 100 if s["t"] else 0
        print(f"{method:<20} {s['c']:>8} {s['t']:>7} {s['null']:>6} {acc:>7.1f}%")
    print()


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prob",    nargs="*", help="Specific prob_id(s)")
    parser.add_argument("--method",  choices=["self", "cross", "both"], default="both")
    parser.add_argument("--resume",  action="store_true", help="Skip already-saved results")
    args = parser.parse_args()

    methods = ["self", "cross"] if args.method == "both" else [args.method]

    print("Loading problem metadata ...")
    prob_meta = load_problem_meta()

    print(f"Loading multirun data from {JSONL_SRC} ...")
    multirun = load_multirun_data(year="2024")

    # Determine target problems
    all_probs = sorted(multirun.keys())
    if args.prob:
        target_probs = [p for p in args.prob if p in multirun]
        missing = [p for p in args.prob if p not in multirun]
        if missing:
            print(f"WARNING: prob_ids not in data: {missing}")
    else:
        target_probs = all_probs

    print(f"\n{'='*72}")
    print(f"Multi-run Feedback POC")
    print(f"  Problems : {len(target_probs)}")
    print(f"  Methods  : {methods}")
    print(f"  Synth    : {SYNTH_MODEL}")
    print(f"{'='*72}")

    all_results = []
    for prob_id in target_probs:
        meta      = prob_meta.get(prob_id, {})
        runs      = multirun[prob_id]
        result    = process_problem(prob_id, meta, runs, methods, args.resume)
        all_results.append(result)

    analyze_results(all_results)


if __name__ == "__main__":
    main()
