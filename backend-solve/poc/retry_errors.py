#!/usr/bin/env python3
"""
에러난 항목(402 크레딧/404 라우팅/JSON parse fail)만 골라서 재시도.

Usage:
    cd /workspace/backend-solve/poc
    python retry_errors.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

POC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(POC_DIR))

# feedback_poc 에서 필요한 것들 임포트
from feedback_poc import (
    OUT_DIR,
    PROBLEMS_META,
    SYNTH_MODEL,
    ANSWER_FORMAT,
    SELF_FEEDBACK_PROMPT,
    CROSS_FEEDBACK_PROMPT,
    AGGREGATION_PROMPT,
    _FLOWMAP_SECTION,
    _steps_to_text,
    _flowmap_to_text,
    _resolve_model_id,
    extract_json,
    load_steps,
    load_flowmap,
    load_problem_text,
)
from llm_client import call_llm


def is_retryable(info: dict) -> bool:
    """에러가 있는 항목 = 재시도 대상."""
    return info.get("error") is not None or info.get("final_answer") is None


def retry_self_entry(
    model_name: str,
    problem_text: str,
    solutions: dict,
    answer_type: str,
) -> dict:
    model_id = _resolve_model_id(model_name)
    own_text = _steps_to_text(solutions[model_name])
    fmt = ANSWER_FORMAT[answer_type]
    prompt = SELF_FEEDBACK_PROMPT.format(
        problem=problem_text,
        own_steps=own_text,
        answer_format=fmt,
    )
    t0 = time.time()
    try:
        raw = call_llm(prompt, model_id, temperature=1.0, max_tokens=4096)
        result = extract_json(raw)
    except Exception as e:
        print(f"ERROR: {e}")
        result = {"error": str(e), "final_answer": None, "revised": False}
    elapsed = time.time() - t0
    print(f"done ({elapsed:.1f}s) → answer={result.get('final_answer')}")
    return {**result, "_model_id": model_id, "_elapsed_s": elapsed}


def retry_cross_entry(
    target_name: str,
    problem_text: str,
    solutions: dict,
    answer_type: str,
) -> dict:
    target_id = _resolve_model_id(target_name)
    own_text = _steps_to_text(solutions[target_name])
    other_parts = [
        f"### {n}\n{_steps_to_text(s)}"
        for n, s in solutions.items() if n != target_name
    ]
    other_text = "\n\n".join(other_parts)
    fmt = ANSWER_FORMAT[answer_type]
    prompt = CROSS_FEEDBACK_PROMPT.format(
        problem=problem_text,
        own_steps=own_text,
        other_solutions=other_text,
        answer_format=fmt,
    )
    t0 = time.time()
    try:
        raw = call_llm(prompt, target_id, temperature=1.0, max_tokens=4096)
        result = extract_json(raw)
    except Exception as e:
        print(f"ERROR: {e}")
        result = {"error": str(e), "final_answer": None, "revised": False}
    elapsed = time.time() - t0
    print(f"done ({elapsed:.1f}s) → answer={result.get('final_answer')}")
    return {**result, "_model_id": target_id, "_elapsed_s": elapsed}


def retry_agg(
    problem_text: str,
    solutions: dict,
    flowmap,
    answer_type: str,
) -> dict:
    fmt = ANSWER_FORMAT[answer_type]
    flowmap_section = (
        _FLOWMAP_SECTION.format(flowmap=_flowmap_to_text(flowmap)) if flowmap else ""
    )
    all_solutions_text = "\n\n".join(
        f"### {n}\n{_steps_to_text(s)}" for n, s in solutions.items()
    )
    prompt = AGGREGATION_PROMPT.format(
        problem=problem_text,
        flowmap_section=flowmap_section,
        all_solutions=all_solutions_text,
        answer_format=fmt,
    )
    t0 = time.time()
    try:
        raw = call_llm(prompt, SYNTH_MODEL, temperature=1.0, max_tokens=6000)
        result = extract_json(raw)
    except Exception as e:
        print(f"ERROR: {e}")
        result = {"error": str(e), "final_answer": None, "confidence": "low"}
    elapsed = time.time() - t0
    print(f"done ({elapsed:.1f}s) → answer={result.get('final_answer')}")
    return {**result, "_model_id": SYNTH_MODEL, "_elapsed_s": elapsed}


def main():
    result_files = sorted(OUT_DIR.glob("result_2024_odd_*.json"))
    print(f"결과 파일 {len(result_files)}개 스캔 중...\n")

    total_retried = 0

    for rfile in result_files:
        data = json.loads(rfile.read_text(encoding="utf-8"))
        prob_id     = data["problem_id"]
        gt          = data["ground_truth"]
        answer_type = data["answer_type"]
        changed     = False

        try:
            solutions = load_steps(prob_id)
        except FileNotFoundError:
            continue
        problem_text = load_problem_text(prob_id)
        flowmap      = load_flowmap(prob_id)
        baseline     = data.get("baseline", {})

        # ── Self-feedback 재시도 ─────────────────────────────────
        for model_name, info in data.get("self_feedback", {}).items():
            if not is_retryable(info):
                continue
            if baseline.get(model_name, {}).get("answer") is None:
                continue  # 원본 풀이가 없으면 skip
            print(f"  [self] {prob_id} / {model_name} ... ", end="", flush=True)
            new_info = retry_self_entry(model_name, problem_text, solutions, answer_type)
            if new_info.get("final_answer") is not None:
                data["self_feedback"][model_name] = new_info
                changed = True
                total_retried += 1

        # ── Cross-feedback 재시도 ────────────────────────────────
        for model_name, info in data.get("cross_feedback", {}).items():
            if not is_retryable(info):
                continue
            if baseline.get(model_name, {}).get("answer") is None:
                continue
            print(f"  [cross] {prob_id} / {model_name} ... ", end="", flush=True)
            new_info = retry_cross_entry(model_name, problem_text, solutions, answer_type)
            if new_info.get("final_answer") is not None:
                data["cross_feedback"][model_name] = new_info
                changed = True
                total_retried += 1

        # ── Aggregation 재시도 ───────────────────────────────────
        agg = data.get("aggregation", {})
        if agg and is_retryable(agg):
            print(f"  [agg]  {prob_id} ... ", end="", flush=True)
            new_agg = retry_agg(problem_text, solutions, flowmap, answer_type)
            if new_agg.get("final_answer") is not None:
                data["aggregation"] = new_agg
                changed = True
                total_retried += 1

        if changed:
            rfile.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  → {rfile.name} 저장 완료")

    print(f"\n총 {total_retried}개 항목 재시도 완료.")


if __name__ == "__main__":
    main()
