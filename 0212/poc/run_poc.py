#!/usr/bin/env python3
"""
PoC script for ACL 2026 demo – LLM step-by-step solution alignment.


Tasks:
  1. Compare prompt versions (simple / step-by-step / titled variants)
  2. Cosine-similarity based step alignment across solutions
  3. Single-model diversity (temperature=1.0, 5 runs)
  4. Multi-model diversity (N different models, 1 run each)
  5. Temperature/model statistics (v4 prompt, 6 models, 4 temps, 3 runs each)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import csv
from collections import Counter
from itertools import combinations
from datetime import datetime

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

from config import (
    ALL_PROBLEMS,
    DEFAULT_MODEL,
    encode_image_base64,
    MODELS,
    PROBLEMS,
    PROMPT_VERSIONS,
)
from llm_client import call_llm

OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)
FLAT_RESULTS_PATH = os.path.join(OUT_DIR, "results_flat.csv")
FLAT_RESULTS_HEADER = [
    "model",
    "temperature",
    "run_index",
    "problem_id",
    "step_count",
    "correct",
    "titles",
    "elapsed_time_seconds",
]


# ───────────────────────────────────────────────────────────────
# Parsing helpers
# ───────────────────────────────────────────────────────────────

def parse_v3_steps(text: str) -> list[dict]:
    """Parse [STEP title="..."] ... [FINAL_ANSWER] format."""
    marker_re = re.compile(
        r'\[STEP\s+title="([^"]+)"\]|\[FINAL_ANSWER\]'
    )
    markers = []
    for m in marker_re.finditer(text):
        if m.group(0).startswith("[STEP"):
            markers.append({"type": "step", "title": m.group(1), "end": m.end()})
        else:
            markers.append({"type": "final", "title": "Final Answer", "end": m.end()})

    steps = []
    for i, marker in enumerate(markers):
        next_start = markers[i + 1].get("end", len(text)) if i + 1 < len(markers) else len(text)
        # body = text between current marker end and next marker start
        next_marker_match = marker_re.search(text, marker["end"])
        body_end = next_marker_match.start() if next_marker_match else len(text)
        body = text[marker["end"]:body_end].strip()
        steps.append({"title": marker["title"], "body": body, "type": marker["type"]})
    return steps


def heuristic_split_steps(text: str) -> list[dict]:
    """Heuristic step splitting for v1/v2 responses.

    Tries numbered patterns like '1.', 'Step 1:', '단계 1:', '**Step 1**' etc.
    Falls back to paragraph splitting.
    """
    # Try numbered patterns
    patterns = [
        r'(?:^|\n)\s*(?:\*\*)?(?:Step|단계|STEP)\s*(\d+)[.:)\s]',
        r'(?:^|\n)\s*(\d+)\.\s',
    ]
    for pat in patterns:
        splits = list(re.finditer(pat, text))
        if len(splits) >= 2:
            steps = []
            for i, m in enumerate(splits):
                start = m.end()
                end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
                body = text[start:end].strip()
                title = f"Step {m.group(1)}"
                steps.append({"title": title, "body": body, "type": "step"})
            return steps

    # Fallback: paragraph splitting
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return [
        {"title": f"Part {i+1}", "body": p, "type": "step"}
        for i, p in enumerate(paragraphs)
    ]


def extract_steps(text: str, prompt_version: str) -> list[dict]:
    """Extract steps from LLM response based on prompt version."""
    if prompt_version in ("v3_titled", "v4_flow_titled"):
        steps = parse_v3_steps(text)
        if steps:
            return steps
    return heuristic_split_steps(text)


def _format_prompt_for_problem(template: str, prob: dict, problem_text: str | None = None) -> str:
    """Format prompt template with backward-compatible keys."""
    if problem_text is None:
        problem_text = prob.get("text", "")
    return template.format(
        problem=problem_text,
        answer_format_rule=prob.get("answer_format_rule", ""),
        problem_type=prob.get("prob_type_en", "Unknown"),
    )


def _build_task5_user_messages(prompt_text: str, prob: dict, require_image: bool = True) -> list[dict]:
    """Build user messages: image-only block (base64), prompt is sent via system message."""
    content = []
    image_path = str(prob.get("prob_img_abs_path", "")).strip()
    if image_path and os.path.exists(image_path):
        b64, media_type = encode_image_base64(image_path)
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64,
                },
            }
        )
    elif require_image:
        raise FileNotFoundError(
            f"Problem image not found for {prob.get('id')}: {image_path} "
            f"(prob_img_path={prob.get('prob_img_path', '')})"
        )
    else:
        content.append({"type": "text", "text": f"문제 원문:\n{prob.get('text', '').strip()}"})
    return [{"role": "user", "content": content}]


def _normalize_for_match(text: str) -> str:
    """Normalize text for lenient exact/containment matching."""
    return re.sub(r"[^0-9A-Za-z가-힣]+", "", text).lower()


def _extract_final_answer(response: str, steps: list[dict]) -> str:
    """Extract final answer text from parsed steps or raw response fallback."""
    for step in reversed(steps):
        if step.get("type") == "final":
            return step.get("body", "").strip()

    # Fallback: try marker-based extraction from raw response
    m = re.search(r"\[FINAL_ANSWER\](.*)$", response, flags=re.DOTALL)
    if m:
        return m.group(1).strip()

    # Last resort: tail text
    return response.strip()[-300:]


def _is_correct_answer(pred_answer: str, gold_answer: str) -> bool:
    """Heuristic correctness check based on normalized exact/containment match."""
    pred_norm = _normalize_for_match(pred_answer)
    gold_norm = _normalize_for_match(gold_answer)
    if not pred_norm or not gold_norm:
        return False
    if pred_norm == gold_norm:
        return True
    if gold_norm in pred_norm:
        return True
    # Numeric fallback: if all gold numbers appear in prediction numbers
    gold_nums = re.findall(r"-?\d+(?:\.\d+)?", gold_answer)
    pred_nums = re.findall(r"-?\d+(?:\.\d+)?", pred_answer)
    if gold_nums and pred_nums and set(gold_nums).issubset(set(pred_nums)):
        return True
    return False


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _format_duration(seconds: float) -> str:
    seconds = int(max(0, round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _append_flat_result_row(row: dict) -> None:
    """Append one run-level row to results_flat.csv (create header if needed)."""
    file_exists = os.path.exists(FLAT_RESULTS_PATH)
    write_header = (not file_exists) or os.path.getsize(FLAT_RESULTS_PATH) == 0
    with open(FLAT_RESULTS_PATH, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FLAT_RESULTS_HEADER)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _compute_group_stats(
    step_counts: list[int],
    correct_flags: list[bool],
) -> dict:
    """Compute per-(model, temperature) summary stats."""
    step_arr = np.asarray(step_counts, dtype=float)
    acc_arr = np.asarray(correct_flags, dtype=float)

    step_mean = float(step_arr.mean()) if step_arr.size else 0.0
    step_std = float(step_arr.std(ddof=1)) if step_arr.size > 1 else 0.0

    accuracy_mean = float(acc_arr.mean()) if acc_arr.size else 0.0
    accuracy_std = float(acc_arr.std(ddof=1)) if acc_arr.size > 1 else 0.0

    # Normal approximation: p ± 1.96 * sqrt(p(1-p)/n)
    n = int(acc_arr.size)
    if n > 0:
        se = float(np.sqrt(max(0.0, accuracy_mean * (1.0 - accuracy_mean)) / n))
        ci_low = max(0.0, accuracy_mean - 1.96 * se)
        ci_high = min(1.0, accuracy_mean + 1.96 * se)
    else:
        ci_low, ci_high = 0.0, 0.0

    return {
        "step_mean": step_mean,
        "step_std": step_std,
        "accuracy_mean": accuracy_mean,
        "accuracy_std": accuracy_std,
        "accuracy_ci_95": {
            "lower": ci_low,
            "upper": ci_high,
        },
    }


# ───────────────────────────────────────────────────────────────
# Similarity computation
# ───────────────────────────────────────────────────────────────

def compute_step_similarities(all_steps: list[list[dict]]) -> np.ndarray:
    """Compute pairwise cosine similarity between all steps across solutions.

    Args:
        all_steps: list of step-lists, one per solution
    Returns:
        similarity matrix (N_total x N_total)
    """
    texts = []
    for steps in all_steps:
        for step in steps:
            texts.append(f"{step['title']} {step['body']}")

    if not texts:
        return np.array([])

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=10000,
    )
    tfidf = vectorizer.fit_transform(texts)
    return cosine_similarity(tfidf)


def build_alignment_groups(
    all_steps: list[list[dict]],
    sim_matrix: np.ndarray,
    threshold: float = 0.3,
) -> list[dict]:
    """Group similar steps across different solutions using similarity threshold.

    Returns list of alignment groups with members from different solutions.
    """
    # Build flat index → (solution_idx, step_idx)
    flat_index = []
    for sol_idx, steps in enumerate(all_steps):
        for step_idx in range(len(steps)):
            flat_index.append((sol_idx, step_idx))

    n = len(flat_index)
    # Union-Find
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Only merge steps from *different* solutions
    for i in range(n):
        for j in range(i + 1, n):
            if flat_index[i][0] != flat_index[j][0]:  # different solution
                if sim_matrix[i][j] >= threshold:
                    union(i, j)

    # Collect groups
    clusters = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(i)

    groups = []
    for members_flat in clusters.values():
        solutions_in_group = set(flat_index[m][0] for m in members_flat)
        if len(solutions_in_group) < 2:
            continue
        members = []
        for m in members_flat:
            sol_idx, step_idx = flat_index[m]
            step = all_steps[sol_idx][step_idx]
            members.append({
                "solution_idx": sol_idx,
                "step_idx": step_idx,
                "title": step["title"],
                "body_preview": step["body"][:80],
            })
        groups.append({"members": members, "n_solutions": len(solutions_in_group)})

    groups.sort(key=lambda g: g["n_solutions"], reverse=True)
    return groups


# ───────────────────────────────────────────────────────────────
# Task runners
# ───────────────────────────────────────────────────────────────

def task1_prompt_versions(prob_key: str = "prob1"):
    """Task 1: Compare prompt versions for step extraction quality."""
    print("\n" + "=" * 70)
    print("TASK 1: Prompt version comparison")
    print("=" * 70)

    prob = PROBLEMS[prob_key]
    print(f"Problem: {prob['label']}")
    print(f"Answer: {prob['answer']}")
    print(f"Model: {DEFAULT_MODEL}")
    print()

    results = {}
    for ver_name, template in PROMPT_VERSIONS.items():
        prompt = _format_prompt_for_problem(template, prob)
        print(f"--- {ver_name} ---")
        print(f"Calling {DEFAULT_MODEL} ...")
        t0 = time.time()
        response = call_llm(prompt, DEFAULT_MODEL, temperature=0.3)
        elapsed = time.time() - t0
        print(f"  Response received ({elapsed:.1f}s, {len(response)} chars)")

        steps = extract_steps(response, ver_name)
        print(f"  Extracted {len(steps)} steps:")
        for i, s in enumerate(steps):
            print(f"    [{i+1}] {s['title']}: {s['body'][:60]}...")

        results[ver_name] = {
            "raw_response": response,
            "steps": steps,
            "elapsed_s": elapsed,
            "n_steps": len(steps),
        }
        print()

    out_path = os.path.join(OUT_DIR, f"task1_{prob_key}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {out_path}")
    return results


def task2_step_similarity(prob_key: str = "prob1"):
    """Task 2: Cosine-similarity based step alignment verification."""
    print("\n" + "=" * 70)
    print("TASK 2: Step similarity & alignment")
    print("=" * 70)

    prob = PROBLEMS[prob_key]
    template = PROMPT_VERSIONS["v3_titled"]
    prompt = _format_prompt_for_problem(template, prob)

    # Get 3 solutions from the same model with temperature=0.3 for consistency
    # (to test alignment, not diversity)
    models_to_use = list(MODELS.keys())[:3]
    all_steps = []
    solution_labels = []

    for model_key in models_to_use:
        model_id = MODELS[model_key]
        print(f"Calling {model_key} ({model_id}) ...")
        t0 = time.time()
        response = call_llm(prompt, model_id, temperature=0.3)
        elapsed = time.time() - t0
        steps = extract_steps(response, "v3_titled")
        all_steps.append(steps)
        solution_labels.append(model_key)
        print(f"  {model_key}: {len(steps)} steps ({elapsed:.1f}s)")
        for i, s in enumerate(steps):
            print(f"    [{i+1}] {s['title']}")

    print(f"\nComputing cosine similarity ...")
    sim_matrix = compute_step_similarities(all_steps)

    print(f"\nBuilding alignment groups (threshold=0.3) ...")
    groups = build_alignment_groups(all_steps, sim_matrix, threshold=0.3)

    print(f"\nFound {len(groups)} alignment groups:")
    for gi, g in enumerate(groups):
        print(f"  Group {gi+1} ({g['n_solutions']} solutions):")
        for m in g["members"]:
            label = solution_labels[m["solution_idx"]]
            print(f"    - [{label}] Step {m['step_idx']+1}: {m['title']}")

    # Print pairwise similarity between all step pairs (cross-solution)
    print("\n--- Cross-solution step similarities ---")
    flat_index = []
    for sol_idx, steps in enumerate(all_steps):
        for step_idx in range(len(steps)):
            flat_index.append((sol_idx, step_idx))

    for i in range(len(flat_index)):
        for j in range(i + 1, len(flat_index)):
            si, sti = flat_index[i]
            sj, stj = flat_index[j]
            if si == sj:
                continue
            score = sim_matrix[i][j]
            if score >= 0.15:
                label_i = solution_labels[si]
                label_j = solution_labels[sj]
                title_i = all_steps[si][sti]["title"]
                title_j = all_steps[sj][stj]["title"]
                print(f"  {score:.3f}  [{label_i}] {title_i}  <->  [{label_j}] {title_j}")

    # Save
    out_path = os.path.join(OUT_DIR, f"task2_{prob_key}.json")
    result = {
        "solutions": {
            label: [{"title": s["title"], "body": s["body"]} for s in steps]
            for label, steps in zip(solution_labels, all_steps)
        },
        "alignment_groups": groups,
        "sim_matrix": sim_matrix.tolist() if sim_matrix.size else [],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")
    return result


def task3_single_model_diversity(prob_key: str = "prob1", n_runs: int = 5):
    """Task 3: Single model, temperature=1.0, multiple runs → diversity check."""
    print("\n" + "=" * 70)
    print("TASK 3: Single-model diversity (temperature=1.0)")
    print("=" * 70)

    prob = PROBLEMS[prob_key]
    template = PROMPT_VERSIONS["v3_titled"]
    prompt = _format_prompt_for_problem(template, prob)
    model_id = MODELS[list(MODELS.keys())[0]]

    print(f"Problem: {prob['label']}")
    print(f"Model: {model_id}, temperature=1.0, {n_runs} runs")
    print()

    all_steps = []
    run_data = []

    for run_idx in range(n_runs):
        print(f"Run {run_idx+1}/{n_runs} ...")
        t0 = time.time()
        response = call_llm(prompt, model_id, temperature=1.0)
        elapsed = time.time() - t0
        steps = extract_steps(response, "v3_titled")
        all_steps.append(steps)
        run_data.append({
            "raw_response": response,
            "steps": [{"title": s["title"], "body": s["body"]} for s in steps],
            "elapsed_s": elapsed,
        })
        print(f"  {len(steps)} steps ({elapsed:.1f}s): {[s['title'] for s in steps]}")

    # Compute pairwise solution-level similarity
    print("\n--- Pairwise solution similarity ---")
    solution_texts = [
        " ".join(f"{s['title']} {s['body']}" for s in steps) for steps in all_steps
    ]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=10000)
    tfidf = vectorizer.fit_transform(solution_texts)
    sol_sim = cosine_similarity(tfidf)

    for i, j in combinations(range(n_runs), 2):
        print(f"  Run {i+1} vs Run {j+1}: {sol_sim[i][j]:.3f}")

    avg_sim = np.mean([sol_sim[i][j] for i, j in combinations(range(n_runs), 2)])
    print(f"\n  Average pairwise similarity: {avg_sim:.3f}")
    print(f"  (Lower = more diverse)")

    # Step-level alignment
    sim_matrix = compute_step_similarities(all_steps)
    groups = build_alignment_groups(all_steps, sim_matrix, threshold=0.3)
    print(f"\n  Alignment groups found: {len(groups)}")
    for gi, g in enumerate(groups):
        members_str = [
            "Run{}-Step{}:{}".format(m["solution_idx"]+1, m["step_idx"]+1, m["title"])
            for m in g["members"]
        ]
        print(f"    Group {gi+1}: {members_str}")

    out_path = os.path.join(OUT_DIR, f"task3_{prob_key}.json")
    result = {
        "model": model_id,
        "temperature": 1.0,
        "n_runs": n_runs,
        "runs": run_data,
        "solution_similarity_matrix": sol_sim.tolist(),
        "avg_pairwise_similarity": float(avg_sim),
        "alignment_groups": groups,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")
    return result


def task4_multi_model_diversity(prob_key: str = "prob1"):
    """Task 4: Multiple different models → diversity check."""
    print("\n" + "=" * 70)
    print("TASK 4: Multi-model diversity")
    print("=" * 70)

    prob = PROBLEMS[prob_key]
    template = PROMPT_VERSIONS["v3_titled"]
    prompt = _format_prompt_for_problem(template, prob)

    print(f"Problem: {prob['label']}")
    print(f"Models: {list(MODELS.keys())}")
    print()

    all_steps = []
    model_labels = []
    run_data = {}

    for model_key, model_id in MODELS.items():
        print(f"Calling {model_key} ({model_id}) ...")
        t0 = time.time()
        response = call_llm(prompt, model_id, temperature=0.7)
        elapsed = time.time() - t0
        steps = extract_steps(response, "v3_titled")
        all_steps.append(steps)
        model_labels.append(model_key)
        run_data[model_key] = {
            "model_id": model_id,
            "raw_response": response,
            "steps": [{"title": s["title"], "body": s["body"]} for s in steps],
            "elapsed_s": elapsed,
        }
        print(f"  {model_key}: {len(steps)} steps ({elapsed:.1f}s)")
        for i, s in enumerate(steps):
            print(f"    [{i+1}] {s['title']}")

    # Solution-level similarity
    print("\n--- Pairwise solution similarity ---")
    solution_texts = [
        " ".join(f"{s['title']} {s['body']}" for s in steps) for steps in all_steps
    ]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=10000)
    tfidf = vectorizer.fit_transform(solution_texts)
    sol_sim = cosine_similarity(tfidf)

    n_models = len(model_labels)
    for i, j in combinations(range(n_models), 2):
        print(f"  {model_labels[i]} vs {model_labels[j]}: {sol_sim[i][j]:.3f}")

    avg_sim = np.mean([sol_sim[i][j] for i, j in combinations(range(n_models), 2)])
    print(f"\n  Average pairwise similarity: {avg_sim:.3f}")

    # Step alignment
    sim_matrix = compute_step_similarities(all_steps)
    groups = build_alignment_groups(all_steps, sim_matrix, threshold=0.3)
    print(f"\n  Alignment groups found: {len(groups)}")
    for gi, g in enumerate(groups):
        members_str = [
            f"{model_labels[m['solution_idx']]}-Step{m['step_idx']+1}:{m['title']}"
            for m in g["members"]
        ]
        print(f"    Group {gi+1}: {members_str}")

    out_path = os.path.join(OUT_DIR, f"task4_{prob_key}.json")
    result = {
        "models": run_data,
        "solution_similarity_matrix": sol_sim.tolist(),
        "model_labels": model_labels,
        "avg_pairwise_similarity": float(avg_sim),
        "alignment_groups": groups,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")
    return result


def _resolve_task5_problem_items(prob_key: str) -> list[tuple[str, dict]]:
    """Resolve task5 problem targets from alias/prob_id/all."""
    if prob_key == "all":
        return list(ALL_PROBLEMS.items())
    if prob_key in PROBLEMS:
        prob = PROBLEMS[prob_key]
        return [(prob["id"], prob)]
    if prob_key in ALL_PROBLEMS:
        return [(prob_key, ALL_PROBLEMS[prob_key])]
    raise ValueError(
        f"Unknown problem key '{prob_key}'. Use prob1/prob2/prob22, a prob_id, or all."
    )


def _run_task5_for_one_problem(
    prob: dict,
    problem_tag: str,
    temperatures: tuple[float, ...],
    n_runs: int,
    print_run_logs: bool = True,
    progress_bar=None,
    global_state: dict | None = None,
) -> dict:
    """Run task5 statistics for one problem and save per-problem JSON."""
    started_at = _utc_now_iso()
    problem_t0 = time.time()
    prompt_version = "v4_flow_titled"
    template = PROMPT_VERSIONS[prompt_version]
    has_problem_image = bool(
        str(prob.get("prob_img_abs_path", "")).strip()
        and os.path.exists(str(prob.get("prob_img_abs_path", "")).strip())
    )
    problem_for_prompt = (
        "(문제 이미지는 함께 제공됩니다. 이미지 내용을 기준으로 풀이하세요.)"
        if has_problem_image
        else prob["text"]
    )
    answer_format_rule = prob.get("answer_format_rule", "")
    problem_type = prob.get("prob_type_en", "Unknown")
    prompt = _format_prompt_for_problem(
        template=template,
        prob=prob,
        problem_text=problem_for_prompt,
    )
    task_messages = _build_task5_user_messages(prompt, prob, require_image=True)

    print(f"Problem: {prob['label']}")
    print(f"Problem type: {problem_type}")
    print(f"Input mode: {'image' if has_problem_image else 'text'}")
    if answer_format_rule:
        print(f"Answer rule: {answer_format_rule}")
    print(f"Gold answer: {prob['answer']}")
    print(f"Prompt version: {prompt_version}")
    print(f"Models ({len(MODELS)}): {list(MODELS.keys())}")
    print(f"Temperatures: {list(temperatures)}")
    print(f"Runs per temperature/model: {n_runs}")
    print()

    # same_temperature[temp][model] -> run stats
    same_temperature: dict[str, dict] = {}

    # aggregate across all temperature/model/run
    all_step_counts = []
    all_titles = Counter()
    all_correct = 0
    all_total = 0
    problem_calls = 0

    for temp in temperatures:
        temp_key = f"{temp:.1f}"
        print(f"\n--- Temperature {temp_key} ---")
        per_model = {}
        temp_step_counts = []
        temp_title_counter = Counter()
        temp_correct = 0
        temp_total = 0

        for model_key, model_id in MODELS.items():
            print(f"  Model: {model_key} ({model_id})")
            runs = []
            model_step_counts = []
            model_title_counter = Counter()
            model_correct = 0
            model_correct_flags = []

            for run_idx in range(n_runs):
                t0 = time.time()
                response = call_llm(
                    prompt=prompt,
                    model=model_id,
                    temperature=temp,
                    messages=task_messages,
                    system_prompt=prompt,
                )
                call_elapsed = time.time() - t0
                steps = extract_steps(response, prompt_version)

                step_only = [s for s in steps if s.get("type") != "final"]
                step_count = len(step_only)
                titles = [s["title"] for s in step_only]
                final_answer = _extract_final_answer(response, steps)
                is_correct = _is_correct_answer(final_answer, prob["answer"])

                runs.append(
                    {
                        "run_idx": run_idx + 1,
                        "elapsed_s": call_elapsed,
                        "step_count": step_count,
                        "step_titles": titles,
                        "final_answer": final_answer,
                        "is_correct": is_correct,
                    }
                )

                model_step_counts.append(step_count)
                model_title_counter.update(titles)
                model_correct += int(is_correct)
                model_correct_flags.append(bool(is_correct))

                temp_step_counts.append(step_count)
                temp_title_counter.update(titles)
                temp_correct += int(is_correct)
                temp_total += 1

                all_step_counts.append(step_count)
                all_titles.update(titles)
                all_correct += int(is_correct)
                all_total += 1
                problem_calls += 1

                if global_state is not None:
                    global_state["done_calls"] += 1
                    done = global_state["done_calls"]
                    global_elapsed = time.time() - global_state["start_time"]
                    avg = global_elapsed / done if done else 0.0
                    eta = avg * (global_state["total_calls"] - done)
                else:
                    eta = 0.0

                if print_run_logs:
                    print(
                        f"    Run {run_idx+1}/{n_runs}: "
                        f"steps={step_count}, correct={is_correct}, {call_elapsed:.1f}s"
                    )
                if progress_bar is not None:
                    progress_bar.update(1)
                    progress_bar.set_postfix_str(
                        f"prob={problem_tag} temp={temp_key} model={model_key} "
                        f"run={run_idx+1}/{n_runs} eta={_format_duration(eta)}"
                    )

                _append_flat_result_row(
                    {
                        "model": model_key,
                        "temperature": temp,
                        "run_index": run_idx + 1,
                        "problem_id": prob["id"],
                        "step_count": step_count,
                        "correct": bool(is_correct),
                        "titles": json.dumps(titles, ensure_ascii=False),
                        "elapsed_time_seconds": round(call_elapsed, 6),
                    }
                )

            model_stats = _compute_group_stats(
                step_counts=model_step_counts,
                correct_flags=model_correct_flags,
            )
            model_accuracy = model_stats["accuracy_mean"]
            model_avg_steps = model_stats["step_mean"]
            per_model[model_key] = {
                "model_id": model_id,
                "runs": runs,
                "raw_runs": runs,
                "avg_step_count": model_avg_steps,
                "title_frequency": dict(model_title_counter),
                "accuracy": model_accuracy,
                "step_mean": model_stats["step_mean"],
                "step_std": model_stats["step_std"],
                "accuracy_mean": model_stats["accuracy_mean"],
                "accuracy_std": model_stats["accuracy_std"],
                "accuracy_ci_95": model_stats["accuracy_ci_95"],
                "raw_title_counts": dict(model_title_counter),
            }
            if not print_run_logs:
                print(
                    f"    avg_step={model_avg_steps:.2f}, "
                    f"accuracy={model_accuracy:.3f}, "
                    f"acc_ci95=[{model_stats['accuracy_ci_95']['lower']:.3f}, "
                    f"{model_stats['accuracy_ci_95']['upper']:.3f}]"
                )

        same_temperature[temp_key] = {
            "per_model": per_model,
            "temperature_summary": {
                "avg_step_count_all_models": (
                    float(np.mean(temp_step_counts)) if temp_step_counts else 0.0
                ),
                "title_frequency_all_models": dict(temp_title_counter),
                "accuracy_all_models": (temp_correct / temp_total) if temp_total else 0.0,
                "n_samples": temp_total,
            },
        }

    cross_temperature = {
        "per_temperature": {
            temp_key: same_temperature[temp_key]["temperature_summary"]
            for temp_key in same_temperature
        },
        "overall": {
            "avg_step_count": float(np.mean(all_step_counts)) if all_step_counts else 0.0,
            "title_frequency": dict(all_titles),
            "accuracy": (all_correct / all_total) if all_total else 0.0,
            "n_samples": all_total,
        },
    }

    result = {
        "task": "5",
        "meta": {
            "started_at_utc": started_at,
            "ended_at_utc": _utc_now_iso(),
            "elapsed_seconds": round(time.time() - problem_t0, 3),
            "problem_tag": problem_tag,
            "n_expected_calls": len(temperatures) * len(MODELS) * n_runs,
            "n_actual_calls": problem_calls,
        },
        "problem": {
            "id": prob["id"],
            "label": prob["label"],
            "gold_answer": prob["answer"],
        },
        "prompt_version": prompt_version,
        "temperatures": list(temperatures),
        "n_runs_per_temperature_model": n_runs,
        "models": MODELS,
        "same_temperature": same_temperature,
        "cross_temperature": cross_temperature,
    }

    out_path = os.path.join(OUT_DIR, f"task5_{problem_tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n--- Task 5 Summary ---")
    for temp_key, temp_data in same_temperature.items():
        summary = temp_data["temperature_summary"]
        print(
            f"  Temp {temp_key}: avg_step={summary['avg_step_count_all_models']:.2f}, "
            f"accuracy={summary['accuracy_all_models']:.3f}, n={summary['n_samples']}"
        )
    overall = cross_temperature["overall"]
    print(
        f"  Overall: avg_step={overall['avg_step_count']:.2f}, "
        f"accuracy={overall['accuracy']:.3f}, n={overall['n_samples']}"
    )
    print(f"\nResults saved to {out_path}")
    return result


def task5_temperature_model_stats(
    prob_key: str = "prob1",
    temperatures: tuple[float, ...] = (0.0, 0.5, 0.7, 1.0),
    n_runs: int = 3,
):
    """Task 5: v4 prompt 기반 temperature/model 통계."""
    print("\n" + "=" * 70)
    print("TASK 5: Temperature/model statistics (v4 prompt)")
    print("=" * 70)

    problem_items = _resolve_task5_problem_items(prob_key)
    n_problems = len(problem_items)
    total_calls = n_problems * len(temperatures) * len(MODELS) * n_runs
    task_started_at = _utc_now_iso()
    task_t0 = time.time()
    print(
        f"Target problems: {n_problems} | "
        f"Expected LLM calls: {total_calls} "
        f"({len(temperatures)} temps x {len(MODELS)} models x {n_runs} runs)"
    )

    print_run_logs = n_problems == 1
    last_result = {}
    all_problem_summary = {}
    global_state = {
        "start_time": time.time(),
        "done_calls": 0,
        "total_calls": total_calls,
    }
    problem_bar = None
    progress_bar = None
    if tqdm:
        if n_problems > 1:
            problem_bar = tqdm(total=n_problems, desc="Problems", unit="prob", position=0)
            progress_bar = tqdm(total=total_calls, desc="Calls", unit="call", position=1)
        else:
            progress_bar = tqdm(total=total_calls, desc="Task5", unit="call", position=0)

    for idx, (problem_tag, prob) in enumerate(problem_items, start=1):
        print("\n" + "-" * 70)
        print(f"[{idx}/{n_problems}] Running problem: {prob['label']}")
        print("-" * 70)
        result = _run_task5_for_one_problem(
            prob=prob,
            problem_tag=problem_tag,
            temperatures=temperatures,
            n_runs=n_runs,
            print_run_logs=print_run_logs,
            progress_bar=progress_bar,
            global_state=global_state,
        )
        last_result = result
        overall = result["cross_temperature"]["overall"]
        all_problem_summary[problem_tag] = {
            "problem_id": result["problem"]["id"],
            "avg_step_count": overall["avg_step_count"],
            "accuracy": overall["accuracy"],
            "n_samples": overall["n_samples"],
            "elapsed_seconds": result["meta"]["elapsed_seconds"],
        }
        done = global_state["done_calls"]
        elapsed = time.time() - global_state["start_time"]
        avg = elapsed / done if done else 0.0
        eta = avg * (global_state["total_calls"] - done)
        print(
            f"Progress: {done}/{global_state['total_calls']} calls | "
            f"elapsed={_format_duration(elapsed)} | eta={_format_duration(eta)}"
        )
        if problem_bar is not None:
            problem_bar.update(1)
            prob_elapsed = time.time() - task_t0
            avg_prob = prob_elapsed / idx if idx else 0.0
            prob_eta = avg_prob * (n_problems - idx)
            problem_bar.set_postfix_str(
                f"last={problem_tag} eta={_format_duration(prob_eta)}"
            )

    if progress_bar is not None:
        progress_bar.close()
    if problem_bar is not None:
        problem_bar.close()

    if n_problems > 1:
        task_elapsed = time.time() - task_t0
        summary = {
            "task": "5",
            "scope": "all_problems",
            "meta": {
                "started_at_utc": task_started_at,
                "ended_at_utc": _utc_now_iso(),
                "elapsed_seconds": round(task_elapsed, 3),
            },
            "n_problems": n_problems,
            "temperatures": list(temperatures),
            "n_models": len(MODELS),
            "n_runs_per_temperature_model": n_runs,
            "expected_total_calls": total_calls,
            "actual_total_calls": global_state["done_calls"],
            "per_problem_overall_summary": all_problem_summary,
        }
        summary_path = os.path.join(OUT_DIR, "task5_all_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("\n" + "=" * 70)
        print(f"All-problem summary saved to {summary_path}")
        print(
            f"Total elapsed: {_format_duration(task_elapsed)} | "
            f"Calls: {global_state['done_calls']}/{total_calls}"
        )
        print("=" * 70)
        return summary

    return last_result


# ───────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────

def main():
    # Parse --prob option
    args = sys.argv[1:]
    prob_key = "prob1"
    if "--prob" in args:
        idx = args.index("--prob")
        prob_key = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    tasks = args if args else ["5"]
    print(f"Problem key: {prob_key}")
    print(f"Tasks: {tasks}")

    if "1" in tasks:
        task1_prompt_versions(prob_key)

    if "2" in tasks:
        task2_step_similarity(prob_key)

    if "3" in tasks:
        task3_single_model_diversity(prob_key)

    if "4" in tasks:
        task4_multi_model_diversity(prob_key)

    if "5" in tasks:
        task5_temperature_model_stats(prob_key)

    print("\n" + "=" * 70)
    print("All requested tasks complete.")
    print(f"Results directory: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
