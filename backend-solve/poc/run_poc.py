#!/usr/bin/env python3
"""
PoC script for ACL 2026 demo – LLM step-by-step solution alignment.


Tasks:
  1. Compare 3 prompt versions (simple / step-by-step / step-by-step+titled)
  2. Cosine-similarity based step alignment across solutions
  3. Single-model diversity (temperature=1.0, 5 runs)
  4. Multi-model diversity (5 different models, 1 run each)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from itertools import combinations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    DEFAULT_MODEL,
    MODELS,
    PROBLEMS,
    PROMPT_VERSIONS,
)
from llm_client import call_llm

OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)


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
    if prompt_version == "v3_titled":
        steps = parse_v3_steps(text)
        if steps:
            return steps
    return heuristic_split_steps(text)


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
    """Task 1: Compare 3 prompt versions for step extraction quality."""
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
        prompt = template.format(problem=prob["text"])
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
    prompt = template.format(problem=prob["text"])

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
    prompt = template.format(problem=prob["text"])
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
    prompt = template.format(problem=prob["text"])

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

    tasks = args if args else ["1", "2", "3", "4"]
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

    print("\n" + "=" * 70)
    print("All requested tasks complete.")
    print(f"Results directory: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
