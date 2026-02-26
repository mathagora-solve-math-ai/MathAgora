#!/usr/bin/env python3
"""Pipeline: Test v4 prompt with a few problems

v4 changes:
- Added guidance: "STEP은 풀이의 흐름(전략/접근)이 바뀌는 지점마다 구분하되, 의미 없는 세분화는 피하고 필요한 만큼만 나누세요."
"""

import csv
import json
import os
import sys
import time
from typing import List, Dict

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'poc'))

from config import DATA_ROOT
from llm_client import call_llm
from generator import generate_flow_map
from schemas import FlowMapInput, ModelSolution, Step

# Increase CSV field size limit
csv.field_size_limit(10_000_000)

# Models to use
MODELS = {
    "gpt-5.2": "openai/gpt-5.2",
    "claude-opus-4.5": "anthropic/claude-opus-4.5",
    "gemini-3-pro": "google/gemini-3-pro-preview",
}

# v4 Prompt
PROMPT_V4 = """다음 수학 문제를 단계별(step-by-step)로 풀어주세요.
각 단계마다 간결한 제목(예: "조건 정리", "인수분해")을 붙여주세요.
STEP은 풀이의 흐름(전략/접근)이 바뀌는 지점마다 구분하되, 의미 없는 세분화는 피하고 필요한 만큼만 나누세요.

출력 형식:
[STEP title="<간결한 제목>"]
<풀이 내용>

[FINAL_ANSWER]
<최종 답>

문제:
{problem}
"""


def parse_v3_steps(text: str) -> List[Dict]:
    """Parse [STEP title="..."] format"""
    import re

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
        # body = text between current marker end and next marker start
        next_marker_match = marker_re.search(text, marker["end"])
        body_end = next_marker_match.start() if next_marker_match else len(text)
        body = text[marker["end"]:body_end].strip()
        steps.append({"title": marker["title"], "body": body, "type": marker["type"]})

    return steps


def load_all_problems(csv_filename: str) -> List[Dict]:
    """Load all problems from CSV"""
    csv_path = os.path.join(DATA_ROOT, csv_filename)
    problems = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            problems.append({
                "prob_id": row["prob_id"],
                "prob_area": row["prob_area"],
                "prob_point": row["prob_point"],
                "prob_desc": row["prob_desc"],
                "answer": row["answer"],
            })

    return problems


def generate_solutions_for_problem(problem: Dict) -> FlowMapInput:
    """Generate solutions from multiple models for one problem"""
    prob_text = f"{problem['prob_id']} ({problem['prob_area']}, {problem['prob_point']}점)\n\n{problem['prob_desc']}"
    prompt = PROMPT_V4.format(problem=problem['prob_desc'])

    solutions = []

    for model_name, model_id in MODELS.items():
        print(f"  Calling {model_name}...", end=" ", flush=True)

        try:
            t0 = time.time()
            response = call_llm(prompt, model_id, temperature=0.3, max_tokens=4096)
            elapsed = time.time() - t0

            # Parse steps
            steps_data = parse_v3_steps(response)

            steps = [
                Step(step_idx=i, title=s["title"], content=s["body"])
                for i, s in enumerate(steps_data)
            ]

            solutions.append(ModelSolution(
                model_name=model_name,
                steps=steps
            ))

            print(f"{len(steps)} steps ({elapsed:.1f}s)")

        except Exception as e:
            print(f"FAILED: {e}")
            # Add empty solution to maintain model list
            solutions.append(ModelSolution(
                model_name=model_name,
                steps=[]
            ))

    return FlowMapInput(
        problem_text=prob_text,
        solutions=solutions
    )


def main():
    """Run v4 pipeline on test problems"""

    # Configuration
    csv_file = "2024_math_odd.csv"
    output_dir = os.path.join(os.path.dirname(__file__), "outputs", "v4_test")
    os.makedirs(output_dir, exist_ok=True)

    # Get limit from command line (default: 5 problems)
    limit = 5
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
        except ValueError:
            print(f"Invalid limit: {sys.argv[1]}, using default 5")

    print(f"v4 Prompt Test - Processing {limit} problems")

    # Load problems
    print("Loading problems from CSV...")
    problems = load_all_problems(csv_file)
    problems = problems[:limit]
    print(f"Selected {len(problems)} problems")

    # Process each problem
    results_summary = []

    for idx, problem in enumerate(problems):
        prob_id = problem["prob_id"]

        print(f"\n{'=' * 70}")
        print(f"[{idx+1}/{len(problems)}] {prob_id} ({problem['prob_area']}, {problem['prob_point']}점)")
        print('=' * 70)

        # Generate solutions
        print("Generating solutions from LLMs...")
        input_data = generate_solutions_for_problem(problem)

        # Save step JSON
        step_json_path = os.path.join(output_dir, f"steps_{prob_id}.json")
        step_data = {
            "problem": {
                "prob_id": prob_id,
                "prob_area": problem["prob_area"],
                "prob_point": problem["prob_point"],
                "prob_desc": problem["prob_desc"],
                "answer": problem["answer"]
            },
            "solutions": {
                sol.model_name: [
                    {"title": s.title, "body": s.content}
                    for s in sol.steps
                ]
                for sol in input_data.solutions
            }
        }

        with open(step_json_path, 'w', encoding='utf-8') as f:
            json.dump(step_data, f, ensure_ascii=False, indent=2)

        print(f"Saved steps to {step_json_path}")

        # Generate flow map
        print("Generating Flow Map...")

        try:
            flow_map = generate_flow_map(input_data)

            # Save flow map JSON
            flowmap_path = os.path.join(output_dir, f"flowmap_{prob_id}.json")
            with open(flowmap_path, 'w', encoding='utf-8') as f:
                json.dump(flow_map.to_dict(), f, ensure_ascii=False, indent=2)

            print(f"Flow Map: {len(flow_map.groups)} groups, {len(flow_map.flows)} flows")
            print(f"Saved to {flowmap_path}")

            # Summary
            results_summary.append({
                "prob_id": prob_id,
                "prob_area": problem["prob_area"],
                "prob_point": problem["prob_point"],
                "n_groups": len(flow_map.groups),
                "n_flows": len(flow_map.flows),
                "models_participated": list(set(
                    s.model for g in flow_map.groups for s in g.steps
                )),
                "success": True
            })

        except Exception as e:
            print(f"Flow Map generation FAILED: {e}")
            results_summary.append({
                "prob_id": prob_id,
                "prob_area": problem["prob_area"],
                "prob_point": problem["prob_point"],
                "success": False,
                "error": str(e)
            })

    # Save summary
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 70}")
    print("v4 TEST COMPLETE")
    print(f"Processed {len(problems)} problems")
    print(f"Success: {sum(1 for r in results_summary if r['success'])}/{len(problems)}")
    print(f"Summary saved to {summary_path}")
    print('=' * 70)


if __name__ == "__main__":
    main()
