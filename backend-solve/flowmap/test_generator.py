#!/usr/bin/env python3
"""Test Flow Map Generator with PoC results"""

import json
import os
import sys

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'poc'))

from generator import generate_flow_map, load_from_poc_result
from schemas import FlowMapInput, ModelSolution, Step
from config import PROBLEMS


def load_input_from_poc(prob_key: str) -> FlowMapInput:
    """Load FlowMapInput from PoC task2 result

    Args:
        prob_key: "prob1" or "prob22"

    Returns:
        FlowMapInput
    """
    # Load problem text from config
    problem = PROBLEMS[prob_key]
    problem_text = f"{problem['label']}\n\n{problem['text']}"

    # Load task2 result
    poc_results_dir = os.path.join(os.path.dirname(__file__), '..', 'poc', 'results')
    task2_path = os.path.join(poc_results_dir, f'task2_{prob_key}.json')

    with open(task2_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert to FlowMapInput format
    solutions = []
    for model_name, steps_data in data["solutions"].items():
        steps = [
            Step(
                step_idx=i,
                title=s["title"],
                content=s["body"]
            )
            for i, s in enumerate(steps_data)
        ]

        solutions.append(ModelSolution(
            model_name=model_name,
            steps=steps
        ))

    return FlowMapInput(
        problem_text=problem_text,
        solutions=solutions
    )


def main():
    """Test generator on prob1 and prob22"""

    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    for prob_key in ["prob1", "prob22"]:
        print(f"\n{'=' * 70}")
        print(f"Generating Flow Map for {prob_key}")
        print('=' * 70)

        # Load input
        print(f"Loading data from task2_{prob_key}.json...")
        input_data = load_input_from_poc(prob_key)

        print(f"Loaded {len(input_data.solutions)} model solutions:")
        for sol in input_data.solutions:
            print(f"  - {sol.model_name}: {len(sol.steps)} steps")

        # Generate flow map
        flow_map = generate_flow_map(input_data)

        print(f"\nGenerated Flow Map:")
        print(f"  - {len(flow_map.groups)} groups")
        print(f"  - {len(flow_map.flows)} flow connections")

        for group in flow_map.groups:
            print(f"\n  Group {group.group_id}: {group.group_name}")
            for step in group.steps:
                print(f"    - [{step.model}] Step {step.step_idx}: {step.title}")

        # Save output
        output_path = os.path.join(output_dir, f'flowmap_{prob_key}.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(flow_map.to_dict(), f, ensure_ascii=False, indent=2)

        print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
