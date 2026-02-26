#!/usr/bin/env python3
"""Visualize v3 Flow Maps with Sub-row Fix + Answer Checker

Fixed: Arrow offset adjusted for sub-row boxes
Added: Answer correctness checking
"""

import json
import sys
import os
import csv
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from answer_checker import check_answer, get_answer_display, get_answer_color

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
csv.field_size_limit(10_000_000)


def load_flowmap(prob_id, output_dir='outputs/all_problems'):
    """Load flow map JSON from v3 results"""
    flowmap_path = os.path.join(output_dir, f'flowmap_{prob_id}.json')

    if not os.path.exists(flowmap_path):
        raise FileNotFoundError(f"Flow map not found: {flowmap_path}")

    with open(flowmap_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_problem_info(prob_id, output_dir='outputs/all_problems'):
    """Load problem metadata from steps JSON and CSV"""
    # Load basic info from steps JSON
    steps_path = os.path.join(output_dir, f'steps_{prob_id}.json')
    prob_info = {}

    if os.path.exists(steps_path):
        with open(steps_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            prob_info = data.get('problem', {})

    # Add prob_type from CSV (not in steps JSON)
    csv_path = '../data/2024_math_odd.csv'
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['prob_id'] == prob_id:
                    prob_info['prob_type'] = row['prob_type']
                    break

    return prob_info


def visualize_flowmap(flowmap, title="Flow Map", figsize=(14, 10), save_path=None, prob_info=None):
    """Visualize flow map with sub-row fix and answer checking

    Layout:
    - Columns: Models
    - Rows: Groups
    - Sub-rows: Multiple steps from same model in same group
    - Arrows: Adjusted for box height
    - Final Answer boxes: Green (correct) / Red (incorrect)
    """
    # Check answers if prob_info provided
    answer_results = {}
    ground_truth = None
    if prob_info and 'answer' in prob_info:
        answer_results = check_answer(flowmap, prob_info)
        ground_truth = str(prob_info['answer']).strip()
    # Extract unique models
    models = set()
    for group in flowmap['groups']:
        for step in group['steps']:
            models.add(step['model'])
    models = sorted(models)

    n_models = len(models)
    n_groups = len(flowmap['groups'])

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-0.5, n_models - 0.5)
    ax.set_ylim(-0.5, n_groups - 0.5)
    ax.invert_yaxis()

    # Set ticks
    ax.set_xticks(range(n_models))
    ax.set_xticklabels(models, fontsize=12, fontweight='bold')
    ax.set_yticks(range(n_groups))
    ax.set_yticklabels([g['group_name'] for g in flowmap['groups']], fontsize=11)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Build position map: (model, step_idx) -> (col, row, box_height)
    step_positions = {}

    # Draw boxes
    colors = plt.cm.Set3(np.linspace(0, 1, n_groups))

    for group_idx, group in enumerate(flowmap['groups']):
        row = group_idx
        color = colors[group_idx]

        # Count steps per model in this group
        model_steps_in_group = {}
        for step in group['steps']:
            model = step['model']
            if model not in model_steps_in_group:
                model_steps_in_group[model] = []
            model_steps_in_group[model].append(step)

        # Place each step with sub-row offset
        for step in group['steps']:
            model = step['model']
            col = models.index(model)
            step_idx = step['step_idx']

            # Find position in this model's steps
            model_steps = model_steps_in_group[model]
            sub_idx = next(i for i, s in enumerate(model_steps) if s['step_idx'] == step_idx)

            # Calculate sub-row offset
            if len(model_steps) > 1:
                total_spread = 0.5
                offset = (sub_idx - (len(model_steps) - 1) / 2) * (total_spread / (len(model_steps) - 1))
            else:
                offset = 0

            actual_row = row + offset

            # Box size (smaller if multiple)
            if len(model_steps) > 1:
                box_height = 0.7 / len(model_steps)
                box_width = 0.8
            else:
                box_height = 0.7
                box_width = 0.8

            # Store position WITH box height for arrow calculation
            step_positions[(model, step_idx)] = (col, actual_row, box_height)

            # Check if this is a Final Answer step
            is_final_answer = 'final' in step['title'].lower() or 'answer' in step['title'].lower()

            # Get edge color (green/red for Final Answer, black otherwise)
            if is_final_answer and answer_results:
                edge_color = get_answer_color(model, answer_results)
                edge_width = 3.0  # Thicker for emphasis
            else:
                edge_color = 'black'
                edge_width = 1.5

            # Draw box
            box = FancyBboxPatch(
                (col - box_width/2, actual_row - box_height/2),
                box_width, box_height,
                boxstyle="round,pad=0.02",
                facecolor=color,
                edgecolor=edge_color,
                linewidth=edge_width,
                alpha=0.7
            )
            ax.add_patch(box)

            # Text (with step number if multiple, or answer display)
            if is_final_answer and ground_truth and prob_info:
                prob_type = prob_info.get('prob_type', '단답형')
                prob_id = prob_info.get('prob_id')
                title_text = get_answer_display(step['content'], ground_truth, prob_type, prob_id)
                fontsize = 6.5  # Smaller font for multi-line answer display
            else:
                title_text = step['title'][:20] + '...' if len(step['title']) > 20 else step['title']
                # Escape $ for matplotlib
                title_text = title_text.replace('$', r'\$')
                if len(model_steps) > 1:
                    title_text = f"[{step_idx}] {title_text}"
                fontsize = 8 if len(model_steps) > 1 else 9

            ax.text(col, actual_row, title_text, ha='center', va='center',
                   fontsize=fontsize, wrap=True)

    # Draw flow connections with adjusted offsets
    for flow in flowmap['flows']:
        model = flow['model']
        from_step = flow['from_step']
        to_step = flow['to_step']

        if (model, from_step) not in step_positions or (model, to_step) not in step_positions:
            continue

        from_col, from_row, from_height = step_positions[(model, from_step)]
        to_col, to_row, to_height = step_positions[(model, to_step)]

        # Arrow offset = half of box height (adjusted for each step)
        from_offset = from_height / 2
        to_offset = to_height / 2

        # Draw arrow
        arrow = FancyArrowPatch(
            (from_col, from_row + from_offset),
            (to_col, to_row - to_offset),
            arrowstyle='-|>',
            mutation_scale=15,
            linewidth=2,
            color='darkblue',
            alpha=0.6
        )
        ax.add_patch(arrow)

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python3 visualize_v3_fixed.py <prob_id>")
        print("\nExample:")
        print("  python3 visualize_v3_fixed.py 2024_odd_common_14")
        sys.exit(1)

    prob_id = sys.argv[1]

    print(f"Loading Flow Map: {prob_id}")
    flowmap = load_flowmap(prob_id)
    prob_info = load_problem_info(prob_id)

    # Print details
    print(f"\nProblem: {prob_info.get('prob_id')} ({prob_info.get('prob_area')}, {prob_info.get('prob_point')}점)")
    print(f"Type: {prob_info.get('prob_type')}")
    print(f"Answer: {prob_info.get('answer')}")
    print(f"Groups: {len(flowmap['groups'])}")
    print(f"Flows: {len(flowmap['flows'])}")

    # Visualize
    n_groups = len(flowmap['groups'])
    figsize = (14, max(10, n_groups * 1.2))

    title = f"Flow Map (v3 Fixed): {prob_id}"
    if prob_info:
        title += f" ({prob_info.get('prob_area')}, {prob_info.get('prob_point')}점)"

    output_dir = 'outputs/v3_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'{prob_id}_fixed.png')

    fig = visualize_flowmap(flowmap, title=title, figsize=figsize, save_path=save_path, prob_info=prob_info)

    plt.show()


if __name__ == "__main__":
    main()
