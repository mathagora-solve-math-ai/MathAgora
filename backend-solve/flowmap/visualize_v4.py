#!/usr/bin/env python3
"""Visualize v4 Flow Maps

Usage:
    python3 visualize_v4.py 2024_odd_common_1
    python3 visualize_v4.py 2024_odd_common_1 2024_odd_common_7 2024_odd_common_22
"""

import json
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Configure matplotlib for Korean
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


def load_flowmap(prob_id, output_dir='outputs/v4_all'):
    """Load flow map JSON from v4 results"""
    flowmap_path = os.path.join(output_dir, f'flowmap_{prob_id}.json')

    if not os.path.exists(flowmap_path):
        # Try v4_test folder
        flowmap_path = os.path.join('outputs/v4_test', f'flowmap_{prob_id}.json')

    if not os.path.exists(flowmap_path):
        raise FileNotFoundError(f"Flow map not found: {flowmap_path}")

    with open(flowmap_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_problem_info(prob_id, output_dir='outputs/v4_all'):
    """Load problem metadata"""
    steps_path = os.path.join(output_dir, f'steps_{prob_id}.json')

    if not os.path.exists(steps_path):
        steps_path = os.path.join('outputs/v4_test', f'steps_{prob_id}.json')

    if os.path.exists(steps_path):
        with open(steps_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('problem', {})

    return {}


def visualize_flowmap(flowmap, title="Flow Map", figsize=(14, 10), save_path=None):
    """Visualize flow map as a grid with connections

    Layout:
    - Columns: Models
    - Rows: Groups (in order)
    - Arrows: Flow connections
    """
    # Extract unique models
    models = set()
    for group in flowmap['groups']:
        for step in group['steps']:
            models.add(step['model'])
    models = sorted(models)

    n_models = len(models)
    n_groups = len(flowmap['groups'])

    if n_groups == 0:
        print("No groups to visualize!")
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-0.5, n_models - 0.5)
    ax.set_ylim(-0.5, n_groups - 0.5)
    ax.invert_yaxis()  # Top to bottom

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

    # Build position map: (model, step_idx) -> (col, row)
    step_positions = {}

    # Draw boxes for each step
    colors = plt.cm.Set3(np.linspace(0, 1, n_groups))

    for group_idx, group in enumerate(flowmap['groups']):
        row = group_idx
        color = colors[group_idx]

        for step in group['steps']:
            model = step['model']
            col = models.index(model)
            step_idx = step['step_idx']

            # Store position
            step_positions[(model, step_idx)] = (col, row)

            # Draw box
            box = FancyBboxPatch(
                (col - 0.4, row - 0.35),
                0.8, 0.7,
                boxstyle="round,pad=0.05",
                facecolor=color,
                edgecolor='black',
                linewidth=1.5,
                alpha=0.7
            )
            ax.add_patch(box)

            # Add step title
            title_text = step['title'][:25] + '...' if len(step['title']) > 25 else step['title']
            ax.text(col, row, title_text, ha='center', va='center', fontsize=9, wrap=True)

    # Draw flow connections
    for flow in flowmap['flows']:
        model = flow['model']
        from_step = flow['from_step']
        to_step = flow['to_step']

        if (model, from_step) not in step_positions or (model, to_step) not in step_positions:
            continue

        from_col, from_row = step_positions[(model, from_step)]
        to_col, to_row = step_positions[(model, to_step)]

        # Draw arrow
        arrow = FancyArrowPatch(
            (from_col, from_row + 0.35),
            (to_col, to_row - 0.35),
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


def print_flowmap_details(flowmap, prob_info=None):
    """Print flow map statistics"""
    print("\n" + "=" * 70)
    if prob_info:
        print(f"Problem: {prob_info.get('prob_id', 'Unknown')}")
        print(f"Area: {prob_info.get('prob_area', 'Unknown')}")
        print(f"Points: {prob_info.get('prob_point', 'Unknown')}")
    print("=" * 70)

    print(f"\nGroups: {len(flowmap['groups'])}")
    print(f"Flows: {len(flowmap['flows'])}")

    # Models
    models = set(s['model'] for g in flowmap['groups'] for s in g['steps'])
    print(f"Models: {len(models)} ({', '.join(sorted(models))})")

    print("\nGroup Details:")
    for g in flowmap['groups']:
        print(f"\n  [{g['group_id']}] {g['group_name']}")
        for s in g['steps']:
            print(f"      • [{s['model']}] Step {s['step_idx']}: {s['title'][:40]}")


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python3 visualize_v4.py <prob_id1> [prob_id2] ...")
        print("\nExamples:")
        print("  python3 visualize_v4.py 2024_odd_common_1")
        print("  python3 visualize_v4.py 2024_odd_common_1 2024_odd_common_7")
        sys.exit(1)

    prob_ids = sys.argv[1:]

    output_dir = 'outputs/v4_visualizations'
    os.makedirs(output_dir, exist_ok=True)

    for prob_id in prob_ids:
        print(f"\n{'=' * 70}")
        print(f"Visualizing: {prob_id}")
        print('=' * 70)

        try:
            # Load flow map
            flowmap = load_flowmap(prob_id)
            prob_info = load_problem_info(prob_id)

            # Print details
            print_flowmap_details(flowmap, prob_info)

            # Visualize
            title = f"Flow Map: {prob_id}"
            if prob_info:
                title += f" ({prob_info.get('prob_area', '')}, {prob_info.get('prob_point', '')}점)"

            save_path = os.path.join(output_dir, f'{prob_id}.png')

            # Dynamic figure size based on groups
            n_groups = len(flowmap['groups'])
            figsize = (14, max(8, n_groups * 1.2))

            fig = visualize_flowmap(flowmap, title=title, figsize=figsize, save_path=save_path)

            if fig:
                plt.show()

        except Exception as e:
            print(f"Error processing {prob_id}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
