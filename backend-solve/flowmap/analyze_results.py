#!/usr/bin/env python3
"""Analyze Flow Map results from all problems"""

import json
import os
from collections import defaultdict

def main():
    """Analyze summary.json and generate statistics"""

    summary_path = "outputs/all_problems/summary.json"

    if not os.path.exists(summary_path):
        print(f"Summary file not found: {summary_path}")
        print("Run pipeline first!")
        return

    with open(summary_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    print("=" * 70)
    print("FLOW MAP PIPELINE ANALYSIS")
    print("=" * 70)
    print()

    # Overall statistics
    total = len(results)
    success = sum(1 for r in results if r.get("success"))
    failed = total - success

    print(f"Total problems: {total}")
    print(f"Success: {success} ({success/total*100:.1f}%)")
    print(f"Failed: {failed} ({failed/total*100:.1f}%)")
    print()

    # Group statistics (only for successful ones)
    successful_results = [r for r in results if r.get("success")]

    if successful_results:
        groups_per_problem = [r["n_groups"] for r in successful_results]
        flows_per_problem = [r["n_flows"] for r in successful_results]

        print("Flow Map Statistics (successful problems):")
        print(f"  Groups per problem:")
        print(f"    Min: {min(groups_per_problem)}")
        print(f"    Max: {max(groups_per_problem)}")
        print(f"    Avg: {sum(groups_per_problem)/len(groups_per_problem):.1f}")
        print()
        print(f"  Flows per problem:")
        print(f"    Min: {min(flows_per_problem)}")
        print(f"    Max: {max(flows_per_problem)}")
        print(f"    Avg: {sum(flows_per_problem)/len(flows_per_problem):.1f}")
        print()

    # By difficulty (prob_point)
    by_point = defaultdict(list)
    for r in successful_results:
        by_point[r["prob_point"]].append(r)

    print("By Difficulty:")
    for point in sorted(by_point.keys()):
        probs = by_point[point]
        avg_groups = sum(p["n_groups"] for p in probs) / len(probs)
        print(f"  {point}점: {len(probs)} problems, avg {avg_groups:.1f} groups")
    print()

    # Model participation
    print("Model Participation:")
    model_counts = defaultdict(int)
    for r in successful_results:
        for model in r.get("models_participated", []):
            model_counts[model] += 1

    for model, count in sorted(model_counts.items()):
        print(f"  {model}: {count}/{len(successful_results)} problems ({count/len(successful_results)*100:.1f}%)")
    print()

    # Failed problems
    if failed > 0:
        print(f"Failed Problems ({failed}):")
        for r in results:
            if not r.get("success"):
                print(f"  - {r['prob_id']} ({r['prob_area']}, {r['prob_point']}점)")
                if "error" in r:
                    print(f"    Error: {r['error'][:100]}")
        print()

    # Top/bottom by complexity
    print("Most Complex Problems (by # groups):")
    sorted_by_groups = sorted(successful_results, key=lambda r: r["n_groups"], reverse=True)
    for r in sorted_by_groups[:5]:
        print(f"  {r['prob_id']} ({r['prob_point']}점): {r['n_groups']} groups, {r['n_flows']} flows")
    print()

    print("Simplest Problems (by # groups):")
    for r in sorted_by_groups[-5:]:
        print(f"  {r['prob_id']} ({r['prob_point']}점): {r['n_groups']} groups, {r['n_flows']} flows")
    print()

    print("=" * 70)


if __name__ == "__main__":
    main()
