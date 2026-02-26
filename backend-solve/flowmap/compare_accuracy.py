#!/usr/bin/env python3
"""Compare v3 vs v4 accuracy"""

import json
import os
import csv
import pandas as pd
from answer_checker import check_answer

csv.field_size_limit(10_000_000)


def load_all_results(output_dir):
    """Load all flowmap results from directory"""
    import glob

    flowmap_files = glob.glob(os.path.join(output_dir, 'flowmap_*.json'))
    results = []

    for file in flowmap_files:
        prob_id = os.path.basename(file).replace('flowmap_', '').replace('.json', '')

        with open(file, 'r', encoding='utf-8') as f:
            flowmap = json.load(f)

        # Load prob_info
        steps_file = os.path.join(output_dir, f'steps_{prob_id}.json')
        if os.path.exists(steps_file):
            with open(steps_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                prob_info = data.get('problem', {})

                # Add prob_type from CSV
                csv_path = '../data/2024_math_odd.csv'
                if os.path.exists(csv_path):
                    with open(csv_path, 'r', encoding='utf-8') as csvf:
                        reader = csv.DictReader(csvf)
                        for row in reader:
                            if row['prob_id'] == prob_id:
                                prob_info['prob_type'] = row['prob_type']
                                break

                results.append({
                    'prob_id': prob_id,
                    'flowmap': flowmap,
                    'prob_info': prob_info
                })

    return results


def calculate_accuracy(results):
    """Calculate accuracy per model"""
    model_correct = {}
    model_total = {}

    for result in results:
        flowmap = result['flowmap']
        prob_info = result['prob_info']

        if 'answer' not in prob_info or 'prob_type' not in prob_info:
            continue

        answer_results = check_answer(flowmap, prob_info)

        for model, (extracted, is_correct) in answer_results.items():
            if model not in model_correct:
                model_correct[model] = 0
                model_total[model] = 0

            if is_correct:
                model_correct[model] += 1
            model_total[model] += 1

    # Calculate percentages
    accuracy = {}
    for model in model_correct:
        accuracy[model] = {
            'correct': model_correct[model],
            'total': model_total[model],
            'accuracy': (model_correct[model] / model_total[model] * 100) if model_total[model] > 0 else 0
        }

    return accuracy


def main():
    print("=== v3 vs v4 정답률 비교 ===\n")

    # Load v3 results
    print("Loading v3 results...")
    v3_results = load_all_results('outputs/all_problems')
    v3_accuracy = calculate_accuracy(v3_results)

    # Load v4 results
    print("Loading v4 results...")
    v4_dirs = ['outputs/v4_all', 'outputs/v4_test']
    v4_results = []
    for v4_dir in v4_dirs:
        if os.path.exists(v4_dir):
            v4_results.extend(load_all_results(v4_dir))
    v4_accuracy = calculate_accuracy(v4_results)

    # Get all models
    all_models = sorted(set(list(v3_accuracy.keys()) + list(v4_accuracy.keys())))

    # Create comparison table
    comparison_data = []
    for model in all_models:
        v3_acc = v3_accuracy.get(model, {'correct': 0, 'total': 0, 'accuracy': 0})
        v4_acc = v4_accuracy.get(model, {'correct': 0, 'total': 0, 'accuracy': 0})

        comparison_data.append({
            '모델': model,
            'v3 정답': f"{v3_acc['correct']}/{v3_acc['total']}",
            'v3 정답률': f"{v3_acc['accuracy']:.1f}%",
            'v4 정답': f"{v4_acc['correct']}/{v4_acc['total']}",
            'v4 정답률': f"{v4_acc['accuracy']:.1f}%",
            '차이': f"{v4_acc['accuracy'] - v3_acc['accuracy']:+.1f}%p"
        })

    df = pd.DataFrame(comparison_data)

    print("\n📊 모델별 정답률 비교:")
    print(df.to_string(index=False))

    # Overall accuracy
    v3_total_correct = sum(acc['correct'] for acc in v3_accuracy.values())
    v3_total = sum(acc['total'] for acc in v3_accuracy.values())
    v4_total_correct = sum(acc['correct'] for acc in v4_accuracy.values())
    v4_total = sum(acc['total'] for acc in v4_accuracy.values())

    print(f"\n📈 전체 정답률:")
    print(f"  v3: {v3_total_correct}/{v3_total} = {v3_total_correct/v3_total*100:.1f}%")
    print(f"  v4: {v4_total_correct}/{v4_total} = {v4_total_correct/v4_total*100:.1f}%")
    print(f"  차이: {(v4_total_correct/v4_total - v3_total_correct/v3_total)*100:+.1f}%p")

    print(f"\n📁 문제 수:")
    print(f"  v3: {len(v3_results)} 문제")
    print(f"  v4: {len(v4_results)} 문제")


if __name__ == "__main__":
    main()
