#!/usr/bin/env python3
"""
JSON Enforced 결과 vs v3/v4 비교 분석
"""

import json
import os
import glob
from collections import defaultdict

def analyze_results(output_dir):
    """결과 파일들 분석"""

    # Input 파일들에서 step 수 추출
    input_files = glob.glob(os.path.join(output_dir, 'input_*.json'))
    flowmap_files = glob.glob(os.path.join(output_dir, 'flowmap_*.json'))

    step_counts = {}  # {prob_id: {model: count}}
    group_counts = []
    flow_counts = []

    for input_file in input_files:
        prob_id = os.path.basename(input_file).replace('input_', '').replace('.json', '')

        with open(input_file, 'r') as f:
            data = json.load(f)

        # 각 모델의 step 수
        step_counts[prob_id] = {}
        for solution in data['solutions']:
            model = solution['model_name']
            num_steps = len(solution['steps'])
            step_counts[prob_id][model] = num_steps

    # Flowmap 분석
    for flowmap_file in flowmap_files:
        with open(flowmap_file, 'r') as f:
            flowmap = json.load(f)

        group_counts.append(len(flowmap['groups']))
        flow_counts.append(len(flowmap['flows']))

    return step_counts, group_counts, flow_counts


def load_v3_v4_results():
    """v3, v4 결과 로드"""

    v3_dir = 'outputs/v3_all'
    v4_dir = 'outputs/v4_all'

    results = {'v3': {}, 'v4': {}}

    for version, output_dir in [('v3', v3_dir), ('v4', v4_dir)]:
        if not os.path.exists(output_dir):
            continue

        step_files = glob.glob(os.path.join(output_dir, 'steps_*.json'))

        for step_file in step_files:
            prob_id = os.path.basename(step_file).replace('steps_', '').replace('.json', '')

            with open(step_file, 'r') as f:
                steps_data = json.load(f)

            # 각 모델의 step 수 추출
            model_steps = {}

            # v4 형식: {"problem": {...}, "solutions": {"model": [steps]}}
            if 'solutions' in steps_data:
                for model, steps in steps_data['solutions'].items():
                    if isinstance(steps, list):
                        model_steps[model] = len(steps)
            # v3 형식 또는 다른 형식
            else:
                for model, steps in steps_data.items():
                    if isinstance(steps, list):
                        model_steps[model] = len(steps)

            results[version][prob_id] = model_steps

    return results


def compare_versions():
    """버전별 비교"""

    print("="*70)
    print(" JSON ENFORCED vs v3/v4 비교 분석")
    print("="*70)
    print()

    # JSON Enforced 결과
    print("📊 JSON Enforced 분석 중...")
    json_steps, json_groups, json_flows = analyze_results('outputs/json_enforced')

    # v3, v4 결과
    print("📊 v3/v4 결과 로드 중...")
    old_results = load_v3_v4_results()

    # 통계 계산
    print("\n" + "="*70)
    print(" STEP 수 비교 (평균)")
    print("="*70)
    print()

    # JSON Enforced 평균
    json_avg = {'gpt-5.2': [], 'claude-opus-4.5': [], 'gemini-3-pro': []}
    for prob_id, models in json_steps.items():
        for model, count in models.items():
            json_avg[model].append(count)

    print("JSON Enforced:")
    for model in ['gpt-5.2', 'claude-opus-4.5', 'gemini-3-pro']:
        if json_avg[model]:
            avg = sum(json_avg[model]) / len(json_avg[model])
            print(f"  {model:20s}: {avg:.2f} steps/problem")

    print()

    # v3 평균
    if old_results['v3']:
        v3_avg = defaultdict(list)
        for prob_id, models in old_results['v3'].items():
            for model, count in models.items():
                v3_avg[model].append(count)

        print("v3:")
        for model in ['gpt-5.2', 'claude-opus-4.5', 'gemini-3-pro']:
            if v3_avg[model]:
                avg = sum(v3_avg[model]) / len(v3_avg[model])
                print(f"  {model:20s}: {avg:.2f} steps/problem")

    print()

    # v4 평균
    if old_results['v4']:
        v4_avg = defaultdict(list)
        for prob_id, models in old_results['v4'].items():
            for model, count in models.items():
                v4_avg[model].append(count)

        print("v4:")
        for model in ['gpt-5.2', 'claude-opus-4.5', 'gemini-3-pro']:
            if v4_avg[model]:
                avg = sum(v4_avg[model]) / len(v4_avg[model])
                print(f"  {model:20s}: {avg:.2f} steps/problem")

    # Flow Map 비교
    print()
    print("="*70)
    print(" FLOW MAP 비교")
    print("="*70)
    print()

    print(f"JSON Enforced:")
    print(f"  Average Groups: {sum(json_groups) / len(json_groups):.2f}")
    print(f"  Average Flows:  {sum(json_flows) / len(json_flows):.2f}")

    # v3 flowmap
    v3_flowmaps = glob.glob('outputs/v3_all/flowmap_*.json')
    if v3_flowmaps:
        v3_groups = []
        v3_flows = []
        for fm_file in v3_flowmaps:
            with open(fm_file, 'r') as f:
                fm = json.load(f)
            v3_groups.append(len(fm['groups']))
            v3_flows.append(len(fm['flows']))

        print()
        print(f"v3:")
        print(f"  Average Groups: {sum(v3_groups) / len(v3_groups):.2f}")
        print(f"  Average Flows:  {sum(v3_flows) / len(v3_flows):.2f}")

    # v4 flowmap
    v4_flowmaps = glob.glob('outputs/v4_all/flowmap_*.json')
    if v4_flowmaps:
        v4_groups = []
        v4_flows = []
        for fm_file in v4_flowmaps:
            with open(fm_file, 'r') as f:
                fm = json.load(f)
            v4_groups.append(len(fm['groups']))
            v4_flows.append(len(fm['flows']))

        print()
        print(f"v4:")
        print(f"  Average Groups: {sum(v4_groups) / len(v4_groups):.2f}")
        print(f"  Average Flows:  {sum(v4_flows) / len(v4_flows):.2f}")

    # 개별 비교 (step 수 변화가 큰 문제들)
    print()
    print("="*70)
    print(" STEP 수 변화가 큰 문제 (v4 → JSON Enforced)")
    print("="*70)
    print()

    if old_results['v4']:
        changes = []
        for prob_id in json_steps.keys():
            if prob_id in old_results['v4']:
                v4_total = sum(old_results['v4'][prob_id].values())
                json_total = sum(json_steps[prob_id].values())
                diff = json_total - v4_total
                changes.append((prob_id, v4_total, json_total, diff))

        # 차이가 큰 순으로 정렬
        changes.sort(key=lambda x: abs(x[3]), reverse=True)

        print("Top 10 변화:")
        print(f"{'Problem ID':<30} {'v4':>8} {'JSON':>8} {'Diff':>8}")
        print("-" * 70)
        for prob_id, v4_count, json_count, diff in changes[:10]:
            diff_str = f"{diff:+d}" if diff != 0 else "0"
            print(f"{prob_id:<30} {v4_count:8d} {json_count:8d} {diff_str:>8}")


if __name__ == "__main__":
    compare_versions()
