#!/usr/bin/env python3
"""
v3, v4, JSON Enforced 정답률 비교
"""

import json
import os
import glob
import csv
import sys
from collections import defaultdict

# answer_checker 임포트
sys.path.insert(0, os.path.dirname(__file__))
import re


def extract_answer_improved(content, prob_type):
    """개선된 답 추출 - 마지막 등호 뒤의 숫자 찾기"""
    content = content.strip()

    if prob_type == "5지선다형":
        match = re.search(r'[①②③④⑤]', content)
        if match:
            char = match.group()
            mapping = {'①': '1', '②': '2', '③': '3', '④': '4', '⑤': '5'}
            return mapping.get(char)

        match = re.search(r'\b([1-5])\b', content)
        if match:
            return match.group(1)

    else:  # 단답형
        # 마지막 등호 뒤의 숫자 찾기
        last_eq_idx = content.rfind('=')
        if last_eq_idx != -1:
            after_eq = content[last_eq_idx+1:].strip()
            match = re.search(r'(\d+)', after_eq)
            if match:
                num = int(match.group(1))
                if 0 <= num <= 999:
                    return str(num)

        # Fallback: 마지막 숫자 찾기
        all_numbers = re.findall(r'\b(\d{1,3})\b', content)
        if all_numbers:
            num = int(all_numbers[-1])
            if 0 <= num <= 999:
                return str(num)

    return None


def is_correct(extracted, truth, prob_type):
    """답이 맞는지 체크"""
    if extracted is None or extracted == "":
        return False

    extracted_str = str(extracted).strip()
    truth_str = str(truth).strip()

    return extracted_str == truth_str

csv.field_size_limit(10_000_000)


def load_ground_truth():
    """CSV에서 정답 로드"""
    csv_path = '../data/2024_math_odd.csv'

    ground_truth = {}
    prob_types = {}

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prob_id = row['prob_id']
            ground_truth[prob_id] = row['answer']
            prob_types[prob_id] = row.get('prob_type', '5지선다형')

    return ground_truth, prob_types


def check_json_enforced_accuracy(output_dir='outputs/json_enforced'):
    """JSON Enforced 정답률 체크"""

    ground_truth, prob_types = load_ground_truth()

    input_files = glob.glob(os.path.join(output_dir, 'input_*.json'))

    results = defaultdict(lambda: {'correct': 0, 'total': 0})
    all_results = []

    for input_file in input_files:
        prob_id = os.path.basename(input_file).replace('input_', '').replace('.json', '')

        if prob_id not in ground_truth:
            continue

        truth = ground_truth[prob_id]
        prob_type = prob_types[prob_id]

        with open(input_file, 'r') as f:
            data = json.load(f)

        for solution in data['solutions']:
            model = solution['model_name']
            steps = solution['steps']

            # Final Answer 찾기 (마지막 step)
            if steps:
                final_step = steps[-1]
                final_content = final_step.get('content', '')

                # 답 추출
                extracted = extract_answer_improved(final_content, prob_type)

                # 정답 체크
                correct = is_correct(extracted, truth, prob_type)

                results[model]['total'] += 1
                if correct:
                    results[model]['correct'] += 1

                all_results.append({
                    'prob_id': prob_id,
                    'model': model,
                    'extracted': extracted,
                    'truth': truth,
                    'correct': correct
                })

    return results, all_results


def check_v4_accuracy(output_dir='outputs/v4_all'):
    """v4 정답률 체크"""

    ground_truth, prob_types = load_ground_truth()

    step_files = glob.glob(os.path.join(output_dir, 'steps_*.json'))

    results = defaultdict(lambda: {'correct': 0, 'total': 0})

    for step_file in step_files:
        prob_id = os.path.basename(step_file).replace('steps_', '').replace('.json', '')

        if prob_id not in ground_truth:
            continue

        truth = ground_truth[prob_id]
        prob_type = prob_types[prob_id]

        with open(step_file, 'r') as f:
            data = json.load(f)

        # v4 형식: {"problem": {...}, "solutions": {"model": [steps]}}
        if 'solutions' in data:
            solutions = data['solutions']
        else:
            solutions = data

        for model, steps in solutions.items():
            if not isinstance(steps, list) or len(steps) == 0:
                continue

            # Final Answer 찾기
            final_step = None
            for step in reversed(steps):
                if isinstance(step, dict) and 'title' in step:
                    if 'final' in step['title'].lower() or 'answer' in step['title'].lower():
                        final_step = step
                        break

            if final_step is None:
                final_step = steps[-1]

            final_content = final_step.get('body', final_step.get('content', ''))

            # 답 추출
            extracted = extract_answer_improved(final_content, prob_type)

            # 정답 체크
            correct = is_correct(extracted, truth, prob_type)

            results[model]['total'] += 1
            if correct:
                results[model]['correct'] += 1

    return results


def main():
    print("="*70)
    print(" v3/v4/JSON Enforced 정답률 비교")
    print("="*70)
    print()

    # JSON Enforced
    print("📊 JSON Enforced 정답률 분석 중...")
    json_results, json_details = check_json_enforced_accuracy()

    # v4
    print("📊 v4 정답률 분석 중...")
    v4_results = check_v4_accuracy()

    # 결과 출력
    print()
    print("="*70)
    print(" 정답률 비교")
    print("="*70)
    print()

    # 모델별 정답률
    models = ['gpt-5.2', 'claude-opus-4.5', 'gemini-3-pro']

    print(f"{'Model':<25} {'v4':>15} {'JSON Enforced':>15} {'변화':>10}")
    print("-"*70)

    for model in models:
        # v4
        v4_total = v4_results[model]['total']
        v4_correct = v4_results[model]['correct']
        v4_acc = (v4_correct / v4_total * 100) if v4_total > 0 else 0

        # JSON Enforced
        json_total = json_results[model]['total']
        json_correct = json_results[model]['correct']
        json_acc = (json_correct / json_total * 100) if json_total > 0 else 0

        # 변화
        diff = json_acc - v4_acc
        diff_str = f"{diff:+.1f}%p" if diff != 0 else "0.0%p"

        print(f"{model:<25} {v4_acc:>6.1f}% ({v4_correct:>2}/{v4_total:<2}) {json_acc:>6.1f}% ({json_correct:>2}/{json_total:<2}) {diff_str:>10}")

    print()
    print("-"*70)

    # 전체 평균
    v4_total_correct = sum(r['correct'] for r in v4_results.values())
    v4_total_all = sum(r['total'] for r in v4_results.values())
    v4_overall = (v4_total_correct / v4_total_all * 100) if v4_total_all > 0 else 0

    json_total_correct = sum(r['correct'] for r in json_results.values())
    json_total_all = sum(r['total'] for r in json_results.values())
    json_overall = (json_total_correct / json_total_all * 100) if json_total_all > 0 else 0

    diff_overall = json_overall - v4_overall
    diff_str = f"{diff_overall:+.1f}%p" if diff_overall != 0 else "0.0%p"

    print(f"{'전체 평균':<25} {v4_overall:>6.1f}% ({v4_total_correct:>3}/{v4_total_all:<3}) {json_overall:>6.1f}% ({json_total_correct:>3}/{json_total_all:<3}) {diff_str:>10}")

    # 틀린 문제 분석
    print()
    print("="*70)
    print(" 틀린 문제 분석 (JSON Enforced)")
    print("="*70)
    print()

    wrong_by_model = defaultdict(list)
    for result in json_details:
        if not result['correct']:
            wrong_by_model[result['model']].append(result)

    for model in models:
        wrongs = wrong_by_model[model]
        if wrongs:
            print(f"\n[{model}] 틀린 문제 {len(wrongs)}개:")
            for w in wrongs[:5]:  # 최대 5개만 표시
                print(f"  - {w['prob_id']}: 추출={w['extracted']}, 정답={w['truth']}")
            if len(wrongs) > 5:
                print(f"  ... 외 {len(wrongs)-5}개")

    print()
    print("="*70)
    print()


if __name__ == "__main__":
    main()
