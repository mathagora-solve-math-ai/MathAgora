#!/usr/bin/env python3
"""
Structured JSON Output Pipeline for Flow Map Generation

강제된 JSON 스키마로 step과 flowmap을 생성합니다.
- OpenAI: response_format={"type": "json_schema"}
- Anthropic: Tool use로 JSON 강제
"""

import json
import os
import csv
import time
from anthropic import Anthropic
from openai import OpenAI

# API clients
anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

csv.field_size_limit(10_000_000)


# JSON Schema 정의
STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "description": "풀이 단계들의 배열",
            "items": {
                "type": "object",
                "properties": {
                    "step_idx": {
                        "type": "integer",
                        "description": "단계 번호 (0부터 시작)"
                    },
                    "title": {
                        "type": "string",
                        "description": "단계의 제목 (간결하게, 10단어 이내)"
                    },
                    "content": {
                        "type": "string",
                        "description": "단계의 상세 내용"
                    }
                },
                "required": ["step_idx", "title", "content"],
                "additionalProperties": False
            }
        }
    },
    "required": ["steps"],
    "additionalProperties": False
}

FLOWMAP_SCHEMA = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "description": "의미적으로 유사한 step들의 그룹",
            "items": {
                "type": "object",
                "properties": {
                    "group_id": {"type": "string"},
                    "group_name": {"type": "string"},
                    "step_ids": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "model": {"type": "string"},
                                "step_idx": {"type": "integer"}
                            },
                            "required": ["model", "step_idx"]
                        }
                    }
                },
                "required": ["group_id", "group_name", "step_ids"]
            }
        }
    },
    "required": ["groups"]
}


def generate_steps_structured_openai(model_name, problem_text):
    """OpenAI Structured Output으로 step 생성"""

    prompt = f"""다음 수학 문제를 단계별로 풀어주세요.

<problem>
{problem_text}
</problem>

**중요**: 반드시 JSON 형식으로 응답하세요. 각 단계는 다음 형식을 따라야 합니다:
- step_idx: 단계 번호 (0부터 시작)
- title: 단계의 간결한 제목 (10단어 이내)
- content: 단계의 상세한 풀이 내용

단계는 풀이의 흐름(전략/접근)이 바뀌는 지점마다 구분하되, 의미 없는 세분화는 피하고 필요한 만큼만 나누세요."""

    response = openai_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "당신은 수학 문제 풀이 전문가입니다. 문제를 단계별로 나누어 풀이합니다."},
            {"role": "user", "content": prompt}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "step_response",
                "strict": True,
                "schema": STEP_SCHEMA
            }
        },
        temperature=0.7
    )

    return json.loads(response.choices[0].message.content)


def generate_steps_structured_anthropic(model_name, problem_text):
    """Anthropic Tool Use로 step 생성"""

    prompt = f"""다음 수학 문제를 단계별로 풀어주세요.

<problem>
{problem_text}
</problem>

단계는 풀이의 흐름(전략/접근)이 바뀌는 지점마다 구분하되, 의미 없는 세분화는 피하고 필요한 만큼만 나누세요.

각 단계는:
- step_idx: 단계 번호 (0부터 시작)
- title: 단계의 간결한 제목 (10단어 이내)
- content: 단계의 상세한 풀이 내용

반드시 submit_steps 도구를 사용하여 결과를 제출하세요."""

    tools = [{
        "name": "submit_steps",
        "description": "풀이 단계들을 제출합니다",
        "input_schema": STEP_SCHEMA
    }]

    response = anthropic_client.messages.create(
        model=model_name,
        max_tokens=8000,
        temperature=0.7,
        tools=tools,
        messages=[{"role": "user", "content": prompt}]
    )

    # Tool use 결과 추출
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_steps":
            return block.input

    raise ValueError("No tool use found in response")


def generate_flowmap_structured(all_steps, models):
    """Structured JSON으로 flowmap 생성"""

    # 모든 step 정보 준비
    step_info = []
    for model, steps_data in zip(models, all_steps):
        for step in steps_data['steps']:
            step_info.append({
                'model': model,
                'step_idx': step['step_idx'],
                'title': step['title'],
                'content': step['content'][:200]  # 요약
            })

    prompt = f"""다음은 3개 모델이 같은 수학 문제를 푼 단계들입니다.

{json.dumps(step_info, ensure_ascii=False, indent=2)}

**작업**: 의미적으로 유사한 step들을 그룹으로 묶어주세요.

**규칙**:
1. 그룹은 시간 순서대로 배열되어야 합니다 (먼저 수행하는 단계가 앞에)
2. 각 그룹은 명확한 이름을 가져야 합니다
3. 그룹 ID는 "G0", "G1", ... 형식입니다
4. 불필요하게 세분화하지 마세요

반드시 submit_flowmap 도구를 사용하여 결과를 제출하세요."""

    tools = [{
        "name": "submit_flowmap",
        "description": "그룹핑된 flowmap을 제출합니다",
        "input_schema": FLOWMAP_SCHEMA
    }]

    response = anthropic_client.messages.create(
        model="claude-opus-4-20250514",
        max_tokens=8000,
        temperature=0.7,
        tools=tools,
        messages=[{"role": "user", "content": prompt}]
    )

    # Tool use 결과 추출
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_flowmap":
            return block.input

    raise ValueError("No tool use found in response")


def process_problem_structured(prob_id, prob_text, output_dir='outputs/structured_json'):
    """문제 하나를 structured JSON으로 처리"""

    print(f"\n{'='*60}")
    print(f"Processing: {prob_id}")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    models = {
        'gpt-5.2': 'gpt-4o-2024-11-20',  # 실제 모델명으로 교체 필요
        'claude-opus-4.5': 'claude-opus-4-20250514',
        'gemini-3-pro': None  # Gemini는 별도 처리 필요
    }

    all_steps = {}

    # 1. 각 모델에서 structured step 생성
    for display_name, model_name in models.items():
        if model_name is None:
            continue

        print(f"\n[{display_name}] Generating structured steps...")

        try:
            if 'gpt' in model_name or 'o1' in model_name:
                steps = generate_steps_structured_openai(model_name, prob_text)
            else:
                steps = generate_steps_structured_anthropic(model_name, prob_text)

            all_steps[display_name] = steps
            print(f"  ✓ Generated {len(steps['steps'])} steps")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue

        time.sleep(2)  # Rate limit

    # 2. Structured flowmap 생성
    print(f"\n[Flowmap] Generating structured grouping...")
    try:
        flowmap_groups = generate_flowmap_structured(
            list(all_steps.values()),
            list(all_steps.keys())
        )

        # 전체 flowmap 구성
        flowmap = {
            'groups': [],
            'flows': []
        }

        for group in flowmap_groups['groups']:
            group_steps = []
            for step_id in group['step_ids']:
                model = step_id['model']
                step_idx = step_id['step_idx']

                if model in all_steps:
                    step_data = all_steps[model]['steps'][step_idx]
                    group_steps.append({
                        'model': model,
                        'step_idx': step_idx,
                        'title': step_data['title'],
                        'content': step_data['content']
                    })

            flowmap['groups'].append({
                'group_id': group['group_id'],
                'group_name': group['group_name'],
                'steps': group_steps
            })

        # Flows 생성 (step 간 연결)
        for model, steps_data in all_steps.items():
            for i in range(len(steps_data['steps']) - 1):
                flowmap['flows'].append({
                    'model': model,
                    'from_step': i,
                    'to_step': i + 1
                })

        print(f"  ✓ Generated {len(flowmap['groups'])} groups")

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None

    # 3. 저장
    output_path = os.path.join(output_dir, f'flowmap_{prob_id}.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(flowmap, f, ensure_ascii=False, indent=2)

    steps_path = os.path.join(output_dir, f'steps_{prob_id}.json')
    with open(steps_path, 'w', encoding='utf-8') as f:
        json.dump(all_steps, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved to {output_dir}/")

    return flowmap


def main():
    """2024년도 46문제 처리"""

    print("=== Structured JSON Pipeline ===\n")

    # 2024 수능 홀수 문제 로드
    csv_path = '../data/2024_math_odd.csv'

    problems = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            problems.append({
                'prob_id': row['prob_id'],
                'prob_desc': row['prob_desc']
            })

    print(f"Total problems: {len(problems)}")

    # 처리
    output_dir = 'outputs/structured_json'
    success = 0

    for i, prob in enumerate(problems, 1):
        print(f"\n[{i}/{len(problems)}]")

        try:
            result = process_problem_structured(
                prob['prob_id'],
                prob['prob_desc'],
                output_dir
            )

            if result:
                success += 1

        except Exception as e:
            print(f"✗ Failed: {e}")
            continue

        # Rate limiting
        if i < len(problems):
            time.sleep(5)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total: {len(problems)}")
    print(f"Success: {success}")
    print(f"Failed: {len(problems) - success}")
    print(f"Output: {output_dir}/")


if __name__ == "__main__":
    main()
