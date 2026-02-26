#!/usr/bin/env python3
"""
JSON Schema 강제 출력 파이프라인

각 모델(GPT-5.2, Claude Opus 4.5, Gemini 3 Pro)이
Generator 입력 스키마 형식으로 직접 JSON을 출력하도록 강제합니다.

Input Schema:
{
  "model_name": "gpt-5.2",
  "steps": [
    {"step_idx": 0, "title": "...", "content": "..."},
    {"step_idx": 1, "title": "...", "content": "..."}
  ]
}
"""

import json
import os
import csv
import time
from openai import OpenAI

csv.field_size_limit(10_000_000)

# API Keys and Configuration (set via environment variables)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# API Clients
def _get_openai_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)

def _get_openrouter_client() -> OpenAI:
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)


# ═══════════════════════════════════════════════════════════════
# JSON Schema 정의 (Generator 입력 형식)
# ═══════════════════════════════════════════════════════════════

def make_solution_schema(prob_type: str) -> dict:
    """문제 유형에 따른 JSON Schema 생성

    Args:
        prob_type: '5지선다형' 또는 '단답형'
    """
    if prob_type == "5지선다형":
        final_answer_schema = {
            "type": "integer",
            "description": "최종 답 (객관식 선택지 번호: 1~5)",
            "minimum": 1,
            "maximum": 5,
        }
    else:  # 단답형
        final_answer_schema = {
            "type": "integer",
            "description": "최종 답 (단답형 정수: 0~999)",
            "minimum": 0,
            "maximum": 999,
        }

    return {
        "type": "object",
        "properties": {
            "model_name": {
                "type": "string",
                "description": "모델 이름"
            },
            "steps": {
                "type": "array",
                "description": "풀이 단계 배열",
                "items": {
                    "type": "object",
                    "properties": {
                        "step_idx": {
                            "type": "integer",
                            "description": "단계 번호 (0부터 시작)"
                        },
                        "title": {
                            "type": "string",
                            "description": "단계 제목 (간결하게, 10단어 이내)"
                        },
                        "content": {
                            "type": "string",
                            "description": "단계 내용 (상세한 풀이)"
                        }
                    },
                    "required": ["step_idx", "title", "content"],
                    "additionalProperties": False
                }
            },
            "final_answer": final_answer_schema,
        },
        "required": ["model_name", "steps", "final_answer"],
        "additionalProperties": False
    }


# ═══════════════════════════════════════════════════════════════
# v4 프롬프트 (불필요한 세분화 방지)
# ═══════════════════════════════════════════════════════════════

V4_PROMPT_TEMPLATE = """다음 수학 문제를 단계별로 풀어주세요.

<problem>
{problem_text}
</problem>

**중요 지침**:
1. 단계(STEP)는 풀이의 흐름(전략/접근)이 바뀌는 지점마다 구분하되, **의미 없는 세분화는 피하고 필요한 만큼만** 나누세요
2. 각 단계는 명확한 목적을 가져야 합니다
3. 불필요하게 세분화하지 마세요 (예: "식 정리" → "괄호 풀기" → "동류항 정리" 대신 "식 정리"로 통합)

**출력 형식**:
반드시 다음 JSON 형식으로 응답하세요:

{{
  "model_name": "{model_name}",
  "steps": [
    {{
      "step_idx": 0,
      "title": "단계 제목 (간결하게)",
      "content": "상세한 풀이 내용"
    }},
    {{
      "step_idx": 1,
      "title": "...",
      "content": "..."
    }}
  ],
  "final_answer": {final_answer_hint}
}}

**step_idx는 0부터 시작**하며, 순차적으로 증가합니다.
**title은 10단어 이내**로 간결하게 작성합니다.
**content는 해당 단계의 구체적인 풀이 과정**을 포함합니다.
**final_answer는 {final_answer_desc}**"""


# ═══════════════════════════════════════════════════════════════
# OpenAI Structured Output
# ═══════════════════════════════════════════════════════════════

def _make_prompt(problem_text: str, model_name: str, prob_type: str) -> str:
    """prob_type에 맞는 final_answer 힌트를 포함한 프롬프트 생성"""
    if prob_type == "5지선다형":
        hint = "1"
        desc = "객관식 선택지 번호 정수 (1~5 중 하나)"
    else:
        hint = "0"
        desc = "단답형 정답 정수 (0~999 사이)"
    return V4_PROMPT_TEMPLATE.format(
        problem_text=problem_text,
        model_name=model_name,
        final_answer_hint=hint,
        final_answer_desc=desc,
    )


def solve_with_openai_structured(problem_text, model_name="openai/gpt-5.2", display_name="gpt-5.2", prob_type="단답형"):
    """OpenAI Structured Output으로 풀이 (OpenAI API 직접 사용)"""

    prompt = _make_prompt(problem_text, display_name, prob_type)

    # OpenAI API 직접 사용 (structured output 지원)
    client = _get_openai_client()
    native_model = model_name.split("/", 1)[1] if "/" in model_name else model_name

    response = client.chat.completions.create(
        model=native_model,
        messages=[
            {
                "role": "system",
                "content": "당신은 수학 문제 풀이 전문가입니다. 문제를 명확한 단계로 나누어 풀이하되, 불필요한 세분화는 피합니다."
            },
            {"role": "user", "content": prompt}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "model_solution",
                "strict": True,
                "schema": make_solution_schema(prob_type)
            }
        },
        temperature=0.7
    )

    return json.loads(response.choices[0].message.content)


# ═══════════════════════════════════════════════════════════════
# Anthropic Tool Use (JSON 강제)
# ═══════════════════════════════════════════════════════════════

def solve_with_openrouter_json(problem_text, model_name="anthropic/claude-opus-4.5", display_name="claude-opus-4.5", prob_type="단답형"):
    """OpenRouter API로 JSON 강제 (Claude, Gemini 등)"""

    prompt = _make_prompt(problem_text, display_name, prob_type)

    # OpenRouter API 사용
    client = _get_openrouter_client()

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "당신은 수학 문제 풀이 전문가입니다. 문제를 명확한 단계로 나누어 풀이하되, 불필요한 세분화는 피합니다. 반드시 JSON 형식으로만 응답하세요."
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},  # JSON mode
            temperature=0.7,
            max_tokens=8000
        )

        result = json.loads(response.choices[0].message.content)
        return result

    except Exception as e:
        print(f"  ⚠️  OpenRouter JSON mode error, trying without format constraint: {e}")
        # Fallback: without response_format
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "당신은 수학 문제 풀이 전문가입니다. 문제를 명확한 단계로 나누어 풀이하되, 불필요한 세분화는 피합니다. 반드시 JSON 형식으로만 응답하세요."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=8000
        )

        content = response.choices[0].message.content
        # Try to extract JSON from markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        return json.loads(content)


# ═══════════════════════════════════════════════════════════════
# 모든 OpenRouter 모델은 solve_with_openrouter_json 사용
# ═══════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════

def process_problem_json_enforced(prob_id, prob_text, prob_type="단답형", output_dir='outputs/json_enforced'):
    """
    1개 문제를 JSON 강제 방식으로 처리

    각 모델이 Generator 입력 스키마 형식으로 직접 출력
    → Generator에 입력 → Flow Map 생성
    """

    print(f"\n{'='*60}")
    print(f"Processing: {prob_id}")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    # 모델 설정 (OpenRouter model IDs)
    models = {
        'gpt-5.2': {
            'api_model': 'openai/gpt-5.2',  # OpenAI API 직접 사용
            'function': solve_with_openai_structured
        },
        'claude-opus-4.5': {
            'api_model': 'anthropic/claude-opus-4.5',  # OpenRouter 사용
            'function': solve_with_openrouter_json
        },
        'gemini-3-pro': {
            'api_model': 'google/gemini-3-pro-preview',  # OpenRouter 사용
            'function': solve_with_openrouter_json
        }
    }

    solutions = []

    # 1. 각 모델에서 JSON 강제 풀이 생성
    for display_name, config in models.items():
        print(f"\n[{display_name}] Solving with JSON schema enforcement...")

        try:
            solution = config['function'](
                problem_text=prob_text,
                model_name=config['api_model'],
                display_name=display_name,
                prob_type=prob_type,
            )

            if solution:
                solutions.append(solution)
                print(f"  ✓ Generated {len(solution['steps'])} steps")
            else:
                print(f"  ✗ Failed to generate solution")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue

        time.sleep(2)  # Rate limit

    # 2. JSON 저장 (Generator 입력 형식)
    generator_input = {
        'problem_text': prob_text,
        'solutions': solutions
    }

    input_path = os.path.join(output_dir, f'input_{prob_id}.json')
    with open(input_path, 'w', encoding='utf-8') as f:
        json.dump(generator_input, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved Generator input: {input_path}")

    # 3. Generator 호출하여 Flow Map 생성
    try:
        from generator import generate_flow_map
        from schemas import FlowMapInput, ModelSolution, Step

        # Convert dict to schema objects
        model_solutions = []
        for sol_dict in solutions:
            steps = [
                Step(
                    step_idx=s['step_idx'],
                    title=s['title'],
                    content=s['content']
                )
                for s in sol_dict['steps']
            ]
            model_solutions.append(
                ModelSolution(
                    model_name=sol_dict['model_name'],
                    steps=steps
                )
            )

        flowmap_input = FlowMapInput(
            problem_text=prob_text,
            solutions=model_solutions
        )

        flowmap = generate_flow_map(flowmap_input)

        flowmap_path = os.path.join(output_dir, f'flowmap_{prob_id}.json')
        with open(flowmap_path, 'w', encoding='utf-8') as f:
            json.dump(flowmap.to_dict(), f, ensure_ascii=False, indent=2)

        print(f"✅ Generated Flow Map: {flowmap_path}")
        print(f"   - Groups: {len(flowmap.groups)}")
        print(f"   - Flows: {len(flowmap.flows)}")

        return flowmap

    except Exception as e:
        print(f"⚠️  Flow Map generation failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    """2024년도 46문제 처리"""

    print("="*70)
    print(" JSON Schema 강제 출력 파이프라인 (v4 Prompt)")
    print("="*70)
    print()
    print("각 모델이 Generator 입력 스키마 형식으로 직접 JSON 출력")
    print("→ Generator에 입력 → Flow Map 생성")
    print()

    # 2024 수능 홀수 문제 로드
    csv_path = '../data/2024_math_odd.csv'

    problems = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            problems.append({
                'prob_id': row['prob_id'],
                'prob_type': row.get('prob_type', '단답형'),
                'prob_desc': row['prob_desc'],
                'prob_area': row.get('prob_area', ''),
                'prob_point': row.get('prob_point', ''),
                'answer': row.get('answer', '')
            })

    print(f"📚 Total problems: {len(problems)}\n")

    # 처리
    output_dir = 'outputs/json_enforced'
    success_count = 0
    summary = []

    for i, prob in enumerate(problems, 1):
        print(f"\n{'#'*70}")
        print(f"# Problem {i}/{len(problems)}")
        print(f"{'#'*70}")

        try:
            flowmap = process_problem_json_enforced(
                prob['prob_id'],
                prob['prob_desc'],
                prob['prob_type'],
                output_dir
            )

            if flowmap:
                success_count += 1
                summary.append({
                    'prob_id': prob['prob_id'],
                    'success': True,
                    'n_groups': len(flowmap.groups),
                    'n_flows': len(flowmap.flows),
                    'prob_area': prob['prob_area'],
                    'prob_point': prob['prob_point']
                })
            else:
                summary.append({
                    'prob_id': prob['prob_id'],
                    'success': False
                })

        except Exception as e:
            print(f"✗ Fatal error: {e}")
            summary.append({
                'prob_id': prob['prob_id'],
                'success': False
            })
            continue

        # Rate limiting
        if i < len(problems):
            print("\n⏳ Waiting 5 seconds...")
            time.sleep(5)

    # Summary 저장
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 최종 리포트
    print(f"\n{'='*70}")
    print(" SUMMARY")
    print(f"{'='*70}")
    print(f"Total problems:  {len(problems)}")
    print(f"Success:         {success_count}")
    print(f"Failed:          {len(problems) - success_count}")
    print(f"Success rate:    {success_count/len(problems)*100:.1f}%")

    if success_count > 0:
        successful = [s for s in summary if s['success']]
        avg_groups = sum(s['n_groups'] for s in successful) / len(successful)
        avg_flows = sum(s['n_flows'] for s in successful) / len(successful)

        print(f"\nAverage groups:  {avg_groups:.1f}")
        print(f"Average flows:   {avg_flows:.1f}")

    print(f"\nOutput directory: {output_dir}/")
    print(f"Summary file:     {summary_path}")
    print()


if __name__ == "__main__":
    main()
