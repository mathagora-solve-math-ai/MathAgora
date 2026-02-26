"""Flow Map Generator - Main Agent"""

import json
import sys
import os

# Add parent directory to path to import from poc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'poc'))

from llm_client import call_llm
from schemas import (
    FlowMapInput, FlowMap, StepGroup, GroupedStep, FlowConnection
)


GENERATOR_PROMPT = """당신은 여러 LLM의 수학 풀이를 분석하여 Flow Map을 생성하는 전문가입니다.

주어진 문제와 여러 모델의 step-by-step 풀이를 보고, 다음을 수행하세요:

1. **유사한 풀이 단계끼리 그룹핑**: 서로 다른 모델의 step이라도 같은 풀이 단계(예: "도함수 구하기")를 수행하면 같은 그룹으로 묶습니다.
2. **그룹 이름 작명**: 각 그룹에 간결하고 명확한 중제목을 붙입니다 (예: "도함수 구하기", "임계점 찾기", "극대·극소 판별").
3. **그룹 순서 결정**: 풀이의 논리적 흐름에 따라 그룹에 순서를 매깁니다.

# 입력

## 문제
{problem}

## 모델별 풀이

{solutions}

# 출력 형식

JSON 형태로 출력하세요:

```json
{{
  "groups": [
    {{
      "group_id": 0,
      "group_name": "그룹 이름 (간결하게)",
      "steps": [
        {{"model": "모델명", "step_idx": 원본_인덱스}}
      ]
    }}
  ]
}}
```

**중요:**
- 각 step은 정확히 하나의 그룹에만 속합니다.
- group_id는 0부터 시작하며, 풀이 순서대로 증가합니다.
- steps 배열에는 해당 그룹에 속하는 step의 model과 step_idx만 기록합니다 (title/content는 제외).
- 모든 모델의 모든 step이 어떤 그룹에든 포함되어야 합니다.

JSON만 출력하고, 다른 설명은 붙이지 마세요.
"""


def generate_flow_map(input_data: FlowMapInput, generator_model: str = "anthropic/claude-opus-4.5") -> FlowMap:
    """Generate Flow Map from multiple model solutions

    Args:
        input_data: FlowMapInput with problem and solutions
        generator_model: LLM to use for generation

    Returns:
        FlowMap with groups and flows
    """
    # Format solutions for prompt
    solutions_text = ""
    for sol in input_data.solutions:
        solutions_text += f"\n### {sol.model_name}\n"
        for step in sol.steps:
            solutions_text += f"Step {step.step_idx}: {step.title}\n"
            solutions_text += f"{step.content}\n\n"

    # Call LLM to generate grouping
    prompt = GENERATOR_PROMPT.format(
        problem=input_data.problem_text,
        solutions=solutions_text
    )

    print("Calling LLM to generate Flow Map...")
    response = call_llm(prompt, generator_model, temperature=0.3, max_tokens=4096)

    # Parse JSON response
    try:
        # Extract JSON from response (in case there's markdown wrapping)
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        json_str = response[json_start:json_end]

        grouping = json.loads(json_str)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Failed to parse LLM response as JSON: {e}")
        print(f"Response: {response[:500]}...")
        raise

    # Build FlowMap from LLM response + original data
    groups = []

    # Create lookup: (model, step_idx) -> Step object
    step_lookup = {}
    for sol in input_data.solutions:
        for step in sol.steps:
            step_lookup[(sol.model_name, step.step_idx)] = step

    for group_data in grouping["groups"]:
        group_steps = []
        for step_ref in group_data["steps"]:
            model = step_ref["model"]
            step_idx = step_ref["step_idx"]

            # Get original step data
            original_step = step_lookup.get((model, step_idx))
            if original_step is None:
                print(f"Warning: Step not found: {model} step {step_idx}")
                continue

            group_steps.append(GroupedStep(
                model=model,
                step_idx=step_idx,
                title=original_step.title,
                content=original_step.content
            ))

        groups.append(StepGroup(
            group_id=group_data["group_id"],
            group_name=group_data["group_name"],
            steps=group_steps
        ))

    # Generate flow connections (sequential within each model)
    flows = []
    for sol in input_data.solutions:
        for i in range(len(sol.steps) - 1):
            flows.append(FlowConnection(
                model=sol.model_name,
                from_step=sol.steps[i].step_idx,
                to_step=sol.steps[i + 1].step_idx
            ))

    return FlowMap(groups=groups, flows=flows)


def load_from_poc_result(json_path: str) -> FlowMapInput:
    """Load input data from PoC task2 result JSON

    Args:
        json_path: Path to task2_probX.json

    Returns:
        FlowMapInput
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract problem (we'll need to load it from config)
    # For now, use a placeholder
    problem_text = "문제 텍스트 (JSON에서 추출 필요)"

    solutions = []
    for model_name, steps_data in data["solutions"].items():
        from schemas import Step, ModelSolution

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
