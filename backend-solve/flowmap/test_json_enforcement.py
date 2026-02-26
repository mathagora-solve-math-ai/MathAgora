#!/usr/bin/env python3
"""
JSON Schema Enforcement 방법별 비교 테스트

Methods:
  1. OpenRouter structured_outputs  → Claude-opus-4.5, Gemini-3-pro
  2. Anthropic API native json_schema → Claude-opus-4.5
  3. Google GenAI (Vertex AI) response_schema → Gemini-3-pro

평가 지표:
  - 파싱 성공 여부 (valid JSON)
  - final_answer 유효성 (integer, 범위 준수)
  - 정답 일치 여부
"""

import csv
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Optional

csv.field_size_limit(10_000_000)

# ─── API Keys (set via environment variables) ──────────────────────────────────

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GOOGLE_CLOUD_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")

# ─── Schema ───────────────────────────────────────────────────────────────────

def make_solution_schema(prob_type: str, supports_minmax: bool = True) -> dict:
    """JSON Schema 생성.

    Args:
        prob_type: '5지선다형' 또는 '단답형'
        supports_minmax: False이면 integer의 minimum/maximum 제거
                         (Claude는 min/max 미지원)
    """
    if prob_type == "5지선다형":
        fa: dict = {"type": "integer", "description": "최종 답 (1~5)"}
        if supports_minmax:
            fa["minimum"] = 1
            fa["maximum"] = 5
    else:
        fa = {"type": "integer", "description": "최종 답 (0~999)"}
        if supports_minmax:
            fa["minimum"] = 0
            fa["maximum"] = 999

    return {
        "type": "object",
        "properties": {
            "model_name": {"type": "string"},
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "step_idx": {"type": "integer"},
                        "title": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["step_idx", "title", "content"],
                    "additionalProperties": False,
                },
            },
            "final_answer": fa,
        },
        "required": ["model_name", "steps", "final_answer"],
        "additionalProperties": False,
    }


# ─── Prompt ───────────────────────────────────────────────────────────────────

def make_prompt(problem_text: str, model_name: str, prob_type: str) -> str:
    if prob_type == "5지선다형":
        fa_hint, fa_desc = "3", "객관식 선택지 번호 정수 (1~5 중 하나)"
    else:
        fa_hint, fa_desc = "42", "단답형 정답 정수 (0~999 사이)"

    return f"""다음 수학 문제를 단계별로 풀어주세요.

<problem>
{problem_text}
</problem>

중요 지침:
1. STEP은 풀이의 흐름(전략/접근)이 바뀌는 지점마다 구분하되, 의미 없는 세분화는 피하고 필요한 만큼만 나누세요
2. 각 단계는 명확한 목적을 가져야 합니다

반드시 다음 JSON 형식으로 응답하세요:

{{
  "model_name": "{model_name}",
  "steps": [
    {{"step_idx": 0, "title": "단계 제목", "content": "풀이 내용"}},
    {{"step_idx": 1, "title": "...", "content": "..."}}
  ],
  "final_answer": {fa_hint}
}}

final_answer는 {fa_desc}"""


# ─── Result ───────────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    method: str
    model: str
    prob_id: str
    prob_type: str
    correct_answer: str
    parse_ok: bool = False
    final_answer_valid: bool = False
    final_answer_value: Optional[int] = None
    correct: bool = False
    elapsed: float = 0.0
    error: str = ""
    n_steps: int = 0


def validate_final_answer(value, prob_type: str) -> bool:
    if not isinstance(value, int):
        return False
    if prob_type == "5지선다형":
        return 1 <= value <= 5
    else:
        return 0 <= value <= 999


def check_correct(predicted: Optional[int], correct_answer: str) -> bool:
    if predicted is None:
        return False
    try:
        return predicted == int(correct_answer)
    except (ValueError, TypeError):
        return False


# ─── Method 1: OpenRouter structured_outputs ──────────────────────────────────

def test_openrouter_structured(problem_text, prob_type, model_id, display_name,
                               supports_minmax: bool = True):
    from openai import OpenAI
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    schema = make_solution_schema(prob_type, supports_minmax=supports_minmax)
    prompt = make_prompt(problem_text, display_name, prob_type)

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "수학 문제 풀이 전문가입니다. 반드시 JSON으로만 응답하세요."},
            {"role": "user", "content": prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "model_solution",
                "strict": True,
                "schema": schema,
            },
        },
        temperature=0.3,
        max_tokens=4096,
    )
    return json.loads(response.choices[0].message.content)


# ─── Method 2: Anthropic native output_config ─────────────────────────────────

def test_anthropic_native(problem_text, prob_type, display_name="claude-opus-4-5"):
    import anthropic

    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    # Claude는 integer min/max 미지원
    schema = make_solution_schema(prob_type, supports_minmax=False)
    prompt = make_prompt(problem_text, display_name, prob_type)

    # output_config은 beta 기능 → client.beta.messages 사용
    # SDK 구조: output_config.format.{type, schema}
    response = client.beta.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=4096,
        system="수학 문제 풀이 전문가입니다. 반드시 JSON으로만 응답하세요.",
        messages=[{"role": "user", "content": prompt}],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": schema,
            }
        },
    )
    text = response.content[0].text
    return json.loads(text)


# ─── Method 3: Google GenAI (Vertex AI or AI Studio) response_schema ──────────

def _build_genai_schema(prob_type: str) -> dict:
    """google-genai SDK가 요구하는 대문자 타입 스키마"""
    if prob_type == "5지선다형":
        fa = {"type": "INTEGER", "description": "최종 답 (1~5)", "minimum": 1, "maximum": 5}
    else:
        fa = {"type": "INTEGER", "description": "최종 답 (0~999)", "minimum": 0, "maximum": 999}
    return {
        "type": "OBJECT",
        "properties": {
            "model_name": {"type": "STRING"},
            "steps": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "step_idx": {"type": "INTEGER"},
                        "title": {"type": "STRING"},
                        "content": {"type": "STRING"},
                    },
                    "required": ["step_idx", "title", "content"],
                },
            },
            "final_answer": fa,
        },
        "required": ["model_name", "steps", "final_answer"],
    }


def test_google_genai(problem_text, prob_type, display_name="gemini-3-pro"):
    from google import genai
    from google.genai.types import HttpOptions, GenerateContentConfig

    # Vertex AI 우선, 없으면 AI Studio
    use_vertex = bool(GOOGLE_CLOUD_PROJECT)
    if use_vertex:
        client = genai.Client(
            http_options=HttpOptions(api_version="v1"),
            vertexai=True,
            project=GOOGLE_CLOUD_PROJECT,
            location="us-central1",
        )
        model_id = "gemini-2.5-flash"   # Vertex AI model ID
    else:
        if not GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY (or GOOGLE_CLOUD_PROJECT) not set")
        client = genai.Client(api_key=GOOGLE_API_KEY)
        model_id = "gemini-2.5-flash"

    prompt = make_prompt(problem_text, display_name, prob_type)
    schema = _build_genai_schema(prob_type)

    response = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=GenerateContentConfig(
            system_instruction="수학 문제 풀이 전문가입니다. 반드시 JSON으로만 응답하세요.",
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.3,
        ),
    )
    return json.loads(response.text)


# ─── Test Runner ──────────────────────────────────────────────────────────────

METHODS = [
    # (method_label, model_label, call_fn)
    ("openrouter_structured",  "claude-opus-4.5",
     lambda prob_text, prob_type: test_openrouter_structured(
         prob_text, prob_type,
         model_id="anthropic/claude-opus-4.5",
         display_name="claude-opus-4.5",
         supports_minmax=False,          # Claude: min/max 미지원
     )),
    ("openrouter_structured",  "gemini-3-pro",
     lambda prob_text, prob_type: test_openrouter_structured(
         prob_text, prob_type,
         model_id="google/gemini-3-pro-preview",
         display_name="gemini-3-pro",
         supports_minmax=True,           # Gemini: min/max 지원
     )),
    ("anthropic_native",       "claude-opus-4.5",
     lambda prob_text, prob_type: test_anthropic_native(prob_text, prob_type)),
    ("google_genai_schema",    "gemini",
     lambda prob_text, prob_type: test_google_genai(prob_text, prob_type)),
]


def run_tests(problems: list) -> list[TestResult]:
    results = []

    for method_label, model_label, call_fn in METHODS:
        print(f"\n{'='*60}")
        print(f"Method: {method_label} | Model: {model_label}")
        print(f"{'='*60}")

        for prob in problems:
            prob_id = prob["prob_id"]
            prob_type = prob["prob_type"]
            correct_answer = prob["answer"]
            prob_text = prob["prob_desc"]

            print(f"  [{prob_id}] ({prob_type}) ... ", end="", flush=True)
            result = TestResult(
                method=method_label,
                model=model_label,
                prob_id=prob_id,
                prob_type=prob_type,
                correct_answer=correct_answer,
            )

            t0 = time.time()
            try:
                data = call_fn(prob_text, prob_type)
                result.elapsed = time.time() - t0

                # 1) 파싱 성공
                result.parse_ok = isinstance(data, dict)

                if result.parse_ok:
                    fa = data.get("final_answer")
                    result.final_answer_value = fa
                    result.final_answer_valid = validate_final_answer(fa, prob_type)
                    result.correct = check_correct(fa, correct_answer)
                    result.n_steps = len(data.get("steps", []))

                status = "✓" if result.correct else ("△" if result.final_answer_valid else "✗")
                print(f"{status} answer={result.final_answer_value} "
                      f"(correct={correct_answer}) steps={result.n_steps} {result.elapsed:.1f}s")

            except Exception as e:
                result.elapsed = time.time() - t0
                result.error = str(e)
                print(f"✗ ERROR: {e}")

            results.append(result)
            time.sleep(2)  # rate limit

    return results


def print_summary(results: list[TestResult]):
    print(f"\n{'='*70}")
    print(" SUMMARY")
    print(f"{'='*70}")

    # per method+model
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        groups[(r.method, r.model)].append(r)

    header = f"{'Method':<28} {'Model':<18} {'Parse':>6} {'Valid FA':>8} {'Correct':>8} {'Avg t':>6}"
    print(header)
    print("-" * 76)

    for (method, model), rs in sorted(groups.items()):
        n = len(rs)
        parse_ok = sum(1 for r in rs if r.parse_ok)
        valid_fa = sum(1 for r in rs if r.final_answer_valid)
        correct  = sum(1 for r in rs if r.correct)
        avg_t    = sum(r.elapsed for r in rs) / n if n else 0
        print(f"{method:<28} {model:<18} "
              f"{parse_ok}/{n:>4} "
              f"{valid_fa}/{n:>6} "
              f"{correct}/{n:>6} "
              f"{avg_t:>5.1f}s")

    print()

    # error details
    errors = [r for r in results if r.error]
    if errors:
        print("Errors:")
        for r in errors:
            print(f"  [{r.method}/{r.model}/{r.prob_id}] {r.error[:120]}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def load_test_problems(n_per_type: int = 3) -> list:
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "2024_math_odd.csv")
    all_rows = []
    with open(csv_path, encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    cho = [r for r in all_rows if r["prob_type"] == "5지선다형"][:n_per_type]
    dan = [r for r in all_rows if r["prob_type"] == "단답형"][:n_per_type]
    return cho + dan


if __name__ == "__main__":
    print("JSON Schema Enforcement 방법별 테스트")
    print(f"ANTHROPIC_API_KEY: {'설정됨' if ANTHROPIC_API_KEY else '없음 ← Anthropic native 테스트 불가'}")
    print(f"GOOGLE_API_KEY:    {'설정됨' if GOOGLE_API_KEY else '없음'}")
    print(f"GOOGLE_CLOUD_PROJECT: {'설정됨' if GOOGLE_CLOUD_PROJECT else '없음 ← Vertex AI 불가 (AI Studio fallback)'}")

    problems = load_test_problems(n_per_type=3)
    print(f"\n테스트 문제: {len(problems)}개")
    for p in problems:
        print(f"  {p['prob_id']} [{p['prob_type']}] 정답={p['answer']}")

    results = run_tests(problems)

    # JSON 저장
    out_path = os.path.join(os.path.dirname(__file__), "outputs", "enforcement_test_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            [
                {k: v for k, v in r.__dict__.items()}
                for r in results
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )

    print_summary(results)
    print(f"Results saved: {out_path}")
