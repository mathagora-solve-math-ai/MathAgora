"""OpenRouter API configuration and shared constants."""

import base64
import csv
import os
from pathlib import Path

csv.field_size_limit(10_000_000)

OPENROUTER_API_KEY = os.environ.get(
    "OPENROUTER_API_KEY",
    # 편하게 하려면 openrouter key 하드코딩
)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ---------------------------------------------------------------------------
# Models (OpenRouter model IDs)
# ---------------------------------------------------------------------------
MODELS = {
    "gpt-5-codex": "openai/gpt-5-codex",
    "claude-opus-4.5": "anthropic/claude-opus-4.5",
    "gemini-3-pro": "google/gemini-3-pro-preview",
    "grok-4": "x-ai/grok-4",
    "deepseek-v3.2": "deepseek/deepseek-v3.2",
    "qwen3-vl-235b-a22b": "qwen/qwen3-vl-235b-a22b-instruct",
}

DEFAULT_MODEL = "openai/gpt-5-codex"

# ---------------------------------------------------------------------------
# Load problems from CSV
# ---------------------------------------------------------------------------
DATA_ROOT = "/workspace/data"
DATA_FILE = os.environ.get("MATH_DATA_FILE", "2026_math_odd.tsv")

PROB_TYPE_MAPPING = {
    "5지선다형": "Multiple-choice (5 options)",
    "단답형": "Short-answer",
}

ANSWER_FORMAT_RULES = {
    "Multiple-choice (5 options)": (
        "정답은 반드시 1~5 사이의 정수 하나만 출력하세요. "
        "원문자(①~⑤), 문자, 단위, 추가 설명을 쓰지 마세요."
    ),
    "Short-answer": (
        "정답은 반드시 0~999 사이의 정수 하나만 출력하세요. "
        "단위, 추가 설명, 선행 0을 쓰지 마세요."
    ),
}

DEFAULT_ANSWER_FORMAT_RULE = (
    "정답은 반드시 정수 하나만 출력하세요. 불필요한 텍스트를 포함하지 마세요."
)

FINAL_ANSWER_HINTS = {
    "5지선다형": {
        "hint": "1",
        "desc": "객관식 선택지 번호 정수 (1~5 중 하나)",
    },
    "단답형": {
        "hint": "0",
        "desc": "단답형 정답 정수 (0~999 사이)",
    },
}

DEFAULT_FINAL_ANSWER_HINT = "0"
DEFAULT_FINAL_ANSWER_DESC = "정답 정수"


def get_final_answer_hint_desc(prob_type: str) -> tuple[str, str]:
    info = FINAL_ANSWER_HINTS.get(prob_type, None)
    if not info:
        return DEFAULT_FINAL_ANSWER_HINT, DEFAULT_FINAL_ANSWER_DESC
    return info["hint"], info["desc"]


def _infer_delimiter(csv_path: str) -> str:
    """Infer delimiter for mixed CSV/TSV files."""
    with open(csv_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
    # Some files use .csv extension but are tab-separated.
    return "\t" if first_line.count("\t") > first_line.count(",") else ","


def _resolve_data_asset_path(path_str: str) -> str:
    """Resolve image path with /workspace/data as the fixed base directory."""
    rel = (path_str or "").strip()
    if not rel:
        return ""

    # Treat DATA_ROOT as the current root.
    # Example: "data/crop_image/a.png" -> "/workspace/data/crop_image/a.png"
    cleaned = rel.lstrip("/").replace("\\", "/")
    if cleaned.startswith("data/"):
        cleaned = cleaned[len("data/"):]
    return str((Path(DATA_ROOT) / cleaned).resolve())


def encode_image_base64(image_path: str) -> tuple[str, str]:
    """Encode an image file to base64 and infer media type."""
    with open(image_path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode("utf-8")

    suffix = Path(image_path).suffix.lower()
    media_type_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(suffix, "image/png")
    return b64, media_type


def _build_problem_payload(row: dict) -> dict:
    """Convert one CSV row to internal problem payload."""
    prob_id = row["prob_id"]
    prob_type_kr = (row.get("prob_type") or "").strip()
    prob_type_en = PROB_TYPE_MAPPING.get(prob_type_kr, "Unknown")
    answer_format_rule = ANSWER_FORMAT_RULES.get(prob_type_en, DEFAULT_ANSWER_FORMAT_RULE)
    final_answer_hint, final_answer_desc = get_final_answer_hint_desc(prob_type_kr)
    prob_img_path = (row.get("prob_img_path") or "").strip()
    prob_fig_img_path = (row.get("prob_fig_img_path") or "").strip()
    return {
        "id": prob_id,
        "label": f"{prob_id} ({row['prob_area']}, {row['prob_point']}점)",
        "text": row["prob_desc"],
        "answer": row["answer"],
        "prob_type": prob_type_kr,
        "prob_type_en": prob_type_en,
        "answer_format_rule": answer_format_rule,
        "final_answer_hint": final_answer_hint,
        "final_answer_desc": final_answer_desc,
        "prob_img_path": prob_img_path,
        "prob_img_abs_path": _resolve_data_asset_path(prob_img_path),
        "prob_fig_img_path": prob_fig_img_path,
        "prob_fig_img_abs_path": _resolve_data_asset_path(prob_fig_img_path),
    }


def load_problem_from_csv(csv_filename: str, prob_id: str) -> dict:
    """Load a single problem by prob_id from a CSV file."""
    csv_path = os.path.join(DATA_ROOT, csv_filename)
    delimiter = _infer_delimiter(csv_path)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            if row["prob_id"] == prob_id:
                return _build_problem_payload(row)
    raise ValueError(f"Problem {prob_id} not found in {csv_path}")


def load_all_problems_from_csv(csv_filename: str) -> dict[str, dict]:
    """Load all problems from a CSV/TSV file, keyed by prob_id."""
    csv_path = os.path.join(DATA_ROOT, csv_filename)
    delimiter = _infer_delimiter(csv_path)
    problems = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            prob_id = row.get("prob_id")
            if not prob_id:
                continue
            problems[prob_id] = _build_problem_payload(row)
    return problems

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

ALL_PROBLEMS = load_all_problems_from_csv(DATA_FILE)

PROBLEMS = {
    "prob1": ALL_PROBLEMS["2026_odd_common_7"],
    "prob2": ALL_PROBLEMS["2026_odd_common_8"],
    "prob22": ALL_PROBLEMS["2026_odd_common_22"],
}

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
PROMPT_V1_SIMPLE = """\
다음 수학 문제를 풀어주세요.

문제:
{problem}
"""

PROMPT_V2_STEPBYSTEP = """\
다음 수학 문제를 단계별(step-by-step)로 풀어주세요.

문제:
{problem}
"""

PROMPT_V3_STEPBYSTEP_TITLED = """\
다음 수학 문제를 단계별(step-by-step)로 풀어주세요.
각 단계마다 **간결한 제목**(예: "조건 정리", "인수분해")을 붙여주세요.

출력 형식:
[STEP title="<간결한 제목>"]
<풀이 내용>

[FINAL_ANSWER]
<최종 답>

문제:
{problem}
"""

PROMPT_V4_STEPBYSTEP_FLOW_TITLED = """\
다음 수학 문제를 단계별(step-by-step)로 풀어주세요.

**중요 지침**:
1. 단계(STEP)는 풀이의 흐름(전략/접근)이 바뀌는 지점마다 구분하되, **의미 없는 세분화는 피하고 필요한 만큼만** 나누세요
2. 각 단계는 명확한 목적을 가져야 합니다
3. 불필요하게 세분화하지 마세요 (예: "식 정리" → "괄호 풀기" → "동류항 정리" 대신 "식 정리"로 통합)

**출력 형식**:
반드시 아래 JSON 객체 하나만 출력하세요. (코드블록/추가 텍스트 금지)
여기서 "final_answer"는 [FINAL_ANSWER]에 해당하는 값만 넣습니다.

{{
  "model_name": "모델 이름",
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

PROMPT_VERSIONS = {
    "v1_simple": PROMPT_V1_SIMPLE,
    "v2_step": PROMPT_V2_STEPBYSTEP,
    "v3_titled": PROMPT_V3_STEPBYSTEP_TITLED,
    "v4_flow_titled": PROMPT_V4_STEPBYSTEP_FLOW_TITLED,
}
