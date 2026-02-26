"""OpenRouter API configuration and shared constants."""

import csv
import os

csv.field_size_limit(10_000_000)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ---------------------------------------------------------------------------
# Models (OpenRouter model IDs)
# ---------------------------------------------------------------------------
MODELS = {
    "gpt-5.2": "openai/gpt-5.2",
    "claude-opus-4.5": "anthropic/claude-opus-4.5",
    "gemini-3-pro": "google/gemini-3-pro-preview",
    "grok-4": "x-ai/grok-4",
    "minimax-m2.1": "minimax/minimax-m2.1",
}

DEFAULT_MODEL = "openai/gpt-5.2"

# ---------------------------------------------------------------------------
# Load problems from CSV
# ---------------------------------------------------------------------------
DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")


def load_problem_from_csv(csv_filename: str, prob_id: str) -> dict:
    """Load a single problem by prob_id from a CSV file."""
    csv_path = os.path.join(DATA_ROOT, csv_filename)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["prob_id"] == prob_id:
                return {
                    "id": row["prob_id"],
                    "label": f"{row['prob_id']} ({row['prob_area']}, {row['prob_point']}점)",
                    "text": row["prob_desc"],
                    "answer": row["answer"],
                }
    raise ValueError(f"Problem {prob_id} not found in {csv_path}")


PROBLEMS = {
    "prob1": load_problem_from_csv("2024_math_odd.csv", "2024_odd_common_7"),
    "prob2": load_problem_from_csv("2024_math_odd.csv", "2024_odd_common_8"),
    "prob22": load_problem_from_csv("2024_math_odd.csv", "2024_odd_common_22"),
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

PROMPT_VERSIONS = {
    "v1_simple": PROMPT_V1_SIMPLE,
    "v2_step": PROMPT_V2_STEPBYSTEP,
    "v3_titled": PROMPT_V3_STEPBYSTEP_TITLED,
}
