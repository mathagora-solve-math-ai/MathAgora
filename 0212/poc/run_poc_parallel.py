#!/usr/bin/env python3
"""
PoC script for ACL 2026 demo – LLM step-by-step solution alignment.


Tasks:
  1. Compare prompt versions (simple / step-by-step / titled variants)
  2. Cosine-similarity based step alignment across solutions
  3. Single-model diversity (temperature=1.0, 5 runs)
  4. Multi-model diversity (N different models, 1 run each)
  5. Structured-output statistics (v4 prompt, 6 models, temperature 0.7 fixed)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import csv
import glob
import threading
from collections import Counter
from itertools import combinations
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

from config import (
    ALL_PROBLEMS,
    DEFAULT_MODEL,
    encode_image_base64,
    make_solution_schema,
    MODELS,
    PROBLEMS,
    PROMPT_VERSIONS,
)
from llm_client import call_llm

OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)
FLAT_RESULTS_PATH = os.path.join(OUT_DIR, "results_flat.csv")
RAW_RESPONSES_PATH = os.path.join(OUT_DIR, "task5_raw_responses.jsonl")
CSV_APPEND_LOCK = threading.Lock()
RAW_APPEND_LOCK = threading.Lock()
FLAT_HEADER_READY = False
FLAT_RESULTS_HEADER = [
    "model",
    "temperature",
    "run_index",
    "problem_id",
    "step_count",
    "correct",
    "correct_recheck",
    "gold_answer",
    "model_final_answer",
    "gold_choice",
    "model_choice",
    "parse_ok",
    "raw_response_saved",
    "titles",
    "elapsed_time_seconds",
]

# ───────────────────────────────────────────────────────────────
# Parsing helpers
# ───────────────────────────────────────────────────────────────

def parse_v3_steps(text: str) -> list[dict]:
    """Parse [STEP title="..."] ... [FINAL_ANSWER] format."""
    marker_re = re.compile(
        r'\[STEP\s+title="([^"]+)"\]|\[FINAL_ANSWER\]'
    )
    markers = []
    for m in marker_re.finditer(text):
        if m.group(0).startswith("[STEP"):
            markers.append({"type": "step", "title": m.group(1), "end": m.end()})
        else:
            markers.append({"type": "final", "title": "Final Answer", "end": m.end()})

    steps = []
    for i, marker in enumerate(markers):
        next_start = markers[i + 1].get("end", len(text)) if i + 1 < len(markers) else len(text)
        # body = text between current marker end and next marker start
        next_marker_match = marker_re.search(text, marker["end"])
        body_end = next_marker_match.start() if next_marker_match else len(text)
        body = text[marker["end"]:body_end].strip()
        steps.append({"title": marker["title"], "body": body, "type": marker["type"]})
    return steps


def heuristic_split_steps(text: str) -> list[dict]:
    """Heuristic step splitting for v1/v2 responses.

    Tries numbered patterns like '1.', 'Step 1:', '단계 1:', '**Step 1**' etc.
    Falls back to paragraph splitting.
    """
    # Try numbered patterns
    patterns = [
        r'(?:^|\n)\s*(?:\*\*)?(?:Step|단계|STEP)\s*(\d+)[.:)\s]',
        r'(?:^|\n)\s*(\d+)\.\s',
    ]
    for pat in patterns:
        splits = list(re.finditer(pat, text))
        if len(splits) >= 2:
            steps = []
            for i, m in enumerate(splits):
                start = m.end()
                end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
                body = text[start:end].strip()
                title = f"Step {m.group(1)}"
                steps.append({"title": title, "body": body, "type": "step"})
            return steps

    # Fallback: paragraph splitting
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return [
        {"title": f"Part {i+1}", "body": p, "type": "step"}
        for i, p in enumerate(paragraphs)
    ]


def extract_steps(text: str, prompt_version: str) -> list[dict]:
    """Extract steps from LLM response based on prompt version."""
    if prompt_version in ("v3_titled", "v4_flow_titled"):
        steps = parse_v3_steps(text)
        if steps:
            return steps
    return heuristic_split_steps(text)


def _format_prompt_for_problem(template: str, prob: dict, problem_text: str | None = None) -> str:
    """Format prompt template with backward-compatible keys."""
    if problem_text is None:
        problem_text = prob.get("text", "")
    return template.format(
        problem=problem_text,
        answer_format_rule=prob.get("answer_format_rule", ""),
        final_answer_hint=prob.get("final_answer_hint", "0"),
        final_answer_desc=prob.get("final_answer_desc", "정답 정수"),
        problem_type=prob.get("prob_type_en", "Unknown"),
    )


def _build_task5_user_messages(prompt_text: str, prob: dict, require_image: bool = True) -> list[dict]:
    """Build user messages: image-only block (base64), prompt is sent via system message."""
    content = []
    image_path = str(prob.get("prob_img_abs_path", "")).strip()
    if image_path and os.path.exists(image_path):
        b64, media_type = encode_image_base64(image_path)
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64,
                },
            }
        )
    elif require_image:
        raise FileNotFoundError(
            f"Problem image not found for {prob.get('id')}: {image_path} "
            f"(prob_img_path={prob.get('prob_img_path', '')})"
        )
    else:
        content.append({"type": "text", "text": f"문제 원문:\n{prob.get('text', '').strip()}"})
    return [{"role": "user", "content": content}]


def _normalize_for_match(text: str) -> str:
    """Normalize text for lenient exact/containment matching."""
    return re.sub(r"[^0-9A-Za-z가-힣]+", "", text).lower()


def _extract_final_answer(response: str, steps: list[dict]) -> str:
    """Extract final answer text from parsed steps or raw response fallback."""
    for step in reversed(steps):
        if step.get("type") == "final":
            return step.get("body", "").strip()

    # Fallback: try marker-based extraction from raw response
    m = re.search(r"\[FINAL_ANSWER\](.*)$", response, flags=re.DOTALL)
    if m:
        return m.group(1).strip()

    # Last resort: tail text
    return response.strip()[-300:]


def _parse_model_solution_json(response: str) -> dict | None:
    """Parse schema-enforced model solution JSON response."""
    try:
        payload = json.loads(response)
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    model_name = str(payload.get("model_name", "")).strip()
    final_answer = str(payload.get("final_answer", "")).strip()
    raw_steps = payload.get("steps", [])
    if not isinstance(raw_steps, list):
        return None

    steps = []
    titles = []
    for idx, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, dict):
            continue
        try:
            step_idx = int(raw_step.get("step_idx", idx))
        except Exception:
            step_idx = idx
        title = str(raw_step.get("title", "")).strip()
        content = str(raw_step.get("content", "")).strip()
        steps.append(
            {
                "step_idx": step_idx,
                "title": title,
                "content": content,
            }
        )
        if title:
            titles.append(title)

    return {
        "model_name": model_name,
        "steps": steps,
        "step_count": len(steps),
        "titles": titles,
        "final_answer": final_answer,
    }


_CIRCLED_NUM_MAP = str.maketrans({
    "①": "1",
    "②": "2",
    "③": "3",
    "④": "4",
    "⑤": "5",
})


def _extract_choice_label(text: str) -> str:
    """Extract choice label such as 1-5/A-E from answer text when present."""
    if not text:
        return ""
    s = text.translate(_CIRCLED_NUM_MAP)

    # Prefer explicit answer cues first.
    explicit_patterns = [
        r"(?:정답|답)\s*[:：]?\s*(?:은|는)?\s*([1-5A-Ea-e])\s*(?:번|choice|option)?",
        r"(?:final\s*answer)\s*[:：]?\s*([1-5A-Ea-e])\b",
        r"\b([1-5A-Ea-e])\s*(?:번|choice|option)\b",
    ]
    for pat in explicit_patterns:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # If response is short, allow bare token.
    short = s.strip()
    if len(short) <= 6:
        m = re.fullmatch(r"[ \t()]*([1-5A-Ea-e])[ \t()]*", short)
        if m:
            return m.group(1).upper()

    return ""


def _is_correct_answer(pred_answer: str, gold_answer: str) -> bool:
    """Heuristic correctness check based on normalized exact/containment match."""
    pred_norm = _normalize_for_match(pred_answer)
    gold_norm = _normalize_for_match(gold_answer)
    if not pred_norm or not gold_norm:
        return False
    if pred_norm == gold_norm:
        return True
    if gold_norm in pred_norm:
        return True
    # Numeric fallback: if all gold numbers appear in prediction numbers
    gold_nums = re.findall(r"-?\d+(?:\.\d+)?", gold_answer)
    pred_nums = re.findall(r"-?\d+(?:\.\d+)?", pred_answer)
    if gold_nums and pred_nums and set(gold_nums).issubset(set(pred_nums)):
        return True
    return False


def _is_correct_answer_recheck(pred_answer: str, gold_answer: str) -> bool:
    """Extended correctness check: choice-label match + legacy heuristic."""
    gold_choice = _extract_choice_label(gold_answer)
    pred_choice = _extract_choice_label(pred_answer)
    if gold_choice and pred_choice and gold_choice == pred_choice:
        return True
    return _is_correct_answer(pred_answer, gold_answer)


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _format_duration(seconds: float) -> str:
    seconds = int(max(0, round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _ensure_flat_results_header() -> None:
    """Upgrade results_flat.csv header in place when new columns are added."""
    global FLAT_HEADER_READY
    if FLAT_HEADER_READY:
        return

    with CSV_APPEND_LOCK:
        if FLAT_HEADER_READY:
            return
        if not os.path.exists(FLAT_RESULTS_PATH) or os.path.getsize(FLAT_RESULTS_PATH) == 0:
            FLAT_HEADER_READY = True
            return

        with open(FLAT_RESULTS_PATH, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            current_header = reader.fieldnames or []
            if current_header == FLAT_RESULTS_HEADER:
                FLAT_HEADER_READY = True
                return
            rows = list(reader)

        tmp_path = FLAT_RESULTS_PATH + ".tmp"
        with open(tmp_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FLAT_RESULTS_HEADER)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in FLAT_RESULTS_HEADER})
        os.replace(tmp_path, FLAT_RESULTS_PATH)
        FLAT_HEADER_READY = True


def _append_flat_result_row(row: dict) -> None:
    """Append one run-level row to results_flat.csv (create header if needed)."""
    _ensure_flat_results_header()
    with CSV_APPEND_LOCK:
        file_exists = os.path.exists(FLAT_RESULTS_PATH)
        write_header = (not file_exists) or os.path.getsize(FLAT_RESULTS_PATH) == 0
        with open(FLAT_RESULTS_PATH, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FLAT_RESULTS_HEADER)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def _append_raw_response_record(record: dict) -> bool:
    """Append one raw response record to task5_raw_responses.jsonl."""
    try:
        line = json.dumps(record, ensure_ascii=False)
    except Exception:
        return False
    with RAW_APPEND_LOCK:
        with open(RAW_RESPONSES_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    return True


def _compute_group_stats(
    step_counts: list[int],
    correct_flags: list[bool],
) -> dict:
    """Compute per-(model, temperature) summary stats."""
    step_arr = np.asarray(step_counts, dtype=float)
    acc_arr = np.asarray(correct_flags, dtype=float)

    step_mean = float(step_arr.mean()) if step_arr.size else 0.0
    step_std = float(step_arr.std(ddof=1)) if step_arr.size > 1 else 0.0

    accuracy_mean = float(acc_arr.mean()) if acc_arr.size else 0.0
    accuracy_std = float(acc_arr.std(ddof=1)) if acc_arr.size > 1 else 0.0

    # Normal approximation: p ± 1.96 * sqrt(p(1-p)/n)
    n = int(acc_arr.size)
    if n > 0:
        se = float(np.sqrt(max(0.0, accuracy_mean * (1.0 - accuracy_mean)) / n))
        ci_low = max(0.0, accuracy_mean - 1.96 * se)
        ci_high = min(1.0, accuracy_mean + 1.96 * se)
    else:
        ci_low, ci_high = 0.0, 0.0

    return {
        "step_mean": step_mean,
        "step_std": step_std,
        "accuracy_mean": accuracy_mean,
        "accuracy_std": accuracy_std,
        "accuracy_ci_95": {
            "lower": ci_low,
            "upper": ci_high,
        },
    }


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def _parse_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _parse_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_temp_key(value) -> str | None:
    try:
        return f"{float(value):.1f}"
    except Exception:
        return None


def _parse_titles_json(value: str) -> list[str]:
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    return []


def _load_existing_run_index(
    temperatures: tuple[float, ...],
    n_runs: int,
) -> dict[str, dict[tuple[str, str, int], dict]]:
    """Load existing flat CSV rows into an index keyed by problem/run slot.

    Returns:
      {
        problem_id: {
          (model_key, temp_key, run_index): {
            step_count, titles, is_correct, elapsed_s
          }
        }
      }
    """
    if not os.path.exists(FLAT_RESULTS_PATH):
        return {}
    _ensure_flat_results_header()

    allowed_temps = {f"{t:.1f}" for t in temperatures}
    allowed_models = set(MODELS.keys())
    max_run = int(n_runs)
    index: dict[str, dict[tuple[str, str, int], dict]] = {}

    with CSV_APPEND_LOCK:
        try:
            with open(FLAT_RESULTS_PATH, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    model = row.get("model", "")
                    temp_key = _normalize_temp_key(row.get("temperature"))
                    run_idx = _parse_int(row.get("run_index"), 0)
                    problem_id = row.get("problem_id", "")
                    if not problem_id or model not in allowed_models:
                        continue
                    if temp_key is None or temp_key not in allowed_temps:
                        continue
                    if run_idx < 1 or run_idx > max_run:
                        continue

                    key = (model, temp_key, run_idx)
                    slot = {
                        "step_count": _parse_int(row.get("step_count"), 0),
                        "titles": _parse_titles_json(row.get("titles", "")),
                        "is_correct": _parse_bool(row.get("correct", False)),
                        "is_correct_recheck": _parse_bool(
                            row.get("correct_recheck", row.get("correct", False))
                        ),
                        "elapsed_s": _parse_float(row.get("elapsed_time_seconds"), 0.0),
                        "final_answer": str(row.get("model_final_answer", "")).strip(),
                        "gold_answer": str(row.get("gold_answer", "")).strip(),
                        "gold_choice": str(row.get("gold_choice", "")).strip(),
                        "model_choice": str(row.get("model_choice", "")).strip(),
                        "parse_ok": _parse_bool(row.get("parse_ok", False)),
                        "raw_response_saved": _parse_bool(
                            row.get("raw_response_saved", False)
                        ),
                        "has_final_answer": bool(
                            str(row.get("model_final_answer", "")).strip()
                        ),
                    }
                    index.setdefault(problem_id, {})[key] = slot
        except FileNotFoundError:
            return {}
    return index


def _build_task5_answer_index_from_json() -> dict[tuple[str, str, str, int], dict]:
    """Index final answers from task5_<problem>.json for CSV backfill."""
    index: dict[tuple[str, str, str, int], dict] = {}
    pattern = os.path.join(OUT_DIR, "task5_*.json")
    for path in glob.glob(pattern):
        if path.endswith("task5_all_summary.json"):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        problem = data.get("problem", {})
        problem_id = str(problem.get("id", "")).strip()
        gold_answer = str(problem.get("gold_answer", "")).strip()
        same_temperature = data.get("same_temperature", {})
        if not problem_id or not isinstance(same_temperature, dict):
            continue

        for temp_key, temp_data in same_temperature.items():
            per_model = temp_data.get("per_model", {})
            if not isinstance(per_model, dict):
                continue
            for model_key, model_data in per_model.items():
                runs = model_data.get("raw_runs") or model_data.get("runs") or []
                for run in runs:
                    run_idx = _parse_int(run.get("run_idx"), 0)
                    if run_idx <= 0:
                        continue
                    final_answer = str(run.get("final_answer", "")).strip()
                    if not final_answer:
                        continue
                    slot_key = (problem_id, str(model_key), str(temp_key), run_idx)
                    index[slot_key] = {
                        "gold_answer": gold_answer,
                        "model_final_answer": final_answer,
                    }
    return index


def _count_problem_remaining_calls(
    existing_slots: dict[tuple[str, str, int], dict],
    expected_calls: int,
    rerun_empty_final: bool,
) -> int:
    """Count remaining calls for one problem under current resume policy."""
    if not existing_slots:
        return expected_calls
    if rerun_empty_final:
        done_count = sum(
            1 for slot in existing_slots.values() if bool(slot.get("has_final_answer", False))
        )
    else:
        done_count = len(existing_slots)
    return max(0, expected_calls - min(expected_calls, done_count))


def enrich_flat_results_csv() -> None:
    """Backfill gold/model answers and recheck correctness in results_flat.csv."""
    if not os.path.exists(FLAT_RESULTS_PATH):
        print(f"No flat CSV found: {FLAT_RESULTS_PATH}")
        return

    _ensure_flat_results_header()
    answer_index = _build_task5_answer_index_from_json()
    updated = 0
    filled_model_answer = 0
    filled_gold_answer = 0
    recalculated = 0

    with CSV_APPEND_LOCK:
        with open(FLAT_RESULTS_PATH, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for row in rows:
            problem_id = str(row.get("problem_id", "")).strip()
            model_key = str(row.get("model", "")).strip()
            temp_key = _normalize_temp_key(row.get("temperature")) or ""
            run_idx = _parse_int(row.get("run_index"), 0)
            slot_key = (problem_id, model_key, temp_key, run_idx)

            before = dict(row)

            if not row.get("gold_answer"):
                if problem_id in ALL_PROBLEMS:
                    row["gold_answer"] = str(ALL_PROBLEMS[problem_id].get("answer", "")).strip()
                elif slot_key in answer_index:
                    row["gold_answer"] = answer_index[slot_key]["gold_answer"]

            if not row.get("model_final_answer") and slot_key in answer_index:
                row["model_final_answer"] = answer_index[slot_key]["model_final_answer"]

            if row.get("gold_answer"):
                row["gold_choice"] = _extract_choice_label(row["gold_answer"])
            if row.get("model_final_answer"):
                row["model_choice"] = _extract_choice_label(row["model_final_answer"])

            if row.get("gold_answer") and row.get("model_final_answer"):
                row["correct_recheck"] = bool(
                    _is_correct_answer_recheck(row["model_final_answer"], row["gold_answer"])
                )
                recalculated += 1
            elif row.get("correct_recheck", "") == "":
                row["correct_recheck"] = row.get("correct", "")

            if row.get("parse_ok", "") == "":
                row["parse_ok"] = bool(str(row.get("model_final_answer", "")).strip())
            if row.get("raw_response_saved", "") == "":
                row["raw_response_saved"] = False

            if row != before:
                updated += 1
                if before.get("gold_answer", "") != row.get("gold_answer", "") and row.get("gold_answer"):
                    filled_gold_answer += 1
                if (
                    before.get("model_final_answer", "") != row.get("model_final_answer", "")
                    and row.get("model_final_answer")
                ):
                    filled_model_answer += 1

        tmp_path = FLAT_RESULTS_PATH + ".tmp"
        with open(tmp_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FLAT_RESULTS_HEADER)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in FLAT_RESULTS_HEADER})
        os.replace(tmp_path, FLAT_RESULTS_PATH)

    print("Flat CSV enrichment complete.")
    print(f"  rows_total={len(rows)}")
    print(f"  rows_updated={updated}")
    print(f"  gold_answer_filled={filled_gold_answer}")
    print(f"  model_final_answer_filled={filled_model_answer}")
    print(f"  correct_recheck_recomputed={recalculated}")


# ───────────────────────────────────────────────────────────────
# Similarity computation
# ───────────────────────────────────────────────────────────────

def compute_step_similarities(all_steps: list[list[dict]]) -> np.ndarray:
    """Compute pairwise cosine similarity between all steps across solutions.

    Args:
        all_steps: list of step-lists, one per solution
    Returns:
        similarity matrix (N_total x N_total)
    """
    texts = []
    for steps in all_steps:
        for step in steps:
            texts.append(f"{step['title']} {step['body']}")

    if not texts:
        return np.array([])

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=10000,
    )
    tfidf = vectorizer.fit_transform(texts)
    return cosine_similarity(tfidf)


def build_alignment_groups(
    all_steps: list[list[dict]],
    sim_matrix: np.ndarray,
    threshold: float = 0.3,
) -> list[dict]:
    """Group similar steps across different solutions using similarity threshold.

    Returns list of alignment groups with members from different solutions.
    """
    # Build flat index → (solution_idx, step_idx)
    flat_index = []
    for sol_idx, steps in enumerate(all_steps):
        for step_idx in range(len(steps)):
            flat_index.append((sol_idx, step_idx))

    n = len(flat_index)
    # Union-Find
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Only merge steps from *different* solutions
    for i in range(n):
        for j in range(i + 1, n):
            if flat_index[i][0] != flat_index[j][0]:  # different solution
                if sim_matrix[i][j] >= threshold:
                    union(i, j)

    # Collect groups
    clusters = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(i)

    groups = []
    for members_flat in clusters.values():
        solutions_in_group = set(flat_index[m][0] for m in members_flat)
        if len(solutions_in_group) < 2:
            continue
        members = []
        for m in members_flat:
            sol_idx, step_idx = flat_index[m]
            step = all_steps[sol_idx][step_idx]
            members.append({
                "solution_idx": sol_idx,
                "step_idx": step_idx,
                "title": step["title"],
                "body_preview": step["body"][:80],
            })
        groups.append({"members": members, "n_solutions": len(solutions_in_group)})

    groups.sort(key=lambda g: g["n_solutions"], reverse=True)
    return groups


# ───────────────────────────────────────────────────────────────
# Task runners
# ───────────────────────────────────────────────────────────────

def task1_prompt_versions(prob_key: str = "prob1"):
    """Task 1: Compare prompt versions for step extraction quality."""
    print("\n" + "=" * 70)
    print("TASK 1: Prompt version comparison")
    print("=" * 70)

    prob = PROBLEMS[prob_key]
    print(f"Problem: {prob['label']}")
    print(f"Answer: {prob['answer']}")
    print(f"Model: {DEFAULT_MODEL}")
    print()

    results = {}
    for ver_name, template in PROMPT_VERSIONS.items():
        prompt = _format_prompt_for_problem(template, prob)
        print(f"--- {ver_name} ---")
        print(f"Calling {DEFAULT_MODEL} ...")
        t0 = time.time()
        response = call_llm(prompt, DEFAULT_MODEL, temperature=0.3)
        elapsed = time.time() - t0
        print(f"  Response received ({elapsed:.1f}s, {len(response)} chars)")

        steps = extract_steps(response, ver_name)
        print(f"  Extracted {len(steps)} steps:")
        for i, s in enumerate(steps):
            print(f"    [{i+1}] {s['title']}: {s['body'][:60]}...")

        results[ver_name] = {
            "raw_response": response,
            "steps": steps,
            "elapsed_s": elapsed,
            "n_steps": len(steps),
        }
        print()

    out_path = os.path.join(OUT_DIR, f"task1_{prob_key}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {out_path}")
    return results


def task2_step_similarity(prob_key: str = "prob1"):
    """Task 2: Cosine-similarity based step alignment verification."""
    print("\n" + "=" * 70)
    print("TASK 2: Step similarity & alignment")
    print("=" * 70)

    prob = PROBLEMS[prob_key]
    template = PROMPT_VERSIONS["v3_titled"]
    prompt = _format_prompt_for_problem(template, prob)

    # Get 3 solutions from the same model with temperature=0.3 for consistency
    # (to test alignment, not diversity)
    models_to_use = list(MODELS.keys())[:3]
    all_steps = []
    solution_labels = []

    for model_key in models_to_use:
        model_id = MODELS[model_key]
        print(f"Calling {model_key} ({model_id}) ...")
        t0 = time.time()
        response = call_llm(prompt, model_id, temperature=0.3)
        elapsed = time.time() - t0
        steps = extract_steps(response, "v3_titled")
        all_steps.append(steps)
        solution_labels.append(model_key)
        print(f"  {model_key}: {len(steps)} steps ({elapsed:.1f}s)")
        for i, s in enumerate(steps):
            print(f"    [{i+1}] {s['title']}")

    print(f"\nComputing cosine similarity ...")
    sim_matrix = compute_step_similarities(all_steps)

    print(f"\nBuilding alignment groups (threshold=0.3) ...")
    groups = build_alignment_groups(all_steps, sim_matrix, threshold=0.3)

    print(f"\nFound {len(groups)} alignment groups:")
    for gi, g in enumerate(groups):
        print(f"  Group {gi+1} ({g['n_solutions']} solutions):")
        for m in g["members"]:
            label = solution_labels[m["solution_idx"]]
            print(f"    - [{label}] Step {m['step_idx']+1}: {m['title']}")

    # Print pairwise similarity between all step pairs (cross-solution)
    print("\n--- Cross-solution step similarities ---")
    flat_index = []
    for sol_idx, steps in enumerate(all_steps):
        for step_idx in range(len(steps)):
            flat_index.append((sol_idx, step_idx))

    for i in range(len(flat_index)):
        for j in range(i + 1, len(flat_index)):
            si, sti = flat_index[i]
            sj, stj = flat_index[j]
            if si == sj:
                continue
            score = sim_matrix[i][j]
            if score >= 0.15:
                label_i = solution_labels[si]
                label_j = solution_labels[sj]
                title_i = all_steps[si][sti]["title"]
                title_j = all_steps[sj][stj]["title"]
                print(f"  {score:.3f}  [{label_i}] {title_i}  <->  [{label_j}] {title_j}")

    # Save
    out_path = os.path.join(OUT_DIR, f"task2_{prob_key}.json")
    result = {
        "solutions": {
            label: [{"title": s["title"], "body": s["body"]} for s in steps]
            for label, steps in zip(solution_labels, all_steps)
        },
        "alignment_groups": groups,
        "sim_matrix": sim_matrix.tolist() if sim_matrix.size else [],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")
    return result


def task3_single_model_diversity(prob_key: str = "prob1", n_runs: int = 5):
    """Task 3: Single model, temperature=1.0, multiple runs → diversity check."""
    print("\n" + "=" * 70)
    print("TASK 3: Single-model diversity (temperature=1.0)")
    print("=" * 70)

    prob = PROBLEMS[prob_key]
    template = PROMPT_VERSIONS["v3_titled"]
    prompt = _format_prompt_for_problem(template, prob)
    model_id = MODELS[list(MODELS.keys())[0]]

    print(f"Problem: {prob['label']}")
    print(f"Model: {model_id}, temperature=1.0, {n_runs} runs")
    print()

    all_steps = []
    run_data = []

    for run_idx in range(n_runs):
        print(f"Run {run_idx+1}/{n_runs} ...")
        t0 = time.time()
        response = call_llm(prompt, model_id, temperature=1.0)
        elapsed = time.time() - t0
        steps = extract_steps(response, "v3_titled")
        all_steps.append(steps)
        run_data.append({
            "raw_response": response,
            "steps": [{"title": s["title"], "body": s["body"]} for s in steps],
            "elapsed_s": elapsed,
        })
        print(f"  {len(steps)} steps ({elapsed:.1f}s): {[s['title'] for s in steps]}")

    # Compute pairwise solution-level similarity
    print("\n--- Pairwise solution similarity ---")
    solution_texts = [
        " ".join(f"{s['title']} {s['body']}" for s in steps) for steps in all_steps
    ]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=10000)
    tfidf = vectorizer.fit_transform(solution_texts)
    sol_sim = cosine_similarity(tfidf)

    for i, j in combinations(range(n_runs), 2):
        print(f"  Run {i+1} vs Run {j+1}: {sol_sim[i][j]:.3f}")

    avg_sim = np.mean([sol_sim[i][j] for i, j in combinations(range(n_runs), 2)])
    print(f"\n  Average pairwise similarity: {avg_sim:.3f}")
    print(f"  (Lower = more diverse)")

    # Step-level alignment
    sim_matrix = compute_step_similarities(all_steps)
    groups = build_alignment_groups(all_steps, sim_matrix, threshold=0.3)
    print(f"\n  Alignment groups found: {len(groups)}")
    for gi, g in enumerate(groups):
        members_str = [
            "Run{}-Step{}:{}".format(m["solution_idx"]+1, m["step_idx"]+1, m["title"])
            for m in g["members"]
        ]
        print(f"    Group {gi+1}: {members_str}")

    out_path = os.path.join(OUT_DIR, f"task3_{prob_key}.json")
    result = {
        "model": model_id,
        "temperature": 1.0,
        "n_runs": n_runs,
        "runs": run_data,
        "solution_similarity_matrix": sol_sim.tolist(),
        "avg_pairwise_similarity": float(avg_sim),
        "alignment_groups": groups,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")
    return result


def task4_multi_model_diversity(prob_key: str = "prob1"):
    """Task 4: Multiple different models → diversity check."""
    print("\n" + "=" * 70)
    print("TASK 4: Multi-model diversity")
    print("=" * 70)

    prob = PROBLEMS[prob_key]
    template = PROMPT_VERSIONS["v3_titled"]
    prompt = _format_prompt_for_problem(template, prob)

    print(f"Problem: {prob['label']}")
    print(f"Models: {list(MODELS.keys())}")
    print()

    all_steps = []
    model_labels = []
    run_data = {}

    for model_key, model_id in MODELS.items():
        print(f"Calling {model_key} ({model_id}) ...")
        t0 = time.time()
        response = call_llm(prompt, model_id, temperature=0.7)
        elapsed = time.time() - t0
        steps = extract_steps(response, "v3_titled")
        all_steps.append(steps)
        model_labels.append(model_key)
        run_data[model_key] = {
            "model_id": model_id,
            "raw_response": response,
            "steps": [{"title": s["title"], "body": s["body"]} for s in steps],
            "elapsed_s": elapsed,
        }
        print(f"  {model_key}: {len(steps)} steps ({elapsed:.1f}s)")
        for i, s in enumerate(steps):
            print(f"    [{i+1}] {s['title']}")

    # Solution-level similarity
    print("\n--- Pairwise solution similarity ---")
    solution_texts = [
        " ".join(f"{s['title']} {s['body']}" for s in steps) for steps in all_steps
    ]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=10000)
    tfidf = vectorizer.fit_transform(solution_texts)
    sol_sim = cosine_similarity(tfidf)

    n_models = len(model_labels)
    for i, j in combinations(range(n_models), 2):
        print(f"  {model_labels[i]} vs {model_labels[j]}: {sol_sim[i][j]:.3f}")

    avg_sim = np.mean([sol_sim[i][j] for i, j in combinations(range(n_models), 2)])
    print(f"\n  Average pairwise similarity: {avg_sim:.3f}")

    # Step alignment
    sim_matrix = compute_step_similarities(all_steps)
    groups = build_alignment_groups(all_steps, sim_matrix, threshold=0.3)
    print(f"\n  Alignment groups found: {len(groups)}")
    for gi, g in enumerate(groups):
        members_str = [
            f"{model_labels[m['solution_idx']]}-Step{m['step_idx']+1}:{m['title']}"
            for m in g["members"]
        ]
        print(f"    Group {gi+1}: {members_str}")

    out_path = os.path.join(OUT_DIR, f"task4_{prob_key}.json")
    result = {
        "models": run_data,
        "solution_similarity_matrix": sol_sim.tolist(),
        "model_labels": model_labels,
        "avg_pairwise_similarity": float(avg_sim),
        "alignment_groups": groups,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")
    return result


def _resolve_task5_problem_items(prob_key: str) -> list[tuple[str, dict]]:
    """Resolve task5 problem targets from alias/prob_id/all."""
    if prob_key == "all":
        return list(ALL_PROBLEMS.items())
    if prob_key in PROBLEMS:
        prob = PROBLEMS[prob_key]
        return [(prob["id"], prob)]
    if prob_key in ALL_PROBLEMS:
        return [(prob_key, ALL_PROBLEMS[prob_key])]
    raise ValueError(
        f"Unknown problem key '{prob_key}'. Use prob1/prob2/prob22, a prob_id, or all."
    )


def _task5_result_path(problem_tag: str) -> str:
    return os.path.join(OUT_DIR, f"task5_{problem_tag}.json")


def _load_completed_task5_result(
    problem_tag: str,
    temperatures: tuple[float, ...],
    n_runs: int,
) -> dict | None:
    """Load task5 result only if it looks complete for current settings."""
    path = _task5_result_path(problem_tag)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    expected = len(temperatures) * len(MODELS) * n_runs
    meta = data.get("meta", {})
    actual = meta.get("n_actual_calls")
    recorded_expected = meta.get("n_expected_calls")
    if actual != expected:
        return None
    if recorded_expected is not None and recorded_expected != expected:
        return None
    return data


def _task5_summary_entry_from_result(result: dict) -> dict:
    overall = result.get("cross_temperature", {}).get("overall", {})
    problem = result.get("problem", {})
    meta = result.get("meta", {})
    return {
        "problem_id": problem.get("id"),
        "avg_step_count": overall.get("avg_step_count", 0.0),
        "accuracy": overall.get("accuracy", 0.0),
        "n_samples": overall.get("n_samples", 0),
        "elapsed_seconds": meta.get("elapsed_seconds", 0.0),
        "source": "existing" if meta else "computed",
    }


def _call_llm_with_retry(
    prompt: str,
    model_id: str,
    temperature: float,
    messages: list[dict] | None = None,
    system_prompt: str | None = None,
    response_format: dict | None = None,
    max_retries: int = 3,
    base_backoff_s: float = 2.0,
) -> str:
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return call_llm(
                prompt=prompt,
                model=model_id,
                temperature=temperature,
                messages=messages,
                system_prompt=system_prompt,
                response_format=response_format,
            )
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                wait_s = base_backoff_s * (2 ** (attempt - 1))
                print(
                    f"      [retry] model={model_id} temp={temperature} "
                    f"attempt={attempt}/{max_retries} err={type(e).__name__} wait={wait_s:.1f}s"
                )
                time.sleep(wait_s)
    raise RuntimeError(
        f"LLM call failed after retries (model={model_id}, temp={temperature}): {last_err}"
    ) from last_err


def _run_task5_for_one_problem(
    prob: dict,
    problem_tag: str,
    temperatures: tuple[float, ...],
    n_runs: int,
    intra_problem_workers: int = 1,
    rerun_empty_final: bool = False,
    print_run_logs: bool = True,
    progress_bar=None,
    global_state: dict | None = None,
    state_lock: threading.Lock | None = None,
    existing_runs: dict[tuple[str, str, int], dict] | None = None,
) -> dict:
    """Run task5 statistics for one problem and save per-problem JSON."""
    started_at = _utc_now_iso()
    problem_t0 = time.time()
    prompt_version = "v4_flow_titled"
    template = PROMPT_VERSIONS[prompt_version]
    has_problem_image = bool(
        str(prob.get("prob_img_abs_path", "")).strip()
        and os.path.exists(str(prob.get("prob_img_abs_path", "")).strip())
    )
    problem_for_prompt = (
        "(문제 이미지는 함께 제공됩니다. 이미지 내용을 기준으로 풀이하세요.)"
        if has_problem_image
        else prob["text"]
    )
    answer_format_rule = prob.get("answer_format_rule", "")
    problem_type = prob.get("prob_type_en", "Unknown")
    prob_type_kr = prob.get("prob_type", "단답형")
    prompt = _format_prompt_for_problem(
        template=template,
        prob=prob,
        problem_text=problem_for_prompt,
    )
    task_messages = _build_task5_user_messages(prompt, prob, require_image=True)

    print(f"Problem: {prob['label']}")
    print(f"Problem type: {problem_type}")
    print(f"Input mode: {'image' if has_problem_image else 'text'}")
    if answer_format_rule:
        print(f"Answer rule: {answer_format_rule}")
    print("Structured temperature: 0.7 (fixed)")
    print(f"Gold answer: {prob['answer']}")
    print(f"Prompt version: {prompt_version}")
    print(f"Models ({len(MODELS)}): {list(MODELS.keys())}")
    print(f"Temperatures: {list(temperatures)}")
    print(f"Runs per temperature/model: {n_runs}")
    print()

    # same_temperature[temp][model] -> run stats
    same_temperature: dict[str, dict] = {}

    # aggregate across all temperature/model/run
    all_step_counts = []
    all_titles = Counter()
    all_correct = 0
    all_total = 0
    problem_calls = 0
    api_calls_this_run = 0
    resumed_runs = 0

    intra_problem_workers = max(1, int(intra_problem_workers))
    if intra_problem_workers > 1:
        print(f"Intra-problem workers: {intra_problem_workers} (slot-level)")

    slot_results: dict[tuple[str, str, int], dict] = {}
    pending_slots = []

    for temp in temperatures:
        temp_key = f"{temp:.1f}"
        for model_key, model_id in MODELS.items():
            for run_idx in range(n_runs):
                run_number = run_idx + 1
                slot_key = (model_key, temp_key, run_number)
                slot_cached = (
                    existing_runs.get(slot_key) if existing_runs is not None else None
                )
                can_resume_from_csv = slot_cached is not None
                if (
                    can_resume_from_csv
                    and rerun_empty_final
                    and not bool(slot_cached.get("has_final_answer", False))
                ):
                    can_resume_from_csv = False

                if can_resume_from_csv:
                    cached = slot_cached
                    slot_results[(temp_key, model_key, run_number)] = {
                        "temp": temp,
                        "temp_key": temp_key,
                        "model_key": model_key,
                        "model_id": model_id,
                        "run_number": run_number,
                        "step_count": _parse_int(cached.get("step_count"), 0),
                        "titles": list(cached.get("titles", [])),
                        "is_correct": _parse_bool(cached.get("is_correct", False)),
                        "is_correct_recheck": _parse_bool(
                            cached.get("is_correct_recheck", cached.get("is_correct", False))
                        ),
                        "elapsed_s": _parse_float(cached.get("elapsed_s"), 0.0),
                        "final_answer": str(cached.get("final_answer", "")).strip(),
                        "gold_answer": str(cached.get("gold_answer", prob["answer"])).strip(),
                        "gold_choice": str(cached.get("gold_choice", "")).strip(),
                        "model_choice": str(cached.get("model_choice", "")).strip(),
                        "parse_ok": _parse_bool(
                            cached.get(
                                "parse_ok",
                                bool(str(cached.get("final_answer", "")).strip()),
                            )
                        ),
                        "raw_response_saved": _parse_bool(
                            cached.get("raw_response_saved", False)
                        ),
                        "resumed_from_csv": True,
                        "error": "",
                    }
                    resumed_runs += 1
                else:
                    pending_slots.append(
                        {
                            "temp": temp,
                            "temp_key": temp_key,
                            "model_key": model_key,
                            "model_id": model_id,
                            "run_number": run_number,
                        }
                    )

    def _execute_slot(slot: dict) -> dict:
        t0 = time.time()
        error_message = ""
        response = ""
        raw_response_saved = False
        parse_ok = False
        try:
            response = _call_llm_with_retry(
                prompt=prompt,
                model_id=slot["model_id"],
                temperature=0.7,
                messages=task_messages,
                system_prompt=prompt,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "model_solution",
                        "strict": True,
                        "schema": make_solution_schema(prob_type_kr),
                    },
                },
            )
            call_elapsed = time.time() - t0
            raw_response_saved = _append_raw_response_record(
                {
                    "recorded_at_utc": _utc_now_iso(),
                    "task": "5",
                    "prompt_version": prompt_version,
                    "problem_id": prob["id"],
                    "problem_tag": problem_tag,
                    "model": slot["model_key"],
                    "model_id": slot["model_id"],
                    "temperature": slot["temp"],
                    "run_index": slot["run_number"],
                    "response_text": response,
                    "elapsed_time_seconds": round(call_elapsed, 6),
                    "error": "",
                }
            )

            parsed = _parse_model_solution_json(response)
            if parsed is not None:
                step_count = int(parsed["step_count"])
                titles = list(parsed["titles"])
                final_answer = str(parsed["final_answer"]).strip()
                parse_ok = bool(step_count > 0 and final_answer)
            else:
                steps = extract_steps(response, prompt_version)
                step_only = [s for s in steps if s.get("type") != "final"]
                step_count = len(step_only)
                titles = [s["title"] for s in step_only]
                final_answer = _extract_final_answer(response, steps)
                parse_ok = bool(steps) or bool(final_answer.strip())

            is_correct = _is_correct_answer(final_answer, prob["answer"])
            is_correct_recheck = _is_correct_answer_recheck(final_answer, prob["answer"])
            gold_choice = _extract_choice_label(prob["answer"])
            model_choice = _extract_choice_label(final_answer)
        except Exception as e:
            call_elapsed = time.time() - t0
            step_count = 0
            titles = []
            final_answer = ""
            is_correct = False
            is_correct_recheck = False
            gold_choice = _extract_choice_label(prob["answer"])
            model_choice = ""
            error_message = f"{type(e).__name__}: {e}"
            raw_response_saved = _append_raw_response_record(
                {
                    "recorded_at_utc": _utc_now_iso(),
                    "task": "5",
                    "prompt_version": prompt_version,
                    "problem_id": prob["id"],
                    "problem_tag": problem_tag,
                    "model": slot["model_key"],
                    "model_id": slot["model_id"],
                    "temperature": slot["temp"],
                    "run_index": slot["run_number"],
                    "response_text": response,
                    "elapsed_time_seconds": round(call_elapsed, 6),
                    "error": error_message,
                }
            )

        return {
            "temp": slot["temp"],
            "temp_key": slot["temp_key"],
            "model_key": slot["model_key"],
            "model_id": slot["model_id"],
            "run_number": slot["run_number"],
            "step_count": step_count,
            "titles": titles,
            "is_correct": is_correct,
            "is_correct_recheck": is_correct_recheck,
            "elapsed_s": call_elapsed,
            "final_answer": final_answer,
            "gold_answer": prob["answer"],
            "gold_choice": gold_choice,
            "model_choice": model_choice,
            "parse_ok": parse_ok,
            "raw_response_saved": raw_response_saved,
            "resumed_from_csv": False,
            "error": error_message,
        }

    def _record_fresh_slot_result(slot_result: dict) -> None:
        nonlocal api_calls_this_run
        api_calls_this_run += 1

        if slot_result["error"]:
            print(
                f"      [run_error] prob={problem_tag} model={slot_result['model_key']} "
                f"temp={slot_result['temp_key']} run={slot_result['run_number']}/{n_runs} "
                f"err={slot_result['error']}"
            )

        if global_state is not None:
            if state_lock is not None:
                with state_lock:
                    global_state["done_calls"] += 1
                    done = global_state["done_calls"]
            else:
                global_state["done_calls"] += 1
                done = global_state["done_calls"]
            global_elapsed = time.time() - global_state["start_time"]
            avg = global_elapsed / done if done else 0.0
            eta = avg * (global_state["total_calls"] - done)
        else:
            eta = 0.0

        if progress_bar is not None:
            progress_bar.update(1)
            progress_bar.set_postfix_str(
                f"prob={problem_tag} temp={slot_result['temp_key']} "
                f"model={slot_result['model_key']} run={slot_result['run_number']}/{n_runs} "
                f"eta={_format_duration(eta)}"
            )

        _append_flat_result_row(
            {
                "model": slot_result["model_key"],
                "temperature": slot_result["temp"],
                "run_index": slot_result["run_number"],
                "problem_id": prob["id"],
                "step_count": slot_result["step_count"],
                "correct": bool(slot_result["is_correct"]),
                "correct_recheck": bool(slot_result["is_correct_recheck"]),
                "gold_answer": slot_result["gold_answer"],
                "model_final_answer": slot_result["final_answer"],
                "gold_choice": slot_result["gold_choice"],
                "model_choice": slot_result["model_choice"],
                "parse_ok": bool(slot_result["parse_ok"]),
                "raw_response_saved": bool(slot_result["raw_response_saved"]),
                "titles": json.dumps(slot_result["titles"], ensure_ascii=False),
                "elapsed_time_seconds": round(slot_result["elapsed_s"], 6),
            }
        )

    if pending_slots:
        if intra_problem_workers > 1 and len(pending_slots) > 1:
            with ThreadPoolExecutor(max_workers=min(intra_problem_workers, len(pending_slots))) as executor:
                futures = {executor.submit(_execute_slot, slot): slot for slot in pending_slots}
                for fut in as_completed(futures):
                    slot_result = fut.result()
                    key = (
                        slot_result["temp_key"],
                        slot_result["model_key"],
                        slot_result["run_number"],
                    )
                    slot_results[key] = slot_result
                    _record_fresh_slot_result(slot_result)
        else:
            for slot in pending_slots:
                slot_result = _execute_slot(slot)
                key = (
                    slot_result["temp_key"],
                    slot_result["model_key"],
                    slot_result["run_number"],
                )
                slot_results[key] = slot_result
                _record_fresh_slot_result(slot_result)

    for temp in temperatures:
        temp_key = f"{temp:.1f}"
        print(f"\n--- Temperature {temp_key} ---")
        per_model = {}
        temp_step_counts = []
        temp_title_counter = Counter()
        temp_correct = 0
        temp_total = 0

        for model_key, model_id in MODELS.items():
            print(f"  Model: {model_key} ({model_id})")
            runs = []
            model_step_counts = []
            model_title_counter = Counter()
            model_correct = 0
            model_correct_flags = []

            for run_idx in range(n_runs):
                run_number = run_idx + 1
                slot_result = slot_results[(temp_key, model_key, run_number)]
                step_count = slot_result["step_count"]
                titles = slot_result["titles"]
                is_correct = bool(slot_result["is_correct"])
                is_correct_recheck = bool(slot_result.get("is_correct_recheck", is_correct))
                call_elapsed = slot_result["elapsed_s"]
                final_answer = slot_result["final_answer"]
                parse_ok = bool(slot_result.get("parse_ok", bool(final_answer.strip())))
                raw_response_saved = bool(slot_result.get("raw_response_saved", False))
                resumed_from_csv = bool(slot_result["resumed_from_csv"])
                error_message = slot_result["error"]

                runs.append(
                    {
                        "run_idx": run_number,
                        "elapsed_s": call_elapsed,
                        "step_count": step_count,
                        "step_titles": titles,
                        "final_answer": final_answer,
                        "is_correct": is_correct,
                        "is_correct_recheck": is_correct_recheck,
                        "parse_ok": parse_ok,
                        "raw_response_saved": raw_response_saved,
                        "resumed_from_csv": resumed_from_csv,
                        "error": error_message,
                    }
                )

                model_step_counts.append(step_count)
                model_title_counter.update(titles)
                model_correct += int(is_correct)
                model_correct_flags.append(bool(is_correct))

                temp_step_counts.append(step_count)
                temp_title_counter.update(titles)
                temp_correct += int(is_correct)
                temp_total += 1

                all_step_counts.append(step_count)
                all_titles.update(titles)
                all_correct += int(is_correct)
                all_total += 1
                problem_calls += 1

                if print_run_logs:
                    mode = "resume" if resumed_from_csv else "run"
                    print(
                        f"    [{mode}] Run {run_number}/{n_runs}: "
                        f"steps={step_count}, correct={is_correct}, {call_elapsed:.1f}s"
                    )

            model_stats = _compute_group_stats(
                step_counts=model_step_counts,
                correct_flags=model_correct_flags,
            )
            model_accuracy = model_stats["accuracy_mean"]
            model_avg_steps = model_stats["step_mean"]
            per_model[model_key] = {
                "model_id": model_id,
                "runs": runs,
                "raw_runs": runs,
                "avg_step_count": model_avg_steps,
                "title_frequency": dict(model_title_counter),
                "accuracy": model_accuracy,
                "step_mean": model_stats["step_mean"],
                "step_std": model_stats["step_std"],
                "accuracy_mean": model_stats["accuracy_mean"],
                "accuracy_std": model_stats["accuracy_std"],
                "accuracy_ci_95": model_stats["accuracy_ci_95"],
                "raw_title_counts": dict(model_title_counter),
            }
            if not print_run_logs:
                print(
                    f"    avg_step={model_avg_steps:.2f}, "
                    f"accuracy={model_accuracy:.3f}, "
                    f"acc_ci95=[{model_stats['accuracy_ci_95']['lower']:.3f}, "
                    f"{model_stats['accuracy_ci_95']['upper']:.3f}]"
                )

        same_temperature[temp_key] = {
            "per_model": per_model,
            "temperature_summary": {
                "avg_step_count_all_models": (
                    float(np.mean(temp_step_counts)) if temp_step_counts else 0.0
                ),
                "title_frequency_all_models": dict(temp_title_counter),
                "accuracy_all_models": (temp_correct / temp_total) if temp_total else 0.0,
                "n_samples": temp_total,
            },
        }

    cross_temperature = {
        "per_temperature": {
            temp_key: same_temperature[temp_key]["temperature_summary"]
            for temp_key in same_temperature
        },
        "overall": {
            "avg_step_count": float(np.mean(all_step_counts)) if all_step_counts else 0.0,
            "title_frequency": dict(all_titles),
            "accuracy": (all_correct / all_total) if all_total else 0.0,
            "n_samples": all_total,
        },
    }

    result = {
        "task": "5",
        "meta": {
            "started_at_utc": started_at,
            "ended_at_utc": _utc_now_iso(),
            "elapsed_seconds": round(time.time() - problem_t0, 3),
            "problem_tag": problem_tag,
            "n_expected_calls": len(temperatures) * len(MODELS) * n_runs,
            "n_actual_calls": problem_calls,
            "n_api_calls_this_run": api_calls_this_run,
            "n_resumed_runs_from_flat_csv": resumed_runs,
            "raw_responses_jsonl": RAW_RESPONSES_PATH,
        },
        "problem": {
            "id": prob["id"],
            "label": prob["label"],
            "gold_answer": prob["answer"],
        },
        "prompt_version": prompt_version,
        "temperatures": list(temperatures),
        "n_runs_per_temperature_model": n_runs,
        "models": MODELS,
        "same_temperature": same_temperature,
        "cross_temperature": cross_temperature,
    }

    out_path = os.path.join(OUT_DIR, f"task5_{problem_tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n--- Task 5 Summary ---")
    for temp_key, temp_data in same_temperature.items():
        summary = temp_data["temperature_summary"]
        print(
            f"  Temp {temp_key}: avg_step={summary['avg_step_count_all_models']:.2f}, "
            f"accuracy={summary['accuracy_all_models']:.3f}, n={summary['n_samples']}"
        )
    overall = cross_temperature["overall"]
    print(
        f"  Overall: avg_step={overall['avg_step_count']:.2f}, "
        f"accuracy={overall['accuracy']:.3f}, n={overall['n_samples']}"
    )
    print(f"\nResults saved to {out_path}")
    return result


def task5_temperature_model_stats(
    prob_key: str = "prob1",
    temperatures: tuple[float, ...] = (0.7,),
    n_runs: int = 3,
    max_workers: int = 1,
    resume: bool = True,
    rerun_empty_final: bool = False,
):
    """Task 5: v4 prompt 기반 temperature/model 통계."""
    print("\n" + "=" * 70)
    print("TASK 5: Temperature/model statistics (v4 prompt)")
    print("=" * 70)

    max_workers = max(1, int(max_workers))
    problem_items = _resolve_task5_problem_items(prob_key)
    n_problems_total = len(problem_items)
    calls_per_problem = len(temperatures) * len(MODELS) * n_runs
    existing_run_index = _load_existing_run_index(temperatures, n_runs)

    completed_results = {}
    pending_items = []
    if resume:
        for problem_tag, prob in problem_items:
            problem_slots = existing_run_index.get(prob["id"], {})
            remaining_for_problem = _count_problem_remaining_calls(
                existing_slots=problem_slots,
                expected_calls=calls_per_problem,
                rerun_empty_final=rerun_empty_final,
            )
            existing = _load_completed_task5_result(problem_tag, temperatures, n_runs)
            if existing is not None and remaining_for_problem == 0:
                completed_results[problem_tag] = existing
            else:
                pending_items.append((problem_tag, prob))
    else:
        pending_items = list(problem_items)

    n_skipped = len(completed_results)
    n_pending = len(pending_items)
    remaining_calls_by_problem = {}
    for problem_tag, prob in pending_items:
        existing_slots = existing_run_index.get(prob["id"], {})
        remaining_calls_by_problem[problem_tag] = _count_problem_remaining_calls(
            existing_slots=existing_slots,
            expected_calls=calls_per_problem,
            rerun_empty_final=rerun_empty_final,
        )
    total_calls = sum(remaining_calls_by_problem.values())
    task_started_at = _utc_now_iso()
    task_t0 = time.time()
    print(
        f"Target problems: {n_problems_total} | "
        f"Pending: {n_pending}, Skipped(completed): {n_skipped} | "
        f"Remaining calls: {total_calls} "
        f"({len(temperatures)} temps x {len(MODELS)} models x {n_runs} runs)"
    )
    if n_pending > 1:
        print(f"Workers: {min(max_workers, n_pending)}")
    if n_pending == 1 and max_workers > 1:
        print(f"Workers: 1 (problem-level), {max_workers} (slot-level for final problem)")
    if rerun_empty_final:
        print("Task5 mode: rerun slots with empty model_final_answer")

    print_run_logs = n_pending == 1 and n_problems_total == 1
    last_result = {}
    all_problem_summary = {
        tag: _task5_summary_entry_from_result(result)
        for tag, result in completed_results.items()
    }
    global_state = {
        "start_time": time.time(),
        "done_calls": 0,
        "total_calls": total_calls,
    }
    state_lock = threading.Lock()
    problem_bar = None
    progress_bar = None
    if tqdm:
        if n_problems_total > 1:
            problem_bar = tqdm(total=n_problems_total, desc="Problems", unit="prob", position=0)
            if n_skipped:
                problem_bar.update(n_skipped)
                problem_bar.set_postfix_str(f"skipped={n_skipped}")
            if total_calls > 0:
                progress_bar = tqdm(total=total_calls, desc="Calls", unit="call", position=1)
        elif total_calls > 0:
            progress_bar = tqdm(total=total_calls, desc="Task5", unit="call", position=0)

    if n_pending == 0:
        print("All target problems already completed. Nothing to run.")
        if progress_bar is not None:
            progress_bar.close()
        if problem_bar is not None:
            problem_bar.close()
        if n_problems_total > 1:
            summary = {
                "task": "5",
                "scope": "all_problems",
                "meta": {
                    "started_at_utc": task_started_at,
                    "ended_at_utc": _utc_now_iso(),
                    "elapsed_seconds": round(time.time() - task_t0, 3),
                    "resume_enabled": resume,
                    "rerun_empty_final": rerun_empty_final,
                },
                "n_problems": n_problems_total,
                "n_pending": 0,
                "n_skipped_completed": n_skipped,
                "temperatures": list(temperatures),
                "n_models": len(MODELS),
                "n_runs_per_temperature_model": n_runs,
                "expected_total_calls": n_problems_total * calls_per_problem,
                "actual_total_calls": n_problems_total * calls_per_problem,
                "executed_calls_this_run": 0,
                "per_problem_overall_summary": all_problem_summary,
            }
            summary_path = os.path.join(OUT_DIR, "task5_all_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"All-problem summary saved to {summary_path}")
            return summary
        only_tag = problem_items[0][0]
        return completed_results.get(only_tag, {})

    def _record_problem_result(problem_tag: str, result: dict) -> None:
        nonlocal last_result
        last_result = result
        entry = _task5_summary_entry_from_result(result)
        entry["source"] = "computed"
        all_problem_summary[problem_tag] = entry

        done = global_state["done_calls"]
        elapsed = time.time() - global_state["start_time"]
        avg = elapsed / done if done else 0.0
        eta = avg * (global_state["total_calls"] - done)
        print(
            f"Progress: {done}/{global_state['total_calls']} calls | "
            f"elapsed={_format_duration(elapsed)} | eta={_format_duration(eta)}"
        )
        if problem_bar is not None:
            problem_bar.update(1)
            processed_problems = len(all_problem_summary)
            prob_elapsed = time.time() - task_t0
            avg_prob = prob_elapsed / processed_problems if processed_problems else 0.0
            prob_eta = avg_prob * (n_problems_total - processed_problems)
            problem_bar.set_postfix_str(
                f"last={problem_tag} eta={_format_duration(prob_eta)}"
            )

    effective_workers = min(max_workers, n_pending)
    if effective_workers > 1:
        print(f"Running in parallel with {effective_workers} workers (problem-level).")
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                executor.submit(
                    _run_task5_for_one_problem,
                    prob=prob,
                    problem_tag=problem_tag,
                    temperatures=temperatures,
                    n_runs=n_runs,
                    intra_problem_workers=1,
                    rerun_empty_final=rerun_empty_final,
                    print_run_logs=False,
                    progress_bar=progress_bar,
                    global_state=global_state,
                    state_lock=state_lock,
                    existing_runs=existing_run_index.get(prob["id"], {}),
                ): (problem_tag, prob)
                for problem_tag, prob in pending_items
            }
            for fut in as_completed(futures):
                problem_tag, prob = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    print(f"ERROR on {problem_tag}: {type(e).__name__}: {e}")
                    raise
                print("\n" + "-" * 70)
                print(f"Completed problem: {prob['label']}")
                print("-" * 70)
                _record_problem_result(problem_tag, result)
    else:
        intra_problem_workers = max_workers if n_pending == 1 else 1
        for idx, (problem_tag, prob) in enumerate(pending_items, start=1):
            print("\n" + "-" * 70)
            print(f"[{idx}/{n_pending}] Running problem: {prob['label']}")
            print("-" * 70)
            result = _run_task5_for_one_problem(
                prob=prob,
                problem_tag=problem_tag,
                temperatures=temperatures,
                n_runs=n_runs,
                intra_problem_workers=intra_problem_workers,
                rerun_empty_final=rerun_empty_final,
                print_run_logs=print_run_logs,
                progress_bar=progress_bar,
                global_state=global_state,
                state_lock=state_lock,
                existing_runs=existing_run_index.get(prob["id"], {}),
            )
            _record_problem_result(problem_tag, result)

    if progress_bar is not None:
        progress_bar.close()
    if problem_bar is not None:
        problem_bar.close()

    if n_problems_total > 1:
        task_elapsed = time.time() - task_t0
        collected_total_calls = sum(
            int(v.get("n_samples", 0))
            for v in all_problem_summary.values()
        )
        summary = {
            "task": "5",
            "scope": "all_problems",
            "meta": {
                "started_at_utc": task_started_at,
                "ended_at_utc": _utc_now_iso(),
                "elapsed_seconds": round(task_elapsed, 3),
                "resume_enabled": resume,
                "rerun_empty_final": rerun_empty_final,
                "max_workers": effective_workers,
            },
            "n_problems": n_problems_total,
            "n_pending": n_pending,
            "n_skipped_completed": n_skipped,
            "temperatures": list(temperatures),
            "n_models": len(MODELS),
            "n_runs_per_temperature_model": n_runs,
            "expected_total_calls": n_problems_total * calls_per_problem,
            "actual_total_calls": collected_total_calls,
            "executed_calls_this_run": global_state["done_calls"],
            "remaining_calls_before_this_run": total_calls,
            "per_problem_overall_summary": all_problem_summary,
        }
        summary_path = os.path.join(OUT_DIR, "task5_all_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("\n" + "=" * 70)
        print(f"All-problem summary saved to {summary_path}")
        print(
            f"Total elapsed: {_format_duration(task_elapsed)} | "
            f"Calls: {global_state['done_calls']}/{total_calls}"
        )
        print("=" * 70)
        return summary

    return last_result


# ───────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────

def main():
    # Parse options
    args = sys.argv[1:]
    prob_key = "prob1"
    max_workers = 1
    resume = True
    enrich_flat = False
    rerun_empty_final = False

    if "--prob" in args:
        idx = args.index("--prob")
        if idx + 1 >= len(args):
            raise ValueError("--prob requires a value")
        prob_key = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    if "--workers" in args:
        idx = args.index("--workers")
        if idx + 1 >= len(args):
            raise ValueError("--workers requires a value")
        max_workers = int(args[idx + 1])
        args = args[:idx] + args[idx + 2:]

    if "--no-resume" in args:
        resume = False
        args = [a for a in args if a != "--no-resume"]

    if "--enrich-flat" in args:
        enrich_flat = True
        args = [a for a in args if a != "--enrich-flat"]

    if "--rerun-empty-final" in args:
        rerun_empty_final = True
        args = [a for a in args if a != "--rerun-empty-final"]

    tasks = args if args else ([] if enrich_flat else ["5"])
    print(f"Problem key: {prob_key}")
    print(f"Tasks: {tasks}")
    print(f"Task5 options: workers={max_workers}, resume={resume}")
    if enrich_flat:
        print("Flat CSV enrich mode: enabled")
    if rerun_empty_final:
        print("Task5 rerun mode: empty model_final_answer slots")

    if enrich_flat:
        enrich_flat_results_csv()

    if not tasks:
        print("\nNo task requested. Exiting after flat CSV enrichment.")
        return

    if "1" in tasks:
        task1_prompt_versions(prob_key)

    if "2" in tasks:
        task2_step_similarity(prob_key)

    if "3" in tasks:
        task3_single_model_diversity(prob_key)

    if "4" in tasks:
        task4_multi_model_diversity(prob_key)

    if "5" in tasks:
        task5_temperature_model_stats(
            prob_key=prob_key,
            max_workers=max_workers,
            resume=resume,
            rerun_empty_final=rerun_empty_final,
        )

    print("\n" + "=" * 70)
    print("All requested tasks complete.")
    print(f"Results directory: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
