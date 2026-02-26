#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2023년도 1번 페이지 문항별로 다음 세 가지 값을 출력:
1. DeepSeek-OCR 결과값 그 자체 (raw)
2. DeepSeek-OCR postcorrection 적용 값 (postprocess_deepseek_ocr)
3. ocr_text_converter.py 동작 후 값 (태그만 제거 + OpenRouter)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))

# 2023 page 1 raw infer log (full page)
RAW_LOG = WORKSPACE / "data/outputs_parsing/2023_math_odd/page_001/raw/page_001__infer_20251026_185000.log"


def _load_backend_env() -> None:
    env_path = WORKSPACE / "backend" / ".env"
    if not env_path.exists():
        return
    import os
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key:
            continue
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _split_raw_log_by_question(raw_text: str) -> dict[int, str]:
    """Parse full-page raw log and return { 1: raw_q1, 2: raw_q2, 3: raw_q3, 4: raw_q4 }."""
    if "===============save results:===============" in raw_text:
        raw_text = raw_text.split("===============save results:===============")[0]
    # Split by block start <|ref|>; each segment is one block (tag line + content until next tag)
    parts = re.split(r"(?=<\|ref\|>)", raw_text)
    blocks = []
    for p in parts:
        p = p.strip()
        if not p or p.startswith("="):
            continue
        # First line is tag, rest is content
        lines = p.split("\n", 1)
        tag_line = lines[0]
        content = (lines[1] or "").strip()
        box_m = re.search(r"\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]", tag_line)
        x, y = (int(box_m.group(1)), int(box_m.group(2))) if box_m else (0, 0)
        blocks.append({"x": x, "y": y, "full": (tag_line + "\n" + content).strip()})
    # Left col x<400: y<500 -> 1, else 2. Right col x>=400: y<400 -> 3, else 4
    def q_for(b):
        if b["x"] < 400:
            return 1 if b["y"] < 500 else 2
        return 3 if b["y"] < 400 else 4
    by_q: dict[int, list[str]] = {1: [], 2: [], 3: [], 4: []}
    for b in blocks:
        by_q[q_for(b)].append(b["full"])
    return {q: "\n\n".join(by_q[q]) for q in (1, 2, 3, 4)}


def main() -> int:
    _load_backend_env()
    if not RAW_LOG.exists():
        print(f"Raw log not found: {RAW_LOG}", file=sys.stderr)
        return 1
    raw_full = RAW_LOG.read_text(encoding="utf-8")
    by_question = _split_raw_log_by_question(raw_full)
    from dla.deepseek_ocr_postprocess import postprocess_deepseek_ocr
    from backend.ocr_text_converter import get_converted_prob_desc

    for qid in (1, 2, 3, 4):
        raw_q = by_question.get(qid, "")
        problem_id = f"2023_math_odd_page_001_{qid}"
        postcorrected = postprocess_deepseek_ocr(raw_q) if raw_q else ""
        try:
            converter_out = get_converted_prob_desc(problem_id, raw_q, run_converter_if_missing=True)
        except Exception as e:
            converter_out = f"[변환 실패] {e}"
        print("=" * 80)
        print(f"문항 {qid}")
        print("=" * 80)
        print("\n[1] DeepSeek-OCR 결과값 그 자체 (raw):")
        print("-" * 40)
        print(raw_q or "(없음)")
        print("\n[2] DeepSeek-OCR postcorrection 적용 값:")
        print("-" * 40)
        print(postcorrected or "(없음)")
        print("\n[3] ocr_text_converter.py 동작 후 값 (태그 제거 + OpenRouter):")
        print("-" * 40)
        print(converter_out or "(없음)")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
