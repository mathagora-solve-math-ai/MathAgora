#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
데모와 동일한 흐름으로 2023년 1페이지 4문항에 대해 get_converted_prob_desc 호출 후
각 문항 text를 출력. (page_id=2023_math_odd_page_001, payload from page_001.json)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))

# backend .env 로드
def _load_env():
    p = WORKSPACE / "backend" / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        os.environ.setdefault(k, v)

_load_env()

PAYLOAD_PATH = WORKSPACE / "data/outputs_parsing/2023_math_odd/page_001/page_001.json"
PAGE_ID = "2023_math_odd_page_001"


def main() -> int:
    from backend.ocr_text_converter import get_converted_prob_desc, get_raw_ocr_from_infer_log

    if not PAYLOAD_PATH.exists():
        print(f"Payload not found: {PAYLOAD_PATH}", file=sys.stderr)
        return 1
    data = json.loads(PAYLOAD_PATH.read_text(encoding="utf-8"))
    questions = data.get("questions") or []
    if len(questions) < 4:
        print(f"Expected 4 questions, got {len(questions)}", file=sys.stderr)
        return 1

    print("=== 데모와 동일: page_id=%s, infer 로그 raw 우선 → get_converted_prob_desc ===\n" % PAGE_ID)
    for q in questions:
        qid = str(q.get("qid", ""))
        conversion_id = f"{PAGE_ID}_{qid}"
        raw_text = (get_raw_ocr_from_infer_log(PAGE_ID, qid) or (q.get("merged_text") or "")).strip()
        crop_b64 = q.get("crop_b64") or ""
        text = get_converted_prob_desc(
            conversion_id,
            raw_text,
            crop_b64=crop_b64 or None,
            run_converter_if_missing=True,
        )
        print("-" * 60)
        print(f"문항 {qid} (conversion_id={conversion_id})")
        print("-" * 60)
        print(text)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
