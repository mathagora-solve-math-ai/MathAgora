#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
저장된 OCR converter 캐시를 지우고, (선택) detect payload로 OpenRouter 변환을 다시 수행해 캐시 저장.

- DeepSeek-OCR 결과에서 "<|ref|>...<|/ref|><|det|>[[...]]<|/det|>" 태그만 제거 후 OpenRouter 보정.
- 사용: python scripts/clear_and_warmup_ocr_cache.py [--payload PATH] [--page-id ID]
  --payload 생략 시 캐시만 삭제하고, 웹에서 Detect 실행 시 새로 저장됨.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))


def _load_backend_env() -> None:
    env_path = WORKSPACE / "backend" / ".env"
    if not env_path.exists():
        return
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Clear OCR converter cache and optionally warm up from payload.")
    parser.add_argument("--payload", type=str, help="Path to detect payload JSON (questions with qid, merged_text)")
    parser.add_argument("--page-id", type=str, default="page_2", help="page_id for conversion_id (default: page_2)")
    parser.add_argument("--dry-run", action="store_true", help="Only clear cache, do not call converter")
    args = parser.parse_args()
    _load_backend_env()

    from backend.ocr_text_converter import clear_all_converter_caches, get_converted_prob_desc

    clear_all_converter_caches(delete_disk_files=True)
    print("Converter cache cleared (before_raw_ocr, after_converted_prob_desc).")

    if args.dry_run or not args.payload:
        if not args.payload:
            print("No --payload given. Run Detect from the UI to repopulate cache with tag-only strip + OpenRouter.")
        return 0

    path = Path(args.payload)
    if not path.exists():
        print(f"Payload file not found: {path}", file=sys.stderr)
        return 1
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Failed to load payload: {e}", file=sys.stderr)
        return 1

    questions = data.get("questions") if isinstance(data, dict) else []
    if not questions:
        print("No 'questions' in payload.", file=sys.stderr)
        return 1

    page_id = args.page_id or "page_2"
    n_ok = 0
    for q in questions:
        qid = str(q.get("qid", ""))
        merged_text = (q.get("merged_text") or "").strip()
        if not merged_text:
            continue
        problem_id = f"{page_id}_{qid}"
        try:
            get_converted_prob_desc(problem_id, merged_text, run_converter_if_missing=True)
            n_ok += 1
            print(f"  {problem_id} OK")
        except Exception as e:
            print(f"  {problem_id} FAIL: {e}", file=sys.stderr)
    print(f"Warmup done: {n_ok}/{len(questions)} converted and saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
