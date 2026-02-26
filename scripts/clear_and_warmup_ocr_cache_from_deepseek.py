#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
캐시를 전부 지운 뒤, after_convertor_deepseek 폴더에 있는 샘플만으로
get_converted_prob_desc를 호출해 after_converted_prob_desc 캐시를 다시 채움.

- raw 입력: data/ocr_text_conversion_debug/after_convertor_deepseek/*__{problem_id}.txt (최신 파일)
- 사용: python scripts/clear_and_warmup_ocr_cache_from_deepseek.py
"""
from __future__ import annotations

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
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def main() -> int:
    _load_backend_env()

    from backend.ocr_text_converter import (
        AFTER_CONVERTOR_DEEPSEEK_DIR,
        clear_all_converter_caches,
        get_converted_prob_desc,
    )

    clear_all_converter_caches(delete_disk_files=True)
    print("Converter cache cleared (before_raw_ocr, after_converted_prob_desc).", flush=True)

    if not AFTER_CONVERTOR_DEEPSEEK_DIR.is_dir():
        print(f"after_convertor_deepseek not found: {AFTER_CONVERTOR_DEEPSEEK_DIR}", file=sys.stderr)
        return 1

    # *__{problem_id}.txt 형태만, .ipynb_checkpoints 제외
    files = [
        f
        for f in AFTER_CONVERTOR_DEEPSEEK_DIR.iterdir()
        if f.is_file() and f.suffix == ".txt" and "__" in f.stem and ".ipynb_checkpoints" not in str(f)
    ]
    # problem_id별로 최신 파일 하나만 (파일명 정렬 시 타임스탬프 순)
    by_problem: dict[str, Path] = {}
    for f in files:
        problem_id = f.stem.split("__", 1)[-1].strip()
        if not problem_id:
            continue
        if problem_id not in by_problem or f.name > by_problem[problem_id].name:
            by_problem[problem_id] = f

    n_ok = 0
    n_fail = 0
    for problem_id, path in sorted(by_problem.items()):
        try:
            content = path.read_text(encoding="utf-8").strip()
        except Exception as e:
            print(f"  {problem_id} read FAIL: {e}", file=sys.stderr)
            n_fail += 1
            continue
        if not content:
            continue
        try:
            get_converted_prob_desc(problem_id, content, run_converter_if_missing=True)
            n_ok += 1
            print(f"  {problem_id} OK", flush=True)
        except Exception as e:
            print(f"  {problem_id} FAIL: {e}", file=sys.stderr)
            n_fail += 1

    print(f"Warmup done: {n_ok} OK, {n_fail} FAIL (total {len(by_problem)} from after_convertor_deepseek).", flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
