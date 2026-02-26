#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
outputs_parsing에서 2023·2024·2025년도 샘플만 대상으로:
- before_final: OCR 원시 출력에서 <|ref|>...<|/ref|><|det|>[[x,y,w,h]]<|/det|> 태그만 제거한 텍스트
- after_final:  웹 데모와 동일한 완전 변환 (get_converted_prob_desc 결과)

를 data/ocr_text_conversion_debug/before_final, after_final 에 각각 {problem_id}.txt 로 저장.
(웹 데모: get_raw_ocr_from_infer_log(page_id, qid) → get_converted_prob_desc(problem_id, raw, ...))
"""
from __future__ import annotations

import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))

OUTPUTS_PARSING = WORKSPACE / "data" / "outputs_parsing"
DEBUG_DIR = WORKSPACE / "data" / "ocr_text_conversion_debug"
BEFORE_DIR = DEBUG_DIR / "before_final"
AFTER_DIR = DEBUG_DIR / "after_final"

# 웹 데모와 동일: 2023·2024·2025년도만
ALLOWED_DATASETS = {"2023_math_odd", "2024_math_odd", "2025_math_odd"}


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


def _collect_page_ids() -> list[tuple[str, str]]:
    """(dataset, page_folder) 목록 수집. 2023·2024·2025만. page_id = f'{dataset}_{page_folder}'."""
    out = []
    if not OUTPUTS_PARSING.is_dir():
        return out
    for dataset_dir in sorted(OUTPUTS_PARSING.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name
        if dataset not in ALLOWED_DATASETS:
            continue
        for page_dir in sorted(dataset_dir.iterdir()):
            if not page_dir.is_dir():
                continue
            raw_dir = page_dir / "raw"
            if not raw_dir.is_dir():
                continue
            logs = list(raw_dir.glob("*__infer_*.log"))
            if not logs:
                continue
            page_folder = page_dir.name
            out.append((dataset, page_folder))
    return out


def main() -> int:
    _load_backend_env()
    from backend.ocr_text_converter import (
        get_raw_ocr_from_infer_log,
        get_converted_prob_desc,
        _postprocess_ocr_text,
    )

    BEFORE_DIR.mkdir(parents=True, exist_ok=True)
    AFTER_DIR.mkdir(parents=True, exist_ok=True)

    pages = _collect_page_ids()
    if not pages:
        print("No pages with infer logs found (2023/2024/2025) under data/outputs_parsing", file=sys.stderr)
        return 1
    print(f"Target datasets: {sorted(ALLOWED_DATASETS)}")
    print(f"Pages to process: {len(pages)}")

    total = 0
    failed = []
    for dataset, page_folder in pages:
        # page_id 형식: get_raw_ocr_from_infer_log가 기대하는 것과 동일
        # (dataset = "2023_math_odd", page_num = "001" -> page_id = "2023_math_odd_page_001")
        page_num = page_folder.replace("page_", "") if page_folder.startswith("page_") else page_folder
        page_id = f"{dataset}_page_{page_num}"
        for qid in (1, 2, 3, 4):
            try:
                raw = get_raw_ocr_from_infer_log(page_id, qid)
            except Exception as e:
                print(f"  {page_id}_{qid} get_raw FAIL: {e}", file=sys.stderr)
                failed.append((f"{page_id}_{qid}", e))
                continue
            if not raw or not raw.strip():
                continue
            problem_id = f"{page_id}_{qid}"
            try:
                before_text = _postprocess_ocr_text(raw)
                after_text = get_converted_prob_desc(
                    problem_id, raw, run_converter_if_missing=True
                )
                (BEFORE_DIR / f"{problem_id}.txt").write_text(before_text, encoding="utf-8")
                (AFTER_DIR / f"{problem_id}.txt").write_text(after_text, encoding="utf-8")
                total += 1
                print(f"  {problem_id}")
            except Exception as e:
                failed.append((problem_id, e))
                print(f"  {problem_id} FAIL: {e}", file=sys.stderr)

    print(f"\nSaved {total} samples to {BEFORE_DIR} and {AFTER_DIR}")
    if failed:
        print(f"Failed: {len(failed)}", file=sys.stderr)
        for pid, err in failed:
            print(f"  {pid}: {err}", file=sys.stderr)
    return 0 if total > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
