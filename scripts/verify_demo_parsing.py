#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_parsing 저장 연동 검증 스크립트.
- save_demo_parsing() 직접 호출로 저장 동작 확인
- (선택) 실제 이미지로 detect 호출 시뮬레이션

사용: python scripts/verify_demo_parsing.py [이미지경로]
  이미지 경로 생략 시 mock payload로만 저장 테스트.
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))

DATA_ROOT = WORKSPACE / "data"
DEMO_PARSING = DATA_ROOT / "demo_parsing"


def verify_with_mock():
    """Mock payload로 save_demo_parsing 호출 후 디렉터리/TSV 확인."""
    from backend.demo_parsing_io import save_demo_parsing

    # 테스트용 이미지 (없으면 더미 경로로 페이지 이미지만 스킵)
    img = WORKSPACE / "dla" / "data" / "processed_images" / "2023_csat_9_problem_page_015.png"
    if not img.exists():
        imgs = list((WORKSPACE / "dla" / "data" / "processed_images").glob("*.png"))[:1]
        img = imgs[0] if imgs else None
    if not img or not img.exists():
        # 페이지 이미지 없이 payload만 테스트 (페이지 복사 단계는 실패할 수 있음)
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82")
        tmp.close()
        img = Path(tmp.name)
        try:
            _run_mock_save(str(img))
        finally:
            Path(tmp.name).unlink(missing_ok=True)
        return
    _run_mock_save(str(img))


def _run_mock_save(page_image_path: str):
    from backend.demo_parsing_io import save_demo_parsing

    payload = {
        "questions": [
            {"qid": "1", "bbox": [0, 0, 100, 100], "merged_text": "문제1 텍스트", "crop_b64": ""},
            {"qid": "2", "bbox": [0, 110, 100, 210], "merged_text": "문제2 텍스트", "crop_b64": ""},
        ],
        "image_width": 800,
        "image_height": 1200,
    }
    out = save_demo_parsing(
        page_image_path, payload, "csat", "page_2", run_ocr_per_crop=False
    )
    print("save_demo_parsing OK")
    print("  tsv_path:", out.get("tsv_path"))
    print("  problems:", len(out.get("problems", [])))
    for p in [DEMO_PARSING / "page_2.png", DEMO_PARSING / "llm_input.tsv", DEMO_PARSING / "crops"]:
        exists = p.exists()
        size = p.stat().st_size if p.exists() and p.is_file() else "-"
        print(f"  {p.relative_to(WORKSPACE)}: exists={exists}" + (f" size={size}" if p.is_file() else ""))
    if (DEMO_PARSING / "llm_input.tsv").exists():
        print("  llm_input.tsv preview:")
        for line in (DEMO_PARSING / "llm_input.tsv").read_text().strip().split("\n")[:4]:
            print("   ", line[:100])


if __name__ == "__main__":
    print("Verify demo_parsing save (mock payload)...")
    verify_with_mock()
    print("Done.")
