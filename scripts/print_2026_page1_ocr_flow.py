#!/usr/bin/env python3
"""
2026년도 1페이지: OCR 원문, _postprocess_ocr_text 이후, 최종 웹데모(get_converted_prob_desc) 3가지를 출력.
+ 데이터셋 JSON의 merged_text (프론트가 처음 로드할 때 쓰는 값).
"""
import json
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))

from backend.ocr_text_converter import (
    get_raw_ocr_from_infer_log,
    _postprocess_ocr_text,
    get_converted_prob_desc,
)

PAGE_ID = "2026_math_odd_page_001"
QIDS = [1, 2, 3, 4]
PAGE_JSON = WORKSPACE / "data" / "outputs_parsing" / "2026_math_odd" / "page_001" / "page_001.json"


def main():
    print("=" * 70)
    print("2026년도 1페이지 — OCR 원문 / 후처리 / 웹데모 최종 표시")
    print("=" * 70)

    # 데이터셋 JSON에 들어 있는 merged_text (프론트 로드 시 사용)
    if PAGE_JSON.is_file():
        data = json.loads(PAGE_JSON.read_text(encoding="utf-8"))
        questions = data.get("questions") or []
        print("\n[참고] data/outputs_parsing/2026_math_odd/page_001/page_001.json 의 questions 수:", len(questions))
        for q in questions:
            print(f"  - qid={q.get('qid')}, merged_text 길이={len((q.get('merged_text') or ''))}")
    else:
        print("\n[참고] page_001.json 없음:", PAGE_JSON)

    for qid in QIDS:
        problem_id = f"{PAGE_ID}_{qid}"
        print(f"\n{'#' * 70}")
        print(f"# 문항 {qid} (problem_id={problem_id})")
        print(f"{'#' * 70}\n")

        # 1) OCR 그대로 (infer 로그에서 추출)
        raw = get_raw_ocr_from_infer_log(PAGE_ID, qid)
        print("--- 1. OCR 그대로 (infer 로그 수행 결과) ---")
        if raw is None:
            print("(없음 — 2026 페이지는 파서 infer 로그가 없음)")
        else:
            print(raw)
        print()

        # 2) _postprocess_ocr_text 이후
        post_input = raw if raw else ""
        post = _postprocess_ocr_text(post_input)
        print("--- 2. _postprocess_ocr_text 이후 ---")
        if not post:
            print("(비어 있음)")
        else:
            print(post)
        print()

        # 3) 최종 웹데모에서 보여지는 값 (get_converted_prob_desc)
        # 실제 요청 시 raw_ocr_text는 infer 로그 또는 JSON의 merged_text
        raw_for_convert = raw if raw else ""
        final = get_converted_prob_desc(problem_id, raw_for_convert)
        print("--- 3. 최종 웹데모에서 보여지는 값 (get_converted_prob_desc) ---")
        print(final if final else "(비어 있음)")
        print()

    print("=" * 70)
    print("끝")
    print("=" * 70)


if __name__ == "__main__":
    main()
