# -*- coding: utf-8 -*-
"""
detect 결과를 /workspace/data/demo_parsing 에 저장.
- 2번째 페이지(또는 업로드 페이지) 이미지 및 문항별 crop 이미지 저장
- 문항별 prob_desc = ocr_text_converter(태그 제거 + OpenRouter) 동작 후 값 (sample/데모 동일)
- LLM 입력용 TSV 생성 (config에서 MATH_DATA_FILE 경로만 바꿔서 사용 가능)
"""
from __future__ import annotations

import base64
import csv
import os
import shutil
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
DATA_ROOT = WORKSPACE / "data"
DEMO_PARSING_DIR = DATA_ROOT / "demo_parsing"
DEMO_PARSING_CROPS = DEMO_PARSING_DIR / "crops"

# TSV 컬럼 (POC config와 동일 포맷)
LLM_TSV_COLUMNS = [
    "prob_id",
    "prob_type",
    "prob_area",
    "prob_point",
    "prob_desc",
    "prob_img_path",
    "prob_fig_img_path",
    "answer",
    "solution",
]


def _run_ocr_on_crop(crop_image_path: str) -> str:
    """단일 crop 이미지에 대해 DeepSeek-OCR 실행 후 후처리된 텍스트 반환 (run_deepseek_ocr → deepseek_ocr_postprocess)."""
    import sys
    if str(WORKSPACE) not in sys.path:
        sys.path.insert(0, str(WORKSPACE))
    try:
        from dla.ocr_compare_utils import run_deepseek_ocr
        text = run_deepseek_ocr(crop_image_path)
        if text and not text.strip().startswith("[DeepSeek"):
            return text.strip()
        return ""
    except Exception as e:
        return f"[OCR error] {e}"


def _get_converted_prob_desc_for_demo(prob_id: str, merged_text: str, crop_b64: str | None = None) -> str:
    """데모용 prob_desc: ocr_text_converter(태그 제거 + OpenRouter) 결과 사용."""
    if not (merged_text or "").strip():
        return "(텍스트 없음)"
    try:
        from backend.ocr_text_converter import get_converted_prob_desc
        return get_converted_prob_desc(
            prob_id,
            merged_text.strip(),
            crop_b64=crop_b64,
            run_converter_if_missing=True,
        ) or merged_text.strip()
    except Exception:
        return (merged_text or "").strip()


def save_demo_parsing(
    page_image_path: str,
    payload: dict,
    document_type: str,
    page_id: str = "page_2",
    run_ocr_per_crop: bool = True,
) -> dict:
    """
    detect 결과를 demo_parsing 폴더에 저장.
    - page_image_path: 업로드된 원본 페이지 이미지 경로
    - payload: parser process_single_image 반환값 (questions, image_width, image_height)
    - document_type: "csat" | "sat"
    - page_id: 페이지 식별자 (파일명에 사용, 기본 "page_2")
    - run_ocr_per_crop: True면 각 crop에 대해 DeepSeek-OCR 실행하여 문제별 텍스트 생성

    Returns:
        {"demo_parsing_dir": str, "page_image": str, "tsv_path": str, "problems": list}
    """
    demo_dir = Path(DEMO_PARSING_DIR)
    crops_dir = Path(DEMO_PARSING_CROPS)
    demo_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    # 1) 페이지 전체 이미지 저장
    page_image_name = f"{page_id}.png"
    page_image_dest = demo_dir / page_image_name
    try:
        shutil.copy2(page_image_path, page_image_dest)
    except Exception:
        shutil.copy(page_image_path, page_image_dest)

    questions = payload.get("questions") or []
    rows = []

    for q in questions:
        qid = q.get("qid", "")
        crop_b64 = q.get("crop_b64") or ""
        merged_text_from_parser = (q.get("merged_text") or "").strip()

        # 2) crop 이미지 저장 (crop_b64 우선, 없으면 페이지 이미지에서 bbox로 crop)
        crop_name = f"{page_id}_{qid}.png"
        crop_rel = f"demo_parsing/crops/{crop_name}"
        crop_abs = crops_dir / crop_name

        if crop_b64:
            try:
                raw = base64.b64decode(crop_b64)
                crop_abs.write_bytes(raw)
            except Exception:
                pass
        else:
            # crop_b64가 없어도 bbox가 있으면 페이지 이미지에서 crop하여 저장
            bbox = q.get("bbox")
            if bbox and len(bbox) >= 4:
                try:
                    from PIL import Image
                    with Image.open(page_image_path).convert("RGB") as im:
                        x1, y1, x2, y2 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                        w, h = im.size
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        if x2 > x1 and y2 > y1:
                            im.crop((x1, y1, x2, y2)).save(crop_abs, format="PNG")
                except Exception:
                    pass

        # 3) 문제별 텍스트: infer 로그 raw OCR 있으면 사용(output 파일과 동일), 없으면 merged_text → converter
        try:
            from backend.ocr_text_converter import get_raw_ocr_from_infer_log
            raw_for_convert = get_raw_ocr_from_infer_log(page_id, qid) or merged_text_from_parser
        except Exception:
            raw_for_convert = merged_text_from_parser
        if not raw_for_convert and run_ocr_per_crop and crop_abs.exists():
            raw_for_convert = _run_ocr_on_crop(str(crop_abs))
        prob_id = f"{page_id}_{qid}"
        prob_desc = _get_converted_prob_desc_for_demo(prob_id, raw_for_convert or "", crop_b64=crop_b64 or None)

        row = {
            "prob_id": prob_id,
            "prob_type": "5지선다형",
            "prob_area": "수학",
            "prob_point": "2",
            "prob_desc": prob_desc.replace("\t", " ").replace("\r\n", "\n"),
            "prob_img_path": crop_rel,
            "prob_fig_img_path": "",
            "answer": "",
            "solution": "",
        }
        rows.append(row)

    # 4) LLM 입력용 TSV 저장 (config에서 DATA_ROOT + 이 파일 경로로 로드 가능)
    tsv_name = "llm_input.tsv"
    tsv_path = demo_dir / tsv_name
    with open(tsv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LLM_TSV_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    return {
        "demo_parsing_dir": str(demo_dir),
        "page_image": str(page_image_dest),
        "tsv_path": str(tsv_path),
        "problems": rows,
        "document_type": document_type,
    }
