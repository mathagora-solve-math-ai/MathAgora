#!/usr/bin/env python3
"""
PaddleOCR 2.x API로 단일 이미지 OCR 실행. oneDNN 오류가 나는 환경 대신
venv_paddle2 (paddlepaddle==2.6, paddleocr==2.7, numpy<2)에서 실행용.

사용: python run_paddleocr_v2.py <image_path> [--output result.json]
결과를 JSON으로 저장 (박스 좌표, 인식 텍스트, 신뢰도).
"""
import sys
import json
import time
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: run_paddleocr_v2.py <image_path> [--output result.json]", file=sys.stderr)
        sys.exit(1)
    image_path = sys.argv[1]
    out_path = "paddleocr_result.json"
    if len(sys.argv) >= 4 and sys.argv[2] == "--output":
        out_path = sys.argv[3]

    if not Path(image_path).exists():
        print(f"Error: image not found: {image_path}", file=sys.stderr)
        sys.exit(2)

    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    t0 = time.perf_counter()
    result = ocr.ocr(image_path, cls=True)
    elapsed = time.perf_counter() - t0

    # result: list of pages; each page is list of [box, (text, score)]
    # JSON 직렬화: box는 리스트의 리스트, (text, score)는 리스트로
    out = {
        "elapsed_seconds": round(elapsed, 3),
        "pages": []
    }
    if result:
        for page in result:
            if not page:
                out["pages"].append([])
                continue
            rows = []
            for item in page:
                box, rec = item
                text = rec[0] if isinstance(rec, (list, tuple)) else str(rec[0])
                score = float(rec[1]) if isinstance(rec, (list, tuple)) and len(rec) > 1 else 0.0
                # box: [[x,y],...] -> list of lists for JSON
                box_ser = [[float(x), float(y)] for x, y in box]
                rows.append({"box": box_ser, "text": text, "score": score})
            out["pages"].append(rows)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=0)
    print(out_path)
    print(elapsed)


if __name__ == "__main__":
    main()
