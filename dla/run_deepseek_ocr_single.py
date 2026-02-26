#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
단일 이미지에 대해 DeepSeek-OCR 또는 DeepSeek-OCR-2를 실행하고 마크다운 결과를 stdout에 출력합니다.
venv_ocr 환경에서 실행할 때 사용합니다. 로직은 ocr_compare_utils.run_deepseek_inprocess 와 동일.

사용법:
  python run_deepseek_ocr_single.py <image_path> [DeepSeek-OCR|DeepSeek-OCR-2]
"""
import sys
from pathlib import Path

# 스크립트 디렉터리를 path에 넣어 같은 폴더의 ocr_compare_utils를 import 가능하게 함
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))


def main():
    if len(sys.argv) < 2:
        print("Usage: run_deepseek_ocr_single.py <image_path> [DeepSeek-OCR|DeepSeek-OCR-2]", file=sys.stderr)
        sys.exit(1)
    image_path = sys.argv[1]
    model_arg = sys.argv[2] if len(sys.argv) > 2 else "DeepSeek-OCR"
    model_key = "v2" if model_arg in ("DeepSeek-OCR-2", "DeepSeek-OCR2") else "v1"

    from ocr_compare_utils import run_deepseek_inprocess
    raw = run_deepseek_inprocess(image_path, model_key)
    print(raw)


if __name__ == "__main__":
    main()
