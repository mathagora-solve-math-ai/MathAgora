#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
단일 이미지에 대해 GLM-OCR을 실행하고 결과를 stdout에 출력합니다.
DLA_SYSTEM_PYTHON(시스템/기본 Python)으로 실행할 때 사용. venv_ocr에서 실행하지 마세요.

사용법: python run_glm_ocr_single.py <image_path>
"""
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))


def main():
    if len(sys.argv) < 2:
        print("Usage: run_glm_ocr_single.py <image_path>", file=sys.stderr)
        sys.exit(1)
    from ocr_compare_utils import run_glm_ocr
    result = run_glm_ocr(sys.argv[1])
    print(result)


if __name__ == "__main__":
    main()
