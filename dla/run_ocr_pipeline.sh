#!/usr/bin/env bash
# OCR 파이프라인: venv_ocr에서 실행. DeepSeek은 in-process, Paddle/GLM은 시스템 Python subprocess.
# - DeepSeek: 같은 프로세스(venv_ocr)에서 실행 (subprocess 사용 안 함)
# - Paddle, GLM: DLA_SYSTEM_PYTHON으로 run_*_single.py 실행 (venv_ocr에서 돌리면 안 되므로)
#
# 사용법: DLA_SYSTEM_PYTHON=/usr/bin/python3 ./run_ocr_pipeline.sh [--year 2022] [--limit N] ...
set -e
DLA_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DLA_DIR"
# Paddle/GLM subprocess용 Python (venv_ocr이 아닌, paddle·glm 설치된 환경)
export DLA_SYSTEM_PYTHON="${DLA_SYSTEM_PYTHON:-python3}"
exec ./venv_ocr/bin/python ocr_pipeline.py "$@"
