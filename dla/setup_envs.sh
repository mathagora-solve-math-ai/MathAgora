#!/bin/bash
# =============================================================================
# setup_envs.sh
# DLA 파이프라인을 위한 두 개의 가상환경을 생성합니다.
#   1) venv_ocr  : DeepSeek-OCR + PDF 변환 파이프라인
#   2) venv_yolo : YOLO12 학습 / 추론
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo " [1/2] Creating venv_ocr (DeepSeek-OCR)"
echo "============================================"
if [ ! -d "venv_ocr" ]; then
    python3 -m venv venv_ocr
    echo "  -> venv_ocr created."
else
    echo "  -> venv_ocr already exists, skipping creation."
fi

source venv_ocr/bin/activate
pip install --upgrade pip
pip install -r requirements_ocr.txt
deactivate
echo "  -> venv_ocr dependencies installed."

echo ""
echo "============================================"
echo " [2/2] Creating venv_yolo (YOLO12)"
echo "============================================"
if [ ! -d "venv_yolo" ]; then
    python3 -m venv venv_yolo
    echo "  -> venv_yolo created."
else
    echo "  -> venv_yolo already exists, skipping creation."
fi

source venv_yolo/bin/activate
pip install --upgrade pip
pip install -r requirements_yolo.txt
deactivate
echo "  -> venv_yolo dependencies installed."

echo ""
echo "============================================"
echo " Setup complete!"
echo "  - OCR env  : source venv_ocr/bin/activate"
echo "  - YOLO env : source venv_yolo/bin/activate"
echo "============================================"
