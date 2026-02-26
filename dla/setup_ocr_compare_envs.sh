#!/bin/bash
# =============================================================================
# setup_ocr_compare_envs.sh
# OCR 비교(ocr_compare_notebook)용 가상환경을 만듭니다.
#   - venv_ocr_compare : GLM-OCR, MinerU2.5, PaddleOCR-VL 등 동작 (권장)
#   - venv_deepseek    : DeepSeek-OCR/2 (flash-attn 필요 시 선택)
# dots.ocr는 현재 transformers processor 호환 이슈로 별도 환경이 필요할 수 있음.
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo " [1/2] Creating venv_ocr_compare (GLM, MinerU, Paddle 등)"
echo "============================================"
if [ ! -d "venv_ocr_compare" ]; then
    python3 -m venv venv_ocr_compare
    echo "  -> venv_ocr_compare created."
else
    echo "  -> venv_ocr_compare already exists."
fi

source venv_ocr_compare/bin/activate
pip install --upgrade pip
pip install -r requirements-ocr-compare.txt
deactivate
echo "  -> venv_ocr_compare dependencies installed."

echo ""
echo "============================================"
echo " [2/2] Optional: venv_deepseek (DeepSeek-OCR/2, flash-attn)"
echo "============================================"
if [ ! -d "venv_deepseek" ]; then
    python3 -m venv venv_deepseek
    echo "  -> venv_deepseek created."
    source venv_deepseek/bin/activate
    pip install --upgrade pip
    pip install torch transformers scipy accelerate
    echo "  -> Install flash-attn manually if needed: pip install flash-attn --no-build-isolation"
    echo "     (requires python3-dev, CUDA toolkit; build can fail without them)"
    deactivate
else
    echo "  -> venv_deepseek already exists."
fi

echo ""
echo "============================================"
echo " OCR compare envs ready."
echo "  - Main (GLM, MinerU, Paddle): source venv_ocr_compare/bin/activate"
echo "  - DeepSeek (optional):       source venv_deepseek/bin/activate"
echo "  - Run notebook from dla:     jupyter notebook ocr_compare_notebook.ipynb"
echo "============================================"
