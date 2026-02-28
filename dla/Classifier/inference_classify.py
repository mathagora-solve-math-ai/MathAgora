# -*- coding: utf-8 -*-
"""
단일 이미지 입력 → CSAT / SAT 분류(추론).

사용법 (dla 폴더에서):
  python -m Classifier.inference_classify <이미지경로>
  python -m Classifier.inference_classify /path/to/page.png --method cnn
  python -m Classifier.inference_classify /path/to/page.png --method heuristic
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if SCRIPT_DIR.parent not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

# 학습된 CNN 기본 경로 (Classifier/classifier_checkpoints/best.pt)
DEFAULT_CNN_PATH = SCRIPT_DIR / "classifier_checkpoints" / "best.pt"


def main() -> None:
    ap = argparse.ArgumentParser(description="단일 시험지 이미지 → CSAT 또는 SAT 분류")
    ap.add_argument("image", type=str, help="분류할 이미지 파일 경로")
    ap.add_argument(
        "--method",
        choices=["heuristic", "cnn", "hybrid"],
        default=None,
        help="heuristic=OCR+키워드, cnn=학습모델, hybrid=휴리스틱 후 불확실시 CNN. 미지정 시 best.pt 있으면 cnn, 없으면 heuristic",
    )
    ap.add_argument(
        "--cnn",
        type=str,
        default=None,
        help="CNN 체크포인트 경로 (기본: Classifier/classifier_checkpoints/best.pt)",
    )
    ap.add_argument(
        "--default",
        choices=["csat", "sat"],
        default="csat",
        help="휴리스틱 불확실 시 기본값 (기본: csat)",
    )
    args = ap.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    cnn_path = args.cnn or str(DEFAULT_CNN_PATH)
    if args.method is None:
        method = "cnn" if Path(cnn_path).exists() else "heuristic"
    else:
        method = args.method

    from Classifier import classify

    label = classify(
        str(image_path),
        method=method,
        default_if_uncertain=args.default,
        cnn_model_path=cnn_path if method in ("cnn", "hybrid") else None,
    )
    print(label)


if __name__ == "__main__":
    main()
