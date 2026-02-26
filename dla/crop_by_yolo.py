# -*- coding: utf-8 -*-
"""
학습된 YOLO 문항 탐지 모델로 이미지에서 문항 bbox를 예측하고, 각 문항 영역을 crop하여 저장합니다.
사용법:
  python crop_by_yolo.py --model runs/yolo12_problem_epoch50_batch32/weights/best.pt \\
      --source data/test_sat/png_pages --output_dir data/test_sat/crops_yolo
"""
import argparse
import sys
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="YOLO 문항 탐지 → 문항별 crop 저장")
    parser.add_argument("--model", type=str, required=True, help="학습된 YOLO 가중치 경로 (best.pt)")
    parser.add_argument("--source", type=str, required=True, help="입력 이미지 디렉토리 또는 파일")
    parser.add_argument("--output_dir", type=str, required=True, help="crop 저장 디렉토리")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold (default: 0.25)")
    parser.add_argument("--device", type=str, default=None, help="cuda device (e.g. 0)")
    args = parser.parse_args()

    from ultralytics import YOLO

    source = Path(args.source)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if source.is_dir():
        images = sorted(source.glob("*.png")) + sorted(source.glob("*.PNG"))
        images += sorted(source.glob("*.jpg")) + sorted(source.glob("*.JPG"))
        if not images:
            print(f"No images found in {source}")
            sys.exit(1)
        source_list = [str(p) for p in images]
    else:
        source_list = [str(source)]

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    predict_kw = dict(imgsz=640, conf=args.conf, verbose=False)
    if args.device is not None:
        predict_kw["device"] = args.device

    total_crops = 0
    for path in source_list:
        p = Path(path)
        stem = p.stem
        img = Image.open(path).convert("RGB")
        w, h = img.size
        results = model.predict(path, **predict_kw)
        if not results:
            continue
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        for i, (x1, y1, x2, y2) in enumerate(xyxy):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img.crop((x1, y1, x2, y2))
            out_name = f"{stem}__q{i+1:02d}.png"
            out_path = output_dir / out_name
            crop.save(out_path)
            total_crops += 1
        img.close()

    print(f"Saved {total_crops} crops to {output_dir}")


if __name__ == "__main__":
    main()
