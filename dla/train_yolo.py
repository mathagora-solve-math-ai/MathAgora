# -*- coding: utf-8 -*-
"""
train_yolo.py
Ultralytics YOLO12 모델을 사용하여 문항(problem) 객체 탐지 모델을 학습합니다.

사용법:
  python train_yolo.py --data yolo_dataset/dataset.yaml --model yolo12n.pt --epochs 100
  python train_yolo.py --data yolo_dataset/dataset.yaml --mode val          # 검증만
  python train_yolo.py --data yolo_dataset/dataset.yaml --mode predict --source test.png

참고: https://docs.ultralytics.com/ko/models/yolo12/
"""

import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO12 문항 탐지 학습/추론")

    # 모드 선택
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "val", "predict", "export"],
                        help="실행 모드: train / val / predict / export (default: train)")

    # 모델
    parser.add_argument("--model", type=str, default="yolo12n.pt",
                        help="사전학습 모델 또는 학습된 모델 경로 (default: yolo12n.pt)")

    # 데이터셋
    parser.add_argument("--data", type=str, default="yolo_dataset/dataset.yaml",
                        help="dataset.yaml 경로 (default: yolo_dataset/dataset.yaml)")

    # 학습 하이퍼파라미터
    parser.add_argument("--epochs", type=int, default=100,
                        help="학습 에폭 수 (default: 100)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="입력 이미지 크기 (default: 640)")
    parser.add_argument("--batch", type=int, default=-1,
                        help="배치 사이즈, -1=auto (default: -1, GPU 메모리 자동 최적화)")
    parser.add_argument("--device", type=str, default=None,
                        help="학습 디바이스: 0 / 0,1,2,3 / cpu (default: auto)")
    parser.add_argument("--workers", type=int, default=8,
                        help="GPU당 데이터 로더 워커 수 (default: 8)")
    parser.add_argument("--project", type=str, default="runs",
                        help="프로젝트 저장 경로 (default: runs)")
    parser.add_argument("--name", type=str, default="yolo12_problem",
                        help="실험 이름 (default: yolo12_problem)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (default: 20)")
    parser.add_argument("--lr0", type=float, default=0.01,
                        help="초기 학습률 (default: 0.01)")
    parser.add_argument("--resume", action="store_true",
                        help="마지막 학습 체크포인트에서 재개")

    # 추론용
    parser.add_argument("--source", type=str, default=None,
                        help="추론 대상 이미지/디렉토리 경로 (predict 모드)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="추론 confidence threshold (default: 0.25)")

    # 내보내기용
    parser.add_argument("--export_format", type=str, default="onnx",
                        help="내보내기 형식: onnx, torchscript, engine 등 (default: onnx)")

    return parser.parse_args()


def train(args):
    """YOLO12 모델 학습"""
    from ultralytics import YOLO

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        lr0=args.lr0,
        exist_ok=True,
        verbose=True,
    )

    if args.device is not None:
        # "0,1,2,3" → [0,1,2,3] 형태로 변환 (ultralytics가 리스트도 지원)
        device_str = args.device.strip()
        if "," in device_str:
            train_kwargs["device"] = [int(x) for x in device_str.split(",")]
        else:
            train_kwargs["device"] = device_str

    if args.resume:
        train_kwargs["resume"] = True

    device_info = args.device if args.device else "auto"
    print(f"Starting training...")
    print(f"  Dataset : {args.data}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  ImgSize : {args.imgsz}")
    print(f"  Batch   : {args.batch} {'(auto)' if args.batch == -1 else ''}")
    print(f"  Device  : {device_info}")
    print(f"  Project : {args.project}/{args.name}")

    results = model.train(**train_kwargs)
    print(f"\nTraining complete. Best model saved to: {args.project}/{args.name}/weights/best.pt")
    return results


def validate(args):
    """학습된 모델 검증"""
    from ultralytics import YOLO

    print(f"Loading model for validation: {args.model}")
    model = YOLO(args.model)

    val_kwargs = dict(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name + "_val",
        exist_ok=True,
    )

    if args.device is not None:
        val_kwargs["device"] = args.device

    results = model.val(**val_kwargs)
    print(f"\nValidation complete.")
    return results


def predict(args):
    """학습된 모델로 추론"""
    from ultralytics import YOLO

    if not args.source:
        print("Error: --source is required for predict mode.")
        sys.exit(1)

    print(f"Loading model for prediction: {args.model}")
    model = YOLO(args.model)

    predict_kwargs = dict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        project=args.project,
        name=args.name + "_predict",
        save=True,
        exist_ok=True,
    )

    if args.device is not None:
        predict_kwargs["device"] = args.device

    results = model.predict(**predict_kwargs)
    print(f"\nPrediction complete. Results saved to: {args.project}/{args.name}_predict/")
    return results


def export_model(args):
    """모델 내보내기"""
    from ultralytics import YOLO

    print(f"Loading model for export: {args.model}")
    model = YOLO(args.model)

    model.export(format=args.export_format, imgsz=args.imgsz)
    print(f"\nExport complete (format={args.export_format}).")


def main():
    args = parse_args()

    mode_fn = {
        "train": train,
        "val": validate,
        "predict": predict,
        "export": export_model,
    }

    fn = mode_fn[args.mode]
    fn(args)


if __name__ == "__main__":
    main()
