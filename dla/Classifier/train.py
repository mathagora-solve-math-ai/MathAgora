# -*- coding: utf-8 -*-
"""
Train 2-class CNN (CSAT vs SAT) for exam type classification.
- Data: CSV from prepare_data.py (Classifier/classifier_data/).
- Model: ResNet18 (or MobileNetV2), ImageNet pretrained, FC → 2 classes (0=CSAT, 1=SAT).
- Output: Classifier/classifier_checkpoints/best.pt, last.pt.

실행: python -m Classifier.train [--data_dir Classifier/classifier_data] [--output_dir Classifier/classifier_checkpoints]
  (dla 폴더에서 실행 시)
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent


class ExamTypeDataset(Dataset):
    def __init__(self, csv_path: Path, transform=None):
        self.samples = []
        with open(csv_path, encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                path = Path(row["path"])
                if not path.exists():
                    continue
                label = row["label"].strip().lower()
                y = 1 if label == "sat" else 0
                self.samples.append((path, y))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, y


def get_transforms(is_train: bool):
    base = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if is_train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.98, 1.02)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            *base,
        ])
    return transforms.Compose(base)


def build_model(arch: str = "resnet18", num_classes: int = 2):
    def _resnet18():
        try:
            return models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except AttributeError:
            return models.resnet18(pretrained=True)
    def _mobilenet_v2():
        try:
            return models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        except AttributeError:
            return models.mobilenet_v2(pretrained=True)

    if arch == "resnet18":
        model = _resnet18()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "mobilenet_v2":
        model = _mobilenet_v2()
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return model


def main():
    ap = argparse.ArgumentParser(description="Train CSAT vs SAT classifier (2-class CNN).")
    ap.add_argument(
        "--data_dir",
        type=Path,
        default=SCRIPT_DIR / "classifier_data",
        help="Directory with exam_classifier_train.csv, exam_classifier_val.csv (default: Classifier/classifier_data)",
    )
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=SCRIPT_DIR / "classifier_checkpoints",
        help="Where to save best.pt and last.pt (default: Classifier/classifier_checkpoints)",
    )
    ap.add_argument("--arch", choices=["resnet18", "mobilenet_v2"], default="resnet18")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default=None, help="Device: cuda or cpu (ignored if --gpus set)")
    ap.add_argument("--gpus", nargs="*", type=int, default=None, help="GPU IDs to use (e.g. --gpus 1 2 3). Multi-GPU: DataParallel.")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    # GPU 선택: --gpus가 있으면 CUDA_VISIBLE_DEVICES 설정 후 cuda 사용, 여러 개면 DataParallel
    if args.gpus is not None and len(args.gpus) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpus))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        multi_gpu = torch.cuda.is_available() and len(args.gpus) > 1
        if torch.cuda.is_available():
            print(f"Using GPUs: {args.gpus} (visible as cuda:0..cuda:{torch.cuda.device_count()-1})" + (" [DataParallel]" if multi_gpu else ""))
    else:
        if args.device is None:
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(args.device)
        multi_gpu = False

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_csv = data_dir / "exam_classifier_train.csv"
    val_csv = data_dir / "exam_classifier_val.csv"
    if not train_csv.exists():
        print(f"Train CSV not found: {train_csv}. Run: python -m Classifier.prepare_data")
        return
    if not val_csv.exists():
        print(f"Val CSV not found: {val_csv}. Run: python -m Classifier.prepare_data")
        return

    train_ds = ExamTypeDataset(train_csv, transform=get_transforms(is_train=True))
    val_ds = ExamTypeDataset(val_csv, transform=get_transforms(is_train=False))
    if len(train_ds) == 0:
        print("No training samples found. Check exam_classifier_train.csv paths.")
        return
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    has_val = len(val_ds) > 0
    if not has_val:
        print("Warning: no validation samples (val split empty). Validation accuracy will be 0.")

    model = build_model(arch=args.arch).to(device)
    if multi_gpu:
        model = nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_acc = 0.0

    pbar_epoch = tqdm(range(args.epochs), desc="Epoch", unit="epoch", ncols=100)
    for epoch in pbar_epoch:
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        t0 = time.time()
        pbar_epoch.set_postfix_str("train...")
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]", leave=False, unit="batch", ncols=90):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = logits.argmax(dim=1)
            train_correct += (pred == labels).sum().item()
            train_total += labels.size(0)
        scheduler.step()
        train_acc = train_correct / train_total if train_total else 0

        model.eval()
        val_correct = 0
        val_total = 0
        if has_val:
            pbar_epoch.set_postfix_str("val...")
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [val]", leave=False, unit="batch", ncols=90):
                    images, labels = images.to(device), labels.to(device)
                    logits = model(images)
                    pred = logits.argmax(dim=1)
                    val_correct += (pred == labels).sum().item()
                    val_total += labels.size(0)
        val_acc = val_correct / val_total if val_total else 0.0
        elapsed = time.time() - t0

        pbar_epoch.set_postfix_str(f"loss={train_loss/len(train_loader):.3f} acc={train_acc:.3f} val_acc={val_acc:.3f} {elapsed:.0f}s")
        state = model.module.state_dict() if multi_gpu else model.state_dict()
        torch.save(state, output_dir / "last.pt")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(state, output_dir / "best.pt")
            tqdm.write(f"  -> new best saved (val_acc={best_acc:.4f})")

    pbar_epoch.close()
    print(f"Done. Best val_acc={best_acc:.4f}. Checkpoints in {output_dir}")


if __name__ == "__main__":
    main()
