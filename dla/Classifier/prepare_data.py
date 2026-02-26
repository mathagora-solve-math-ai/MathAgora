# -*- coding: utf-8 -*-
"""
Prepare train/val/test CSV for CSAT vs SAT exam type classifier.
- SAT: data/SAT/*/ (folders matching N-images) → label sat.
- CSAT: data/2022_math_odd, ... (전체 페이지 + crop 이미지 모두), data/crop_image/ → label csat.
- Split: by_group(기본, 시험 단위) 또는 random(이미지 단위, 비율 균형).

실행: python -m Classifier.prepare_data [--data_dir ...] [--output_dir ...] [--include_crops] [--split_mode]
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent.parent

IMAGE_EXTS = {".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"}


def _is_sat_folder(name: str) -> bool:
    if not name or "-" not in name:
        return False
    prefix = name.split("-")[0]
    return prefix.isdigit()


def _is_csat_folder(name: str) -> bool:
    return "math_odd" in name and len(name) >= 4 and name[:4].isdigit()


def collect_sat_paths(data_root: Path) -> list[tuple[str, Path]]:
    """SAT: N-images 폴더 내 전체 이미지."""
    out = []
    sat_root = data_root / "SAT"
    if not sat_root.is_dir():
        return out
    for folder in sat_root.iterdir():
        if not folder.is_dir() or not _is_sat_folder(folder.name):
            continue
        group_id = f"sat_{folder.name}"
        for p in folder.iterdir():
            if p.suffix in IMAGE_EXTS:
                out.append((group_id, p.resolve()))
    return out


def collect_csat_paths(
    data_root: Path,
    include_crops: bool = True,
) -> list[tuple[str, Path]]:
    """
    CSAT: *_math_odd 폴더 내 전체 이미지(전체 페이지 + crop).
    include_crops True면 _crop 포함, data/crop_image/ 폴더도 수집.
    """
    out = []
    for folder in data_root.iterdir():
        if not folder.is_dir():
            continue
        if not _is_csat_folder(folder.name):
            continue
        group_id = f"csat_{folder.name}"
        for p in folder.iterdir():
            if p.suffix not in IMAGE_EXTS:
                continue
            if not include_crops and "_crop" in p.name:
                continue
            out.append((group_id, p.resolve()))

    if include_crops:
        crop_dir = data_root / "crop_image"
        if crop_dir.is_dir():
            group_id = "csat_crop_image"
            for p in crop_dir.iterdir():
                if p.suffix in IMAGE_EXTS:
                    out.append((group_id, p.resolve()))
    return out


def split_by_group(
    items: list[tuple[str, Path]],
    label: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> list[tuple[Path, str, str]]:
    """시험/폴더 단위로 split → 같은 시험이 train/val/test에 섞이지 않음."""
    groups = {}
    for gid, path in items:
        groups.setdefault(gid, []).append(path)
    group_ids = sorted(groups.keys())
    random.seed(seed)
    random.shuffle(group_ids)
    n = len(group_ids)
    n_train = max(1, int(n * train_ratio))
    n_val = max(0, int(n * val_ratio))
    n_test = n - n_train - n_val
    train_ids = set(group_ids[:n_train])
    val_ids = set(group_ids[n_train : n_train + n_val])
    test_ids = set(group_ids[n_train + n_val :])
    result = []
    for gid, path in items:
        if gid in train_ids:
            result.append((path, label, "train"))
        elif gid in val_ids:
            result.append((path, label, "val"))
        else:
            result.append((path, label, "test"))
    return result


def split_random(
    items: list[tuple[str, Path]],
    label: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> list[tuple[Path, str, str]]:
    """이미지 단위 랜덤 split → train/val/test 비율 균형 (같은 시험이 여러 split에 나올 수 있음)."""
    paths = [p for _, p in items]
    random.seed(seed)
    random.shuffle(paths)
    n = len(paths)
    n_train = max(0, int(n * train_ratio))
    n_val = max(0, int(n * val_ratio))
    n_test = n - n_train - n_val
    result = []
    for i, path in enumerate(paths):
        if i < n_train:
            result.append((path, label, "train"))
        elif i < n_train + n_val:
            result.append((path, label, "val"))
        else:
            result.append((path, label, "test"))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare train/val/test CSV for exam type classifier.")
    ap.add_argument(
        "--data_dir",
        type=Path,
        default=WORKSPACE_ROOT / "data",
        help="Root data directory (contains SAT/, 2022_math_odd/, crop_image/, ...)",
    )
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=SCRIPT_DIR / "classifier_data",
        help="Output directory for CSV files (default: Classifier/classifier_data)",
    )
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--include_crops",
        action="store_true",
        default=True,
        help="Include crop images in CSAT (default: True). Use --no_include_crops to disable.",
    )
    ap.add_argument(
        "--no_include_crops",
        action="store_false",
        dest="include_crops",
        help="Exclude crop images (only full-page).",
    )
    ap.add_argument(
        "--split_mode",
        choices=["by_group", "random"],
        default="by_group",
        help="by_group: 시험 단위 split (같은 시험은 한 split만). random: 이미지 단위 랜덤 (비율 균형).",
    )
    args = ap.parse_args()

    data_root = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sat_items = collect_sat_paths(data_root)
    csat_items = collect_csat_paths(data_root, include_crops=args.include_crops)
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    split_fn = split_random if args.split_mode == "random" else split_by_group
    sat_split = split_fn(sat_items, "sat", train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=test_ratio, seed=args.seed)
    csat_split = split_fn(csat_items, "csat", train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=test_ratio, seed=args.seed)
    all_rows = sat_split + csat_split
    if not all_rows:
        print("No images found. Check --data_dir (SAT/ and *_math_odd/ folders).")
        return

    csv_path = output_dir / "exam_classifier_splits.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "split"])
        for path, label, split in all_rows:
            w.writerow([str(path), label, split])

    for split_name in ("train", "val", "test"):
        rows = [(p, l) for p, l, s in all_rows if s == split_name]
        with open(output_dir / f"exam_classifier_{split_name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "label"])
            for path, label in rows:
                w.writerow([str(path), label])

    n_train = sum(1 for _, _, s in all_rows if s == "train")
    n_val = sum(1 for _, _, s in all_rows if s == "val")
    n_test = sum(1 for _, _, s in all_rows if s == "test")
    print(f"SAT images: {len(sat_items)}, CSAT images: {len(csat_items)} (include_crops={args.include_crops}, split_mode={args.split_mode})")
    print(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")
    print(f"Wrote {csv_path} and exam_classifier_{{train,val,test}}.csv in {output_dir}")


if __name__ == "__main__":
    main()
