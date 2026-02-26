# -*- coding: utf-8 -*-
"""
convert_coco_to_yolo.py
COCO 형식의 어노테이션 JSON을 YOLO 형식으로 변환합니다.
  - COCO bbox [x, y, w, h] (절대 픽셀) → YOLO [class_id, x_center_norm, y_center_norm, w_norm, h_norm]
  - Train / Val 분할
  - dataset.yaml 자동 생성
  - 이미지 파일을 yolo_dataset/images/ 아래로 심볼릭 링크 생성
"""

import json
import os
import random
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

import yaml


def coco_bbox_to_yolo(bbox, img_w, img_h):
    """COCO [x_min, y_min, width, height] → YOLO [x_center, y_center, w, h] (normalized 0-1)"""
    x_min, y_min, w, h = bbox
    x_center = (x_min + w / 2.0) / img_w
    y_center = (y_min + h / 2.0) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    # Clamp to [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w_norm = max(0.0, min(1.0, w_norm))
    h_norm = max(0.0, min(1.0, h_norm))
    return x_center, y_center, w_norm, h_norm


def build_category_mapping(coco_categories):
    """COCO category id → YOLO class index (0-based)"""
    mapping = {}
    names = {}
    for idx, cat in enumerate(sorted(coco_categories, key=lambda c: c["id"])):
        mapping[cat["id"]] = idx
        names[idx] = cat["name"]
    return mapping, names


def main():
    parser = argparse.ArgumentParser(description="Convert COCO JSON to YOLO format dataset")
    parser.add_argument("--coco_json", type=str, required=True,
                        help="Path to COCO instances JSON file")
    parser.add_argument("--output_dir", type=str, default="yolo_dataset",
                        help="Output directory for YOLO dataset")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="Fraction of images for validation (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/val split")
    parser.add_argument("--copy_images", action="store_true",
                        help="Copy images instead of creating symlinks")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Directory containing source images (used to resolve file_name in COCO JSON)")
    args = parser.parse_args()

    random.seed(args.seed)

    # ── 1. Load COCO JSON ──────────────────────────────────────────────
    with open(args.coco_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    cat_map, cat_names = build_category_mapping(coco["categories"])

    # Group annotations by image_id
    ann_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        ann_by_image[ann["image_id"]].append(ann)

    print(f"Loaded COCO JSON: {len(images)} images, {len(coco['annotations'])} annotations, {len(cat_names)} classes")
    for idx, name in cat_names.items():
        print(f"  class {idx}: {name}")

    # ── 2. Create output directory structure ──────────────────────────
    output_dir = Path(args.output_dir)
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ── 3. Train / Val split ──────────────────────────────────────────
    image_ids = sorted(images.keys())
    random.shuffle(image_ids)
    n_val = max(1, int(len(image_ids) * args.val_ratio))
    val_ids = set(image_ids[:n_val])
    train_ids = set(image_ids[n_val:])

    print(f"Split: {len(train_ids)} train, {len(val_ids)} val")

    # ── 4. Write YOLO labels + link images ────────────────────────────
    skipped = 0
    for img_id, img_info in images.items():
        split = "val" if img_id in val_ids else "train"

        # Resolve image source path
        file_name = img_info["file_name"]
        base_name = Path(file_name).name  # 파일명만 추출 (경로가 포함될 수 있으므로)
        src_path = None

        # 탐색 순서: image_dir → 절대경로 → coco json 디렉토리 기준
        candidates = []
        if args.image_dir:
            candidates.append(Path(args.image_dir) / base_name)
        if Path(file_name).is_absolute():
            candidates.append(Path(file_name))
        candidates.append(Path(args.coco_json).parent / file_name)
        candidates.append(Path(args.coco_json).parent / base_name)

        for cand in candidates:
            if cand.exists():
                src_path = cand
                break

        if src_path is None:
            print(f"  Warning: image not found: {base_name}, skipping.")
            skipped += 1
            continue

        img_w = img_info["width"]
        img_h = img_info["height"]

        # Image destination
        dst_img = output_dir / "images" / split / src_path.name
        if not dst_img.exists():
            if args.copy_images:
                shutil.copy2(str(src_path), str(dst_img))
            else:
                # Create symlink (absolute path)
                os.symlink(str(src_path.resolve()), str(dst_img))

        # Label file
        label_name = src_path.stem + ".txt"
        label_path = output_dir / "labels" / split / label_name

        anns = ann_by_image.get(img_id, [])
        lines = []
        for ann in anns:
            class_id = cat_map[ann["category_id"]]
            xc, yc, wn, hn = coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
            lines.append(f"{class_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        with open(label_path, "w") as f:
            f.write("\n".join(lines) + "\n" if lines else "")

    if skipped:
        print(f"  Skipped {skipped} images (not found).")

    # ── 5. Generate dataset.yaml ──────────────────────────────────────
    yaml_path = output_dir / "dataset.yaml"
    dataset_config = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(cat_names),
        "names": [cat_names[i] for i in range(len(cat_names))],
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)

    print(f"\nYOLO dataset created at: {output_dir}")
    print(f"  dataset.yaml: {yaml_path}")
    print(f"  Train images: {len(train_ids)}")
    print(f"  Val images:   {len(val_ids)}")
    print(f"  Classes:      {list(cat_names.values())}")


if __name__ == "__main__":
    main()
