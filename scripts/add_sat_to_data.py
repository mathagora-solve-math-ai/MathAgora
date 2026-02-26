#!/usr/bin/env python3
"""
SAT 샘플을 현재 data 구조와 동일하게 복사/이동합니다.
- data/sat_math_odd/page_<short>.png (전체 페이지 이미지)
- data/outputs_parsing/sat_math_odd/page_<short>/page_<short>.json
- frontend/public/data/datasets.json 에 sat year 추가
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
DLA = WORKSPACE / "dla"
DATA = WORKSPACE / "data"
FRONTEND_PUBLIC = WORKSPACE / "frontend" / "public" / "data"

# (outputs_dla/sat 내 폴더명, 페이지 short id)
SAT_PAGES = [
    ("8-images/sat-practice-test-8-math_page_014", "8_014"),
    ("11-images/sat-practice-test-11-math_page_001", "11_001"),
    ("10-images/sat-practice-test-10-math_page_001", "10_001"),
    ("10-images/sat-practice-test-10-math_page_015", "10_015"),
    ("9-images/sat-practice-test-9-math_page_003", "9_003"),
    ("9-images/sat-practice-test-9-math_page_006", "9_006"),
    ("11-images/sat-practice-test-11-math_page_015", "11_015"),
    ("11-images/sat-practice-test-11-math_page_003", "11_003"),
]


def main() -> None:
    sat_out = DLA / "outputs_dla" / "sat"
    images_dir = DATA / "sat_math_odd"
    parsing_dir = DATA / "outputs_parsing" / "sat_math_odd"
    images_dir.mkdir(parents=True, exist_ok=True)
    parsing_dir.mkdir(parents=True, exist_ok=True)

    sat_pages_for_index: dict[str, dict] = {}

    for rel_folder, short_id in SAT_PAGES:
        page_dir = sat_out / rel_folder
        page_id = page_dir.name
        json_path = page_dir / f"{page_id}.json"
        if not json_path.exists():
            print(f"Skip (no json): {rel_folder}")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        page_image_path = data.get("page_image") or ""
        questions = data.get("questions") or []

        # Resolve page image
        page_img_path = None
        if page_image_path and Path(page_image_path).exists():
            page_img_path = Path(page_image_path)
        if not page_img_path:
            cand = WORKSPACE / "data" / "SAT" / rel_folder.split("/")[0] / f"{page_id}.png"
            if cand.exists():
                page_img_path = cand
        if not page_img_path:
            vis_img = page_dir / "vis" / f"{page_id}_vis.png"
            if vis_img.exists():
                page_img_path = vis_img
        if not page_img_path:
            same = page_dir / f"{page_id}.png"
            if same.exists():
                page_img_path = same
        if not page_img_path:
            print(f"Skip (no page image): {rel_folder}")
            continue

        # Copy page image
        dest_img = images_dir / f"page_{short_id}.png"
        shutil.copy2(page_img_path, dest_img)
        print(f"Copied page image: {dest_img}")

        # Write normalized json (frontend expects page_image + questions + width/height for crop display)
        page_json = {
            "page_image": f"data/sat_math_odd/page_{short_id}.png",
            "width": data.get("width"),
            "height": data.get("height"),
            "questions": [{"qid": q["qid"], "merged_text": q.get("merged_text", ""), "bbox": q["bbox"]} for q in questions],
        }
        page_dir_dest = parsing_dir / f"page_{short_id}"
        page_dir_dest.mkdir(parents=True, exist_ok=True)
        out_json_path = page_dir_dest / f"page_{short_id}.json"
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(page_json, f, ensure_ascii=False, indent=2)
        print(f"Wrote json: {out_json_path}")

        sat_pages_for_index[short_id] = {
            "json": f"/data/outputs_parsing/sat_math_odd/page_{short_id}/page_{short_id}.json",
        }

    # Update datasets.json (frontend/public and keep same structure)
    datasets_path = FRONTEND_PUBLIC / "datasets.json"
    if not datasets_path.exists():
        datasets_path = DATA / "datasets.json"
    if not datasets_path.exists():
        print("datasets.json not found, creating minimal one with sat only.")
        index = {"years": {"sat": {"pages": sat_pages_for_index}}}
        FRONTEND_PUBLIC.mkdir(parents=True, exist_ok=True)
        with open(FRONTEND_PUBLIC / "datasets.json", "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
        print("Created", FRONTEND_PUBLIC / "datasets.json")
        return

    with open(datasets_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    if "years" not in index:
        index["years"] = {}
    index["years"]["sat"] = {"pages": sat_pages_for_index}
    with open(datasets_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    print(f"Updated {datasets_path} with sat year ({len(sat_pages_for_index)} pages).")

    # Copy SAT data into frontend/public/data so fetch("/data/...") works when serving from Vite
    public_data = WORKSPACE / "frontend" / "public" / "data"
    if public_data.exists():
        (public_data / "outputs_parsing").mkdir(parents=True, exist_ok=True)
        dest_parsing = public_data / "outputs_parsing" / "sat_math_odd"
        if dest_parsing.exists():
            shutil.rmtree(dest_parsing)
        shutil.copytree(parsing_dir, dest_parsing)
        dest_images = public_data / "sat_math_odd"
        dest_images.mkdir(parents=True, exist_ok=True)
        for f in images_dir.glob("*.png"):
            shutil.copy2(f, dest_images / f.name)
        print(f"Copied SAT data to {public_data} for frontend /data/ serving.")


if __name__ == "__main__":
    main()
