# -*- coding: utf-8 -*-
"""
연도별 OCR 출력 CSV 생성 (4개 모델 + GT).
- 모델: paddle, DeepSeek-OCR, DeepSeek-OCR2, GLM (4개)
- 연도: 2022, 2023, 2024, 2025, 2026 (실행 순서: 2022 → 2024 → 2025 → 2026 → 2023)
- 각 행: prob_id, paddle, deepseek_ocr, deepseek_ocr2, glm, GT
- 2022: data/2022_math_odd.tsv (GT), data/2022_math_odd/*.png (이미지, *_crop* 제외)
- 2023~2025: _dla.csv (prob_base64), GT는 _math_odd.csv
- 2026: 2026_math_odd.tsv (prob_img_path, prob_desc)
"""
from __future__ import annotations

import argparse
import base64
import csv
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent


def _cell_one_line(s: str) -> str:
    """CSV 셀을 한 줄로 만듦 (Excel/다운로드에서 한 행이 여러 줄로 깨지지 않도록)."""
    if s is None:
        return ""
    s = str(s)
    return s.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").strip()


def _ensure_data_dir(data_dir: Path) -> Path:
    if data_dir.is_absolute():
        return data_dir
    return WORKSPACE_ROOT / data_dir


def load_gt_csv(path: Path) -> dict[str, str]:
    """prob_id -> prob_desc (GT) from CSV."""
    import pandas as pd
    df = pd.read_csv(path)
    desc = df["prob_desc"].fillna("").astype(str)
    return dict(zip(df["prob_id"].astype(str), desc))


def load_gt_tsv(path: Path) -> dict[str, str]:
    """prob_id -> prob_desc (GT) from TSV."""
    import pandas as pd
    df = pd.read_csv(path, sep="\t")
    desc = df["prob_desc"].fillna("").astype(str)
    return dict(zip(df["prob_id"].astype(str), desc))


def load_problems_dla_csv(path: Path) -> list[tuple[str, bytes | None]]:
    """(prob_id, prob_base64_bytes or None) list from _dla CSV. Image from prob_base64."""
    import pandas as pd
    df = pd.read_csv(path)
    out = []
    for _, row in df.iterrows():
        pid = str(row["prob_id"])
        b64 = row.get("prob_base64")
        if pd.isna(b64) or not b64:
            out.append((pid, None))
            continue
        try:
            out.append((pid, base64.b64decode(b64)))
        except Exception:
            out.append((pid, None))
    return out


def load_problems_tsv(path: Path, data_dir: Path) -> list[tuple[str, Path | None]]:
    """(prob_id, image_path or None) list from TSV. Path from prob_img_path, resolved against data_dir."""
    import pandas as pd
    df = pd.read_csv(path, sep="\t")
    out = []
    for _, row in df.iterrows():
        pid = str(row["prob_id"])
        raw_path = row.get("prob_img_path")
        if pd.isna(raw_path) or not raw_path:
            out.append((pid, None))
            continue
        raw_path = str(raw_path).strip()
        name = Path(raw_path).name
        # TSV 내 위치가 실제 폴더와 다를 수 있음: 여러 후보 시도
        candidates = [
            data_dir / raw_path,
            data_dir / name,
            data_dir / "crop_image" / name,
            WORKSPACE_ROOT / raw_path,
            WORKSPACE_ROOT / name,
            WORKSPACE_ROOT / "crop_image" / name,
            path.parent / raw_path,
            path.parent / name,
            path.parent / "crop_image" / name,
        ]
        found = None
        for c in candidates:
            if c.exists():
                found = c
                break
        out.append((pid, found))
    return out


def load_problems_2022_image_dir(image_dir: Path) -> list[tuple[str, Path]]:
    """2022: (prob_id, image_path) from image_dir/*.png, *_crop* 파일 제외. prob_id = stem."""
    if not image_dir.exists():
        return []
    out = []
    for p in sorted(image_dir.glob("*.png")):
        if "_crop" in p.name:
            continue
        out.append((p.stem, p))
    return out


def _run_one_model(model_key: str, image_path: str) -> tuple[str, str]:
    """단일 모델 실행 (병렬용). 반환 (model_key, text)."""
    sys.path.insert(0, str(SCRIPT_DIR))
    from ocr_compare_utils import MODEL_RUNNERS
    runner = MODEL_RUNNERS.get(model_key)
    if runner is None:
        return (model_key, "")
    try:
        return (model_key, runner(image_path))
    except Exception as e:
        return (model_key, f"[{model_key} error] {e}")


def run_ocr_and_collect(
    image_path: str,
    models: list[str],
    parallel_models: bool = True,
) -> dict[str, str]:
    """Run OCR models (4개 병렬) and return model_name -> text."""
    if not parallel_models or len(models) <= 1:
        sys.path.insert(0, str(SCRIPT_DIR))
        from ocr_compare_utils import run_ocr_models
        result = run_ocr_models(image_path, models=models)
        return result.get("outputs", {})

    out = {}
    max_workers = min(4, len(models))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_run_one_model, m, image_path): m for m in models}
        for fut in as_completed(futures):
            key, text = fut.result()
            out[key] = text
    return out


def main():
    parser = argparse.ArgumentParser(description="Build year-wise OCR output CSVs (4 models + GT)")
    parser.add_argument("--data_dir", type=Path, default=WORKSPACE_ROOT / "data", help="Data directory (contains *_math_odd*.csv/tsv)")
    parser.add_argument("--output_dir", type=Path, default=SCRIPT_DIR / "ocr_outputs", help="Output directory for CSVs")
    parser.add_argument("--year", type=int, choices=[2022, 2023, 2024, 2025, 2026], default=None, help="Single year to process (default: all, 2023 last)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of problems per year (for testing)")
    parser.add_argument("--dry_run", action="store_true", help="Only build CSV with prob_id and GT; OCR columns empty (no model run)")
    parser.add_argument("--no_parallel_models", action="store_true", help="Run 4 models sequentially per image (default: parallel)")
    args = parser.parse_args()

    data_dir = _ensure_data_dir(args.data_dir)
    output_dir = args.output_dir if args.output_dir.is_absolute() else SCRIPT_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    models = ["paddle_ocr_vl", "deepseek_ocr", "deepseek_ocr2", "glm_ocr"]
    # CSV 컬럼명 및 run_ocr_models 반환 키
    model_columns = ["paddle", "deepseek_ocr", "deepseek_ocr2", "glm"]
    model_keys = ["paddle_ocr_vl", "deepseek_ocr", "deepseek_ocr2", "glm_ocr"]

    # 2023은 맨 마지막에 진행
    years = [args.year] if args.year else [2022, 2024, 2025, 2026, 2023]

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    for year in years:
        if year == 2022:
            tsv_path = data_dir / "2022_math_odd.tsv"
            image_dir = data_dir / "2022_math_odd"
            if not tsv_path.exists():
                print(f"Skip {year}: not found {tsv_path}")
                continue
            gt_map = load_gt_tsv(tsv_path)
            problems = load_problems_2022_image_dir(image_dir)
            problem_inputs = []
            for pid, img_path in problems:
                problem_inputs.append((pid, str(img_path) if img_path else None, "path"))
        elif year == 2026:
            tsv_path = data_dir / "2026_math_odd.tsv"
            if not tsv_path.exists():
                print(f"Skip {year}: not found {tsv_path}")
                continue
            gt_map = load_gt_tsv(tsv_path)
            problems = load_problems_tsv(tsv_path, data_dir)
            # 이미지: 경로 사용. 경로 없어도 GT만이라도 넣기 위해 (pid, path_or_None, "path") 유지
            problem_inputs = []
            for pid, img_path in problems:
                if img_path is None:
                    print(f"  [{year}] No image path for {pid}; row will have GT only (OCR empty)")
                problem_inputs.append((pid, str(img_path) if img_path else None, "path"))
        else:
            # 2023, 2024, 2025
            dla_path = data_dir / f"{year}_math_odd_dla.csv"
            gt_path = data_dir / f"{year}_math_odd.csv"
            if not dla_path.exists():
                print(f"Skip {year}: not found {dla_path}")
                continue
            if not gt_path.exists():
                print(f"Skip {year}: not found {gt_path}")
                continue
            gt_map = load_gt_csv(gt_path)
            problems = load_problems_dla_csv(dla_path)
            problem_inputs = []
            for pid, b64 in problems:
                if b64 is None:
                    print(f"  [{year}] Skip {pid}: no prob_base64")
                    continue
                problem_inputs.append((pid, b64, "base64"))

        if args.limit:
            problem_inputs = problem_inputs[: args.limit]

        out_path = output_dir / f"{year}.csv"
        n_probs = len(problem_inputs)
        print(f"Processing {year}: {n_probs} problems -> {out_path}")

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(["prob_id"] + model_columns + ["GT"])
            f.flush()

            iter_problems = tqdm(problem_inputs, desc=f"{year}", unit="prob", total=n_probs) if tqdm else problem_inputs

            for idx, item in enumerate(iter_problems):
                pid = item[0]
                gt = gt_map.get(pid, "")

                if args.dry_run:
                    row = [pid] + [""] * len(model_keys) + [gt]
                elif item[2] == "path" and item[1] is None:
                    row = [pid] + [""] * len(model_keys) + [gt]
                else:
                    if item[2] == "path":
                        image_path = item[1]
                    else:
                        b64 = item[1]
                        suffix = "png" if b64[:8] == b"\x89PNG\r\n\x1a\n" else "jpg"
                        tmp = tempfile.NamedTemporaryFile(suffix="." + suffix, delete=False)
                        tmp.write(b64)
                        tmp.close()
                        image_path = tmp.name

                    try:
                        outputs = run_ocr_and_collect(
                            image_path, models, parallel_models=not args.no_parallel_models
                        )
                    finally:
                        if item[2] == "base64" and image_path.startswith(tempfile.gettempdir()):
                            try:
                                os.unlink(image_path)
                            except Exception:
                                pass

                    row = [pid] + [outputs.get(k, "") for k in model_keys] + [gt]

                # 셀 내 줄바꿈·연속공백 제거 → 한 행 = 한 줄 (Excel/다운로드에서 깨지지 않도록)
                row = [_cell_one_line(c) for c in row]
                writer.writerow(row)
                f.flush()

        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
