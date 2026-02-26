# -*- coding: utf-8 -*-
"""
연도별 OCR 결과 CSV 파이프라인 (GT + 4개 모델 비교용).
- 출력: dla/ocr_outputs/{year}_ocr_results.csv (prob_id, GT, paddle, DeepSeek-OCR, DeepSeek-OCR2, GLM)
- 메타: dla/ocr_outputs/meta_logs/{year}_deepseek_meta.csv (DeepSeek 원시 출력 보관)
- 처리 순서: 2024 → 2025 → 2026 → 2022 → 2023
- DeepSeek은 venv_ocr에서 실행, 후처리 적용 후 메타 별도 저장.
- 병렬: --parallel_models 시 이미지당 4개 모델을 스레드 풀로 동시 실행 (속도 향상, GPU 메모리 사용량 증가).
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
    """CSV 셀을 한 줄로 (Excel 등에서 깨지지 않도록)."""
    if s is None:
        return ""
    return str(s).replace("\r\n", " ").replace("\n", " ").replace("\r", " ").strip()


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
    """(prob_id, prob_base64_bytes or None) from _dla CSV."""
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
    """(prob_id, image_path or None) from TSV. prob_img_path 후보 경로로 해석."""
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
        # path.stem 예: 2026_math_odd → 실제 이미지는 data/2026_math_odd/*.png 에 있을 수 있음
        candidates = [
            data_dir / raw_path,
            data_dir / name,
            data_dir / "crop_image" / name,
            path.parent / path.stem / name,
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
    """2022: (prob_id, image_path) from image_dir/*.png, *_crop* 제외."""
    if not image_dir.exists():
        return []
    out = []
    for p in sorted(image_dir.glob("*.png")):
        if "_crop" in p.name:
            continue
        out.append((p.stem, p))
    return out


def _empty_cuda_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _run_one_model(
    model_key: str,
    image_path: str,
) -> tuple[str, str, str | None]:
    """
    단일 모델 실행. 반환 (model_key, text_for_csv, raw_for_meta_or_None).
    DeepSeek 계열만 raw를 채움.
    """
    sys.path.insert(0, str(SCRIPT_DIR))
    if model_key == "paddle_ocr_vl":
        from ocr_compare_utils import run_paddle_ocr_vl
        try:
            return (model_key, run_paddle_ocr_vl(image_path), None)
        except Exception as e:
            return (model_key, f"[paddle_ocr_vl error] {e}", None)
    if model_key == "deepseek_ocr":
        from ocr_compare_utils import run_deepseek_ocr_with_meta
        try:
            processed, raw = run_deepseek_ocr_with_meta(image_path)
            return (model_key, processed, raw or "")
        except Exception as e:
            return (model_key, f"[deepseek_ocr error] {e}", "")
    if model_key == "deepseek_ocr2":
        from ocr_compare_utils import run_deepseek_ocr2_with_meta
        try:
            processed, raw = run_deepseek_ocr2_with_meta(image_path)
            return (model_key, processed, raw or "")
        except Exception as e:
            return (model_key, f"[deepseek_ocr2 error] {e}", "")
    if model_key == "glm_ocr":
        from ocr_compare_utils import run_glm_ocr
        try:
            return (model_key, run_glm_ocr(image_path), None)
        except Exception as e:
            return (model_key, f"[glm_ocr error] {e}", None)
    return (model_key, "", None)


def run_ocr_parallel(image_path: str) -> tuple[dict[str, str], dict[str, str]]:
    """
    이미지당 4개 모델을 스레드 풀로 병렬 실행. DeepSeek 원시 출력은 meta_raw에 포함.
    """
    model_keys = ["paddle_ocr_vl", "deepseek_ocr", "deepseek_ocr2", "glm_ocr"]
    outputs: dict[str, str] = {}
    meta_raw: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(_run_one_model, k, image_path): k for k in model_keys}
        for fut in as_completed(futures):
            key, text, raw = fut.result()
            outputs[key] = text
            if raw is not None:
                meta_raw["raw_deepseek_ocr" if key == "deepseek_ocr" else "raw_deepseek_ocr2"] = raw
    _empty_cuda_cache()
    return outputs, meta_raw


def run_ocr_sequential(image_path: str) -> tuple[dict[str, str], dict[str, str]]:
    """
    이미지당 4개 모델을 순차 실행. DeepSeek은 subprocess(venv_ocr)라 GPU를 독점하므로
    먼저 실행하고, 종료 후 Paddle/GLM을 실행해 GPU 메모리 충돌을 피함.
    """
    sys.path.insert(0, str(SCRIPT_DIR))
    from ocr_compare_utils import (
        run_paddle_ocr_vl,
        run_deepseek_ocr_with_meta,
        run_deepseek_ocr2_with_meta,
        run_glm_ocr,
    )
    outputs: dict[str, str] = {}
    meta_raw: dict[str, str] = {}

    # DeepSeek 먼저 (subprocess가 GPU 사용 후 종료해 메모리 해제)
    try:
        processed, raw = run_deepseek_ocr_with_meta(image_path)
        outputs["deepseek_ocr"] = processed
        meta_raw["raw_deepseek_ocr"] = raw or ""
    except Exception as e:
        outputs["deepseek_ocr"] = f"[deepseek_ocr error] {e}"
        meta_raw["raw_deepseek_ocr"] = ""
    _empty_cuda_cache()

    try:
        processed, raw = run_deepseek_ocr2_with_meta(image_path)
        outputs["deepseek_ocr2"] = processed
        meta_raw["raw_deepseek_ocr2"] = raw or ""
    except Exception as e:
        outputs["deepseek_ocr2"] = f"[deepseek_ocr2 error] {e}"
        meta_raw["raw_deepseek_ocr2"] = ""
    _empty_cuda_cache()

    # Paddle
    try:
        outputs["paddle_ocr_vl"] = run_paddle_ocr_vl(image_path)
    except Exception as e:
        outputs["paddle_ocr_vl"] = f"[paddle_ocr_vl error] {e}"
    _empty_cuda_cache()

    # GLM
    try:
        outputs["glm_ocr"] = run_glm_ocr(image_path)
    except Exception as e:
        outputs["glm_ocr"] = f"[glm_ocr error] {e}"
    _empty_cuda_cache()

    return outputs, meta_raw


def main() -> None:
    parser = argparse.ArgumentParser(
        description="연도별 OCR 결과 CSV 생성 (GT + paddle, DeepSeek-OCR, DeepSeek-OCR2, GLM)"
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=WORKSPACE_ROOT / "data",
        help="데이터 디렉터리",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=SCRIPT_DIR / "ocr_outputs",
        help="출력 디렉터리 (아래에 {year}_ocr_results.csv, meta_logs/ 생성)",
    )
    parser.add_argument(
        "--year",
        type=int,
        choices=[2022, 2023, 2024, 2025, 2026],
        default=None,
        help="특정 연도만 처리 (미지정 시 전체, 2023은 마지막)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="연도당 처리할 문제 수 제한 (테스트용)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="OCR 실행 없이 prob_id, GT만 있는 CSV만 생성",
    )
    parser.add_argument(
        "--parallel_models",
        action="store_true",
        help="이미지당 4개 모델을 병렬 실행 (기본: 순차 실행, 메모리 절약)",
    )
    args = parser.parse_args()

    data_dir = _ensure_data_dir(args.data_dir)
    output_dir = args.output_dir if args.output_dir.is_absolute() else SCRIPT_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = output_dir / "meta_logs"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # 컬럼: prob_id, GT, paddle, DeepSeek-OCR, DeepSeek-OCR2, GLM
    result_columns = ["prob_id", "GT", "paddle", "DeepSeek-OCR", "DeepSeek-OCR2", "GLM"]
    model_keys = ["paddle_ocr_vl", "deepseek_ocr", "deepseek_ocr2", "glm_ocr"]
    model_col_names = ["paddle", "DeepSeek-OCR", "DeepSeek-OCR2", "GLM"]

    # 처리 순서: 2024 → 2025 → 2026 → 2022 → 2023
    years = [args.year] if args.year else [2024, 2025, 2026, 2022, 2023]

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
            problem_inputs = [
                (pid, str(p) if p else None, "path")
                for pid, p in problems
            ]
        elif year == 2026:
            tsv_path = data_dir / "2026_math_odd.tsv"
            if not tsv_path.exists():
                print(f"Skip {year}: not found {tsv_path}")
                continue
            gt_map = load_gt_tsv(tsv_path)
            problems = load_problems_tsv(tsv_path, data_dir)
            problem_inputs = []
            for pid, img_path in problems:
                if img_path is None:
                    print(f"  [{year}] No image for {pid}; GT only")
                problem_inputs.append((pid, str(img_path) if img_path else None, "path"))
        else:
            # 2023, 2024, 2025 (DLA CSV) — 2023은 DLA 없을 때 TSV 또는 통합 CSV(prob_base64 포함)로 대체
            dla_path = data_dir / f"{year}_math_odd_dla.csv"
            gt_path = data_dir / f"{year}_math_odd.csv"
            tsv_path = data_dir / f"{year}_math_odd.tsv"
            if year == 2023 and not dla_path.exists() and gt_path.exists():
                import pandas as pd
                df_head = pd.read_csv(gt_path, nrows=1)
                if "prob_base64" in df_head.columns:
                    gt_map = load_gt_csv(gt_path)
                    problems = load_problems_dla_csv(gt_path)
                    problem_inputs = []
                    for pid, b64 in problems:
                        if b64 is None:
                            continue
                        problem_inputs.append((pid, b64, "base64"))
                elif tsv_path.exists():
                    gt_map = load_gt_csv(gt_path)
                    problems = load_problems_tsv(tsv_path, data_dir)
                    problem_inputs = []
                    for pid, img_path in problems:
                        if img_path is None:
                            print(f"  [{year}] No image for {pid}; GT only")
                        problem_inputs.append((pid, str(img_path) if img_path else None, "path"))
                else:
                    print(f"Skip {year}: not found {dla_path} and no prob_base64 in {gt_path.name}")
                    continue
            elif not dla_path.exists():
                print(f"Skip {year}: not found {dla_path}")
                continue
            elif not gt_path.exists():
                print(f"Skip {year}: not found {gt_path}")
                continue
            else:
                gt_map = load_gt_csv(gt_path)
                problems = load_problems_dla_csv(dla_path)
                problem_inputs = []
                for pid, b64 in problems:
                    if b64 is None:
                        continue
                    problem_inputs.append((pid, b64, "base64"))

        if args.limit:
            problem_inputs = problem_inputs[: args.limit]

        out_path = output_dir / f"{year}_ocr_results.csv"
        meta_path = meta_dir / f"{year}_deepseek_meta.csv"
        n_probs = len(problem_inputs)
        print(f"Processing {year}: {n_probs} problems -> {out_path}")

        iter_problems = (
            tqdm(problem_inputs, desc=f"{year}", unit="img", total=n_probs)
            if tqdm else problem_inputs
        )

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(result_columns)
            f.flush()

            meta_rows: list[tuple[str, str, str]] = []

            for item in iter_problems:
                pid = item[0]
                gt = gt_map.get(pid, "")

                if args.dry_run:
                    row = [pid, _cell_one_line(gt), "", "", "", ""]
                    writer.writerow(row)
                    f.flush()
                    continue

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
                    if image_path and os.path.exists(image_path):
                        if args.parallel_models:
                            outputs, meta_raw = run_ocr_parallel(image_path)
                        else:
                            outputs, meta_raw = run_ocr_sequential(image_path)
                        row = [pid] + [_cell_one_line(gt)] + [
                            _cell_one_line(outputs.get(k, "")) for k in model_keys
                        ]
                        meta_rows.append((
                            pid,
                            meta_raw.get("raw_deepseek_ocr", ""),
                            meta_raw.get("raw_deepseek_ocr2", ""),
                        ))
                    else:
                        row = [pid, _cell_one_line(gt), "", "", "", ""]
                finally:
                    if item[2] == "base64" and image_path.startswith(tempfile.gettempdir()):
                        try:
                            os.unlink(image_path)
                        except Exception:
                            pass

                writer.writerow(row)
                f.flush()

            # 메타 로그 저장 (원시 DeepSeek 출력)
            if meta_rows and not args.dry_run:
                with open(meta_path, "w", newline="", encoding="utf-8") as mf:
                    mwriter = csv.writer(mf, quoting=csv.QUOTE_NONNUMERIC)
                    mwriter.writerow(["prob_id", "raw_deepseek_ocr", "raw_deepseek_ocr2"])
                    for pid, r1, r2 in meta_rows:
                        mwriter.writerow([pid, r1, r2])
                print(f"  Meta: {meta_path}")

        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
