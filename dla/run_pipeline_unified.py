# -*- coding: utf-8 -*-
"""
Unified pipeline: classify each image as CSAT or SAT, then run the appropriate
parser (CSAT = DeepSeek-OCR default, SAT = DeepSeek-OCR-2 + 768).
- Input: directory or list of directories containing exam page images.
- Output: outputs_dla/sat/<group> for SAT, outputs_dla/<group> for CSAT.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

IMAGE_EXTS = {".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"}


def get_python(venv_name: str) -> str:
    """
    OCR 파이프라인용 Python 선택.
    우선순위:
      1) DLA_OCR_VENV 환경변수
      2) venv_deepseek
      3) 요청된 venv_name (기본: venv_ocr)
      4) 시스템 Python
    """
    explicit = os.environ.get("DLA_OCR_VENV")
    if explicit:
        explicit_python = SCRIPT_DIR / explicit / "bin" / "python"
        if explicit_python.exists():
            return str(explicit_python)
    deepseek_python = SCRIPT_DIR / "venv_deepseek" / "bin" / "python"
    if deepseek_python.exists():
        return str(deepseek_python)
    venv_python = SCRIPT_DIR / venv_name / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    print(f"  Warning: {venv_name} not found, using system Python.", file=sys.stderr)
    return sys.executable


def collect_images(input_dirs: list[Path]) -> list[Path]:
    paths = []
    for d in input_dirs:
        if not d.is_dir():
            continue
        for p in d.iterdir():
            if p.suffix in IMAGE_EXTS:
                paths.append(p.resolve())
    return sorted(set(paths))


def run_parser_csat(
    input_dir: Path,
    output_dir: Path,
    python_path: str,
    parser_script: Path,
    gpu_id: int | None = None,
) -> bool:
    """Run parser with default CSAT config (DeepSeek-OCR, 640)."""
    env = os.environ.copy()
    # 노트북/시스템 PYTHONPATH 오염 차단
    env.pop("PYTHONPATH", None)
    env.pop("VIRTUAL_ENV", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PATH"] = str(Path(python_path).resolve().parent) + os.pathsep + env.get("PATH", "")
    env["OUT_DIR"] = str(output_dir)
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [python_path, "-I", str(parser_script), "--input_dirs", str(input_dir.resolve())]
    print(f"  Running CSAT parser: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd, env=env)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  CSAT parser failed: {e}", file=sys.stderr)
        return False


def run_parser_sat(
    input_dir: Path,
    output_base: Path,
    python_path: str,
    parser_script: Path,
    gpu_id: int | None = None,
) -> bool:
    """Run parser with SAT config (DeepSeek-OCR-2, 768)."""
    sat_out = output_base / "sat"
    env = os.environ.copy()
    # 노트북/시스템 PYTHONPATH 오염 차단
    env.pop("PYTHONPATH", None)
    env.pop("VIRTUAL_ENV", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PATH"] = str(Path(python_path).resolve().parent) + os.pathsep + env.get("PATH", "")
    env["MODEL_NAME"] = "deepseek-ai/DeepSeek-OCR-2"
    env["IMAGE_SIZE"] = "768"
    env["OUT_DIR"] = str(sat_out)
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [python_path, "-I", str(parser_script), "--input_dirs", str(input_dir.resolve())]
    print(f"  Running SAT parser: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd, env=env)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  SAT parser failed: {e}", file=sys.stderr)
        return False


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description="Unified pipeline: classify images as CSAT/SAT then run appropriate parser."
    )
    ap.add_argument(
        "input_dirs",
        nargs="+",
        type=str,
        help="Input directory (or directories) containing exam page images",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default="outputs_dla",
        help="Output root (default: outputs_dla)",
    )
    ap.add_argument(
        "--method",
        choices=["heuristic", "cnn", "hybrid"],
        default="heuristic",
        help="Classification method (default: heuristic)",
    )
    ap.add_argument(
        "--cnn",
        type=str,
        default=None,
        help="Path to CNN checkpoint for cnn/hybrid method",
    )
    ap.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU ID for parser (optional)",
    )
    args = ap.parse_args()

    input_dirs = [Path(d) for d in args.input_dirs]
    for d in input_dirs:
        if not d.exists():
            print(f"  Error: directory not found: {d}", file=sys.stderr)
            sys.exit(1)

    images = collect_images(input_dirs)
    if not images:
        print("  No images found in input directories.")
        sys.exit(1)
    print(f"  Found {len(images)} images.")

    # Classify (Classifier 패키지 사용)
    sys.path.insert(0, str(SCRIPT_DIR))
    from Classifier import classify_batch

    paths_str = [str(p) for p in images]
    labels = classify_batch(
        paths_str,
        method=args.method,
        cnn_model_path=args.cnn,
    )
    sat_paths = [images[i] for i, lb in enumerate(labels) if lb == "sat"]
    csat_paths = [images[i] for i, lb in enumerate(labels) if lb == "csat"]
    print(f"  Classified: SAT={len(sat_paths)}, CSAT={len(csat_paths)}")

    output_base = Path(args.output_dir)
    if not output_base.is_absolute():
        output_base = SCRIPT_DIR / output_base
    output_base.mkdir(parents=True, exist_ok=True)
    python_path = get_python("venv_ocr")
    parser_script = SCRIPT_DIR / "parser.py"
    gpu = args.gpu

    failed = []
    tmp_dirs = []

    try:
        if sat_paths:
            tmp_sat = Path(tempfile.mkdtemp(prefix="unified_sat_", dir=output_base))
            tmp_dirs.append(tmp_sat)
            for i, p in enumerate(sat_paths):
                link = tmp_sat / f"{i:05d}_{p.name}"
                if not link.exists():
                    try:
                        os.symlink(p, link)
                    except OSError:
                        shutil.copy2(p, link)
            if not run_parser_sat(tmp_sat, output_base, python_path, parser_script, gpu):
                failed.append("SAT")

        if csat_paths:
            tmp_csat = Path(tempfile.mkdtemp(prefix="unified_csat_", dir=output_base))
            tmp_dirs.append(tmp_csat)
            for i, p in enumerate(csat_paths):
                link = tmp_csat / f"{i:05d}_{p.name}"
                if not link.exists():
                    try:
                        os.symlink(p, link)
                    except OSError:
                        shutil.copy2(p, link)
            # Parser writes to OUT_DIR / group_name; group_name = input_dir.name
            if not run_parser_csat(tmp_csat, output_base, python_path, parser_script, gpu):
                failed.append("CSAT")
    finally:
        for td in tmp_dirs:
            try:
                shutil.rmtree(td, ignore_errors=True)
            except Exception:
                pass

    if failed:
        print(f"  Failed: {failed}", file=sys.stderr)
        sys.exit(1)
    print("  Unified pipeline completed.")


if __name__ == "__main__":
    main()
