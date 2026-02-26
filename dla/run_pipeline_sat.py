# -*- coding: utf-8 -*-
"""
SAT 전용: DeepSeek-OCR-2로 문항별 crop 파이프라인
- data/SAT 아래 "숫자-" 로 시작하는 폴더(예: 11-images, 9-images)를 찾아
  각 폴더별로 parser를 실행하고, 결과를 outputs_dla/sat/<폴더명> 에 저장합니다.
- CSAT 결과(outputs_dla/_gpu_chunk_*, 2023_csat_* 등)와 구분됩니다.

사용법:
  # 기본: data/SAT 에서 N-images 폴더 탐색 → outputs_dla/sat/ 에 저장
  python run_pipeline_sat.py

  # 데이터/출력 경로 지정
  python run_pipeline_sat.py --data_root /path/to/SAT --output_dir outputs_dla

  # 특정 폴더만
  python run_pipeline_sat.py --input_dirs data/SAT/11-images data/SAT/9-images
"""
import os
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# "숫자-" 로 시작하는 폴더명 (예: 11-images, 9-images, 8-images)
SAT_FOLDER_PATTERN = re.compile(r"^\d+-")


def get_python(venv_name: str) -> str:
    """
    SAT 파이프라인용 Python 선택.
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


def find_sat_folders(data_root: Path) -> list[Path]:
    """data_root 아래에서 이름이 '숫자-'로 시작하는 디렉토리만 반환 (정렬)."""
    if not data_root.is_dir():
        return []
    out = []
    for p in data_root.iterdir():
        if p.is_dir() and SAT_FOLDER_PATTERN.match(p.name):
            out.append(p)
    return sorted(out, key=lambda x: x.name)


def run_parser_for_sat(
    input_dir: Path,
    output_base: Path,
    python_path: str,
    parser_script: Path,
    gpu_id: int | None = None,
) -> bool:
    """
    DeepSeek-OCR-2(v2) 환경으로 parser.py를 한 폴더에 대해 실행.
    결과는 output_base / "sat" / input_dir.name 에 저장됩니다.
    """
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
    env["SAT_MODE"] = "1"  # 검은 박스 내 숫자 별도 탐지 + sub_title 앵커 우선
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [python_path, "-I", str(parser_script), "--input_dirs", str(input_dir.resolve())]
    print(f"  Running: {' '.join(cmd)}")
    print(f"  OUT_DIR={env['OUT_DIR']} (→ {sat_out / input_dir.name})")
    try:
        subprocess.check_call(cmd, env=env)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Failed: {e}", file=sys.stderr)
        return False


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="SAT 전용 DeepSeek-OCR-2 문항별 crop → outputs_dla/sat/<폴더명>"
    )
    ap.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="SAT 이미지 폴더들의 상위 디렉토리 (기본: dla 기준 data/SAT)",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default="outputs_dla",
        help="출력 루트 디렉토리 (기본: outputs_dla). 실제 저장: <output_dir>/sat/<폴더명>",
    )
    ap.add_argument(
        "--input_dirs",
        nargs="*",
        type=str,
        default=None,
        help="실행할 입력 디렉토리 목록. 지정 시 data_root 탐색 없이 이 목록만 사용.",
    )
    ap.add_argument(
        "--gpu",
        type=int,
        default=1,
        help="사용할 GPU ID (미지정 시 CUDA_VISIBLE_DEVICES 유지)",
    )
    args = ap.parse_args()

    # 경로 보정
    output_base = Path(args.output_dir)
    if not output_base.is_absolute():
        output_base = SCRIPT_DIR / output_base

    python_path = get_python("venv_ocr")
    parser_script = SCRIPT_DIR / "parser.py"

    if args.input_dirs:
        input_dirs = [Path(d) for d in args.input_dirs]
        for d in input_dirs:
            if not d.exists():
                print(f"  Error: directory not found: {d}", file=sys.stderr)
                sys.exit(1)
    else:
        # 기본: workspace/data/SAT (dla 상위의 data/SAT)
        data_root = args.data_root or str(SCRIPT_DIR.parent / "data" / "SAT")
        data_root = Path(data_root)
        if not data_root.is_absolute():
            data_root = SCRIPT_DIR / data_root
        input_dirs = find_sat_folders(data_root)
        if not input_dirs:
            print(f"  No 'N-images' style folders under {data_root}", file=sys.stderr)
            sys.exit(1)
        print(f"  Found SAT folders: {[p.name for p in input_dirs]}")

    print(f"  Output base: {output_base} (SAT results → {output_base / 'sat'}/<folder>)")
    (output_base / "sat").mkdir(parents=True, exist_ok=True)

    failed = []
    for inp in input_dirs:
        ok = run_parser_for_sat(
            inp, output_base, python_path, parser_script, gpu_id=args.gpu
        )
        if not ok:
            failed.append(str(inp))

    if failed:
        print(f"  Failed folders: {failed}", file=sys.stderr)
        sys.exit(1)
    print("  All SAT folders completed.")


if __name__ == "__main__":
    main()
