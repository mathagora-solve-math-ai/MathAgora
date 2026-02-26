# -*- coding: utf-8 -*-
"""
run_pipeline.py
DLA 전체 파이프라인 오케스트레이터 (5단계)
  Step 1: PDF → Images        (problem PDF만 필터링)
  Step 2: DeepSeek-OCR Parser  (venv_ocr 사용, 멀티GPU 병렬)
  Step 3: COCO 변환
  Step 4: COCO → YOLO 변환
  Step 5: YOLO12 학습           (venv_yolo 사용, 멀티GPU)

사용법:
  # 전체 파이프라인 실행 (4GPU 자동감지)
  python run_pipeline.py --pdf_dir data/original

  # 특정 단계만 실행
  python run_pipeline.py --steps 1 2 3
  python run_pipeline.py --steps 4 5

  # GPU 지정
  python run_pipeline.py --steps 2 --gpus 0 1 2 3
  python run_pipeline.py --steps 5 --epochs 200 --batch 32 --gpus 0 1 2 3
"""

import os
import glob
import shutil
import argparse
import subprocess
import sys
import time
import math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


SCRIPT_DIR = Path(__file__).resolve().parent


# ─────────────────────────────────────────────────────────────
# GPU 유틸리티
# ─────────────────────────────────────────────────────────────
def detect_gpus():
    """사용 가능한 NVIDIA GPU 목록을 반환합니다."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        )
        return [int(x.strip()) for x in out.strip().split("\n") if x.strip()]
    except Exception:
        return [0]


def get_python(venv_name):
    """가상환경의 Python 인터프리터 경로를 반환합니다."""
    venv_python = SCRIPT_DIR / venv_name / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    print(f"  Warning: {venv_name} not found, using system Python.")
    return sys.executable


def run_command(cmd, env=None):
    """커맨드를 실행하고 에러 시 종료합니다."""
    print(f"  Running: {' '.join(str(c) for c in cmd)}")
    try:
        subprocess.check_call([str(c) for c in cmd], env=env)
    except subprocess.CalledProcessError as e:
        print(f"  Error executing command: {e}")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────
# Step 1: PDF → Images
# ─────────────────────────────────────────────────────────────
def step1_pdf_to_images(args):
    """PDF → Images (problem PDF만)"""
    print("\n" + "=" * 60)
    print(" Step 1: Convert PDFs to Images (problem only)")
    print("=" * 60)

    script = SCRIPT_DIR / "pdf_to_images.py"
    python = get_python("venv_ocr")

    cmd = [
        python, str(script),
        "--input_dir", str(args.pdf_dir),
        "--output_dir", str(args.data_dir),
        "--dpi", str(args.dpi),
        "--filter", "problem",
    ]
    run_command(cmd)


# ─────────────────────────────────────────────────────────────
# Step 2: DeepSeek-OCR Parser (멀티GPU 병렬)
# ─────────────────────────────────────────────────────────────
def _run_parser_on_gpu(gpu_id, image_dir, output_dir, python_path, parser_script):
    """단일 GPU에서 parser.py를 실행하는 워커 함수."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["OUT_DIR"] = str(output_dir)
    env["FAST_SKIP_VIS"] = "0"
    env["FAST_SKIP_B64"] = "1"

    cmd = [python_path, str(parser_script), "--input_dirs", str(image_dir)]
    print(f"  [GPU {gpu_id}] Processing {image_dir.name} ...")
    t0 = time.time()
    try:
        subprocess.check_call(cmd, env=env)
        elapsed = time.time() - t0
        print(f"  [GPU {gpu_id}] Completed {image_dir.name} in {elapsed:.0f}s")
        return gpu_id, True, None
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - t0
        print(f"  [GPU {gpu_id}] FAILED {image_dir.name} after {elapsed:.0f}s: {e}")
        return gpu_id, False, str(e)


def step2_parser(args):
    """Step 2: DeepSeek-OCR 파서 (멀티GPU 병렬)"""
    print("\n" + "=" * 60)
    print(" Step 2: Run Parser (DeepSeek-OCR Auto-labeling)")
    print(f" Using GPUs: {args.gpus}")
    print("=" * 60)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    script = SCRIPT_DIR / "parser.py"
    python = get_python("venv_ocr")
    gpu_ids = args.gpus
    n_gpus = len(gpu_ids)

    # 이미지 파일 수집
    patterns = ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
    all_images = []
    for pat in patterns:
        all_images.extend(data_dir.glob(pat))
    all_images = sorted(set(all_images))

    if not all_images:
        print(f"  No images found in {data_dir}")
        sys.exit(1)

    print(f"  Found {len(all_images)} images, splitting across {n_gpus} GPUs")

    # 이미지를 GPU 수만큼 분할 → 임시 디렉토리에 심볼릭 링크
    chunk_size = math.ceil(len(all_images) / n_gpus)
    tmp_dirs = []

    for i, gpu_id in enumerate(gpu_ids):
        chunk = all_images[i * chunk_size : (i + 1) * chunk_size]
        if not chunk:
            continue

        tmp_dir = data_dir / f"_gpu_chunk_{gpu_id}"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        for img in chunk:
            link = tmp_dir / img.name
            if not link.exists():
                os.symlink(img.resolve(), link)

        tmp_dirs.append((gpu_id, tmp_dir))

    print(f"  Split: " + ", ".join(f"GPU {g}: {len(list(d.glob('*')))} imgs" for g, d in tmp_dirs))

    # 병렬 실행
    futures = []
    with ProcessPoolExecutor(max_workers=n_gpus) as pool:
        for gpu_id, tmp_dir in tmp_dirs:
            fut = pool.submit(
                _run_parser_on_gpu,
                gpu_id, tmp_dir, output_dir, python, script,
            )
            futures.append(fut)

        # 결과 수집
        errors = []
        for fut in as_completed(futures):
            gpu_id, ok, err = fut.result()
            if not ok:
                errors.append((gpu_id, err))

    # 임시 디렉토리 정리
    for _, tmp_dir in tmp_dirs:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if errors:
        print(f"\n  WARNING: {len(errors)} GPU(s) had errors:")
        for gpu_id, err in errors:
            print(f"    GPU {gpu_id}: {err}")
    else:
        print(f"\n  All {n_gpus} GPUs completed successfully.")


# ─────────────────────────────────────────────────────────────
# Step 3: COCO 변환
# ─────────────────────────────────────────────────────────────
def step3_coco(args):
    """Step 3: Parser 출력 → COCO 형식 변환"""
    print("\n" + "=" * 60)
    print(" Step 3: Convert to COCO Format")
    print("=" * 60)

    script = SCRIPT_DIR / "convert_to_coco.py"
    python = get_python("venv_ocr")

    cmd = [
        python, str(script),
        "--input_dir", str(args.output_dir),
        "--output_path", str(args.coco_output),
    ]
    run_command(cmd)


# ─────────────────────────────────────────────────────────────
# Step 4: COCO → YOLO 변환
# ─────────────────────────────────────────────────────────────
def step4_coco_to_yolo(args):
    """Step 4: COCO → YOLO 형식 변환"""
    print("\n" + "=" * 60)
    print(" Step 4: Convert COCO to YOLO Format")
    print("=" * 60)

    script = SCRIPT_DIR / "convert_coco_to_yolo.py"
    python = get_python("venv_yolo")

    cmd = [
        python, str(script),
        "--coco_json", str(args.coco_output),
        "--output_dir", str(args.yolo_dataset),
        "--val_ratio", str(args.val_ratio),
        "--image_dir", str(args.data_dir),
    ]
    if args.copy_images:
        cmd.append("--copy_images")
    run_command(cmd)


# ─────────────────────────────────────────────────────────────
# Step 5: YOLO12 학습 (멀티GPU)
# ─────────────────────────────────────────────────────────────
def step5_train_yolo(args):
    """Step 5: YOLO12 학습 (멀티GPU 지원)"""
    print("\n" + "=" * 60)
    print(" Step 5: Train YOLO12 Model")
    print(f" Using GPUs: {args.gpus}")
    print("=" * 60)

    script = SCRIPT_DIR / "train_yolo.py"
    python = get_python("venv_yolo")

    dataset_yaml = Path(args.yolo_dataset) / "dataset.yaml"
    if not dataset_yaml.exists():
        print(f"  Error: dataset.yaml not found at {dataset_yaml}")
        print("  Please run steps 1-4 first.")
        sys.exit(1)

    # 멀티GPU 디바이스 문자열 생성: "0,1,2,3"
    device_str = ",".join(str(g) for g in args.gpus)

    cmd = [
        python, str(script),
        "--mode", "train",
        "--model", str(args.yolo_model),
        "--data", str(dataset_yaml),
        "--epochs", str(args.epochs),
        "--imgsz", str(args.imgsz),
        "--batch", str(args.batch),
        "--project", str(args.yolo_project),
        "--name", str(args.yolo_name),
        "--patience", str(args.patience),
        "--device", device_str,
    ]
    run_command(cmd)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="DLA 전체 파이프라인: PDF → OCR → COCO → YOLO → 학습",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 실행 단계 선택
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 2, 3, 4, 5],
                        help="실행할 단계 목록 (default: 1 2 3 4 5)")

    # GPU 설정
    parser.add_argument("--gpus", nargs="+", type=int, default=None,
                        help="사용할 GPU 목록 (default: 자동감지)")

    # Step 1: PDF → Images
    parser.add_argument("--pdf_dir", type=str, default="../data/original",
                        help="원본 PDF 디렉토리 (default: ../data/original)")
    parser.add_argument("--data_dir", type=str, default="data/processed_images",
                        help="추출된 이미지 저장 디렉토리 (default: data/processed_images)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="PDF 변환 DPI (default: 300)")

    # Step 2: Parser
    parser.add_argument("--output_dir", type=str, default="outputs_dla",
                        help="Parser 출력 디렉토리 (default: outputs_dla)")

    # Step 3: COCO
    parser.add_argument("--coco_output", type=str,
                        default="coco_dataset/annotations/instances_default.json",
                        help="COCO JSON 저장 경로")

    # Step 4: COCO → YOLO
    parser.add_argument("--yolo_dataset", type=str, default="yolo_dataset",
                        help="YOLO 데이터셋 출력 디렉토리 (default: yolo_dataset)")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="Validation 비율 (default: 0.2)")
    parser.add_argument("--copy_images", action="store_true",
                        help="이미지를 심볼릭 링크 대신 복사")

    # Step 5: YOLO 학습
    parser.add_argument("--yolo_model", type=str, default="yolo12n.pt",
                        help="YOLO 사전학습 모델 (default: yolo12n.pt)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="학습 에폭 수 (default: 100)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="입력 이미지 크기 (default: 640)")
    parser.add_argument("--batch", type=int, default=-1,
                        help="배치 사이즈, -1=auto (default: -1, GPU 메모리 자동 최적화)")
    parser.add_argument("--yolo_project", type=str, default="runs",
                        help="YOLO 프로젝트 저장 경로 (default: runs)")
    parser.add_argument("--yolo_name", type=str, default="yolo12_problem",
                        help="실험 이름 (default: yolo12_problem)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (default: 20)")

    args = parser.parse_args()

    # GPU 자동감지
    if args.gpus is None:
        args.gpus = detect_gpus()
    print(f"  Detected GPUs: {args.gpus}")

    # 경로를 절대경로로 변환 (SCRIPT_DIR 기준)
    for attr in ("pdf_dir", "data_dir", "output_dir", "coco_output", "yolo_dataset", "yolo_project"):
        val = getattr(args, attr)
        p = Path(val)
        if not p.is_absolute():
            setattr(args, attr, str(SCRIPT_DIR / p))

    step_fns = {
        1: step1_pdf_to_images,
        2: step2_parser,
        3: step3_coco,
        4: step4_coco_to_yolo,
        5: step5_train_yolo,
    }

    print("=" * 60)
    print(" DLA Pipeline - Problem Detection Training")
    print(f" Steps to run: {args.steps}")
    print(f" GPUs: {args.gpus}")
    print("=" * 60)

    for step_num in sorted(args.steps):
        if step_num not in step_fns:
            print(f"  Unknown step {step_num}, skipping.")
            continue
        step_fns[step_num](args)

    print("\n" + "=" * 60)
    print(" Pipeline Completed!")
    print("=" * 60)

    if 5 in args.steps:
        print(f"  Trained model: {args.yolo_project}/{args.yolo_name}/weights/best.pt")
    if 4 in args.steps:
        print(f"  YOLO dataset:  {args.yolo_dataset}/")
    if 3 in args.steps:
        print(f"  COCO JSON:     {args.coco_output}")


if __name__ == "__main__":
    main()
