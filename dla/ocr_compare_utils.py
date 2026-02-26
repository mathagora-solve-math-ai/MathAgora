# -*- coding: utf-8 -*-
"""
OCR comparison utilities: load samples from CSV and run multiple OCR models.
Each model runner returns text or an error string; results can be cached to JSON.
"""
from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Optional: PIL for image handling
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Optional: pandas for CSV
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def load_samples_from_csv(
    csv_path: str | Path,
    n: int = 3,
    image_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    Load first n rows from 2025_math_odd.csv with prob_id, prob_desc, and decode prob_base64 to image.
    Returns list of dicts: {prob_id, prob_desc, image_path, image_pil}.
    If image_dir is given, images are saved there and image_path is set; else temp dir used.
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required. pip install pandas")
    if not HAS_PIL:
        raise ImportError("PIL is required. pip install Pillow")

    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)  # multiline prob_desc handled by default quoting
    df = df.head(n)

    if image_dir is None:
        image_dir = Path(tempfile.mkdtemp(prefix="ocr_compare_"))
    else:
        image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    for idx, row in df.iterrows():
        prob_id = str(row.get("prob_id", ""))
        prob_desc = str(row.get("prob_desc", "")).strip()
        b64 = row.get("prob_base64", "")
        if pd.isna(b64) or not b64:
            continue
        try:
            raw = base64.b64decode(b64)
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:
            continue
        ext = "png" if raw[:8] == b"\x89PNG\r\n\x1a\n" else "jpg"
        path = image_dir / f"{prob_id}.{ext}"
        img.save(path)
        samples.append({
            "prob_id": prob_id,
            "prob_desc": prob_desc,
            "image_path": str(path),
            "image_pil": img,
        })
        if len(samples) >= n:
            break
    return samples


def load_sat_samples_from_vis_paths(
    vis_paths: list[str | Path],
    image_dir: str | Path,
    root: str | Path | None = None,
    dla_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    SAT 파이프라인 결과(vis 이미지 경로)에서 해당 페이지의 json을 찾아,
    원본 페이지 이미지에서 문항 bbox로 crop한 이미지를 image_dir에 저장하고
    샘플 목록을 반환합니다. (detect problems 결과를 그대로 사용)
    Returns list of dicts: {prob_id, prob_desc, image_path, image_pil}.
    """
    if not HAS_PIL:
        raise ImportError("PIL is required. pip install Pillow")

    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    root = Path(root) if root else Path(__file__).resolve().parent.parent
    dla_dir = Path(dla_dir) if dla_dir else Path(__file__).resolve().parent

    samples = []
    for vis_path in vis_paths:
        p = Path(vis_path)
        # vis path: .../sat/8-images/sat-practice-test-8-math_page_014/vis/..._vis.png → page_dir = .../sat-practice-test-8-math_page_014
        page_dir = p.parent.parent if p.name.endswith("_vis.png") else p
        page_id = page_dir.name
        folder = page_dir.parent.name  # e.g. 8-images
        json_path = page_dir / f"{page_id}.json"
        if not page_dir.exists() or not json_path.exists():
            continue
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        page_image_path = data.get("page_image") or ""
        questions = data.get("questions") or []
        if not questions:
            continue
        # Resolve original page image: try multiple locations
        page_img_path = None
        if page_image_path and Path(page_image_path).exists():
            page_img_path = Path(page_image_path)
        if not page_img_path:
            candidate = root / "data" / "SAT" / folder / f"{page_id}.png"
            if candidate.exists():
                page_img_path = candidate
        if not page_img_path:
            vis_img = page_dir / "vis" / f"{page_id}_vis.png"
            if vis_img.exists():
                page_img_path = vis_img
        if not page_img_path:
            same_dir = page_dir / f"{page_id}.png"
            if same_dir.exists():
                page_img_path = same_dir
        if not page_img_path:
            continue
        try:
            im_orig = Image.open(str(page_img_path)).convert("RGB")
        except Exception:
            continue
        for q in questions:
            qid = q.get("qid", "")
            bbox = q.get("bbox")
            merged_text = (q.get("merged_text") or "").strip()
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(round(x)) for x in bbox]
            try:
                crop = im_orig.crop((x1, y1, x2, y2))
            except Exception:
                continue
            short_id = f"sat_{folder.replace('-images', '')}_{page_id.replace('sat-practice-test-', '').replace('math_page_', 'p')}_q{qid}"
            safe_id = short_id.replace("/", "_").replace(" ", "_")
            out_path = image_dir / f"{safe_id}.png"
            crop.save(out_path)
            samples.append({
                "prob_id": safe_id,
                "prob_desc": merged_text,
                "image_path": str(out_path),
                "image_pil": crop,
            })
        im_orig.close()
    return samples


# ---------- DeepSeek-OCR (v1) ----------
def _is_venv_ocr() -> bool:
    """현재 인터프리터가 dla/venv_ocr 인지 여부."""
    try:
        return "venv_ocr" in os.path.normpath(sys.executable)
    except Exception:
        return False


def run_deepseek_inprocess(image_path: str, model_key: str = "v1") -> str:
    """
    현재 프로세스에서 DeepSeek-OCR 추론. venv_ocr에서만 사용. raw 출력 반환.
    모델이 infer() 반환값이 아닌 stdout으로 결과를 출력하므로, 호출 중 stdout을 캡처해 사용.
    """
    import io
    import sys
    import tempfile
    import warnings
    model_name = "deepseek-ai/DeepSeek-OCR" if model_key == "v1" else "deepseek-ai/DeepSeek-OCR-2"
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    import torch
    from transformers import AutoModel, AutoTokenizer
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*instantiate a model of type.*")
        try:
            import transformers.utils.logging as tf_logging
            tf_logging.set_verbosity_error()
        except Exception:
            pass
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        try:
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_safetensors=True,
            )
        except Exception:
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
    model = model.eval().cuda().to(torch.bfloat16)
    prompt = "<image>\n<|grounding|>Convert the document to markdown. "
    image_size = 768 if model_key == "v2" else 640
    out_dir = tempfile.mkdtemp(prefix="deepseek_ocr_inprocess_")
    # 모델이 결과를 stdout으로 출력하므로 캡처
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=image_path,
            output_path=out_dir,
            base_size=1024,
            image_size=image_size,
            crop_mode=True,
            save_results=False,
        )
    captured = buf.getvalue().strip()
    # 반환값이 있으면 우선 사용
    if res is not None and isinstance(res, str) and res.strip():
        return res.strip()
    if isinstance(res, (list, tuple)):
        parts = []
        out_path = Path(out_dir)
        for item in res:
            if isinstance(item, str):
                p = Path(item)
                if not p.is_absolute():
                    p = out_path / p
                if p.suffix.lower() == ".md" and p.exists():
                    try:
                        parts.append(p.read_text(encoding="utf-8", errors="replace").strip())
                    except Exception:
                        pass
                elif ("\n" in item or len(item) > 80) and item.strip():
                    parts.append(item.strip())
        if parts:
            return "\n".join(p for p in parts if p)
    try:
        md_files = sorted(
            Path(out_dir).rglob("*.md"),
            key=lambda f: (-f.stat().st_size, f.stat().st_mtime) if f.exists() else (0, 0),
        )
    except (OSError, FileNotFoundError):
        md_files = list(Path(out_dir).rglob("*.md"))
    for f in md_files:
        text = f.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            return text
    # stdout 캡처 내용에서 OCR 본문 추출 (BASE/PATCHES 등 디버그 줄만 제거)
    if captured:
        lines = [
            line for line in captured.splitlines()
            if not line.strip().startswith("====")
            and not line.strip().startswith("BASE:")
            and not line.strip().startswith("PATCHES:")
        ]
        out = "\n".join(lines).strip()
        if out:
            return out
    return str(res).strip() if res else ""


def _run_deepseek_via_venv(image_path: str, model_key: str) -> str | None:
    """venv_ocr의 Python으로 run_deepseek_ocr_single.py 실행. 항상 절대경로 사용, 독립 subprocess."""
    import subprocess
    dla_dir = Path(__file__).resolve().parent
    # 우선순위: DLA_OCR_VENV -> venv_deepseek -> venv_ocr
    explicit = os.environ.get("DLA_OCR_VENV")
    if explicit:
        venv_python = (dla_dir / explicit / "bin" / "python").resolve()
    else:
        deepseek_python = (dla_dir / "venv_deepseek" / "bin" / "python").resolve()
        venv_python = deepseek_python if deepseek_python.exists() else (dla_dir / "venv_ocr" / "bin" / "python").resolve()
    script = (dla_dir / "run_deepseek_ocr_single.py").resolve()
    if not venv_python.exists() or not script.exists():
        return None
    model_arg = "DeepSeek-OCR" if model_key == "v1" else "DeepSeek-OCR-2"
    if not image_path:
        return None
    p = Path(image_path)
    img_abs = p.resolve() if p.is_absolute() else (dla_dir / p).resolve()
    if not img_abs.exists():
        img_abs = Path(image_path).resolve()
    if not img_abs.exists():
        return None
    # 자식은 venv_ocr만 쓰도록 최소 환경 전달 (부모 VIRTUAL_ENV/PATH 등이 넘어가면 오동작)
    venv_bin = str(venv_python.parent)
    env = {
        "PATH": venv_bin + os.pathsep + os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", "/tmp"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "PYTHONWARNINGS": "ignore",  # DeepSeek config model_type 경고가 에러로 취급되지 않도록
        "PYTHONNOUSERSITE": "1",
    }
    for key in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "HF_HOME", "TRANSFORMERS_CACHE", "HF_TOKEN"):
        if key in os.environ:
            env[key] = os.environ[key]
    last_err: str | None = None
    for _ in range(2):
        try:
            out = subprocess.run(
                [str(venv_python), "-I", str(script), str(img_abs), model_arg],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(dla_dir),
                env=env,
            )
            if out.returncode == 0 and out.stdout:
                return out.stdout.strip()
            last_err = f"returncode={out.returncode} stderr={ (out.stderr or '')[:800]}"
        except subprocess.TimeoutExpired as e:
            last_err = f"timeout ({e.timeout}s)"
        except FileNotFoundError as e:
            last_err = f"FileNotFoundError: {e}"
        except Exception as e:
            last_err = str(e)[:400]
    err_msg = f"[DeepSeek-OCR subprocess] {last_err or 'unknown'}"
    return err_msg


def _postprocess_deepseek(raw: str) -> str:
    """DeepSeek 원시 출력 후처리 (메타/태그 제거)."""
    try:
        import importlib.util
        _dla = Path(__file__).resolve().parent
        _spec = importlib.util.spec_from_file_location("deepseek_ocr_postprocess", _dla / "deepseek_ocr_postprocess.py")
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        return _mod.postprocess_deepseek_ocr(raw)
    except Exception:
        return raw


def run_deepseek_ocr(image_path: str, **kwargs) -> str:
    """Run deepseek-ai/DeepSeek-OCR. venv_ocr이면 in-process, 아니면 subprocess."""
    if _is_venv_ocr():
        try:
            return _postprocess_deepseek(run_deepseek_inprocess(image_path, "v1"))
        except Exception as e:
            return f"[DeepSeek-OCR error] {e}"
    res = _run_deepseek_via_venv(image_path, "v1")
    if res is not None and not res.lstrip().startswith("[DeepSeek"):
        return _postprocess_deepseek(res)
    return res if (res and res.strip()) else "[DeepSeek-OCR error] Run pipeline with venv_ocr: ./venv_ocr/bin/python ocr_pipeline.py ..."


# ---------- DeepSeek-OCR-2 ----------
def run_deepseek_ocr2(image_path: str, **kwargs) -> str:
    """Run deepseek-ai/DeepSeek-OCR-2. venv_ocr이면 in-process, 아니면 subprocess."""
    if _is_venv_ocr():
        try:
            return _postprocess_deepseek(run_deepseek_inprocess(image_path, "v2"))
        except Exception as e:
            return f"[DeepSeek-OCR-2 error] {e}"
    res = _run_deepseek_via_venv(image_path, "v2")
    if res is not None and not res.lstrip().startswith("[DeepSeek"):
        return _postprocess_deepseek(res)
    return res if (res and res.strip()) else "[DeepSeek-OCR-2 error] Run pipeline with venv_ocr: ./venv_ocr/bin/python ocr_pipeline.py ..."


def run_deepseek_ocr_with_meta(image_path: str) -> tuple[str, str]:
    """Run DeepSeek-OCR. venv_ocr이면 in-process. Returns (postprocessed_text, raw_text)."""
    if _is_venv_ocr():
        try:
            raw = run_deepseek_inprocess(image_path, "v1")
            return (_postprocess_deepseek(raw), raw)
        except Exception as e:
            return (f"[DeepSeek-OCR error] {e}", "")
    raw = _run_deepseek_via_venv(image_path, "v1")
    if raw is None:
        return ("", "")
    if raw.lstrip().startswith("[DeepSeek"):
        return (raw, raw)
    return (_postprocess_deepseek(raw), raw)


def run_deepseek_ocr2_with_meta(image_path: str) -> tuple[str, str]:
    """Run DeepSeek-OCR-2. venv_ocr이면 in-process. Returns (postprocessed_text, raw_text)."""
    if _is_venv_ocr():
        try:
            raw = run_deepseek_inprocess(image_path, "v2")
            return (_postprocess_deepseek(raw), raw)
        except Exception as e:
            return (f"[DeepSeek-OCR-2 error] {e}", "")
    raw = _run_deepseek_via_venv(image_path, "v2")
    if raw is None:
        return ("", "")
    if raw.lstrip().startswith("[DeepSeek"):
        return (raw, raw)
    return (_postprocess_deepseek(raw), raw)


# ---------- GLM-OCR ----------
def _run_via_system_python(script_name: str, image_path: str, timeout: int = 300) -> str:
    """DLA_SYSTEM_PYTHON으로 단일 스크립트 실행. venv_ocr에서 Paddle/GLM 호출 시 사용."""
    import subprocess
    dla_dir = Path(__file__).resolve().parent
    system_python = os.environ.get("DLA_SYSTEM_PYTHON", "python3")
    script = (dla_dir / script_name).resolve()
    if not script.exists():
        return f"[{script_name} error] script not found"
    p = Path(image_path)
    img_abs = p.resolve() if p.is_absolute() else (dla_dir / p).resolve()
    if not img_abs.exists():
        img_abs = Path(image_path).resolve()
    if not img_abs.exists():
        return f"[{script_name} error] image not found: {image_path}"
    try:
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("VIRTUAL_ENV", None)
        env["PYTHONNOUSERSITE"] = "1"
        out = subprocess.run(
            [system_python, "-I", str(script), str(img_abs)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(dla_dir),
            env=env,
        )
        if out.returncode == 0 and out.stdout is not None:
            return out.stdout.strip()
        err = (out.stderr or out.stdout or "")[:500]
        return f"[{script_name} subprocess] returncode={out.returncode} stderr={err}"
    except subprocess.TimeoutExpired:
        return f"[{script_name} subprocess] timeout ({timeout}s)"
    except FileNotFoundError as e:
        return f"[{script_name} subprocess] {e}"
    except Exception as e:
        return f"[{script_name} subprocess] {e}"


def run_glm_ocr(image_path: str, **kwargs) -> str:
    """Run zai-org/GLM-OCR. venv_ocr이면 시스템 Python subprocess, 아니면 in-process."""
    if _is_venv_ocr():
        return _run_via_system_python("run_glm_ocr_single.py", image_path)
    try:
        from transformers import AutoProcessor, AutoModelForImageTextToText
        import torch
        MODEL_PATH = "zai-org/GLM-OCR"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image_path},
                    {"type": "text", "text": "Text Recognition:"},
                ],
            }
        ]
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name_or_path=MODEL_PATH,
            torch_dtype="auto",
            device_map="auto",
        )
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        inputs.pop("token_type_ids", None)
        generated_ids = model.generate(**inputs, max_new_tokens=8192)
        output_text = processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=False,
        )
        return output_text.strip()
    except Exception as e:
        return f"[GLM-OCR error] {e}"


# ---------- PaddleOCR-VL-1.5 ----------
def run_paddle_ocr_vl(image_path: str, **kwargs) -> str:
    """Run PaddleOCR-VL. venv_ocr이면 시스템 Python subprocess, 아니면 in-process."""
    if _is_venv_ocr():
        return _run_via_system_python("run_paddle_ocr_single.py", image_path)
    try:
        from paddleocr import PaddleOCRVL
        pipeline = PaddleOCRVL()
        output = pipeline.predict(image_path)
        parts = []
        for res in output:
            if hasattr(res, "get_markdown"):
                parts.append(res.get_markdown())
            elif hasattr(res, "_to_markdown"):
                md_info = res._to_markdown()
                if isinstance(md_info, dict) and md_info.get("markdown_texts"):
                    parts.append(md_info["markdown_texts"])
            elif hasattr(res, "to_dict"):
                res_dict = res.to_dict()
                if isinstance(res_dict, dict) and res_dict.get("md"):
                    parts.append(res_dict["md"])
            if not parts and res is not None:
                parts.append(str(res))
        return "\n".join(parts).strip() if parts else "(no output)"
    except Exception as e:
        return f"[PaddleOCR-VL error] {e}"


# ---------- MinerU2.5 ----------
def run_mineru25(image_path: str, **kwargs) -> str:
    """Run opendatalab/MinerU2.5-2509-1.2B. Needs: mineru-vl-utils[transformers], transformers, qwen2_vl."""
    try:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        from mineru_vl_utils import MinerUClient
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "opendatalab/MinerU2.5-2509-1.2B",
            torch_dtype="auto",
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(
            "opendatalab/MinerU2.5-2509-1.2B",
            use_fast=True,
        )
        client = MinerUClient(backend="transformers", model=model, processor=processor)
        if HAS_PIL:
            img = Image.open(image_path).convert("RGB")
        else:
            img = Image.open(image_path).convert("RGB")
        extracted = client.two_step_extract(img)
        if isinstance(extracted, str):
            return extracted.strip()
        if isinstance(extracted, (list, tuple)):
            parts = []
            for x in extracted:
                if isinstance(x, dict) and "content" in x:
                    parts.append(x["content"])
                else:
                    parts.append(str(x))
            return "\n".join(parts).strip() if parts else "\n".join(str(x) for x in extracted).strip()
        return str(extracted).strip()
    except Exception as e:
        return f"[MinerU2.5 error] {e}"


# ---------- dots.ocr ----------
DOTS_OCR_PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""


def _resolve_dots_ocr_model_path(model_path: str) -> str:
    """Resolve dots.ocr model path: HF repo id -> local dir without periods (required by dots.ocr)."""
    if "/" not in model_path or not model_path.startswith("rednote-hilab/"):
        if os.path.isdir(model_path) and "." not in os.path.basename(model_path.rstrip(os.sep)):
            return model_path
        return model_path
    try:
        from huggingface_hub import snapshot_download
        # dots.ocr README: directory name must NOT contain periods (e.g. DotsOCR not dots.ocr)
        cache_base = os.environ.get("HF_HOME") or os.path.join(Path.home(), ".cache", "huggingface", "hub")
        local_dir = os.path.join(cache_base, "dla_DotsOCR")
        return snapshot_download(
            repo_id=model_path,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
    except Exception:
        return model_path


def run_dots_ocr(image_path: str, **kwargs) -> str:
    """Run dots.ocr (DotsOCR). Needs: transformers, qwen_vl_utils.
    Model path: kwargs['dots_ocr_model_path'] or env DOTS_OCR_MODEL_PATH or 'rednote-hilab/dots.ocr'.
    When using HF id, model is resolved to a path without periods (required by dots.ocr)."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        from qwen_vl_utils import process_vision_info

        raw_path = kwargs.get("dots_ocr_model_path") or os.environ.get("DOTS_OCR_MODEL_PATH", "rednote-hilab/dots.ocr")
        model_path = _resolve_dots_ocr_model_path(raw_path)
        prompt = kwargs.get("dots_ocr_prompt", DOTS_OCR_PROMPT)

        # Use single device to avoid "tensors on different devices" with dots.ocr
        device_map = kwargs.get("dots_ocr_device_map") or os.environ.get("DOTS_OCR_DEVICE_MAP", "cuda:0")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                trust_remote_code=True,
            )
        except (ImportError, ValueError):
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                attn_implementation="sdpa",
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                trust_remote_code=True,
            )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        # DotsVLProcessor may lack video_processor, or it may be called with (videos=...) only;
        # use an adapter that returns empty video outputs when videos are empty
        if not getattr(processor, "video_processor", None) and getattr(processor, "image_processor", None):

            class _EmptyVideoAdapter:
                """Adapter so processor __call__ can use video_processor when we only pass images."""

                def __init__(self, image_processor):
                    self._img = image_processor

                def __call__(self, images=None, videos=None, **kwargs):
                    if videos is None or (isinstance(videos, (list, tuple)) and len(videos) == 0):
                        return {
                            "video_grid_thw": [],
                            "pixel_values_videos": None,
                            "video_metadata": [],
                        }
                    return self._img(images=None, videos=videos, **kwargs)

                def __getattr__(self, name):
                    return getattr(self._img, name)

            processor.video_processor = _EmptyVideoAdapter(processor.image_processor)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        video_list = video_inputs if video_inputs is not None else []
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_list,
            padding=True,
            return_tensors="pt",
        )
        # dots.ocr model may not accept video kwargs; drop them to avoid "not used by the model" error
        for key in ("pixel_values_videos", "video_grid_thw", "video_metadata", "second_per_grid_ts"):
            inputs.pop(key, None)
        # ensure inputs on same device as model (device_map='auto' may use multiple devices)
        device = next(model.parameters()).device
        inputs = inputs.to(device)

        generated_ids = model.generate(**inputs, max_new_tokens=24000)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return (output_text[0] if output_text else "").strip()
    except Exception as e:
        return f"[dots.ocr error] {e}"


# ---------- Registry and runner ----------
MODEL_RUNNERS = {
    "deepseek_ocr": run_deepseek_ocr,
    "deepseek_ocr2": run_deepseek_ocr2,
    "glm_ocr": run_glm_ocr,
    "paddle_ocr_vl": run_paddle_ocr_vl,
    "mineru25": run_mineru25,
    "dots_ocr": run_dots_ocr,
}


def run_ocr_models(
    image_path: str,
    models: list[str] | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Run selected OCR models on image_path. Returns dict with 'outputs' (model_name -> text) and 'timings' (model_name -> seconds)."""
    if models is None:
        models = list(MODEL_RUNNERS.keys())
    outputs: dict[str, str] = {}
    timings: dict[str, float] = {}
    for name in models:
        runner = MODEL_RUNNERS.get(name)
        if runner is None:
            outputs[name] = f"[Unknown model] {name}"
            timings[name] = 0.0
            continue
        t0 = time.perf_counter()
        try:
            outputs[name] = runner(image_path, **kwargs)
        except Exception as e:
            outputs[name] = f"[{name} error] {e}"
        timings[name] = round(time.perf_counter() - t0, 2)
    return {"outputs": outputs, "timings": timings}


def save_results_cache(results: list[dict], path: str | Path) -> None:
    """Save comparison results to JSON (images as paths only, no base64). Includes per-model timings (seconds) when present."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for r in results:
        row = {
            "prob_id": r.get("prob_id"),
            "prob_desc": r.get("prob_desc"),
            "image_path": r.get("image_path"),
            "outputs": r.get("outputs", {}),
        }
        if r.get("timings"):
            row["timings"] = r["timings"]
        out.append(row)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def load_results_cache(path: str | Path) -> list[dict]:
    """Load comparison results from JSON."""
    path = Path(path)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))
