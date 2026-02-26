# -*- coding: utf-8 -*-
"""
CSAT vs SAT 시험지 이미지 분류기.
- 휴리스틱: 헤더 영역 OCR 후 한글/키워드 규칙으로 판별.
- CNN: (선택) ResNet18/MobileNetV2 2-class 분류.
- API: classify(image_path) -> "csat" | "sat", classify_batch(paths) -> list[str].
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Literal

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# -----------------------------------------------------------------------------
# Constants (paper-tunable)
# -----------------------------------------------------------------------------
HEADER_CROP_RATIO = 0.12  # top 12% of image for heuristic OCR
KOREAN_RATIO_THRESHOLD = 0.08  # above this → CSAT
DEFAULT_UNCERTAIN = "csat"  # when heuristic is uncertain

# CSAT keywords (Korean exam header)
CSAT_KEYWORDS = [
    "지선다형",
    "단답형",
    "서답형",
    "수학 영역",
    "수학영역",
    "홀수형",
    "짝수형",
    "제 ",
    "교시",
]
# SAT keywords (English exam header/footer)
SAT_KEYWORDS = [
    "Module",
    "CONTINUE",
    "Unauthorized copying",
    "reuse of any part",
    "is illegal",
]

# Korean character ranges (CJK Unified Ideographs, Hangul)
_KOREAN_RE = re.compile(r"[\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318F]")

# -----------------------------------------------------------------------------
# Heuristic: header crop + OCR + rules
# -----------------------------------------------------------------------------


def _crop_header(image_input: str | Path | "Image.Image", ratio: float = HEADER_CROP_RATIO):
    """Return PIL Image of top `ratio` of the page (header)."""
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL is required. pip install Pillow")
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("RGB")
    else:
        img = image_input.convert("RGB") if hasattr(image_input, "convert") else image_input
    w, h = img.size
    top_h = max(1, int(h * ratio))
    return img.crop((0, 0, w, top_h))


def _korean_ratio(text: str) -> float:
    """Ratio of Korean characters in text (0~1)."""
    if not text or not text.strip():
        return 0.0
    total = sum(1 for c in text if c.strip())
    if total == 0:
        return 0.0
    korean = sum(1 for c in text if _KOREAN_RE.search(c))
    return korean / total


def _run_ocr_on_image(image_path: str) -> str:
    """
    Run lightweight OCR on an image file. Tries PaddleOCR (Korean + English merged), then pytesseract.
    Returns concatenated text or empty string.
    """
    parts = []
    # 1) PaddleOCR Korean
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=False, show_log=False, lang="korean")
        result = ocr.ocr(image_path, cls=False)
        if result and result[0]:
            lines = [str(line[1][0]).strip() for line in result[0] if line and len(line) >= 2 and line[1]]
            if lines:
                parts.append("\n".join(lines))
    except Exception:
        pass
    # 2) PaddleOCR English (for SAT "Module", "CONTINUE" etc.)
    try:
        from paddleocr import PaddleOCR
        ocr_en = PaddleOCR(use_angle_cls=False, show_log=False, lang="en")
        result = ocr_en.ocr(image_path, cls=False)
        if result and result[0]:
            lines = [str(line[1][0]).strip() for line in result[0] if line and len(line) >= 2 and line[1]]
            if lines:
                parts.append("\n".join(lines))
    except Exception:
        pass
    text = "\n".join(parts).strip() if parts else ""
    if text:
        return text
    # 3) pytesseract (optional)
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        text = pytesseract.image_to_string(img, lang="kor+eng")
    except Exception:
        pass
    return (text or "").strip()


def classify_heuristic(
    image_input: str | Path | "Image.Image",
    *,
    default_if_uncertain: Literal["csat", "sat"] = DEFAULT_UNCERTAIN,
    use_cnn_if_uncertain: bool = False,
    cnn_model_path: str | Path | None = None,
) -> tuple[str, float]:
    """
    Classify using header crop + OCR + keyword/Korean ratio.
    Returns (label, confidence). confidence in [0, 1]; 0.5 means uncertain.
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL is required. pip install Pillow")

    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.exists():
            return default_if_uncertain, 0.0
        crop = _crop_header(image_input, HEADER_CROP_RATIO)
    else:
        crop = _crop_header(image_input, HEADER_CROP_RATIO)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        crop.save(f.name)
        try:
            ocr_text = _run_ocr_on_image(f.name)
        finally:
            try:
                os.unlink(f.name)
            except Exception:
                pass

    ocr_lower = ocr_text.lower()
    ocr_upper = ocr_text.upper()

    # CSAT: Korean keywords or high Korean ratio
    has_csat_keyword = any(kw in ocr_text for kw in CSAT_KEYWORDS)
    k_ratio = _korean_ratio(ocr_text)
    if has_csat_keyword or k_ratio >= KOREAN_RATIO_THRESHOLD:
        conf = 0.9 if has_csat_keyword else 0.5 + min(0.4, k_ratio)
        return "csat", min(1.0, conf)

    # SAT: English keywords
    has_sat_keyword = any(
        kw.lower() in ocr_lower or kw.upper() in ocr_upper or kw in ocr_text
        for kw in SAT_KEYWORDS
    )
    if has_sat_keyword:
        return "sat", 0.9

    # Uncertain: optional CNN fallback
    if use_cnn_if_uncertain and cnn_model_path:
        label, _ = classify_cnn(image_input, model_path=cnn_model_path)
        return label, 0.6
    return default_if_uncertain, 0.5


# -----------------------------------------------------------------------------
# CNN classifier (optional)
# -----------------------------------------------------------------------------
_CNN_MODEL = None
_CNN_TRANSFORM = None
_CNN_MODEL_PATH: str | None = None


def _load_cnn_model(model_path: str | Path, device: str = "cpu"):
    """Load 2-class ResNet18 (or MobileNetV2) from checkpoint."""
    import torch
    import torch.nn as nn
    from torchvision import models, transforms

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    # Standard ImageNet-style transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Model: ResNet18, 2 classes (0=CSAT, 1=SAT)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model, transform


def classify_cnn(
    image_input: str | Path | "Image.Image",
    model_path: str | Path | None = None,
    device: str = "cpu",
) -> tuple[Literal["csat", "sat"], float]:
    """
    Classify using a trained 2-class CNN. Class 0 = CSAT, 1 = SAT.
    Returns (label, confidence). confidence is max class probability from softmax.
    model_path: path to .pt/.pth checkpoint. If None, uses env EXAM_CLASSIFIER_MODEL or returns (default, 0.0).
    """
    global _CNN_MODEL, _CNN_TRANSFORM, _CNN_MODEL_PATH
    try:
        from PIL import Image
        import torch
    except ImportError:
        raise ImportError("PIL and torch are required for CNN classifier.")

    model_path = model_path or os.environ.get("EXAM_CLASSIFIER_MODEL")
    if not model_path:
        return DEFAULT_UNCERTAIN, 0.0
    model_path = str(model_path)

    if _CNN_MODEL is None or _CNN_MODEL_PATH != model_path:
        _CNN_MODEL, _CNN_TRANSFORM = _load_cnn_model(model_path, device)
        _CNN_MODEL_PATH = model_path

    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("RGB")
    else:
        img = image_input.convert("RGB") if hasattr(image_input, "convert") else image_input

    x = _CNN_TRANSFORM(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = _CNN_MODEL(x)
    probs = torch.softmax(logits, dim=1)
    pred = logits.argmax(dim=1).item()
    confidence = probs[0, pred].item()
    label = "sat" if pred == 1 else "csat"
    return label, confidence


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def classify(
    image_input: str | Path | "Image.Image",
    *,
    method: Literal["heuristic", "cnn", "hybrid", "model_first"] = "heuristic",
    default_if_uncertain: Literal["csat", "sat"] = DEFAULT_UNCERTAIN,
    cnn_model_path: str | Path | None = None,
    confidence_threshold: float = 0.8,
) -> Literal["csat", "sat"]:
    """
    Classify a single exam page image as CSAT or SAT.

    - method "heuristic": header OCR + keyword/Korean ratio (no training).
    - method "cnn": use trained 2-class model (requires cnn_model_path or EXAM_CLASSIFIER_MODEL).
    - method "hybrid": heuristic first; if confidence < 0.6, run CNN if available.
    - method "model_first": CNN first; if confidence >= confidence_threshold use CNN result,
      else fall back to heuristic (OCR-based). Default threshold 0.8.

    Returns "csat" or "sat".
    """
    if method == "model_first":
        cnn_path = cnn_model_path or os.environ.get("EXAM_CLASSIFIER_MODEL")
        if cnn_path:
            label, conf = classify_cnn(image_input, model_path=cnn_path)
            if conf >= confidence_threshold:
                return label
        label, _ = classify_heuristic(
            image_input,
            default_if_uncertain=default_if_uncertain,
            use_cnn_if_uncertain=False,
        )
        return label

    if method == "cnn":
        label, _ = classify_cnn(image_input, model_path=cnn_model_path)
        return label

    if method == "heuristic":
        label, _ = classify_heuristic(
            image_input,
            default_if_uncertain=default_if_uncertain,
            use_cnn_if_uncertain=False,
        )
        return label

    # hybrid
    label, conf = classify_heuristic(
        image_input,
        default_if_uncertain=default_if_uncertain,
        use_cnn_if_uncertain=True,
        cnn_model_path=cnn_model_path,
    )
    if conf >= 0.6:
        return label
    if cnn_model_path or os.environ.get("EXAM_CLASSIFIER_MODEL"):
        label, _ = classify_cnn(image_input, model_path=cnn_model_path)
        return label
    return default_if_uncertain


def classify_with_confidence(
    image_input: str | Path | "Image.Image",
    *,
    method: Literal["model_first"] = "model_first",
    default_if_uncertain: Literal["csat", "sat"] = DEFAULT_UNCERTAIN,
    cnn_model_path: str | Path | None = None,
    confidence_threshold: float = 0.8,
) -> tuple[Literal["csat", "sat"], float]:
    """
    Classify using model_first and return (label, confidence).
    Used by API when response must include confidence.
    """
    cnn_path = cnn_model_path or os.environ.get("EXAM_CLASSIFIER_MODEL")
    if cnn_path:
        label, conf = classify_cnn(image_input, model_path=cnn_path)
        if conf >= confidence_threshold:
            return label, conf
    label, conf = classify_heuristic(
        image_input,
        default_if_uncertain=default_if_uncertain,
        use_cnn_if_uncertain=False,
    )
    return label, conf


def classify_batch(
    image_paths: list[str | Path],
    *,
    method: Literal["heuristic", "cnn", "hybrid", "model_first"] = "heuristic",
    default_if_uncertain: Literal["csat", "sat"] = DEFAULT_UNCERTAIN,
    cnn_model_path: str | Path | None = None,
    confidence_threshold: float = 0.8,
    show_progress: bool = True,
) -> list[Literal["csat", "sat"]]:
    """Classify multiple images. Returns list of "csat" | "sat" in same order."""
    it = image_paths
    if show_progress and len(image_paths) > 1:
        it = tqdm(image_paths, desc="Classify", unit="img", ncols=90)
    return [
        classify(
            p,
            method=method,
            default_if_uncertain=default_if_uncertain,
            cnn_model_path=cnn_model_path,
            confidence_threshold=confidence_threshold,
        )
        for p in it
    ]


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Classify exam page image as CSAT or SAT.")
    ap.add_argument("image", type=str, help="Path to image file")
    ap.add_argument(
        "--method",
        choices=["heuristic", "cnn", "hybrid", "model_first"],
        default="heuristic",
        help="model_first: CNN first, fallback to OCR if confidence < threshold",
    )
    ap.add_argument("--cnn", type=str, default=None, help="Path to CNN checkpoint for cnn/hybrid/model_first")
    ap.add_argument("--default", choices=["csat", "sat"], default="csat", help="Default when uncertain")
    ap.add_argument("--confidence-threshold", type=float, default=0.8, help="For model_first: use OCR below this")
    args = ap.parse_args()
    label = classify(
        args.image,
        method=args.method,
        default_if_uncertain=args.default,
        cnn_model_path=args.cnn,
        confidence_threshold=args.confidence_threshold,
    )
    print(label)


if __name__ == "__main__":
    main()
