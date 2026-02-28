# -*- coding: utf-8 -*-
"""
DeepSeek-OCR 기반 완전 자동 수학 문항 분리 파이프라인 (속도 최적화 버전)
- GPU 추론 단일 스레드 + CPU 후처리 병렬 파이프라인
- 입력 디렉토리별(년도별) 개별 출력 폴더 및 CSV 저장
"""

import os, re, io, time, glob, json, contextlib, base64, unicodedata, warnings, ast
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from datetime import datetime
import transformers
import hashlib
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# =========================
# 0) 런타임/옵션
# =========================
# 환경 변수 설정
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
torch.set_grad_enabled(False)
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-OCR")
# OUT_DIR은 이제 베이스 경로이며, 실제 출력은 하위 폴더에 저장됩니다.
OUT_DIR = os.getenv("OUT_DIR", "outputs_dla") 

# 빠른 실행 옵션
FAST_SKIP_VIS = os.getenv("FAST_SKIP_VIS", "0") == "1"
FAST_SKIP_B64 = os.getenv("FAST_SKIP_B64", "0") == "1" # 기본값 True로 변경 (속도 우선)
FAST_SKIP_JSON = os.getenv("FAST_SKIP_JSON", "0") == "1"
SAVE_LOG_FILES = os.getenv("SAVE_LOG_FILES", "1") == "1"

# 파이프라인 병렬 처리 워커 수
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 8))

# 모델 입력 사이즈
BASE_SIZE = int(os.getenv("BASE_SIZE", "1024"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "640"))

# 기타 임계값
PROBLEM_MARGIN_RATIO = float(os.getenv("PROBLEM_MARGIN_RATIO", "0.05"))
BOTTOM_CUTOFF_RATIO = float(os.getenv("BOTTOM_CUTOFF_RATIO", "0.80"))
CLIP_MARGIN = int(os.getenv("CLIP_MARGIN", "5"))
MIN_BOX_W = int(os.getenv("MIN_BOX_W", "24"))

TMP_DIR = os.getenv("TMP_DIR", os.path.join(OUT_DIR, "_tmp_infer"))
SAT_MODE = os.getenv("SAT_MODE", "0") == "1"  # SAT: 검은 사각형 안 흰 숫자 별도 탐지
# (레거시) 상단 클리핑은 이제 CSAT 분류 시 y_cut 기준으로 적용 (MIN_TOP_CUTOFF_* 미사용)
MIN_TOP_CUTOFF_PX = int(os.getenv("MIN_TOP_CUTOFF_PX", "500"))
MIN_TOP_CUTOFF_WH = (2924, 4136)

# =========================
# 1) 전역 정규식/로깅 설정
# =========================

QUESTION_ANCHOR_RE      = re.compile(r"^\s*(\d{1,3})\s*[\.．\)\:]\s*")
QUESTION_ANCHOR_FUZZY_RE = re.compile(r"(?:^[^\d]{0,3})?(\d{1,3})\s*[\.．\)\:]\s*")
# SAT 등: 문항 번호가 단독 블록("10", "11")으로 나올 때
QUESTION_ANCHOR_STANDALONE_RE = re.compile(r"^\s*(\d{1,3})\s*$")
# sub_title 블록에서 "## 3" 형태 (마크다운 헤딩); 공백 없이 ##3 도 허용
QUESTION_ANCHOR_SUBTITLE_RE = re.compile(r"^\s*#+\s*(\d{1,3})\s*$")
# 한 줄이 숫자만 있을 때 (sub_title/text 블록에서 문항 번호만 OCR된 경우)
QUESTION_ANCHOR_SUBTITLE_LOOSE_RE = re.compile(r"^\s*#*\s*(\d{1,3})\s*$")
# 본문 중간에 줄바꿈 다음 숫자만 단일로 나올 때 문항 번호로 분리
NEWLINE_STANDALONE_NUMBER_RE = re.compile(r"\n\s*(\d{1,3})\s*\n")

EXCLUDE_HEAD_PAT = re.compile(
    r"(지선다형|단답형|서답형|홀수형|짝수형|수학\s*영역|제\s*\d+\s*교시)",
    re.I
)

_PAGE_NUM_RE = re.compile(r"^\s*(\d{1,2})\s*[\s/–-]\s*(\d{1,3})\s*$")
_ZW_RE = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2060-\u206F\u200E\u200D\u202F\u2060\u2061\u2062\u2063\u2064\u206A-\u206F\uFEFF]")

TAG_RE = re.compile(
    # 수정: lookahead에서 '>'를 제거 — 수식/텍스트 내 부등호(>)에서 content가 잘리는 버그 수정
    # r"...\s*(?=(?:<\|ref\||>|$))"  ← 기존: > 문자에서 content 캡처 중단
    r"<\|ref\|>(?P<rtype>\w+)<\|/ref\|>\s*<\|det\|>(?P<box>\[[^\]]+\]|\[\[[^\]]+\]\])<\|/det\|>\s*(?P<content>.*?)\s*(?=(?:<\|ref\||$))",
    re.S
)
_TAG_SEQUENCE_RE = re.compile(r"<\s*\|ref\|\s*>\s*.*?<\s*\|/ref\|\s*>\s*<\s*\|det\|\s*>\s*.*?<\s*\|/det\|\s*>", re.S)

_GLOBAL_FONT = None
def _get_font():
    global _GLOBAL_FONT
    if _GLOBAL_FONT is not None:
        return _GLOBAL_FONT
    try:
        _GLOBAL_FONT = ImageFont.truetype("malgun.ttf", 20)
    except IOError:
        try:
            _GLOBAL_FONT = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            _GLOBAL_FONT = ImageFont.load_default()
    return _GLOBAL_FONT

def setup_logger(out_dir: str):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(out_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{ts}.log"

    logger = logging.getLogger("dla_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    ch = logging.StreamHandler()

    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logging.captureWarnings(True)
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FutureWarning)
    transformers.logging.set_verbosity_error()
    
    # tqdm 충돌 방지 (기존 코드가 이 로직을 사용)
    import tqdm as tq
    tq.utils._supports_unicode = True

    logger.info(f"Logging initialized. Log file → {log_path}")
    return logger

# =========================
# 2) 모델 로드 (1회)
# =========================
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = model.eval().to(dtype).to("cuda" if torch.cuda.is_available() else "cpu")

PROMPT_MAIN = "<image>\n<|grounding|>Convert the document to markdown. "

# =========================
# 3) 공통 유틸/헬퍼 (불필요한 코드 제거 및 간소화)
# =========================
def _clean_ocr_text(s: str) -> str:
    if not s: return ""
    # 수정: NFKC normalization이 원문자(①②③④⑤, U+2460~U+2473)를 일반 숫자로 변환하는 버그 수정
    # 원문자를 임시 placeholder로 치환 → NFKC 적용 → 원문자 복원
    # old: s = unicodedata.normalize("NFKC", s).replace("\u00A0", " ")
    _circled = {chr(0x2460 + i): f"__CIRC{i}__" for i in range(20)}
    for ch, ph in _circled.items():
        s = s.replace(ch, ph)
    s = unicodedata.normalize("NFKC", s).replace("\u00A0", " ")
    for ch, ph in _circled.items():
        s = s.replace(ph, ch)
    # 수정 끝
    s = _ZW_RE.sub("", s)
    return s.strip()

def infer_stdout(model, tokenizer, prompt, image_file, base_size=1024, image_size=640, crop_mode=True):
    # infer_stdout에서 로그 파일 저장을 제거하고, main에서 처리하도록 간소화
    os.makedirs(TMP_DIR, exist_ok=True)
    
    buf = io.StringIO()
    ts = time.strftime("%Y%m%d_%H%M%S")
    with contextlib.redirect_stdout(buf):
        _ = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=image_file,
            output_path=TMP_DIR,
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            save_results=True, # 모델이 임시파일을 생성하도록 유지
        )
    stdout_text = buf.getvalue()
    return stdout_text, ts

def norm999_to_orig_box(bx_norm, W, H):
    x1,y1,x2,y2 = bx_norm
    return [
        int(round(x1/999*W)), int(round(y1/999*H)),
        int(round(x2/999*W)), int(round(y2/999*H))
    ]

def _orig_box_to_norm999(bx_px, W, H):
    x1,y1,x2,y2 = bx_px
    return [
        int(round(x1/W*999)), int(round(y1/H*999)),
        int(round(x2/W*999)), int(round(y2/H*999))
    ]

def _iou_rect(a, b):
    """두 사각형 [x1,y1,x2,y2] 겹침 비율 (a 기준)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1); ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    return inter / max(area_a, 1)


def _contour_box_covered_by_ocr(box_px: list, blocks: list, iou_thresh: float = 0.25) -> bool:
    """Contour로 찾은 box_px [x1,y1,x2,y2]가 이미 OCR 블록 중 하나와 충분히 겹치면 True."""
    for b in blocks:
        ob = b.get("bbox_px") or []
        if len(ob) != 4:
            continue
        if _iou_rect(box_px, ob) >= iou_thresh or _iou_rect(ob, box_px) >= iou_thresh:
            return True
    return False


def _merge_overlapping_boxes(rects, iou_thresh=0.5):
    """겹치는 사각형을 하나로 (작은 것 우선 유지)."""
    if not rects:
        return []
    out = []
    for (x, y, w, h) in rects:
        b = [x, y, x + w, y + h]
        merged = False
        for existing in out:
            ex, ey, ew, eh = existing
            a = [ex, ey, ex + ew, ey + eh]
            if _iou_rect(a, b) >= iou_thresh or _iou_rect(b, a) >= iou_thresh:
                merged = True
                break
        if not merged:
            out.append((x, y, w, h))
    return out


def _log_sat_fallback(msg: str, count: int, tesseract: bool = False):
    try:
        log = logging.getLogger("dla_pipeline")
        if count > 0:
            log.info(f"SAT number boxes: {count} found (tesseract={tesseract})")
        else:
            log.debug(f"SAT number boxes: {msg} -> 0 boxes")
    except Exception:
        pass


def find_sat_number_boxes(img_path: str, W: int, H: int):
    """
    SAT 스타일: 검은색 사각형 박스 내 흰색 숫자 영역을 찾아 (bbox_px, 숫자) 목록 반환.
    - Gaussian blur + Otsu 이진화 + morphology로 박스 후보 추출
    - 크기/위치/어두움/흰색비/컬럼 필터 후 비중첩 선택 (참고 알고리즘 적용)
    - pytesseract는 선택(없으면 읽기 순서로 1,2,3... 부여).
    """
    abs_path = os.path.abspath(img_path)
    out = []
    try:
        import cv2
    except ImportError:
        _log_sat_fallback("cv2 not installed", 0)
        return out
    try:
        img = cv2.imread(abs_path)
        if img is None:
            _log_sat_fallback(f"cv2.imread failed: {abs_path}", 0)
            return out
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        # 참고 알고리즘: blur + Otsu + morphology
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, bin_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bin_inv = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, kernel, iterations=1)
        bin_inv = cv2.morphologyEx(bin_inv, cv2.MORPH_CLOSE, kernel, iterations=2)
        cnts, _ = cv2.findContours(bin_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cands = []
        for c in cnts:
            x, y, ww, hh = cv2.boundingRect(c)
            area = ww * hh
            if area < 150 or area > 20000:
                continue
            if not (20 <= ww <= 90 and 18 <= hh <= 80):
                continue
            if y > int(0.40 * h):
                continue
            roi = gray[y : y + hh, x : x + ww]
            if roi.size == 0:
                continue
            mean_val = float(np.mean(roi))
            white_ratio = float(np.mean(roi > 200))
            if mean_val > 130:
                continue
            if not (0.005 <= white_ratio <= 0.40):
                continue
            # 검은 박스 성질: 매우 어두운 픽셀 비율이 충분히 높아야 함
            dark_ratio = float(np.mean(roi < 60))
            if dark_ratio < 0.55:
                continue
            # 테두리가 어두워야 함 (흰 배경 위 글자 조각 제거)
            b = 2
            top = roi[:b, :]
            bottom = roi[-b:, :]
            left = roi[:, :b]
            right = roi[:, -b:]
            border = np.concatenate([top.ravel(), bottom.ravel(), left.ravel(), right.ravel()])
            border_mean = float(np.mean(border))
            if border_mean > 90:
                continue
            if not (x < int(0.42 * w) or x > int(0.48 * w)):
                continue
            cands.append((x, y, ww, hh, area, mean_val, white_ratio))

        cands = sorted(cands, key=lambda t: (-t[4], t[1], t[0]))
        if not cands:
            _log_sat_fallback("no candidates after filters (dark_ratio/border)", 0)
            return out
        picked = []
        for cand in cands:
            x, y, ww, hh, area, _mv, _wr = cand
            keep = True
            for px, py, pw, ph, *_ in picked:
                xx1 = max(x, px)
                yy1 = max(y, py)
                xx2 = min(x + ww, px + pw)
                yy2 = min(y + hh, py + ph)
                inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
                if inter > 0.5 * min(ww * hh, pw * ph):
                    keep = False
                    break
            if keep:
                picked.append(cand)
            if len(picked) >= 6:
                break
        picked.sort(key=lambda t: (t[1], t[0]))

        if not picked:
            _log_sat_fallback("no boxes found", 0)
            return out

        tesseract_ok = False
        try:
            import pytesseract  # noqa: F401
            tesseract_ok = True
        except ImportError:
            pass

        for idx, (x, y, ww, hh, *_rest) in enumerate(picked):
            box_px = [x, y, x + ww, y + hh]
            crop = gray[y : y + hh, x : x + ww]
            n = None
            if tesseract_ok:
                try:
                    import pytesseract
                    _, crop_inv = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    num_str = pytesseract.image_to_string(crop_inv, config="--psm 7 -c tessedit_char_whitelist=0123456789").strip()
                    if not num_str:
                        num_str = pytesseract.image_to_string(crop, config="--psm 7 -c tessedit_char_whitelist=0123456789").strip()
                    digits = "".join(c for c in num_str if c.isdigit())
                    if 1 <= len(digits) <= 3:
                        nn = int(digits)
                        if 1 <= nn <= 999:
                            n = nn
                except Exception:
                    pass
            if n is None:
                n = idx + 1
            out.append((box_px, n))
        _log_sat_fallback("ok", len(out), tesseract=tesseract_ok)
    except Exception as e:
        _log_sat_fallback(str(e), 0)
    return out

def parse_blocks_from_stdout(raw: str):
    blocks = []
    for m in TAG_RE.finditer(raw):
        typ = (m.group("rtype") or "").strip()
        txt = (m.group("content") or "")
        rb = (m.group("box") or "").strip()
        
        txt = _TAG_SEQUENCE_RE.sub("", txt)
        if typ == "image": txt = ""
        if not typ or not rb: continue
        
        txt = _clean_ocr_text(txt)
        try:
            coords = ast.literal_eval(rb)
        except Exception:
            continue
        if isinstance(coords[0], (int, float)): coords = [coords]
        coords = [bx for bx in coords if len(bx) == 4]
        if not coords: continue

        # 줄바꿈 다음 숫자만 단일로 나오면 문항 번호로 인지해 블록 분리 (OCR이 한 블록에 여러 문항을 묶어 낸 경우)
        if typ in ("text", "sub_title") and NEWLINE_STANDALONE_NUMBER_RE.search(txt):
            parts = NEWLINE_STANDALONE_NUMBER_RE.split(txt)
            # parts: [앞내용, 숫자1, 뒤내용] 또는 [앞, 번호1, 중간, 번호2, ...] → 첫 조각은 coords[0], 숫자·이후는 coords[1] 등
            for i, seg in enumerate(parts):
                seg = seg.strip()
                if not seg:
                    continue
                coord_idx = min(i, len(coords) - 1)
                blocks.append({"type": typ, "text": seg, "bbox_norm": coords[coord_idx]})
        else:
            for bx in coords:
                blocks.append({"type": typ, "text": txt, "bbox_norm": bx})
    return blocks

def is_anchor(text: str):
    t = _clean_ocr_text(text or "")
    if not t:
        return None
    # 전체 문자열로 먼저 시도
    m = (
        QUESTION_ANCHOR_RE.match(t)
        or QUESTION_ANCHOR_FUZZY_RE.match(t)
        or QUESTION_ANCHOR_STANDALONE_RE.match(t)
        or QUESTION_ANCHOR_SUBTITLE_RE.match(t)
        or QUESTION_ANCHOR_SUBTITLE_LOOSE_RE.match(t)
    )
    if m:
        return int(m.group(1))
    # sub_title 등에서 여러 줄일 때 첫 번째로 매칭되는 줄 사용 (예: "\n## 19\n" 또는 "## 19\n\n")
    for line in t.splitlines():
        line = line.strip()
        if not line:
            continue
        m = QUESTION_ANCHOR_SUBTITLE_RE.match(line) or QUESTION_ANCHOR_SUBTITLE_LOOSE_RE.match(line) or QUESTION_ANCHOR_STANDALONE_RE.match(line)
        if m:
            return int(m.group(1))
    return None

def merge_bbox(bbs):
    xs1,ys1,xs2,ys2 = zip(*bbs)
    return [min(xs1),min(ys1),max(xs2),max(ys2)]

def expand_with_margin(bx,W,H,ratio):
    x1,y1,x2,y2=bx; w,h=x2-x1,y2-y1
    mx=int(abs(w)*ratio)
    my=int(abs(h)*ratio)
    mx_right = int(mx * 1.25)   # 우측 여백만 조금 더
    my_bottom = int(my * 3.0)   # 하단 여백 조금 더 (기존 1.5 → 2.0)
    return [max(0, x1-mx), max(0, y1-my), min(W, x2+mx_right), min(H, y2+my_bottom)]

def draw_boxes_on_image(img_path, groups, out_path):
    if FAST_SKIP_VIS: return
    im=Image.open(img_path).convert("RGB")
    W,H=im.size; d=ImageDraw.Draw(im)
    fnt=_get_font()
    for g in (groups or []):
        box = g.get("bbox_problem_with_margin")
        if not box:
            bbs=[b["bbox_px"] for b in g["blocks"]]
            if not bbs: continue
            box=expand_with_margin(merge_bbox(bbs),W,H,PROBLEM_MARGIN_RATIO)
        x1,y1,x2,y2 = box
        d.rectangle([x1,y1,x2,y2],outline=(0,180,255),width=3)
        label=str(g["qnum"])
        try:
            tb = d.textbbox((x1, y1), label, font=fnt, anchor="lt")
            tw, th = tb[2]-tb[0], tb[3]-tb[1]
        except Exception:
            tw, th = 20, 20
        tx, ty = x1+4, y1-th-6
        d.rectangle([tx-4, ty, tx+tw+4, ty+th+4], fill=(0,180,255))
        d.text((tx, ty+1), label, fill=(255,255,255), font=fnt)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)

def dynamic_top_cutoff(blocks, H):
    hdr_bottoms = [b["bbox_px"][3] for b in blocks if b["type"] in {"title","sub_title","header"}]
    anchor_tops = [b["bbox_px"][1] for b in blocks if b["type"]=="text" and is_anchor(b.get("text"))]
    y_cands = []
    if hdr_bottoms: y_cands.append(max(hdr_bottoms) + 8)
    if anchor_tops: y_cands.append(min(anchor_tops) - 18)
    if not y_cands: return 0
    return min(int(min(y_cands)), int(H*0.15))

def remove_bottom_noise(blocks, H):
    cleaned = []
    NOISE_KEYWORDS = ("확인 사항","담당자","답안지","선택과목","기입","표기","문제지","자신이 선택한","제시되었")
    H06, H082 = H*0.6, H*0.82
    for b in blocks:
        x1,y1,x2,y2 = b["bbox_px"]
        if y1 < H*0.3: cleaned.append(b); continue
        if b["type"] != "text": cleaned.append(b); continue
        t = _clean_ocr_text(b.get("text") or "")
        if not t: continue
        if y1 > H06 and (any(k in t for k in NOISE_KEYWORDS) or t.startswith(("○","•","*"))): continue
        if _PAGE_NUM_RE.match(t): continue
        if y1 > H082 and (len(t) < 10 and sum(ch.isdigit() for ch in t)/max(1,len(t)) > 0.6): continue
        cleaned.append(b)
    return cleaned

def detect_divider_or_mid(blocks, W, H):
    return 0.5 * W

# 4) 블록 중복 제거 유틸 추가
def dedup_blocks_exact(blocks):
    seen = set()
    uniq = []
    for b in blocks:
        t  = _clean_ocr_text(b.get("text") or "")
        x1,y1,x2,y2 = map(int, b["bbox_px"])
        key = (b["type"], t, x1//2, y1//2, x2//2, y2//2)  # 2px 그리드로 라운딩
        if key in seen:
            continue
        seen.add(key)
        b["text"] = t
        uniq.append(b)
    return uniq


# 4) 컬럼 침범 방지 헬퍼 (유지)
def clip_box_to_column(bx, x_div, col, W, margin=5):
    x1,y1,x2,y2 = map(int, bx)
    if col == 0:
        x2 = min(x2, int(x_div - margin))
    else:
        x1 = max(x1, int(x_div + margin))
    x1 = max(0, x1); x2 = min(W, x2)
    if x2 - x1 < 2: return None
    return [x1,y1,x2,y2]

def clip_group_blocks_to_column(g, x_div, W, margin=5):
    clipped=[]; dropped=0
    for b in g["blocks"]:
        cb = clip_box_to_column(b["bbox_px"], x_div, g["col"], W, margin)
        if cb is None: dropped += 1; continue
        nb = dict(b); nb["bbox_px"]=cb; clipped.append(nb)
    # logger를 사용하지 않으므로 print 제거
    g["blocks"]=clipped

def enforce_column_box(box, x_div, col, W, margin=5, min_width=24):
    x1,y1,x2,y2 = map(int, box)
    if col==0:
        x2 = min(x2, int(x_div - margin)); x1 = max(0, x1)
        if x2 - x1 < min_width: x1 = max(0, x2 - min_width)
    else:
        x1 = max(x1, int(x_div + margin)); x2 = min(W, x2)
        if x2 - x1 < min_width: x2 = min(W, x1 + min_width)
    if x2 <= x1:
        x1 = max(0, min(int(x_div - margin - min_width), W - min_width))
        x2 = min(W, x1 + min_width)
    return [x1,y1,x2,y2]

# 5) 컬럼 분리/그룹핑 (유지)
def split_blocks_into_columns(blocks_px, W, H):
    x_div = detect_divider_or_mid(blocks_px, W, H)
    buf = int(W*0.003)
    left, right, center = [], [], []
    for b in blocks_px:
        x1,y1,x2,y2 = b["bbox_px"]; cx=(x1+x2)*0.5
        if cx < x_div - buf: left.append(b)
        elif cx > x_div + buf: right.append(b)
        else: center.append(b)
    if center:
        for b in center:
            x_mid = (b["bbox_px"][0] + b["bbox_px"][2])*0.5
            (right if x_mid >= x_div*0.99 else left).append(b)
    if len(left)==0 or len(right)==0:
        left, right = [], []
        for b in blocks_px:
            cx=(b["bbox_px"][0]+b["bbox_px"][2])*0.5
            (left if cx < 0.5*W else right).append(b)
    return left, right, x_div


def _split_block_indices_by_gaps(B, indices, num_segments):
    """
    indices: list of indices into B. Sort by y1, split into num_segments
    by (num_segments - 1) largest vertical gaps. Return list of index lists.
    """
    if num_segments <= 1 or not indices:
        return [indices] if indices else []
    sorted_idx = sorted(indices, key=lambda j: B[j]["bbox_px"][1])
    if len(sorted_idx) < num_segments:
        return [sorted_idx]
    gaps = []
    for i in range(len(sorted_idx) - 1):
        j1, j2 = sorted_idx[i], sorted_idx[i + 1]
        gap = B[j2]["bbox_px"][1] - B[j1]["bbox_px"][3]
        gaps.append((gap, i + 1))
    gaps.sort(key=lambda x: -x[0])
    split_ats = sorted(g[1] for g in gaps[: num_segments - 1])
    result = []
    start = 0
    for at in split_ats:
        result.append(sorted_idx[start:at])
        start = at
    result.append(sorted_idx[start:])
    return result


def group_column_blocks(blocks, y_tol=8):
    if not blocks: return []
    B = sorted(blocks, key=lambda b:(b["bbox_px"][1], b["bbox_px"][0]))
    anchors=[]
    for i,b in enumerate(B):
        if b["type"] in {"text", "sub_title", "equation"}:
            txt = _clean_ocr_text(b.get("text") or "")
            if EXCLUDE_HEAD_PAT.search(txt):
                continue
            qn = is_anchor(txt)
            if qn:
                anchors.append({"idx": i, "qnum": qn, "bbox": b["bbox_px"]})
    if not anchors:
        # text 또는 sub_title 블록에서 첫 앵커 후보 탐색
        first_text_idx = next(
            (i for i, b in enumerate(B) if b["type"] in ("text", "sub_title") and _clean_ocr_text(b.get("text"))),
            None,
        )
        if first_text_idx is None:
            return []
        t0 = _clean_ocr_text(B[first_text_idx]["text"])
        m = (
            QUESTION_ANCHOR_RE.match(t0)
            or QUESTION_ANCHOR_STANDALONE_RE.match(t0)
            or QUESTION_ANCHOR_SUBTITLE_RE.match(t0)
            or QUESTION_ANCHOR_SUBTITLE_LOOSE_RE.match(t0)
        )
        if not m:
            for line in t0.splitlines():
                line = line.strip()
                m = QUESTION_ANCHOR_SUBTITLE_RE.match(line) or QUESTION_ANCHOR_SUBTITLE_LOOSE_RE.match(line) or QUESTION_ANCHOR_STANDALONE_RE.match(line)
                if m:
                    break
        if not m:
            # 앵커 없음 → 해당 컬럼 블록 전부를 1번 문항 하나로 묶기 (fallback)
            anchors = [{"idx": 0, "qnum": 1, "bbox": B[0]["bbox_px"]}]
        else:
            qn = int(m.group(1))
            anchors.append({"idx": first_text_idx, "qnum": qn, "bbox": B[first_text_idx]["bbox_px"]})
    groups=[{"qid":str(a["qnum"]), "qnum":a["qnum"], "anchor":a, "blocks":[B[a["idx"]]]} for a in anchors]
    for j,b in enumerate(B):
        if any(g["anchor"]["idx"]==j for g in groups): continue
        y1=b["bbox_px"][1]
        for k,a in enumerate(anchors):
            a_next = anchors[k+1] if k+1<len(anchors) else None
            top_a=a["bbox"][1]; top_n = a_next["bbox"][1] if a_next else 1e9
            if top_a - y_tol <= y1 < top_n + y_tol:
                groups[k]["blocks"].append(b); break

    # 갭 채우기: 앵커 사이에 누락된 문항 번호(N+1..M-1)를 거리(세로 간격)로 분할 (뒤 쌍부터 처리해 인덱스 유지)
    # SAT 등 sub_title 앵커가 이미 많으면 잘못된 분할로 문항 병합/잘못 분리될 수 있으므로 건너뜀
    sub_title_anchor_count = sum(1 for a in anchors if B[a["idx"]].get("type") == "sub_title")
    do_gap_fill = (sub_title_anchor_count <= 1) or (sub_title_anchor_count < len(anchors) // 2)
    if do_gap_fill:
        for k in range(len(anchors) - 2, -1, -1):
            if k + 1 >= len(anchors): continue
            N, M = anchors[k]["qnum"], anchors[k + 1]["qnum"]
            if M - N <= 1: continue
            g = groups[k]
            anchor_idx = g["anchor"]["idx"]
            non_anchor_indices = [j for j in range(len(B)) if B[j] in g["blocks"] and j != anchor_idx]
            if len(non_anchor_indices) < 2:
                continue
            segments = _split_block_indices_by_gaps(B, non_anchor_indices, M - N)
            if len(segments) != M - N:
                continue
            g["blocks"] = [B[anchor_idx]] + [B[j] for j in segments[0]]
            for i in range(1, M - N):
                qnum = N + i
                seg = segments[i]
                new_anchor = {"idx": seg[0], "qnum": qnum, "bbox": B[seg[0]]["bbox_px"]}
                groups.insert(k + i, {"qid": str(qnum), "qnum": qnum, "anchor": new_anchor, "blocks": [B[j] for j in seg]})

    # 첫 앵커 위 블록 → 누락된 1..(first_qnum-1) (첫 앵커가 2 이상일 때만, 거리로 분할)
    first_top = anchors[0]["bbox"][1]
    first_qnum = anchors[0]["qnum"]
    orphan_indices = [j for j in range(len(B)) if not any(g["anchor"]["idx"] == j for g in groups) and B[j]["bbox_px"][1] < first_top - y_tol]
    if first_qnum > 1 and orphan_indices:
        num_lead = first_qnum - 1
        lead_segments = _split_block_indices_by_gaps(B, orphan_indices, num_lead)
        for i in range(num_lead - 1, -1, -1):
            seg = lead_segments[i] if i < len(lead_segments) else []
            if not seg:
                continue
            qnum = i + 1
            new_anchor = {"idx": seg[0], "qnum": qnum, "bbox": B[seg[0]]["bbox_px"]}
            groups.insert(i, {"qid": str(qnum), "qnum": qnum, "anchor": new_anchor, "blocks": [B[j] for j in seg]})

    groups.sort(key=lambda g: g["qnum"])
    final=[]
    for g in groups:
        sb=sorted(g["blocks"], key=lambda b:b["bbox_px"][1])
        anchor_block=B[g["anchor"]["idx"]]
        if len(sb)<=1: final.append(g); continue
        y2s=[b["bbox_px"][3] for b in sb if b is not anchor_block and b["bbox_px"][3] < 3500]
        if not y2s: final.append(g); continue
        y_thr=float(np.percentile(y2s,60)); keep=[]
        for b in sb:
            y1=b["bbox_px"][1]
            if b is anchor_block or (y1 <= y_thr + 200):
                keep.append(b)
            else:
                t=_clean_ocr_text(b.get("text") or "")
                if not (len(t)<10 or (len(t)<15 and sum(ch.isdigit() for ch in t)/max(1,len(t.replace(' ','')))>0.5)):
                    keep.append(b)
        g["blocks"]=keep; final.append(g)
    return final

def group_questions_two_columns(blocks_px_nohdr, W, H, y_tol=8):
    left, right, x_div = split_blocks_into_columns(blocks_px_nohdr, W, H)
    gL = group_column_blocks(left, y_tol=y_tol)
    gR = group_column_blocks(right, y_tol=y_tol)
    for g in gL: g["col"]=0
    for g in gR: g["col"]=1
    groups = (gL + gR)
    groups.sort(key=lambda g:(g["col"], g["qnum"]))
    return groups, x_div

def merge_texts(items):
    lines=[]; buf=[]; last_y1=-1
    for it in sorted(items, key=lambda x:(x["bbox_px"][1], x["bbox_px"][0])):
        t=_clean_ocr_text(it.get("text") or "")
        # 수정 #
        # if not t or it["type"]!="text": continue
        if not t or it["type"] not in {"text", "equation"}: continue
        # 수정 #
        y1=it["bbox_px"][1]
        if not buf: buf=[it]; last_y1=y1
        elif abs(y1-last_y1)<=16: buf.append(it)
        else:
            lines.append(" ".join(_clean_ocr_text(i.get("text")) for i in buf)); buf=[it]; last_y1=y1
    if buf: lines.append(" ".join(_clean_ocr_text(i.get("text")) for i in buf))
    final="\n".join(lines)
    final=re.sub(r"={10,}\s*save results:\s*={10,}", "", final, flags=re.IGNORECASE)
    final=re.sub(r"\s+\[\d{1,3}점\]\s*(\s*\d{1,5}){1,5}\s*$", r" [4점]", final)
    # 수정: 원문자(①~⑳, U+2460~U+2473) 직후 숫자는 선지 값이므로 제거하지 않도록 negative lookbehind 추가
    # old: final=re.sub(r"(\s*\n\s*|\s+)(\d{1,5}){1,5}\s*$", "", final)
    final=re.sub(r"(?<![\u2460-\u2473])(\s*\n\s*|\s+)(\d{1,5}){1,5}\s*$", "", final)
    # 수정 끝
    final='\n'.join(line for line in final.split('\n') if line.strip())
    return final

def pick_best_groups(groups, W, H):
    cand = {}
    for g in groups:
        # 미리 텍스트/영역 스코어 계산
        t = merge_texts(g["blocks"])
        g["_merged_text"] = t
        bbs = [b["bbox_px"] for b in g["blocks"]]
        if bbs:
            x1,y1,x2,y2 = merge_bbox(bbs)
            area = max(1, (x2-x1)*(y2-y1))
        else:
            area = 1
        score = (len(t), area)
        key = g["qid"]
        if key not in cand or score > cand[key]["_score"]:
            g["_score"] = score
            cand[key] = g
    # 점수 기준으로 하나씩만 반환
    return [cand[k] for k in sorted(cand, key=lambda x:int(x))]

# =========================
# 6) 페이지 후처리(병렬 실행 대상) - OUT_DIR 인수를 받도록 수정
# =========================
def process_page_with_outdir(stdout_text, ts, imgf, current_out_dir, sat_contour=None):
    """sat_contour: if not None, override global SAT_MODE for contour injection (used by process_single_image)."""
    pid = Path(imgf).stem
    
    # 🚨 개별 아웃풋 디렉토리 사용: Path 객체로 관리
    pdir = current_out_dir / pid
    raw = pdir / "raw"
    vis = pdir / "vis"
    
    pdir.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    vis.mkdir(parents=True, exist_ok=True)

    im_orig = Image.open(imgf)
    W,H = im_orig.size

    # Parse
    parsed_blocks = parse_blocks_from_stdout(stdout_text)
    if not parsed_blocks:
        im_orig.close(); return None

    blocks = [{"type": b["type"], "text": b["text"], "bbox_norm": b["bbox_norm"],
               "bbox_px": norm999_to_orig_box(b["bbox_norm"], W, H)}
              for b in parsed_blocks]

    blocks = dedup_blocks_exact(blocks)

    use_sat_contour = sat_contour if sat_contour is not None else SAT_MODE
    # SAT 모드: 항상 contour로 문항 번호 박스 탐지. OCR 영역과 겹치면 패스, 아니면 추가
    contour_injected_qids = []
    if use_sat_contour:
        sat_boxes = find_sat_number_boxes(os.path.abspath(imgf), W, H)
        for (box_px, num) in sat_boxes:
            if _contour_box_covered_by_ocr(box_px, blocks, iou_thresh=0.25):
                continue
            blocks.append({
                "type": "sub_title",
                "text": str(num),
                "bbox_norm": _orig_box_to_norm999(box_px, W, H),
                "bbox_px": list(box_px),
                "_from_contour": True,
            })
            contour_injected_qids.append(num)
        if contour_injected_qids:
            try:
                logging.getLogger("dla_pipeline").info(
                    f"SAT: contour로 문항번호 추가 {len(contour_injected_qids)}개 (qids: {sorted(contour_injected_qids)}) for {Path(imgf).name}"
                )
            except Exception:
                pass

    # Filtering/Grouping
    y_cut = dynamic_top_cutoff(blocks, H)
    # CSAT: 헤더/앵커 미탐지 시 y_cut이 0이면 클리핑이 적용되지 않음 → 최소 상단 구간 사용
    if not use_sat_contour and y_cut <= 0:
        y_cut = max(1, int(H * 0.08))
    blocks = [b for b in blocks if b["bbox_px"][1] >= y_cut]
    Hcut = H * BOTTOM_CUTOFF_RATIO
    blocks = [b for b in blocks if b["bbox_px"][1] < Hcut]
    blocks = remove_bottom_noise(blocks, H)
    blocks_nohdr = []
    for b in blocks:
        t = _clean_ocr_text(b.get("text") or "")
        if b["type"] in {"title", "header"}:
            continue
        # sub_title / title 중 '지선다형/단답형/홀수형…'은 버림
        if b["type"] in {"title", "sub_title"} and EXCLUDE_HEAD_PAT.search(t):
            continue
        blocks_nohdr.append(b)
    groups, x_div = group_questions_two_columns(blocks_nohdr, W, H, y_tol=8)
    if not groups:
        # 문항 그룹이 없어도 vis는 저장 (박스 없이 원본만 → 디버깅/원인 확인용)
        if not FAST_SKIP_VIS:
            draw_boxes_on_image(imgf, [], vis / f"{pid}_vis.png")
        im_orig.close()
        return None
    groups = pick_best_groups(groups, W, H)

    # 컬럼 경계 처리 + 크롭/출력
    rows_q, rows_b64 = [], []
    page_payload = {
        "page_image": imgf, "width": W, "height": H, "y_cut": y_cut, "x_divider": x_div,
        "questions": [],
        "contour_injected_qids": contour_injected_qids,
    }

    for g in groups:
        clip_group_blocks_to_column(g, x_div, W, margin=CLIP_MARGIN)
        t = merge_texts(g["blocks"])
        bbs = [b["bbox_px"] for b in g["blocks"]]
        if not bbs: continue
        box = expand_with_margin(merge_bbox(bbs), W, H, PROBLEM_MARGIN_RATIO)
        box = enforce_column_box(box, x_div, g["col"], W, margin=CLIP_MARGIN, min_width=MIN_BOX_W)
        # CSAT일 때만: 상단 검은선(y_cut) 위로 bbox가 올라가지 않도록 클리핑 (모든 해상도 적용)
        if not use_sat_contour and y_cut > 0 and box[1] < y_cut:
            box = [box[0], max(box[1], y_cut), box[2], box[3]]
        g["bbox_problem_with_margin"] = box[:]

        # 저장 경로 (Path 객체 사용)
        qdir = pdir / "questions" / g['qid']
        qdir.mkdir(parents=True, exist_ok=True)
        qpath = qdir / f"{pid}__{g['qid']}__{ts}.png"
        im_orig.crop(tuple(map(int, box))).save(qpath)

        rows_q.append({"page_id":pid,"qid":g["qid"],"text":t})
        
        if not FAST_SKIP_B64:
            with io.BytesIO() as buf:
                crop = im_orig.crop(tuple(map(int, box)))
                crop.save(buf, format="PNG")
                b = buf.getvalue()
                b64 = base64.b64encode(b).decode()
                h  = hashlib.md5(b).hexdigest()        # 🔹 정확 중복 제거용 해시
                area = max(1, (box[2]-box[0])*(box[3]-box[1]))  # 🔹 우선순위(큰 영역 우선)
                rows_b64.append({
                    "page_id": pid,
                    "qid": g["qid"],
                    "b64": b64,
                    "b64_hash": h,
                    "area": area,
                    "w": box[2]-box[0],
                    "h": box[3]-box[1],
                })

        if not FAST_SKIP_JSON:
            page_payload["questions"].append({"qid":g["qid"],"merged_text":t,"bbox":box})

    # 시각화 저장
    draw_boxes_on_image(imgf, groups, vis / f"{pid}_vis.png")

    # 메타 저장
    if not FAST_SKIP_JSON:
         json.dump(page_payload, open(pdir / f"{pid}.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)

    im_orig.close()
    return {"rows_q": rows_q, "rows_b64": rows_b64, "questions": page_payload["questions"]}


def process_single_image(image_path, out_dir, sat_mode=False):
    """
    Process a single exam page image: OCR, group questions, return list of questions with bbox, text, crop base64.
    sat_mode: if True, use SAT contour injection for question number boxes (same as SAT_MODE=1).
    Returns list of dicts: [{"qid", "bbox", "merged_text", "crop_b64"}, ...] and image size;
    or {"questions": [], "image_width": W, "image_height": H} if no questions found.
    Uses global model/tokenizer (MODEL_NAME, IMAGE_SIZE at import time). For full SAT pipeline
    (e.g. DeepSeek-OCR-2, 768) run parser as subprocess with SAT env.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = Path(image_path)
    if not img_path.exists():
        return {"questions": [], "image_width": 0, "image_height": 0}

    stdout_text, ts = infer_stdout(
        model, tokenizer, PROMPT_MAIN, str(img_path),
        base_size=BASE_SIZE, image_size=IMAGE_SIZE, crop_mode=True
    )
    current_out_dir = out_dir / "_single"
    current_out_dir.mkdir(parents=True, exist_ok=True)
    res = process_page_with_outdir(stdout_text, ts, str(img_path), current_out_dir, sat_contour=sat_mode)
    if res is None:
        try:
            with Image.open(str(img_path)) as im:
                W, H = im.size
        except Exception:
            W, H = 0, 0
        return {"questions": [], "image_width": W, "image_height": H}

    questions = res["questions"]
    rows_b64 = res.get("rows_b64") or []
    b64_by_qid = {r["qid"]: r["b64"] for r in rows_b64}

    with Image.open(str(img_path)) as im:
        W, H = im.size

    out = []
    for q in questions:
        qid = q["qid"]
        out.append({
            "qid": qid,
            "bbox": list(q["bbox"]),
            "merged_text": q.get("merged_text", ""),
            "crop_b64": b64_by_qid.get(qid, ""),
        })
    return {"questions": out, "image_width": W, "image_height": H}


# =========================
# 7) 메인: 그룹별 반복 처리 및 개별 저장 로직 구현
# =========================
def main():
    # ⚠️ OUT_DIR은 기본 출력 폴더이며, 로그 파일이 이 경로에 생성됩니다.
    logger = setup_logger(OUT_DIR)
    logger.info("Starting batch processing with separate output folders per group.")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dirs", nargs="+", help="List of input directories")
    args, unknown = parser.parse_known_args()

    if args.input_dirs:
        BASE_IN_DIRS = args.input_dirs
    else:
        BASE_IN_DIRS = ["data/2023_math_odd", "data/2024_math_odd", "data/2025_math_odd", "data/2026_math_odd"]

    img_patterns = ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]

    # 각 입력 디렉토리별로 전체 파이프라인을 실행합니다.
    for base_in_dir in BASE_IN_DIRS:
        
        # --- (A) 현재 그룹 설정 및 이미지 검색 ---
        
        # 1. 입력 디렉토리명 (예: '2023_math_odd')을 기반으로 아웃풋 디렉토리 생성
        group_name = Path(base_in_dir).name
        current_out_dir = Path(OUT_DIR) / group_name
        current_out_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. 해당 그룹의 이미지 파일만 찾습니다.
        imgs = []
        for pat in img_patterns:
             imgs.extend(glob.glob(os.path.join(base_in_dir, pat)))
        imgs = sorted(list(set(imgs)))

        if not imgs:
            logger.warning(f"Skipping {base_in_dir}: No images found.")
            continue
            
        logger.info(f"\n--- Processing Group: {group_name} ({len(imgs)} pages) ---")

        # --- (B) GPU 추론 및 CPU 후처리 파이프라인 ---
        
        all_rows_q, all_rows_b64 = [], []
        futures = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            # GPU 추론 (메인 스레드)
            for imgf in tqdm(imgs, desc=f"Infer ({group_name})", ncols=90):
                stdout_text, ts = infer_stdout(model, tokenizer, PROMPT_MAIN, imgf,
                                               base_size=BASE_SIZE, image_size=IMAGE_SIZE,
                                               crop_mode=True)
                
                log_dir = current_out_dir / Path(imgf).stem / "raw"
                log_dir.mkdir(parents=True, exist_ok=True)
                (log_dir / f"{Path(imgf).stem}__infer_{ts}.log").write_text(stdout_text, encoding="utf-8")

                # CPU 후처리를 스레드 풀에 제출. 개별 OUT_DIR 전달
                futures.append(pool.submit(process_page_with_outdir, stdout_text, ts, imgf, current_out_dir))

            # 결과 수집
            for fu in tqdm(as_completed(futures), total=len(futures), desc=f"Post-proc ({group_name})", ncols=90):
                try:
                    res = fu.result()
                    if not res: continue
                    all_rows_q.extend(res["rows_q"])
                    all_rows_b64.extend(res["rows_b64"])
                except Exception as e:
                    logger.error(f"Postprocess error in {group_name}: {e}", exc_info=True)

        # --- (C) 그룹별 최종 CSV 저장 ---
        
        if all_rows_q:
            df_q = pd.DataFrame(all_rows_q)
            df_q = df_q.sort_values(by=["page_id","qid"]) \
                       .drop_duplicates(subset=["page_id","qid","text"]) \
                       .reset_index(drop=True)
        
            df_q["text_len"] = df_q["text"].str.len()
            df_q = (df_q
                    .sort_values(by=["page_id","qid","text_len"], ascending=[True, True, False])
                    .drop_duplicates(subset=["page_id","qid"], keep="first")
                    .sort_values(by=["page_id","qid"])
                    .drop(columns=["text_len"])
                    .reset_index(drop=True))
            df_q.to_csv(current_out_dir / "llm_input_text.csv", index=False, encoding="utf-8-sig")
        
        
        if all_rows_b64 and not FAST_SKIP_B64:
            df_b = pd.DataFrame(all_rows_b64)
        
            # 1) 완전 동일(해시 동일) 행 제거
            if "b64_hash" in df_b.columns:
                df_b = df_b.drop_duplicates(subset=["page_id", "qid", "b64_hash"])
        
            # 2) 같은 (page_id, qid)에 여러 이미지가 남아있다면,
            #    면적(area) 큰 것 우선으로 하나만 유지
            if "area" in df_b.columns:
                df_b = (df_b
                        .sort_values(by=["page_id", "qid", "area"], ascending=[True, True, False])
                        .drop_duplicates(subset=["page_id", "qid"], keep="first"))
            else:
                # area가 없으면 b64 길이 기준(대용)으로 정리
                df_b = (df_b
                        .assign(b64_len=df_b["b64"].str.len())
                        .sort_values(by=["page_id", "qid", "b64_len"], ascending=[True, True, False])
                        .drop_duplicates(subset=["page_id", "qid"], keep="first")
                        .drop(columns=["b64_len"]))
        
            # 정렬·저장
            df_b = df_b.sort_values(by=["page_id", "qid"]).reset_index(drop=True)
            df_b.to_csv(current_out_dir / "llm_input_b64.csv", index=False, encoding="utf-8-sig")
            logger.info(f"Saved base64 CSV ({len(df_b)} rows) to {group_name}/llm_input_b64.csv")

            
    logger.info("✅ All batch processing finished successfully.")

if __name__ == "__main__":
    main()