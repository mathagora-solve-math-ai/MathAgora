from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

WORKSPACE = Path(__file__).resolve().parent.parent
OUTPUTS_PARSING = WORKSPACE / "data" / "outputs_parsing"
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OCR_TEXT_CONVERTER_MODEL = os.environ.get("OCR_TEXT_CONVERTER_MODEL", "openai/gpt-5")
OCR_TEXT_CONVERTER_DEBUG_DIR = Path(
    os.environ.get(
        "OCR_TEXT_CONVERTER_DEBUG_DIR",
        str(WORKSPACE / "data" / "ocr_text_conversion_debug"),
    )
)
AFTER_CONVERTOR_DEEPSEEK_DIR = OCR_TEXT_CONVERTER_DEBUG_DIR / "after_convertor_deepseek"

logger = logging.getLogger(__name__)

# DeepSeek-OCR 원시 출력에서 제거할 태그만 매칭 (postprocess_deepseek_ocr 미사용).
# 예: "<|ref|>text<|/ref|><|det|>[[52, 720, 135, 824]]<|/det|>" → 제거 후 본문만 OpenRouter로 전달.
_REF_DET_TAG_BLOCK_RE = re.compile(
    r"<\|ref\|>[^<]*<\|/ref\|><\|det\|>\[\[[^\]]*\]\]<\|/det\|>\s*"
)
_REF_DET_TAG_SINGLE_RE = re.compile(r"<\|det\|>\[[^\]]*\]<\|/det\|>\s*")

CONVERSION_SYSTEM_PROMPT = """You are an expert mathematical typesetter specializing in converting OCR-extracted Korean CSAT (수능) mathematics problems into the Digitalized CSAT-Math format.

## Your Task

Convert the given OCR text of a CSAT math problem into a clean, structured prob_desc field following the rules below.

---

## Conversion Rules

### 1. General Text

- Write regular problem body text as plain Markdown paragraphs.
- Preserve the original sentence structure and Korean text exactly.
- Do NOT add or remove any semantic content.

### 2. Mathematical Expressions

- Convert ALL mathematical expressions to LaTeX.
- Inline math: use $...$
- Block-level (display) math: use $$...$$ on its own line.
- Follow these conventions to match the printed CSAT layout:
  - Non-italic identifiers (e.g., P, E): use \\mathrm{P}, \\mathrm{E}
  - Vectors: \\vec{AB}
  - Segment lengths / line segments: \\overline{AB}
  - Limits and summations: always use \\limits so bounds appear above/below
    - e.g., \\sum\\limits_{k=1}^{n}, \\lim\\limits_{x \\to 0}
  - Fractions: \\frac{a}{b}
  - Roots: \\sqrt{a}, \\sqrt[n]{a}
  - Absolute values: |x| or \\left| x \\right| for complex expressions

### 3. Boxed 보기 / 조건 Sections

- Any boxed passage labeled "보기", "조건", or similar: represent as a Markdown blockquote using >.
- Each line inside the box starts with >.
- Example:

> 보기
> ㄱ. f(x)는 x=0에서 연속이다.
> ㄴ. f'(0) = 0이다.
> ㄷ. f(1) > 0이다.

### 4. Tables

- Represent tables using Markdown table syntax.
- Example:

| 열1 | 열2 | 열3 |
|-----|-----|-----|
| 값1 | 값2 | 값3 |

- If the table contains math, apply LaTeX formatting inside cells.

### 5. Problem Number and Score Label

- Write the problem number and score label exactly as printed.
- Format: {number}. [{score}점] on the first line.
- Example: 23. [3점]

### 6. Multiple-Choice Options

- For multiple-choice problems (객관식), preserve all five circled numerals ①–⑤ and their values; do not omit the last option (⑤).
- Apply LaTeX to any math within options.
- Example:

① $2$ ② $3$ ③ $4$ ④ $5$ ⑤ $6$

- For short-answer problems (단답형), omit choices entirely.

### 7. Figures and Diagrams

- If the OCR text references a figure/diagram that cannot be represented in text, insert a placeholder:

[그림]

- If the OCR includes partial coordinate or geometric information from a figure, reconstruct it as faithfully as possible in text/LaTeX.

---

## Common OCR Artifacts to Fix

- Superscripts misread as separate characters: x2 → $x^2$
- Subscripts misread: an → $a_n$
- Fraction bars misread: a/b → $\\frac{a}{b}$
- Greek letters misread (α, β, θ often corrupted) — infer from context and restore
- Integral/summation symbols misread — reconstruct from surrounding context
- Circled numerals ①–⑤ may appear as 1), (1), or similar — always restore to ①–⑤
- Score labels like [3점], [4점] may be misplaced — always place directly after problem number

---

## Output Format

Return ONLY the converted prob_desc text.
Do NOT add any commentary, explanation, or metadata outside the converted problem text.

---

## Examples

### Example 1: 객관식 (수열)

Input (OCR raw):

23. [3점]
수열 {an}에 대하여 sum(k=1 to n) ak = n^2 + 2n 일 때, a10의 값은?
① 20  ② 21  ③ 22  ④ 23  ⑤ 24

Output:

23. [3점]

수열 $\\{a_n\\}$에 대하여 $\\sum\\limits_{k=1}^{n} a_k = n^2 + 2n$ 일 때, $a_{10}$의 값은?

① $20$ ② $21$ ③ $22$ ④ $23$ ⑤ $24$

---

### Example 2: 보기 포함 문제

Input (OCR raw):

15. [4점]
함수 f(x)에 대하여 <보기>에서 옳은 것만을 있는 대로 고른 것은?

보기
ㄱ. f(x)는 x=0에서 연속이다.
ㄴ. f'(0) = 0이다.
ㄷ. f(1) > 0이다.

① ㄱ  ② ㄴ  ③ ㄱ,ㄴ  ④ ㄴ,ㄷ  ⑤ ㄱ,ㄴ,ㄷ

Output:

15. [4점]

함수 $f(x)$에 대하여 <보기>에서 옳은 것만을 있는 대로 고른 것은?

> 보기
> ㄱ. $f(x)$는 $x=0$에서 연속이다.
> ㄴ. $f'(0) = 0$이다.
> ㄷ. $f(1) > 0$이다.

① ㄱ ② ㄴ ③ ㄱ, ㄴ ④ ㄴ, ㄷ ⑤ ㄱ, ㄴ, ㄷ

---

### Example 3: 단답형 (표 포함)

Input (OCR raw):

29. [4점]
아래 표는 어떤 함수 f(x)의 값을 나타낸 것이다.

x    | -1 | 0 | 1
f(x) |  2 | 0 | 3

integral(-1 to 1) f(x) dx 의 값을 구하시오.

Output:

29. [4점]

아래 표는 어떤 함수 $f(x)$의 값을 나타낸 것이다.

| $x$ | $-1$ | $0$ | $1$ |
|-----|------|-----|-----|
| $f(x)$ | $2$ | $0$ | $3$ |

$$\\int\\limits_{-1}^{1} f(x)\\,dx$$

의 값을 구하시오."""

_OPENROUTER_CLIENT: OpenAI | None = None
_LATEST_AFTER_PATH_BY_ID: dict[str, Path] = {}
_CACHE_INDEX_LOADED = False


def _get_openrouter_client() -> OpenAI:
    global _OPENROUTER_CLIENT
    if _OPENROUTER_CLIENT is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is missing for OCR text conversion")
        _OPENROUTER_CLIENT = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    return _OPENROUTER_CLIENT


def _slug(value: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", (value or "").strip())
    s = s.strip("._-")
    return s or "unknown"


def _ensure_cache_index_loaded() -> None:
    global _CACHE_INDEX_LOADED
    if _CACHE_INDEX_LOADED:
        return
    after_dir = OCR_TEXT_CONVERTER_DEBUG_DIR / "after_converted_prob_desc"
    if after_dir.exists():
        by_key: dict[str, tuple[float, Path]] = {}
        for p in after_dir.glob("*.txt"):
            if "__" not in p.stem:
                continue
            key = p.stem.split("__", 1)[1]
            try:
                mtime = p.stat().st_mtime
            except Exception:
                mtime = 0.0
            prev = by_key.get(key)
            if prev is None or mtime >= prev[0]:
                by_key[key] = (mtime, p)
        _LATEST_AFTER_PATH_BY_ID.update({k: v[1] for k, v in by_key.items()})
    _CACHE_INDEX_LOADED = True


def clear_all_converter_caches(*, delete_disk_files: bool = True) -> None:
    """converter 캐시 전부 제거 (메모리 + 선택 시 디스크)."""
    global _LATEST_AFTER_PATH_BY_ID, _CACHE_INDEX_LOADED
    _LATEST_AFTER_PATH_BY_ID.clear()
    _CACHE_INDEX_LOADED = False
    if not delete_disk_files:
        return
    for subdir in ("before_raw_ocr", "after_converted_prob_desc"):
        d = OCR_TEXT_CONVERTER_DEBUG_DIR / subdir
        if not d.is_dir():
            continue
        try:
            for f in d.glob("*.txt"):
                try:
                    f.unlink()
                except OSError:
                    pass
        except OSError:
            pass
    logger.info("Converter caches cleared (disk files removed)")


def _debug_paths(problem_id: str) -> tuple[Path, Path]:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    stem = f"{ts}__{_slug(problem_id)}"
    before_dir = OCR_TEXT_CONVERTER_DEBUG_DIR / "before_raw_ocr"
    after_dir = OCR_TEXT_CONVERTER_DEBUG_DIR / "after_converted_prob_desc"
    before_dir.mkdir(parents=True, exist_ok=True)
    after_dir.mkdir(parents=True, exist_ok=True)
    return before_dir / f"{stem}.txt", after_dir / f"{stem}.txt"


def _extract_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(str(item.get("text", "")))
        return "".join(chunks).strip()
    return str(content).strip()


def convert_ocr_text_to_prob_desc(
    raw_ocr_text: str,
    *,
    problem_id: str,
) -> str:
    text = (raw_ocr_text or "").strip()
    if not text:
        return text

    before_path, after_path = _debug_paths(problem_id)
    before_path.write_text(text, encoding="utf-8")

    client = _get_openrouter_client()
    response = client.chat.completions.create(
        model=OCR_TEXT_CONVERTER_MODEL,
        messages=[
            {"role": "system", "content": CONVERSION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Convert the OCR raw text below into prob_desc format.\n\n"
                    "OCR raw text:\n"
                    f"{text}"
                ),
            },
        ],
        max_tokens=4096,
    )

    choices = getattr(response, "choices", None)
    if not choices or choices[0] is None:
        raise RuntimeError("OCR text conversion model returned no choices")

    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None) if message is not None else None
    converted = _extract_content_text(content)
    if not converted:
        raise RuntimeError("OCR text conversion output is empty")

    after_path.write_text(converted, encoding="utf-8")
    _LATEST_AFTER_PATH_BY_ID[_slug(problem_id)] = after_path
    logger.info(
        "OCR text conversion saved (problem_id=%s before=%s after=%s)",
        problem_id,
        before_path,
        after_path,
    )
    return converted


def load_latest_converted_prob_desc(problem_id: str) -> str | None:
    """Return latest converted text for problem_id from debug cache."""
    key = _slug(problem_id)
    _ensure_cache_index_loaded()
    p = _LATEST_AFTER_PATH_BY_ID.get(key)
    if p is None:
        return None
    try:
        txt = p.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    return txt or None


# ⑤ 뒤 값이 모델에 의해 잘렸을 때 raw에서 복구 (객관식 마지막 선택지)
_FIFTH_OPTION_RAW_CIRCLE = re.compile(r"⑤\s*(\d+)")  # ⑤ 4 (유니코드)
_FIFTH_OPTION_RAW_5_AFTER = re.compile(r"5\s+(\d+)\s*$", re.MULTILINE)  # 5 4 at end
_FIFTH_OPTION_RAW_BEFORE_5 = re.compile(r"(\d+)\s+5\s*$", re.MULTILINE)  # "72 5" → 72
_FIFTH_OPTION_RAW_XY_5 = re.compile(r"(\d+)\s+\d+\s*5\s*$", re.MULTILINE)  # "4 2 5" → 4 (첫 수가 ⑤ 값)


def _get_fifth_option_value_from_raw(raw_ocr_text: str) -> str | None:
    """raw OCR 텍스트에서 마지막 선택지(⑤) 값 숫자만 추출. 없으면 None."""
    raw = (raw_ocr_text or "").strip()
    if not raw:
        return None
    val = None
    m = _FIFTH_OPTION_RAW_CIRCLE.search(raw)
    if m:
        val = m.group(1)
    if not val:
        m = _FIFTH_OPTION_RAW_5_AFTER.search(raw)
        if m:
            val = m.group(1)
    if not val:
        m = _FIFTH_OPTION_RAW_BEFORE_5.search(raw)
        if m:
            before_5 = m.group(1)
            if len(before_5) >= 2:
                val = before_5
        if not val:
            m = _FIFTH_OPTION_RAW_XY_5.search(raw)
            if m:
                val = m.group(1)
    return val


def _fix_truncated_fifth_option(converted: str, raw_ocr_text: str) -> str:
    """변환 결과가 '⑤'로 끝나면 raw에서 마지막 선택지 숫자를 찾아 보정.
    '⑤ $n$'로 끝나는데 raw의 ⑤ 값이 n과 다르면 raw 값으로 교정."""
    if not converted or not isinstance(converted, str):
        return converted
    s = converted.rstrip()
    raw = (raw_ocr_text or "").strip()
    raw_val = _get_fifth_option_value_from_raw(raw_ocr_text) if raw else None

    # 이미 "⑤ $n$" 형태로 끝나는 경우: raw와 다르면 raw 값으로 교정 (마지막 ⑤만)
    match = re.search(r"⑤\s*\$(\d+)\$\s*$", s)
    if match and raw_val and match.group(1) != raw_val:
        return re.sub(r"⑤\s*\$\d+\$\s*$", f"⑤ ${raw_val}$", s)

    # "⑤" 또는 "⑤ "로 끝나는 경우(잘림): raw 값 붙이기
    if not s.endswith("⑤") and not s.endswith("⑤ "):
        return converted
    if not raw_val:
        return converted
    return s.rstrip() + f" ${raw_val}$"


def _postprocess_ocr_text(raw: str) -> str:
    """
    DeepSeek-OCR 원시 출력에서 <|ref|>...<|/ref|><|det|>[[x,y,w,h]]<|/det|> 형식 태그만 제거.
    수식/줄바꿈 등 추가 후처리 없이, 제거된 본문만 반환 → OpenRouter(CONVERSION_SYSTEM_PROMPT) 보정용.
    """
    if not raw or not isinstance(raw, str):
        return (raw or "").strip()
    s = _REF_DET_TAG_BLOCK_RE.sub("", raw)
    s = _REF_DET_TAG_SINGLE_RE.sub("", s)
    return s.strip()


def _split_raw_log_by_question(raw_text: str) -> dict[int, str]:
    """Parse full-page infer log and return { 1: raw_q1, 2: raw_q2, 3: raw_q3, 4: raw_q4 } (output 파일과 동일 로직)."""
    if "===============save results:===============" in raw_text:
        raw_text = raw_text.split("===============save results:===============")[0]
    parts = re.split(r"(?=<\|ref\|>)", raw_text)
    blocks = []
    for p in parts:
        p = p.strip()
        if not p or p.startswith("="):
            continue
        lines = p.split("\n", 1)
        tag_line = lines[0]
        content = (lines[1] or "").strip() if len(lines) > 1 else ""
        box_m = re.search(r"\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]", tag_line)
        x, y = (int(box_m.group(1)), int(box_m.group(2))) if box_m else (0, 0)
        blocks.append({"x": x, "y": y, "full": (tag_line + "\n" + content).strip()})
    def q_for(b: dict) -> int:
        if b["x"] < 400:
            return 1 if b["y"] < 500 else 2
        return 3 if b["y"] < 400 else 4
    by_q: dict[int, list[str]] = {1: [], 2: [], 3: [], 4: []}
    for b in blocks:
        by_q[q_for(b)].append(b["full"])
    return {q: "\n\n".join(by_q[q]) for q in (1, 2, 3, 4)}


def get_raw_ocr_from_infer_log(page_id: str, qid: str | int) -> str | None:
    """
    page_id(예: 2023_math_odd_page_001), qid(예: 1)에 해당하는
    infer 로그(raw DeepSeek-OCR) 문항별 텍스트를 반환. 없으면 None.
    (output 스크립트와 동일: outputs_parsing/{dataset}/{page_folder}/raw/{page_folder}__infer_*.log)
    """
    if not page_id or not str(qid).strip():
        return None
    try:
        qid_int = int(str(qid).strip())
    except ValueError:
        return None
    if "_page_" not in page_id:
        return None
    parts = page_id.rsplit("_page_", 1)
    if len(parts) != 2:
        return None
    dataset, page_num = parts[0].strip(), parts[1].strip()
    if not dataset or not page_num:
        return None
    page_folder = f"page_{page_num}" if not page_num.startswith("page_") else page_num
    raw_dir = OUTPUTS_PARSING / dataset / page_folder / "raw"
    if not raw_dir.is_dir():
        return None
    logs = sorted(raw_dir.glob(f"{page_folder}__infer_*.log"))
    if not logs:
        return None
    try:
        raw_full = logs[-1].read_text(encoding="utf-8")
    except Exception:
        return None
    by_question = _split_raw_log_by_question(raw_full)
    return by_question.get(qid_int)


def get_converted_prob_desc(
    problem_id: str,
    raw_ocr_text: str = "",
    *,
    crop_b64: str | None = None,
    run_converter_if_missing: bool = True,
) -> str:
    """
    문항의 text = converter(GPT)까지 수행된 텍스트.
    (1) 디스크 캐시 → (2) OCR 후처리(태그 제거) 후 GPT 변환 → (3) fallback.
    """
    text = (raw_ocr_text or "").strip()
    if not text:
        return ""

    # 1) 디스크 캐시 우선 (이미 GPT 변환된 결과)
    cached = load_latest_converted_prob_desc(problem_id)
    if cached:
        return _fix_truncated_fifth_option(cached, text)

    # 2) OCR 후처리(태그 제거) — GPT 입력으로 사용
    postprocessed = _postprocess_ocr_text(text)
    input_to_gpt = postprocessed if postprocessed else text

    # 3) 무조건 GPT converter 수행 → 이 결과가 문항 text
    if run_converter_if_missing:
        try:
            result = convert_ocr_text_to_prob_desc(input_to_gpt, problem_id=problem_id)
            return _fix_truncated_fifth_option(result, text)
        except Exception as exc:
            logger.warning(
                "GPT converter failed (problem_id=%s), using postprocessed/raw: %s",
                problem_id,
                exc,
            )

    return input_to_gpt.strip()
