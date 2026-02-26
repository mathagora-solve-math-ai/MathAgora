# -*- coding: utf-8 -*-
"""
DeepSeek-OCR / DeepSeek-OCR-2 원시 출력 후처리.
- 메타 토큰·shape 로그 제거 (===================== BASE: torch.Size(...) PATCHES: ... 등)
- <|ref|>...<|/ref|><|det|>[[...]]<|/det|> 태그 제거, 본문만 추출
- 읽기 순서로 텍스트·수식만 이어서 저장 (한 줄 또는 적절한 줄바꿈)
"""
from __future__ import annotations

import re


def strip_meta_and_shape_log(text: str) -> str:
    """'===================== BASE: ... PATCHES: ... =====================' 블록 제거."""
    if not text or not isinstance(text, str):
        return text
    # ===== ... ===== 사이의 블록 (BASE:, PATCHES:, torch.Size 포함)
    pattern = r"=+\s*\n\s*BASE:.*?PATCHES:.*?\n\s*=+\s*"
    s = re.sub(pattern, "", text, flags=re.DOTALL)
    # 남은 단일 = 줄 또는 torch.Size만 있는 줄 제거
    s = re.sub(r"\n?=+\s*\n?", "\n", s)
    s = re.sub(r"\n\s*(?:BASE|PATCHES):\s*torch\.Size\([^)]*\)\s*\n", "\n", s)
    return s.strip()


def strip_ref_det_tags(text: str) -> str:
    """<|ref|>text<|/ref|><|det|>[[x,y,w,h]]<|/det|> 또는 equation 등 태그 제거 후 내용만 유지."""
    if not text or not isinstance(text, str):
        return text
    # <|ref|>...<|/ref|><|det|>[[...]]<|/det|> 전체를 제거 (태그만 제거, 그 뒤에 오는 본문은 유지)
    tag_block = r"<\|ref\|>[^<]*<\|/ref\|><\|det\|>\[\[[^\]]*\]\]<\|/det\|>\s*"
    s = re.sub(tag_block, "", text)
    # 단일 괄호 <|det|>[...]<|/det|> 도 제거
    s = re.sub(r"<\|det\|>\[[^\]]*\]<\|/det\|>\s*", "", s)
    return s


def normalize_equation_newlines(text: str) -> str:
    """수식 내 불필요한 줄바꿈 정리. \\[ ... \\] 또는 \\( ... \\) 를 한 줄로."""
    if not text or not isinstance(text, str):
        return text
    # \[ ... \] 블록 내부 줄바꿈을 공백 하나로
    def _repl(m: re.Match) -> str:
        inner = m.group(1)
        inner = re.sub(r"\s+", " ", inner).strip()
        return "\\[ " + inner + " \\]"

    text = re.sub(r"\\\[\s*([\s\S]*?)\s*\\\]", _repl, text)
    # \( ... \) 블록
    def _repl_p(m: re.Match) -> str:
        inner = m.group(1)
        inner = re.sub(r"\s+", " ", inner).strip()
        return "\\(" + inner + "\\)"

    text = re.sub(r"\\\(\s*([\s\S]*?)\s*\\\)", _repl_p, text)
    return text


def collapse_redundant_newlines(text: str) -> str:
    """연속 줄바꿈을 최대 1개로, 앞뒤 공백 정리."""
    if not text or not isinstance(text, str):
        return text
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def postprocess_deepseek_ocr(raw: str) -> str:
    """
    DeepSeek-OCR 원시 출력을 정제된 본문 형식으로 변환.
    - 메타·shape 로그 제거
    - ref/det 태그 제거
    - 수식 줄바꿈 정리
    - 결과: '3. 첫째항과 공비가 ... \\frac{a_4}{a_2} + ... ① 1 ② 2 ...' 형태
    """
    if not raw or not isinstance(raw, str):
        return raw
    # 에러 메시지면 그대로 반환
    if raw.strip().startswith("["):
        return raw
    s = strip_meta_and_shape_log(raw)
    s = strip_ref_det_tags(s)
    s = normalize_equation_newlines(s)
    s = collapse_redundant_newlines(s)
    return s
