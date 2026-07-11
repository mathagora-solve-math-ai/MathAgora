from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_CSAT_PAGE_ID_RE = re.compile(r"^(?P<year>\d{4})_math_odd_page_(?P<page>\d+)$")
_SAT_PAGE_ID_RE = re.compile(r"^sat_math_odd_page_(?P<test>\d+)_(?P<page>\d+)$")


@dataclass(frozen=True)
class _PageSpec:
    document_type: str
    subject_key: str
    page: str
    root_env: str
    default_root: Path
    fallback_root: Path
    id_prefix: str


def _root_candidates(spec: _PageSpec) -> list[Path]:
    raw = os.getenv(spec.root_env, "").strip()
    roots = [Path(raw)] if raw else []
    roots.extend([spec.default_root, spec.fallback_root])

    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        root = root.expanduser()
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        out.append(root)
    return out


def _parse_page_id(workspace: Path, page_id: str | None, document_type: str | None) -> _PageSpec | None:
    value = (page_id or "").strip()
    if not value:
        return None

    sat_match = _SAT_PAGE_ID_RE.match(value)
    if sat_match:
        test_no = sat_match.group("test")
        page = sat_match.group("page").zfill(3)
        return _PageSpec(
            document_type="sat",
            subject_key=f"sat-practice-test-{test_no}-math",
            page=page,
            root_env="SAT_PRECOMPUTED_OUTPUT_ROOT",
            default_root=workspace / "dla" / "outputs_parsing_sat" / "output",
            fallback_root=workspace / "dla_v2" / "experiments" / "outputs_parsing_sat" / "output",
            id_prefix=f"sat_{test_no}_{page}",
        )

    csat_match = _CSAT_PAGE_ID_RE.match(value)
    if csat_match:
        year = csat_match.group("year")
        page = csat_match.group("page").zfill(3)
        return _PageSpec(
            document_type="csat",
            subject_key=f"{year}_math_odd",
            page=page,
            root_env="CSAT_PRECOMPUTED_OUTPUT_ROOT",
            default_root=workspace / "dla" / "outputs_parsing_csat" / "output",
            fallback_root=workspace / "dla_v2" / "experiments" / "outputs_parsing_csat" / "output",
            id_prefix=f"{year}_math_odd_page_{page}",
        )

    if document_type == "sat":
        return None
    return None


def _infer_csat_track(page_num: int) -> str | None:
    if 1 <= page_num <= 8:
        return "common"
    if 9 <= page_num <= 12:
        return "probability"
    if 13 <= page_num <= 16:
        return "calculus"
    if 17 <= page_num <= 20:
        return "geometry"
    return None


def _build_csat_detection_id(subject_key: str, page: str, qid: str) -> str:
    year = subject_key.split("_", 1)[0]
    if year == "2026":
        return f"{subject_key}_page_{page}_{qid}"
    try:
        qnum = int(qid)
    except ValueError:
        return f"{subject_key}_page_{page}_{qid}"
    if qnum <= 22:
        return f"{year}_odd_common_{qnum}"
    track = _infer_csat_track(int(page))
    if not track:
        return f"{subject_key}_page_{page}_{qid}"
    return f"{year}_odd_{track}_{qnum}"


def _build_detection_id(spec: _PageSpec, qid: str) -> str:
    if spec.document_type == "sat":
        return f"{spec.id_prefix}_{qid}"
    return _build_csat_detection_id(spec.subject_key, spec.page, qid)


def _clean_text(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _parse_bbox(value: Any) -> list[int] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return [int(float(v)) for v in value]
    except (TypeError, ValueError):
        return None


def _candidate_paths(
    *,
    value: Any,
    root: Path,
    subject_key: str,
    page: str,
    qid: str,
    filename: str,
) -> list[Path]:
    candidates: list[Path] = []
    raw = str(value or "").strip()
    if raw:
        direct = Path(raw)
        candidates.append(direct if direct.is_absolute() else root / direct)
        marker = f"/{subject_key}/page_{page}/"
        if marker in raw:
            suffix = raw.split(marker, 1)[1]
            candidates.append(root / subject_key / f"page_{page}" / suffix)
    candidates.append(root / subject_key / f"page_{page}" / "questions" / qid / filename)

    out: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _first_existing_path(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.is_file():
            return path
    return None


def _read_question_text(question: dict[str, Any], root: Path, spec: _PageSpec, qid: str) -> str:
    path = _first_existing_path(
        _candidate_paths(
            value=question.get("postcorrection_text_path"),
            root=root,
            subject_key=spec.subject_key,
            page=spec.page,
            qid=qid,
            filename="postcorrection_text.txt",
        )
    )
    if path is not None:
        return path.read_text(encoding="utf-8").strip()
    return _clean_text(
        question.get("postcorrection_text")
        or question.get("text")
        or question.get("merged_text")
    )


def _encode_crop_b64(question: dict[str, Any], root: Path, spec: _PageSpec, qid: str) -> str:
    path = _first_existing_path(
        _candidate_paths(
            value=question.get("crop_path"),
            root=root,
            subject_key=spec.subject_key,
            page=spec.page,
            qid=qid,
            filename="crop.png",
        )
    )
    if path is None:
        return ""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def load_precomputed_detection_payload(
    workspace: Path,
    page_id: str | None,
    document_type: str | None = None,
) -> dict[str, Any] | None:
    spec = _parse_page_id(workspace, page_id, document_type)
    if spec is None:
        return None

    for root in _root_candidates(spec):
        page_dir = root / spec.subject_key / f"page_{spec.page}"
        result_path = page_dir / "result.json"
        if not result_path.is_file():
            continue

        data = json.loads(result_path.read_text(encoding="utf-8"))
        questions: list[dict[str, Any]] = []
        for question in data.get("questions") or []:
            if not isinstance(question, dict):
                continue
            qid = str(question.get("qid") or "").strip()
            bbox = _parse_bbox(question.get("bbox"))
            if not qid or bbox is None:
                continue
            text = _read_question_text(question, root, spec, qid)
            crop_b64 = _encode_crop_b64(question, root, spec, qid)
            detection_id = _build_detection_id(spec, qid)
            questions.append(
                {
                    "qid": qid,
                    "detection_id": detection_id,
                    "label": detection_id,
                    "bbox": bbox,
                    "merged_text": text,
                    "postcorrection_text": text,
                    "text_source": "postcorrection",
                    "crop_b64": crop_b64,
                    "source_crop_path": question.get("crop_path") or "",
                    "source_postcorrection_text_path": question.get("postcorrection_text_path") or "",
                }
            )

        if not questions:
            return None

        return {
            "document_type": spec.document_type,
            "image_width": int(data.get("width") or 0),
            "image_height": int(data.get("height") or 0),
            "questions": questions,
            "precomputed_source": str(result_path),
            "subject_key": spec.subject_key,
            "page": spec.page,
        }

    return None
