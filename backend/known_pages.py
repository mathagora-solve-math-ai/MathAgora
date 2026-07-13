from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

_THUMB_SIZE = (64, 64)
_MAX_VISUAL_MAD_SAME_SIZE = 0.35
_MAX_VISUAL_MAD_RESIZED = 0.20
_MAX_ASPECT_DELTA = 0.005


@dataclass(frozen=True)
class KnownPageMatch:
    page_id: str
    document_type: str
    image_path: str
    score: float
    exact: bool


@dataclass(frozen=True)
class _KnownPage:
    page_id: str
    document_type: str
    image_path: Path
    file_size: int


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _image_thumb_file(path: Path) -> tuple[int, int, bytes]:
    with Image.open(path) as im:
        width, height = im.size
        thumb = im.convert("L").resize(_THUMB_SIZE, Image.Resampling.BILINEAR)
        return width, height, thumb.tobytes()


def _image_size_file(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


@lru_cache(maxsize=512)
def _cached_image_size(path_str: str) -> tuple[int, int]:
    return _image_size_file(Path(path_str))


@lru_cache(maxsize=512)
def _cached_image_thumb(path_str: str) -> tuple[int, int, bytes]:
    return _image_thumb_file(Path(path_str))


def _mean_abs_diff(a: bytes, b: bytes) -> float:
    if len(a) != len(b):
        return 255.0
    return sum(abs(x - y) for x, y in zip(a, b)) / max(len(a), 1)


def _resolve_dataset_image(workspace: Path, image_value: str) -> Path | None:
    raw = image_value.strip()
    if not raw:
        return None
    path = Path(raw)
    candidates = [path] if path.is_absolute() else [workspace / path]
    if raw.startswith("/"):
        candidates.extend([
            workspace / raw.lstrip("/"),
            workspace / "frontend" / "public" / raw.lstrip("/"),
        ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _load_dataset_index(workspace: Path) -> dict[str, Any]:
    index_path = workspace / "frontend" / "public" / "data" / "datasets.json"
    if not index_path.is_file():
        return {}
    return json.loads(index_path.read_text(encoding="utf-8"))


def _iter_dataset_pages(workspace: Path) -> list[tuple[str, str, Path]]:
    index = _load_dataset_index(workspace)
    out: list[tuple[str, str, Path]] = []
    groups = index.get("groups") if isinstance(index, dict) else None
    if not isinstance(groups, dict):
        return out

    for group, group_data in groups.items():
        if group not in {"csat", "sat"} or not isinstance(group_data, dict):
            continue
        years = group_data.get("years")
        if not isinstance(years, dict):
            continue
        for year, year_data in years.items():
            pages = year_data.get("pages") if isinstance(year_data, dict) else None
            if not isinstance(pages, dict):
                continue
            for page, entry in pages.items():
                if not isinstance(entry, dict):
                    continue
                image_value = str(entry.get("image") or "")
                image_path = _resolve_dataset_image(workspace, image_value)
                if image_path is None:
                    continue
                if group == "sat":
                    page_key = str(entry.get("pageKey") or f"{year}_{page}")
                    page_id = f"sat_math_odd_page_{page_key}"
                else:
                    page_id = f"{year}_math_odd_page_{page}"
                out.append((group, page_id, image_path))
    return out


@lru_cache(maxsize=4)
def _known_pages(workspace_str: str) -> tuple[_KnownPage, ...]:
    workspace = Path(workspace_str)
    pages: list[_KnownPage] = []
    seen: set[tuple[str, str]] = set()
    for document_type, page_id, image_path in _iter_dataset_pages(workspace):
        key = (document_type, page_id)
        if key in seen:
            continue
        seen.add(key)
        try:
            pages.append(
                _KnownPage(
                    page_id=page_id,
                    document_type=document_type,
                    image_path=image_path,
                    file_size=image_path.stat().st_size,
                )
            )
        except OSError as exc:
            logger.warning("Skipping known page image %s: %s", image_path, exc)
    logger.info("Known page matcher indexed %d page image(s)", len(pages))
    return tuple(pages)


def match_known_page(workspace: Path, image_path: str | Path) -> KnownPageMatch | None:
    target = Path(image_path)
    try:
        target_size = target.stat().st_size
        target_sha = _sha256_file(target)
        target_width, target_height, target_thumb = _image_thumb_file(target)
    except (OSError, UnidentifiedImageError) as exc:
        logger.debug("Known page matcher could not read upload %s: %s", target, exc)
        return None

    pages = _known_pages(str(workspace))
    for page in pages:
        if page.file_size != target_size:
            continue
        try:
            page_sha = _sha256_file(page.image_path)
        except OSError:
            continue
        if page_sha == target_sha:
            return KnownPageMatch(
                page_id=page.page_id,
                document_type=page.document_type,
                image_path=str(page.image_path),
                score=0.0,
                exact=True,
            )

    target_aspect = target_width / max(target_height, 1)
    best: tuple[float, _KnownPage] | None = None
    for page in pages:
        try:
            page_width, page_height = _cached_image_size(str(page.image_path))
        except (OSError, UnidentifiedImageError):
            continue
        same_size = page_width == target_width and page_height == target_height
        aspect_delta = abs((page_width / max(page_height, 1)) - target_aspect)
        if not same_size and aspect_delta > _MAX_ASPECT_DELTA:
            continue
        try:
            _, _, page_thumb = _cached_image_thumb(str(page.image_path))
        except (OSError, UnidentifiedImageError):
            continue
        diff = _mean_abs_diff(target_thumb, page_thumb)
        limit = _MAX_VISUAL_MAD_SAME_SIZE if same_size else _MAX_VISUAL_MAD_RESIZED
        if diff > limit:
            continue
        if best is None or diff < best[0]:
            best = (diff, page)

    if best is None:
        return None

    score, page = best
    return KnownPageMatch(
        page_id=page.page_id,
        document_type=page.document_type,
        image_path=str(page.image_path),
        score=score,
        exact=False,
    )
