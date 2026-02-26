# -*- coding: utf-8 -*-
"""
Backend API for document classification (CSAT/SAT) and problem detection (crop + OCR).
Run from workspace root: python -m uvicorn backend.main:app --reload
"""
from __future__ import annotations

import base64
import csv
import json
import logging
import os
import sys
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Allow importing DLA modules (run from workspace root)
WORKSPACE = Path(__file__).resolve().parent.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))


def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        os.environ.setdefault(key, value)


_BACKEND_ENV_FILE = Path(
    os.environ.get("BACKEND_ENV_FILE", str(WORKSPACE / "backend" / ".env"))
)
_load_env_file(_BACKEND_ENV_FILE)

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from backend.solve_service import stream_solve_ndjson
from backend.demo_parsing_io import save_demo_parsing
from backend.flowmap_service import generate_flow_map_json
from backend.aggregation_service import generate_aggregation
from backend.parser_runtime import detect_problems as _detect_problems_runtime
from backend.ocr_text_converter import get_converted_prob_desc, get_raw_ocr_from_infer_log

logging.basicConfig(
    level=os.environ.get("BACKEND_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_cors_origins() -> list[str]:
    default_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:20225",
        "http://127.0.0.1:20225",
        "http://localhost:20221",
        "http://127.0.0.1:20221",
    ]
    custom = os.environ.get("BACKEND_CORS_ALLOW_ORIGINS", "").strip()
    if not custom:
        return default_origins
    return [origin.strip() for origin in custom.split(",") if origin.strip()]

# Lazy imports for heavy DLA modules (classifier is lighter, parser loads OCR model)
def _classify_image(image_path: str, method: str = "model_first", confidence_threshold: float = 0.8):
    from dla.Classifier.classifier import classify_with_confidence
    return classify_with_confidence(
        image_path,
        method="model_first",
        confidence_threshold=confidence_threshold,
    )

def _detect_problems(image_path: str, out_dir: str, sat_mode: bool):
    return _detect_problems_runtime(WORKSPACE, image_path, out_dir, sat_mode)


app = FastAPI(title="Document Classify & Detect API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(),
    allow_origin_regex=os.environ.get(
        "BACKEND_CORS_ALLOW_ORIGIN_REGEX",
        r"https?://[^/]+(?::(3000|5173|20221|20225))?$",
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------
class ClassifyResponse(BaseModel):
    label: str  # "csat" | "sat"
    confidence: float


class DetectResponse(BaseModel):
    documentType: str  # "csat" | "sat"
    detections: list[dict]  # Detection-like: id, x, y, w, h, label, text?, cropUrl?


class ClassifyJsonBody(BaseModel):
    image_base64: str  # data:image/...;base64,... or raw base64


class DetectJsonBody(BaseModel):
    image_base64: str
    document_type: str | None = None  # optional "csat" | "sat"
    save_to_demo_parsing: bool | None = None  # default True when missing
    page_id: str | None = None  # default "page_2"
    run_ocr_per_crop: bool | None = None  # default True


class SolveModelSpec(BaseModel):
    modelId: str
    displayName: str


class SolveStreamRequest(BaseModel):
    problemId: str
    problemLabel: str
    cropImageDataUrl: str | None = None  # data URL of the cropped problem image
    problemText: str | None = None
    modality: str | None = None  # image | text | image+text
    models: list[SolveModelSpec] | None = None


class FlowmapSolveStep(BaseModel):
    step_idx: int
    title: str
    content: str


class FlowmapSolution(BaseModel):
    model_name: str
    steps: list[FlowmapSolveStep]


class FlowmapGenerateRequest(BaseModel):
    problem_text: str = ""
    solutions: list[FlowmapSolution]


class AggregationGenerateRequest(BaseModel):
    problem_text: str = ""
    solutions: list[dict]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _decode_image_base64(data: str) -> bytes:
    if data.startswith("data:"):
        # data:image/png;base64,...
        idx = data.find(",")
        if idx == -1:
            raise ValueError("Invalid data URL")
        data = data[idx + 1 :]
    return base64.b64decode(data)


def _save_upload_to_temp(contents: bytes, suffix: str = ".png") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(fd, contents)
    finally:
        os.close(fd)
    return path


# OpenRouter 병렬 호출용 (문항별 변환 동시 실행)
_CONVERTER_EXECUTOR = ThreadPoolExecutor(
    max_workers=int(os.environ.get("OCR_CONVERTER_MAX_WORKERS", "8")),
    thread_name_prefix="ocr_conv",
)


# 2026 페이지당 문항 수(고정); TSV 행 순서 = 전역 문항 1, 2, ...
_2026_PROBLEMS_PER_PAGE = 4
# placeholder bbox (페이지당 4문항 그리드)
_2026_PLACEHOLDER_BBOXES = [[100, 400, 600, 900], [100, 1100, 600, 1600], [700, 400, 1200, 900], [700, 1100, 1200, 1600]]


def _ensure_2026_page_json(page_id: str) -> None:
    """2026 데모: page_id에 해당하는 JSON이 없으면 TSV에서 merged_text로 생성한다."""
    if not page_id or not page_id.startswith("2026_math_odd_page_"):
        return
    parts = page_id.split("_page_", 1)
    if len(parts) != 2:
        return
    page_num_str = parts[1].strip()
    if not page_num_str:
        return
    try:
        page_n = int(page_num_str)
    except ValueError:
        return
    folder = f"page_{page_num_str}" if not page_num_str.startswith("page_") else page_num_str
    json_path = WORKSPACE / "data" / "outputs_parsing" / "2026_math_odd" / folder / f"{folder}.json"
    if json_path.is_file():
        return
    tsv_path = WORKSPACE / "data" / "2026_math_odd.tsv"
    if not tsv_path.is_file():
        logger.warning("2026 TSV not found: %s", tsv_path)
        return
    try:
        rows: list[list[str]] = []
        with open(tsv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            for i, row in enumerate(reader):
                if i == 0:
                    header = row
                    try:
                        prob_desc_idx = header.index("prob_desc")
                    except ValueError:
                        logger.warning("2026 TSV has no prob_desc column")
                        return
                    continue
                rows.append(row)
        total = len(rows)
        start = (page_n - 1) * _2026_PROBLEMS_PER_PAGE
        end = min(page_n * _2026_PROBLEMS_PER_PAGE, total)
        if start >= total:
            return
        questions = []
        for local_i in range(end - start):
            row = rows[start + local_i]
            prob_desc = (row[prob_desc_idx] if len(row) > prob_desc_idx else "").strip()
            qid = str(local_i + 1)
            bbox = _2026_PLACEHOLDER_BBOXES[local_i % len(_2026_PLACEHOLDER_BBOXES)]
            questions.append({"qid": qid, "merged_text": prob_desc, "bbox": bbox})
        data = {
            "page_image": f"data/2026_math_odd/page_{page_num_str}.png",
            "width": 2480,
            "height": 3508,
            "questions": questions,
        }
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=None), encoding="utf-8")
        logger.info("Created 2026 page JSON: %s (%d questions)", json_path, len(questions))
    except Exception as e:
        logger.warning("Failed to create 2026 page JSON %s: %s", json_path, e, exc_info=True)


def _load_2026_page_texts(page_id: str) -> dict[str, str] | None:
    """2026 데모: outputs_parsing/2026_math_odd/page_XXX/page_XXX.json 에서 qid -> merged_text. 없으면 None.
    JSON이 없으면 _ensure_2026_page_json으로 TSV에서 생성한 뒤 로드한다."""
    if not page_id or not page_id.startswith("2026_math_odd_page_"):
        return None
    _ensure_2026_page_json(page_id)
    parts = page_id.split("_page_", 1)
    if len(parts) != 2:
        return None
    page_num = parts[1].strip()
    if not page_num:
        return None
    folder = f"page_{page_num}" if not page_num.startswith("page_") else page_num
    json_path = WORKSPACE / "data" / "outputs_parsing" / "2026_math_odd" / folder / f"{folder}.json"
    if not json_path.is_file():
        return None
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        questions = data.get("questions") or []
        return {str(q.get("qid", "")): (q.get("merged_text") or "") for q in questions if q.get("qid")}
    except Exception:
        return None


def _detections_from_stored_json(page_id: str) -> list[dict] | None:
    """page_id가 2026_math_odd_page_XXX 이고 해당 페이지 JSON이 있으면, parser 없이 JSON만으로
    detections 반환. 없으면 None (호출측에서 실제 parser 실행).
    """
    if not page_id or not page_id.startswith("2026_math_odd_page_"):
        return None
    _ensure_2026_page_json(page_id)
    parts = page_id.split("_page_", 1)
    if len(parts) != 2:
        return None
    page_num = parts[1].strip()
    if not page_num:
        return None
    folder = f"page_{page_num}" if not page_num.startswith("page_") else page_num
    json_path = WORKSPACE / "data" / "outputs_parsing" / "2026_math_odd" / folder / f"{folder}.json"
    if not json_path.is_file():
        logger.warning("Detect: 2026 stored JSON not found (using parser): path=%s", json_path)
        return None
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        questions = data.get("questions") or []
        W = max(1, data.get("width") or 2480)
        H = max(1, data.get("height") or 3508)
        out = []
        for q in questions:
            qid = str(q.get("qid", ""))
            bbox = q.get("bbox") or [0, 0, 0, 0]
            x1, y1, x2, y2 = bbox
            det_id = f"{page_id}_{qid}"
            out.append({
                "id": det_id,
                "x": x1 / W,
                "y": y1 / H,
                "w": (x2 - x1) / W,
                "h": (y2 - y1) / H,
                "label": det_id,
                "text": (q.get("merged_text") or "").strip(),
                "cropUrl": None,
            })
        return out
    except Exception:
        return None


def _questions_to_detections(
    payload: dict,
    image_width: int,
    image_height: int,
    *,
    page_id: str | None = None,
    use_demo_ids: bool = False,
    run_id: str | None = None,
) -> list[dict]:
    """Convert parser output to frontend Detection[] (normalized 0-1 bbox, cropUrl data URL).
    When use_demo_ids and page_id are set, id/label use page_2_1 format so Solve can load from demo_parsing.
    When page_id is None (e.g. upload), run_id makes conversion_id request-unique so converter cache is not shared across uploads.
    2026: page_id가 2026_math_odd_page_XXX 이면 해당 페이지 JSON의 merged_text를 그대로 사용(데모에서 수정한 텍스트 반영).
    """
    questions = payload.get("questions") or []
    W = max(1, image_width)
    H = max(1, image_height)
    if not questions:
        return []

    json_text_map = _load_2026_page_texts(page_id or "") if page_id else None

    def _convert_one(idx: int, q: dict) -> str:
        qid = str(q.get("qid", ""))
        # 2026: lookup by position so parser qid (e.g. 5,6,7,8 on page 2) doesn't matter
        if json_text_map is not None:
            key = str(idx + 1)
            if key in json_text_map:
                return (json_text_map.get(key) or "").strip()
        if page_id and use_demo_ids:
            conversion_id = f"{page_id}_{qid}"
        elif run_id:
            conversion_id = f"detect_{run_id}_{qid}"
        else:
            conversion_id = f"detect_{qid}"
        raw_text = (get_raw_ocr_from_infer_log(page_id, qid) if page_id else None) or (q.get("merged_text") or "")
        crop_b64 = q.get("crop_b64") or ""
        return get_converted_prob_desc(
            conversion_id,
            raw_text,
            crop_b64=crop_b64 or None,
            run_converter_if_missing=True,
        )

    futures = [_CONVERTER_EXECUTOR.submit(_convert_one, idx, q) for idx, q in enumerate(questions)]
    converted_texts = []
    for fut in futures:
        try:
            converted_texts.append(fut.result())
        except Exception as exc:
            logger.warning("OCR converter failed for one question: %s", exc)
            converted_texts.append("")

    out = []
    for q, converted_text in zip(questions, converted_texts):
        qid = str(q.get("qid", ""))
        if use_demo_ids and page_id:
            det_id = f"{page_id}_{qid}"
            label = det_id
        else:
            det_id = qid
            label = qid
        bbox = q.get("bbox") or [0, 0, 0, 0]
        x1, y1, x2, y2 = bbox
        x = x1 / W
        y = y1 / H
        w = (x2 - x1) / W
        h = (y2 - y1) / H
        crop_b64 = q.get("crop_b64") or ""
        crop_url = f"data:image/png;base64,{crop_b64}" if crop_b64 else None
        out.append({
            "id": det_id,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "label": label,
            "text": converted_text,
            "cropUrl": crop_url,
        })
    return out


# ---------------------------------------------------------------------------
# POST /api/document/classify
# ---------------------------------------------------------------------------
@app.post("/api/document/classify", response_model=ClassifyResponse)
async def document_classify(
    file: UploadFile | None = File(None),
    body: ClassifyJsonBody | None = None,
):
    if file:
        contents = await file.read()
    elif body:
        try:
            contents = _decode_image_base64(body.image_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image_base64: {e}")
    else:
        raise HTTPException(status_code=400, detail="Provide either file upload or JSON body with image_base64")

    path = _save_upload_to_temp(contents)
    try:
        label, confidence = _classify_image(path, confidence_threshold=0.8)
        return ClassifyResponse(label=label, confidence=confidence)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# POST /api/problems/detect
# ---------------------------------------------------------------------------
@app.post("/api/problems/detect", response_model=DetectResponse)
async def problems_detect(
    request: Request,
    file: UploadFile | None = File(None),
    document_type: str | None = Form(None),
    save_to_demo_parsing: bool = Form(True),
    page_id: str = Form("page_2"),
    run_ocr_per_crop: bool = Form(True),
    body: DetectJsonBody | None = None,
):
    # 프론트가 application/json 으로 보내면 Form 이 비어서 body 도 None 이 됨 → JSON 직접 파싱
    if body is None and request.headers.get("content-type", "").strip().lower().startswith("application/json"):
        try:
            raw = await request.json()
            body = DetectJsonBody(
                image_base64=raw.get("image_base64", ""),
                document_type=raw.get("document_type"),
                save_to_demo_parsing=raw.get("save_to_demo_parsing"),
                page_id=raw.get("page_id"),
                run_ocr_per_crop=raw.get("run_ocr_per_crop"),
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON body: {e}")

    doc_type_param = document_type
    if file:
        contents = await file.read()
    elif body:
        try:
            contents = _decode_image_base64(body.image_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image_base64: {e}")
        doc_type_param = doc_type_param or body.document_type
    else:
        raise HTTPException(status_code=400, detail="Provide either file upload or JSON body with image_base64")

    document_type = doc_type_param
    # JSON body 사용 시 Form 필드가 비어올 수 있음 → 옵션을 body 기준으로 명시 적용 (미지정이면 True)
    if body is not None:
        save_to_demo_parsing = body.save_to_demo_parsing if body.save_to_demo_parsing is not None else True
        page_id = body.page_id if body.page_id is not None else "page_2"
        run_ocr_per_crop = body.run_ocr_per_crop if body.run_ocr_per_crop is not None else True

    path = _save_upload_to_temp(contents)
    out_dir = tempfile.mkdtemp(prefix="detect_")
    try:
        # 0) 저장된 JSON이 있으면 parser 없이 JSON만 사용 (예: 2026 샘플)
        if page_id:
            from_json = _detections_from_stored_json(page_id)
            if from_json is not None:
                logger.info("Detect: using stored JSON for page_id=%s (%d detections)", page_id, len(from_json))
                return DetectResponse(documentType="csat", detections=from_json)

        # 1) Classify if document_type not provided
        if not document_type or document_type not in ("csat", "sat"):
            try:
                document_type, _ = _classify_image(path, confidence_threshold=0.8)
            except Exception:
                document_type = "csat"

        # 2) Run parser (dla/parser.py) single image; CSAT/SAT 구분 반영
        sat_mode = document_type == "sat"
        payload = _detect_problems(path, out_dir, sat_mode=sat_mode)

        W = payload.get("image_width") or 0
        H = payload.get("image_height") or 0

        # 3) /workspace/data/demo_parsing 에 페이지 이미지·crop·OCR 결과·LLM용 TSV 저장
        saved_demo = False
        if save_to_demo_parsing and payload.get("questions"):
            try:
                save_demo_parsing(
                    page_image_path=path,
                    payload=payload,
                    document_type=document_type,
                    page_id=page_id,
                    run_ocr_per_crop=run_ocr_per_crop,
                )
                saved_demo = True
            except Exception as e:
                logger.warning("demo_parsing save failed: %s", e, exc_info=True)

        # 4) detections: demo 저장했으면 id를 page_2_1 형식으로 반환 → Solve가 demo_parsing TSV 사용
        # 업로드 등 demo 미저장 시 run_id로 conversion 캐시 키를 요청별 고유하게 해서 이전 이미지 변환 결과가 노출되지 않도록 함
        run_id = None if saved_demo else uuid.uuid4().hex[:16]
        detections = _questions_to_detections(
            payload, W, H,
            page_id=page_id if saved_demo else (page_id or None),
            use_demo_ids=saved_demo,
            run_id=run_id,
        )
        return DetectResponse(documentType=document_type, detections=detections)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass
        try:
            import shutil
            shutil.rmtree(out_dir, ignore_errors=True)
        except Exception:
            pass


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# POST /api/flowmap/generate
# ---------------------------------------------------------------------------
@app.post("/api/flowmap/generate")
async def flowmap_generate(body: FlowmapGenerateRequest):
    import asyncio

    solutions = [
        {
            "model_name": sol.model_name,
            "steps": [
                {"step_idx": s.step_idx, "title": s.title, "content": s.content}
                for s in sol.steps
            ],
        }
        for sol in body.solutions
    ]

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None,
            generate_flow_map_json,
            body.problem_text,
            solutions,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result


# ---------------------------------------------------------------------------
# POST /api/aggregation/generate
# ---------------------------------------------------------------------------
@app.post("/api/aggregation/generate")
async def aggregation_generate(body: AggregationGenerateRequest):
    import asyncio

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None,
            generate_aggregation,
            body.problem_text,
            body.solutions,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result


# ---------------------------------------------------------------------------
# POST /api/solve/stream (NDJSON streaming)
# ---------------------------------------------------------------------------
@app.post("/api/solve/stream")
async def solve_stream(body: SolveStreamRequest):
    async def stream():
        models = [{"modelId": m.modelId, "displayName": m.displayName} for m in (body.models or [])]
        async for line in stream_solve_ndjson(
            problem_id=body.problemId,
            problem_label=body.problemLabel,
            crop_image_data_url=body.cropImageDataUrl,
            problem_text=body.problemText,
            modality=body.modality,
            models=models,
        ):
            yield line

    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache"},
    )
