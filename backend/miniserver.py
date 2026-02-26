#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import sys
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

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

from backend.demo_parsing_io import save_demo_parsing
from backend.solve_service import stream_solve_ndjson
from backend.flowmap_service import generate_flow_map_json
from backend.aggregation_service import generate_aggregation
from backend.parser_runtime import detect_problems as _detect_problems_runtime
from backend.ocr_text_converter import (
    clear_all_converter_caches,
    get_converted_prob_desc,
    get_raw_ocr_from_infer_log,
)

# 서버 기동 시 converter 캐시 전부 제거
clear_all_converter_caches(delete_disk_files=True)

LOGGER = logging.getLogger("backend.miniserver")


def _decode_image_base64(data: str) -> bytes:
    if data.startswith("data:"):
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


def _classify_image(image_path: str, confidence_threshold: float = 0.8) -> tuple[str, float]:
    from dla.Classifier.classifier import classify_with_confidence

    return classify_with_confidence(
        image_path,
        method="model_first",
        confidence_threshold=confidence_threshold,
    )


def _detect_problems(image_path: str, out_dir: str, sat_mode: bool) -> dict:
    return _detect_problems_runtime(WORKSPACE, image_path, out_dir, sat_mode)


# OpenRouter 병렬 호출용 (문항별 변환 동시 실행)
_CONVERTER_EXECUTOR = ThreadPoolExecutor(
    max_workers=int(os.environ.get("OCR_CONVERTER_MAX_WORKERS", "8")),
    thread_name_prefix="ocr_conv",
)


def _questions_to_detections(
    payload: dict,
    image_width: int,
    image_height: int,
    *,
    page_id: str | None = None,
    use_demo_ids: bool = False,
    run_id: str | None = None,
) -> list[dict]:
    questions = payload.get("questions") or []
    w = max(1, image_width)
    h = max(1, image_height)
    if not questions:
        return []

    # 문항별 변환: infer 로그 raw OCR 있으면 사용(output 파일과 동일), 없으면 merged_text
    # 업로드 등 demo 미저장 시 run_id로 conversion_id를 요청별 고유하게 함
    def _convert_one(q: dict) -> str:
        qid = str(q.get("qid", ""))
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

    futures = [_CONVERTER_EXECUTOR.submit(_convert_one, q) for q in questions]
    converted_texts: list[str] = []
    for fut in futures:
        try:
            converted_texts.append(fut.result())
        except Exception as exc:
            LOGGER.warning("OCR converter failed for one question: %s", exc)
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
        crop_b64 = q.get("crop_b64") or ""
        out.append(
            {
                "id": det_id,
                "x": x1 / w,
                "y": y1 / h,
                "w": (x2 - x1) / w,
                "h": (y2 - y1) / h,
                "label": label,
                "text": converted_text,
                "cropUrl": f"data:image/png;base64,{crop_b64}" if crop_b64 else None,
            }
        )
    return out


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _set_cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._set_cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._set_cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/health":
            self._send_json(200, {"status": "ok"})
            return
        self._send_json(404, {"detail": "Not Found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            if self.path == "/api/document/classify":
                self._handle_classify()
                return
            if self.path == "/api/problems/detect":
                self._handle_detect()
                return
            if self.path == "/api/solve/stream":
                self._handle_solve_stream()
                return
            if self.path == "/api/flowmap/generate":
                self._handle_flowmap_generate()
                return
            if self.path == "/api/aggregation/generate":
                self._handle_aggregation_generate()
                return
            if self.path == "/api/cache/clear":
                self._handle_cache_clear()
                return
            self._send_json(404, {"detail": "Not Found"})
        except Exception as exc:
            LOGGER.exception("Request failed: %s", exc)
            self._send_json(500, {"detail": str(exc)})

    def _handle_classify(self) -> None:
        body = self._read_json()
        image_base64 = body.get("image_base64")
        if not image_base64:
            self._send_json(400, {"detail": "image_base64 is required"})
            return
        path = _save_upload_to_temp(_decode_image_base64(image_base64))
        try:
            label, confidence = _classify_image(path, confidence_threshold=0.8)
            self._send_json(200, {"label": label, "confidence": confidence})
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass

    def _handle_detect(self) -> None:
        """Run parser (DLA) on the page image: detect problem regions, crop per question (bbox + crop_b64),
        OCR text per crop, convert to frontend detections (normalized bbox, cropUrl, text)."""
        body = self._read_json()
        image_base64 = body.get("image_base64")
        if not image_base64:
            self._send_json(400, {"detail": "image_base64 is required"})
            return

        document_type = body.get("document_type")
        save_to_demo_parsing = (
            True if body.get("save_to_demo_parsing") is None else bool(body.get("save_to_demo_parsing"))
        )
        page_id = body.get("page_id") or "page_2"
        run_ocr_per_crop = True if body.get("run_ocr_per_crop") is None else bool(body.get("run_ocr_per_crop"))

        path = _save_upload_to_temp(_decode_image_base64(image_base64))
        out_dir = tempfile.mkdtemp(prefix="detect_")
        try:
            if document_type not in ("csat", "sat"):
                try:
                    document_type, _ = _classify_image(path, confidence_threshold=0.8)
                except Exception:
                    document_type = "csat"
            sat_mode = document_type == "sat"
            payload = _detect_problems(path, out_dir, sat_mode=sat_mode)
            w = payload.get("image_width") or 0
            h = payload.get("image_height") or 0

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
                except Exception:
                    LOGGER.exception("demo_parsing save failed")

            run_id = None if saved_demo else uuid.uuid4().hex[:16]
            detections = _questions_to_detections(
                payload,
                w,
                h,
                page_id=page_id if saved_demo else None,
                use_demo_ids=saved_demo,
                run_id=run_id,
            )
            self._send_json(
                200,
                {
                    "documentType": document_type,
                    "detections": detections,
                    "imageWidth": w or None,
                    "imageHeight": h or None,
                },
            )
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

    def _handle_solve_stream(self) -> None:
        body = self._read_json()
        problem_id = body.get("problemId")
        problem_label = body.get("problemLabel")
        if not problem_id or not problem_label:
            self._send_json(400, {"detail": "problemId and problemLabel are required"})
            return
        crop_image_data_url = body.get("cropImageDataUrl") or None
        problem_text = body.get("problemText") or None
        modality = body.get("modality") or None

        self.send_response(200)
        self._set_cors()
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        models = body.get("models") or []
        async def _stream() -> None:
            async for line in stream_solve_ndjson(
                problem_id=problem_id,
                problem_label=problem_label,
                crop_image_data_url=crop_image_data_url,
                problem_text=problem_text,
                modality=modality,
                models=models,
            ):
                self.wfile.write(line.encode("utf-8"))
                self.wfile.flush()

        try:
            asyncio.run(_stream())
        except (BrokenPipeError, ConnectionResetError):
            LOGGER.info("Client disconnected from solve stream")

    def _handle_cache_clear(self) -> None:
        """POST /api/cache/clear: converter 캐시(메모리·디스크) 전부 제거."""
        if self.command not in ("POST", "GET"):
            self._send_json(405, {"detail": "Method Not Allowed"})
            return
        try:
            clear_all_converter_caches(delete_disk_files=True)
            self._send_json(200, {"ok": True, "message": "Converter caches cleared"})
        except Exception as exc:
            LOGGER.exception("Cache clear failed: %s", exc)
            self._send_json(500, {"detail": str(exc)})

    def _handle_aggregation_generate(self) -> None:
        body = self._read_json()
        problem_text = body.get("problem_text", "")
        solutions = body.get("solutions", [])
        if not solutions:
            self._send_json(400, {"detail": "solutions is required"})
            return
        try:
            result = generate_aggregation(problem_text, solutions)
            self._send_json(200, result)
        except Exception as exc:
            LOGGER.exception("Aggregation failed: %s", exc)
            self._send_json(500, {"detail": str(exc)})

    def _handle_flowmap_generate(self) -> None:
        body = self._read_json()
        problem_text = body.get("problem_text", "")
        solutions = body.get("solutions", [])
        if not solutions:
            self._send_json(400, {"detail": "solutions is required"})
            return
        try:
            result = generate_flow_map_json(problem_text, solutions)
            self._send_json(200, result)
        except Exception as exc:
            self._send_json(500, {"detail": str(exc)})

    def log_message(self, fmt: str, *args) -> None:
        LOGGER.info("%s - %s", self.address_string(), fmt % args)


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("BACKEND_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    host = os.environ.get("BACKEND_HOST", "0.0.0.0")
    port = int(os.environ.get("BACKEND_PORT", "8000"))
    server = ThreadingHTTPServer((host, port), Handler)
    LOGGER.info("Fallback backend server listening on %s:%s", host, port)
    server.serve_forever()


if __name__ == "__main__":
    main()
