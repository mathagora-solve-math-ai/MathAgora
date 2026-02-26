#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Persistent parser worker.
Loads parser model once, then handles JSON-line requests from stdin.

Request line JSON:
  {"image_path": "...", "out_dir": "...", "sat_mode": false}

Response line JSON (prefixed):
  __RESULT__{"ok": true, "payload": {...}}
  __RESULT__{"ok": false, "error": "..."}
"""
from __future__ import annotations

import json
import os
import sys
import traceback
import warnings
from pathlib import Path


def _print_line(s: str) -> None:
    print(s, flush=True)


def main() -> int:
    warnings.filterwarnings("ignore")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")

    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    try:
        from parser import process_single_image
    except Exception as exc:
        _print_line(f"__FATAL__failed to import parser: {exc}")
        return 2

    _print_line("__READY__")

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            image_path = str(req.get("image_path", ""))
            out_dir = str(req.get("out_dir", ""))
            sat_mode = bool(req.get("sat_mode", False))
            payload = process_single_image(image_path, out_dir, sat_mode=sat_mode)
            _print_line("__RESULT__" + json.dumps({"ok": True, "payload": payload}, ensure_ascii=False))
        except Exception as exc:
            _print_line(
                "__RESULT__"
                + json.dumps(
                    {
                        "ok": False,
                        "error": str(exc),
                        "traceback": traceback.format_exc(limit=3),
                    },
                    ensure_ascii=False,
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
