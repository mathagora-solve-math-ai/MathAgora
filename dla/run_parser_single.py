#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run parser.process_single_image on one image and emit JSON payload.
This script is intended to be launched from a dedicated OCR virtualenv.

Usage:
  python run_parser_single.py <image_path> <out_dir> [--sat-mode]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _usage() -> int:
    print(
        "Usage: run_parser_single.py <image_path> <out_dir> [--sat-mode]",
        file=sys.stderr,
    )
    return 1


def main() -> int:
    if len(sys.argv) < 3:
        return _usage()

    image_path = sys.argv[1]
    out_dir = sys.argv[2]
    sat_mode = "--sat-mode" in sys.argv[3:]

    # Keep imports local so argument validation errors are fast.
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    try:
        from parser import process_single_image
        payload = process_single_image(image_path, out_dir, sat_mode=sat_mode)
    except Exception as exc:
        print(f"[run_parser_single error] {exc}", file=sys.stderr)
        return 2

    # Marker-wrapped JSON allows callers to reliably extract payload
    # even when model libraries emit extra logs to stdout.
    print("___DLA_JSON_START___")
    print(json.dumps(payload, ensure_ascii=False))
    print("___DLA_JSON_END___")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    raise SystemExit(main())
