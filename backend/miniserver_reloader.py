#!/usr/bin/env python3
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict

WORKSPACE = Path(__file__).resolve().parent.parent
BACKEND_DIR = WORKSPACE / "backend"
POLL_INTERVAL_S = float(os.environ.get("BACKEND_RELOAD_POLL_SEC", "1.0"))


def _snapshot_py_mtimes() -> Dict[str, int]:
    mtimes: Dict[str, int] = {}
    for path in BACKEND_DIR.rglob("*.py"):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        mtimes[str(path)] = stat.st_mtime_ns
    return mtimes


def _start_server() -> subprocess.Popen:
    cmd = [sys.executable, str(BACKEND_DIR / "miniserver.py")]
    print(f"[backend-reloader] start: {' '.join(cmd)}", flush=True)
    return subprocess.Popen(cmd, cwd=str(WORKSPACE), env=os.environ.copy())


def _stop_server(proc: subprocess.Popen, timeout: float = 5.0) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def main() -> None:
    running = True

    def _handle_signal(signum: int, _frame) -> None:
        nonlocal running
        print(f"[backend-reloader] signal={signum}, shutting down", flush=True)
        running = False

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    baseline = _snapshot_py_mtimes()
    proc = _start_server()

    try:
        while running:
            time.sleep(POLL_INTERVAL_S)

            if proc.poll() is not None:
                print("[backend-reloader] server exited; restarting", flush=True)
                proc = _start_server()
                baseline = _snapshot_py_mtimes()
                continue

            current = _snapshot_py_mtimes()
            if current != baseline:
                print("[backend-reloader] code change detected; reloading server", flush=True)
                _stop_server(proc)
                proc = _start_server()
                baseline = current
    finally:
        _stop_server(proc)


if __name__ == "__main__":
    main()
