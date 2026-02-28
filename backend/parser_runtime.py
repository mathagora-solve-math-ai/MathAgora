from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path


_WORKER: "ParserWorker | None" = None
_WORKER_GUARD = threading.Lock()


def _abs_no_resolve(p: Path) -> Path:
    # Keep venv symlink target untouched (do not collapse to system python).
    return Path(os.path.abspath(str(p)))


def _compute_parser_python(workspace: Path) -> Path:
    dla_dir = workspace / "dla"
    parser_python_env = os.environ.get("DLA_PARSER_PYTHON", "").strip()
    parser_venv_env = os.environ.get("DLA_PARSER_VENV", "").strip()
    if parser_python_env:
        parser_python = Path(parser_python_env).expanduser()
        if not parser_python.is_absolute():
            parser_python = _abs_no_resolve(workspace / parser_python)
        return parser_python
    if parser_venv_env:
        return _abs_no_resolve(dla_dir / parser_venv_env / "bin" / "python")
    deepseek_python = _abs_no_resolve(dla_dir / "venv_deepseek" / "bin" / "python")
    if deepseek_python.exists():
        return deepseek_python
    return _abs_no_resolve(dla_dir / "venv_ocr" / "bin" / "python")


def _build_parser_env(parser_python: Path, sat_mode: bool = False) -> dict[str, str]:
    env = {
        "PATH": str(parser_python.parent) + os.pathsep + os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", "/tmp"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "PYTHONNOUSERSITE": "1",
        "PYTHONWARNINGS": "ignore",
    }
    for key in (
        "CUDA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
        "HF_HOME",
        "TRANSFORMERS_CACHE",
        "HF_TOKEN",
        "MODEL_NAME",
        "IMAGE_SIZE",
        "BASE_SIZE",
    ):
        if key in os.environ:
            env[key] = os.environ[key]
    # SAT: use DeepSeek-OCR-2 + 768 and contour-based question number detection
    if sat_mode:
        env["SAT_MODE"] = "1"
        env["MODEL_NAME"] = "deepseek-ai/DeepSeek-OCR-2"
        env["IMAGE_SIZE"] = "768"
    return env


def _single_run(workspace: Path, image_path: str, out_dir: str, sat_mode: bool) -> dict:
    dla_dir = workspace / "dla"
    parser_script = dla_dir / "run_parser_single.py"
    parser_python = _compute_parser_python(workspace)
    if not parser_script.exists() or not parser_python.exists():
        raise RuntimeError(f"parser runtime not found: python={parser_python} script={parser_script}")

    cmd = [str(parser_python), "-I", str(parser_script.resolve()), str(image_path), str(out_dir)]
    if sat_mode:
        cmd.append("--sat-mode")

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=int(os.environ.get("DLA_PARSER_TIMEOUT_SEC", "420")),
        cwd=str(dla_dir),
        env=_build_parser_env(parser_python, sat_mode=sat_mode),
    )
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if proc.returncode != 0:
        tail = "\n".join((stderr + "\n" + stdout).splitlines()[-40:])
        raise RuntimeError(f"parser subprocess failed (code={proc.returncode}):\n{tail}")

    start = stdout.rfind("___DLA_JSON_START___")
    end = stdout.rfind("___DLA_JSON_END___")
    if start != -1 and end != -1 and end > start:
        json_part = stdout[start + len("___DLA_JSON_START___") : end].strip()
        return json.loads(json_part)
    raise RuntimeError("parser subprocess output parse failed (missing JSON marker)")


class ParserWorker:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.dla_dir = workspace / "dla"
        self.parser_python = _compute_parser_python(workspace)
        self.worker_script = self.dla_dir / "run_parser_worker.py"
        self.proc: subprocess.Popen | None = None
        self._line_queue: "queue.Queue[str | None]" = queue.Queue()
        self._io_lock = threading.Lock()
        self._reader_thread: threading.Thread | None = None
        self._recent_logs: list[str] = []

    def _append_log(self, line: str) -> None:
        s = line.rstrip("\n")
        if not s:
            return
        self._recent_logs.append(s)
        if len(self._recent_logs) > 200:
            self._recent_logs = self._recent_logs[-200:]

    def _reader_loop(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        try:
            for line in self.proc.stdout:
                self._line_queue.put(line)
        finally:
            self._line_queue.put(None)

    def _start(self) -> None:
        if not self.worker_script.exists() or not self.parser_python.exists():
            raise RuntimeError(
                f"parser worker runtime not found: python={self.parser_python} script={self.worker_script}"
            )
        self.proc = subprocess.Popen(
            [str(self.parser_python), "-I", str(self.worker_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(self.dla_dir),
            env=_build_parser_env(self.parser_python),
            bufsize=1,
        )
        self._line_queue = queue.Queue()
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

        deadline = time.time() + int(os.environ.get("DLA_PARSER_INIT_TIMEOUT_SEC", "180"))
        while time.time() < deadline:
            remaining = max(0.1, deadline - time.time())
            try:
                line = self._line_queue.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                break
            if line.startswith("__READY__"):
                return
            self._append_log(line)
        logs = "\n".join(self._recent_logs[-40:])
        raise RuntimeError(f"parser worker failed to start\n{logs}")

    def _ensure_started(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            return
        self._start()

    def request(self, image_path: str, out_dir: str, sat_mode: bool) -> dict:
        timeout = int(os.environ.get("DLA_PARSER_TIMEOUT_SEC", "420"))
        with self._io_lock:
            self._ensure_started()
            assert self.proc is not None and self.proc.stdin is not None
            req = {"image_path": image_path, "out_dir": out_dir, "sat_mode": bool(sat_mode)}
            self.proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()

            deadline = time.time() + timeout
            while time.time() < deadline:
                remaining = max(0.1, deadline - time.time())
                try:
                    line = self._line_queue.get(timeout=remaining)
                except queue.Empty:
                    continue
                if line is None:
                    break
                if line.startswith("__RESULT__"):
                    body = line[len("__RESULT__") :].strip()
                    try:
                        obj = json.loads(body)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(f"invalid parser worker JSON: {exc}")
                    if obj.get("ok") is True:
                        return obj.get("payload") or {}
                    err = str(obj.get("error") or "unknown parser worker error")
                    tb = str(obj.get("traceback") or "")
                    raise RuntimeError(f"parser worker error: {err}\n{tb}".strip())
                self._append_log(line)
            if self.proc and self.proc.poll() is None:
                self.proc.kill()
            logs = "\n".join(self._recent_logs[-40:])
            raise RuntimeError(f"parser worker timeout/exit after {timeout}s\n{logs}")


def detect_problems(workspace: Path, image_path: str, out_dir: str, sat_mode: bool) -> dict:
    # SAT requires subprocess env (SAT_MODE, MODEL_NAME, IMAGE_SIZE); worker has fixed env at startup
    mode = os.environ.get("DLA_PARSER_MODE", "worker").strip().lower()
    if mode == "single" or sat_mode:
        return _single_run(workspace, image_path, out_dir, sat_mode)

    global _WORKER
    with _WORKER_GUARD:
        if _WORKER is None:
            _WORKER = ParserWorker(workspace)
    try:
        return _WORKER.request(image_path, out_dir, sat_mode)
    except Exception:
        # fallback path for stability/debugging
        return _single_run(workspace, image_path, out_dir, sat_mode)
