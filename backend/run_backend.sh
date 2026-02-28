#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${BACKEND_ENV_FILE:-${SCRIPT_DIR}/.env}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

HOST="${BACKEND_HOST:-0.0.0.0}"
PORT="${BACKEND_PORT:-8000}"
AUTO_RELOAD="${BACKEND_AUTO_RELOAD:-1}"

echo "[backend] starting uvicorn on ${HOST}:${PORT}"
if [[ -z "${OPENAI_API_KEY:-}" || -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "[backend] warning: OPENAI_API_KEY or OPENROUTER_API_KEY is missing; solve stream will return per-model error events."
fi

cd "${WORKSPACE_DIR}"
if python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("fastapi") else 1)
PY
then
  if [[ "${AUTO_RELOAD}" == "1" ]]; then
    echo "[backend] fastapi detected; running uvicorn with --reload"
    exec python -m uvicorn backend.main:app --host "${HOST}" --port "${PORT}" --reload --reload-dir /workspace/backend
  fi
  exec python -m uvicorn backend.main:app --host "${HOST}" --port "${PORT}"
else
  if [[ "${AUTO_RELOAD}" == "1" ]]; then
    echo "[backend] fastapi not installed; running fallback auto-reloader backend/miniserver_reloader.py"
    exec python /workspace/backend/miniserver_reloader.py
  fi
  echo "[backend] fastapi not installed; running fallback server backend/miniserver.py"
  exec python /workspace/backend/miniserver.py
fi
