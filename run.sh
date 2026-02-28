#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run.sh – start the whisper REST API using uv
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")"

HOST="${WHISPER_HOST:-0.0.0.0}"
PORT="${WHISPER_PORT:-8178}"

echo ">>> Starting whisper-rest on ${HOST}:${PORT} ..."
uv run uvicorn whisper_api:APP --host "$HOST" --port "$PORT"
