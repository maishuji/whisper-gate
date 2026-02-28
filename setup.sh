#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup.sh – bootstrap the whisper-rest environment with uv
# ---------------------------------------------------------------------------
set -euo pipefail

# Install uv if not already present
if ! command -v uv &>/dev/null; then
    echo ">>> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Reload PATH so the current shell can find uv
    export PATH="$HOME/.local/bin:$PATH"
fi

cd "$(dirname "$0")"

echo ">>> Creating venv and installing dependencies with uv..."
uv sync

echo ""
echo "Done! Start the API with:"
echo "  uv run uvicorn whisper_api:APP --host 0.0.0.0 --port 8178"
