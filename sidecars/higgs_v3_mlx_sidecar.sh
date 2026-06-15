#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${UNIVERSAL_TTS_REPO:-/Users/liam/voice-lab/universal-tts}"
PYTHON="${HIGGS_MLX_PYTHON:-$REPO_DIR/.higgs-mlx-venv/bin/python}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8806}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Higgs MLX venv not found at $PYTHON" >&2
  echo "Run: cd $REPO_DIR && python3 -m venv .higgs-mlx-venv && .higgs-mlx-venv/bin/pip install mlx-audio torch sympy fastapi uvicorn" >&2
  exit 1
fi

cd "$REPO_DIR"
exec "$PYTHON" -m uvicorn sidecars.higgs_v3_mlx_server:app --host "$HOST" --port "$PORT"
