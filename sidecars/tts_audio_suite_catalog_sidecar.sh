#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${UNIVERSAL_TTS_REPO:-/Users/liam/voice-lab/universal-tts}"
PYTHON="${UNIVERSAL_TTS_PYTHON:-$REPO_DIR/.venv/bin/python}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8801}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Universal TTS venv not found at $PYTHON" >&2
  echo "Run: cd $REPO_DIR && python3 -m venv .venv && .venv/bin/pip install -e ." >&2
  exit 1
fi

cd "$REPO_DIR"
exec "$PYTHON" -m uvicorn sidecars.tts_audio_suite_catalog_server:app --host "$HOST" --port "$PORT"
