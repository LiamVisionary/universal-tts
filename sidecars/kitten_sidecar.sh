#!/usr/bin/env bash
# Universal TTS — KittenTTS sidecar launcher.
#
# Runs the KittenTTS sidecar in the dedicated venv where torch/onnxruntime
# /misaki live, so the universal-tts venv stays lean. The sidecar binds to
# 127.0.0.1:8782 by default; Universal's kitten provider is configured to
# talk to that URL.

set -euo pipefail

# Resolve repo root (this file lives in <repo>/sidecars/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV="${KITTEN_VENV:-${REPO_ROOT}/.kitten-venv}"
MODEL="${KITTEN_MODEL:-KittenML/kitten-tts-mini-0.8}"
HOST="${KITTEN_HOST:-127.0.0.1}"
PORT="${KITTEN_PORT:-8782}"

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "Kitten venv not found at ${VENV}" >&2
  echo "Create it with:" >&2
  echo "  ${REPO_ROOT}/.venv/bin/python -m venv ${VENV}" >&2
  echo "  ${VENV}/bin/pip install -U pip" >&2
  echo "  ${VENV}/bin/pip install -e ${REPO_ROOT}" >&2
  echo "or install the released wheel:" >&2
  echo "  ${VENV}/bin/pip install 'https://github.com/KittenML/KittenTTS/releases/download/0.8.1/kittentts-0.8.1-py3-none-any.whl'" >&2
  exit 1
fi

exec "${VENV}/bin/python" "${SCRIPT_DIR}/kitten_server.py" \
  --host "${HOST}" --port "${PORT}" --model "${MODEL}"
