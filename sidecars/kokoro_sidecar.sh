#!/usr/bin/env bash
# Universal TTS — Kokoro sidecar launcher.
#
# Runs Kokoro-82M in the dedicated .kokoro-venv where torch/kokoro/misaki
# dependencies live. The sidecar binds to 127.0.0.1:8783 by default.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV="${KOKORO_VENV:-${REPO_ROOT}/.kokoro-venv}"
MODEL="${KOKORO_MODEL:-hexgrad/Kokoro-82M}"
HOST="${KOKORO_HOST:-127.0.0.1}"
PORT="${KOKORO_PORT:-8783}"
VOICE="${KOKORO_VOICE:-af_heart}"
DEVICE="${KOKORO_DEVICE:-auto}"
MAX_SEGMENT_CHARS="${KOKORO_MAX_SEGMENT_CHARS:-220}"

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "Kokoro venv not found at ${VENV}" >&2
  echo "Create it with:" >&2
  echo "  ${REPO_ROOT}/.venv/bin/python -m venv ${VENV}" >&2
  echo "  ${VENV}/bin/python -m pip install -U pip setuptools wheel" >&2
  echo "  ${VENV}/bin/python -m pip install 'kokoro>=0.9.4' 'misaki[ja,zh]>=0.9.4' soundfile fastapi pydantic huggingface_hub numpy uvicorn" >&2
  exit 1
fi

export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export KOKORO_MODEL="${MODEL}"
export KOKORO_VOICE="${VOICE}"
export KOKORO_DEVICE="${DEVICE}"
export KOKORO_MAX_SEGMENT_CHARS="${MAX_SEGMENT_CHARS}"

exec "${VENV}/bin/python" "${SCRIPT_DIR}/kokoro_server.py" \
  --host "${HOST}" --port "${PORT}" --model "${MODEL}"
