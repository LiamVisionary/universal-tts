#!/usr/bin/env bash
# Universal TTS — Fish Audio S2 Pro MLX sidecar launcher.
#
# Runs the MLX-Audio 8-bit conversion of Fish Audio S2 Pro in the dedicated
# .fish-s2-venv. Binds to 127.0.0.1:8784 by default.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV="${FISH_S2_VENV:-${REPO_ROOT}/.fish-s2-venv}"
MODEL="${FISH_S2_MODEL:-mlx-community/fish-audio-s2-pro-8bit}"
HOST="${FISH_S2_HOST:-127.0.0.1}"
PORT="${FISH_S2_PORT:-8784}"

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "Fish S2 venv not found at ${VENV}" >&2
  echo "Create it with:" >&2
  echo "  ${REPO_ROOT}/.venv/bin/python -m venv ${VENV}" >&2
  echo "  ${VENV}/bin/python -m pip install -U pip setuptools wheel" >&2
  echo "  ${VENV}/bin/python -m pip install mlx-audio soundfile fastapi pydantic huggingface_hub numpy uvicorn" >&2
  exit 1
fi

export FISH_S2_MODEL="${MODEL}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

exec "${VENV}/bin/python" "${SCRIPT_DIR}/fish_s2_server.py" \
  --host "${HOST}" --port "${PORT}" --model "${MODEL}"
