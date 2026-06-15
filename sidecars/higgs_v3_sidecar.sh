#!/usr/bin/env bash
set -euo pipefail
REPO="${UNIVERSAL_TTS_REPO:-/Users/liam/voice-lab/universal-tts}"
VENV="${HIGGS_V3_VENV:-$REPO/.higgs-v3-venv}"
PORT="${HIGGS_V3_PORT:-8786}"
export TTS_AUDIO_SUITE_REPO="${TTS_AUDIO_SUITE_REPO:-/Users/liam/voice-lab/TTS-Audio-Suite}"
export HIGGS_V3_MODEL_DIR="${HIGGS_V3_MODEL_DIR:-/Users/liam/voice-lab/models/TTS/higgs_audio_v3/higgs-audio-v3-tts-4b}"
if [[ ! -x "$VENV/bin/python" ]]; then
  cat >&2 <<EOF
Higgs Audio v3 sidecar venv is missing: $VENV
Install with:
  "$REPO/.venv/bin/python" -m venv "$VENV"
  "$VENV/bin/pip" install -U pip
  "$VENV/bin/pip" install torch torchaudio torchcodec numpy soundfile fastapi uvicorn pydantic huggingface_hub transformers accelerate safetensors tokenizers sentencepiece protobuf
EOF
  exit 127
fi
exec hive-env-run -- "$VENV/bin/python" -m uvicorn sidecars.higgs_v3_server:app --host 127.0.0.1 --port "$PORT"
