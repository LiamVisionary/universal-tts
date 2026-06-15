#!/usr/bin/env bash
set -euo pipefail
REPO="${UNIVERSAL_TTS_REPO:-/Users/liam/voice-lab/universal-tts}"
VENV="${COSYVOICE_VENV:-$REPO/.cosyvoice-venv}"
SIDE="$REPO/sidecars/cosyvoice3_server.py"
PORT="${COSYVOICE_PORT:-8785}"
export TTS_AUDIO_SUITE_REPO="${TTS_AUDIO_SUITE_REPO:-/Users/liam/voice-lab/TTS-Audio-Suite}"
export COSYVOICE_MODEL_DIR="${COSYVOICE_MODEL_DIR:-/Users/liam/voice-lab/models/TTS/CosyVoice/Fun-CosyVoice3-0.5B}"
export PYTHONPATH="$TTS_AUDIO_SUITE_REPO/engines/cosyvoice/impl:$TTS_AUDIO_SUITE_REPO/engines/cosyvoice/impl/third_party/Matcha-TTS:${PYTHONPATH:-}"
if [[ ! -x "$VENV/bin/python" ]]; then
  cat >&2 <<EOF
CosyVoice3 sidecar venv is missing: $VENV
Install with:
  "$REPO/.venv/bin/python" -m venv "$VENV"
  "$VENV/bin/pip" install -U pip
  "$VENV/bin/pip" install torch torchaudio torchcodec numpy soundfile fastapi uvicorn pydantic huggingface_hub hyperpyyaml modelscope inflect onnxruntime tqdm transformers omegaconf conformer diffusers hydra-core rich wetext tiktoken openai-whisper librosa numba audioread decorator joblib lazy-loader msgpack pooch scikit-learn scipy soxr x-transformers matplotlib pyarrow pyworld
EOF
  exit 127
fi
exec "$VENV/bin/python" -m uvicorn sidecars.cosyvoice3_server:app --host 127.0.0.1 --port "$PORT"
