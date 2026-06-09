#!/usr/bin/env bash
set -euo pipefail
cd /Users/liam/voice-lab/universal-tts
export PYTHONPATH=/Users/liam/voice-lab/universal-tts/src
export UNIVERSAL_TTS_CONFIG=/Users/liam/voice-lab/universal-tts/config.yaml
exec .venv/bin/python -m uvicorn universal_tts.app:app --host 127.0.0.1 --port 8799
