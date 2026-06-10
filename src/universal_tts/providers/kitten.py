"""KittenTTS provider adapter.

The actual KittenTTS runtime (torch + ONNX + misaki) lives in a dedicated
venv and is started by the sidecar launcher at ``sidecars/kitten_sidecar.sh``.
This adapter is a thin HTTP-backed client over :class:`HttpBackedProvider`.

True realtime streaming is provided by the sidecar's
``/v1/audio/speech-stream`` endpoint, which forwards the ONNX model's
``generate_stream`` generator chunk-by-chunk as raw 24 kHz mono PCM frames.
"""

from __future__ import annotations

import json
import os
from typing import Any

from universal_tts.config import ProviderConfig
from universal_tts.providers.base import TTSRequest
from universal_tts.providers.http_backed import HttpBackedProvider

DEFAULT_VOICE = "Bella"


class KittenTTSProvider(HttpBackedProvider):
    """HTTP-backed KittenTTS adapter (sidecar in ``.kitten-venv``)."""

    def _voice_aliases(self) -> dict[str, str]:
        env_value = os.environ.get("KITTEN_VOICE_ALIASES")
        if env_value:
            try:
                return json.loads(env_value)
            except Exception:  # noqa: BLE001
                pass
        return dict(self.cfg.options.get("voice_aliases", {}))

    def _resolve_voice(self, voice: str | None) -> str:
        aliases = self._voice_aliases()
        if voice and voice in aliases:
            return aliases[voice]
        return voice or self.cfg.options.get("default_voice", DEFAULT_VOICE)

    def speech_payload(self, request: TTSRequest) -> dict[str, Any]:
        payload = super().speech_payload(request)
        payload["voice"] = self._resolve_voice(request.voice)
        clean_text = request.options.get(
            "clean_text", self.cfg.options.get("default_clean_text", True)
        )
        payload["clean_text"] = bool(clean_text)
        return payload

    def stream_payload(self, request: TTSRequest) -> dict[str, Any]:
        return self.speech_payload(request)

    def stream_path(self) -> str | None:
        return self.cfg.options.get("stream_path", "/v1/audio/speech-stream")
