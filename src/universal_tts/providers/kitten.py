"""KittenTTS provider adapter.

The actual KittenTTS runtime (torch + ONNX + misaki) lives in a dedicated
venv and is started by the sidecar launcher at ``sidecars/kitten_sidecar.sh``.
This adapter is a thin HTTP-backed client over :class:`HttpBackedProvider`.

True realtime streaming is provided by the sidecar's
``/v1/audio/speech-stream`` endpoint, which forwards the ONNX model's
``generate_stream`` generator chunk-by-chunk as raw 24 kHz mono PCM frames.

Audio conversion
----------------

KittenTTS returns ``float32`` numpy arrays with values roughly in ``[-1, 1]``.
The sidecar converts these to PCM int16 by clipping to ``[-1, 1]`` and
multiplying by ``32767``. Naive ``np.astype(np.int16)`` truncates the entire
signal to silence — that bug is captured by ``test_kitten_int16_scaling.py``.
The two helpers below mirror the sidecar's conversion so the conversion is
testable in isolation without spinning up the sidecar.
"""

from __future__ import annotations

import io
import json
import os
import wave
from typing import Any

import numpy as np

from universal_tts.config import ProviderConfig
from universal_tts.providers.base import TTSRequest
from universal_tts.providers.http_backed import HttpBackedProvider

DEFAULT_VOICE = "Bella"
SAMPLE_RATE = 24000
CHANNELS = 1
SAMPLE_WIDTH = 2


def _samples_to_pcm16(samples: np.ndarray) -> np.ndarray:
    """Convert KittenTTS float32 audio in ``[-1, 1]`` to PCM int16.

    Casts ``float32`` -> ``int16`` with proper scaling: clip to ``[-1, 1]``,
    multiply by ``32767``, round, cast. Naive ``astype(np.int16)`` truncates
    every value to 0 (silence); do not regress to that.
    """
    arr = np.asarray(samples, dtype=np.float32).squeeze()
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    clipped = np.clip(arr, -1.0, 1.0)
    return np.round(clipped * 32767.0).astype(np.int16)


def _samples_to_wav(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Wrap a KittenTTS float32 array as a 16-bit mono WAV byte string."""
    pcm = _samples_to_pcm16(samples)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


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
