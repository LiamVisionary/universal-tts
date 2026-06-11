"""Tests for the Kokoro sidecar/provider wiring."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from universal_tts.config import ProviderConfig
from universal_tts.providers.base import TTSRequest
from universal_tts.providers.http_backed import HttpBackedProvider


def _load_kokoro_server_module():
    path = Path(__file__).resolve().parents[1] / "sidecars" / "kokoro_server.py"
    spec = importlib.util.spec_from_file_location("kokoro_server_for_tests", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_kokoro_provider_generic_http_payload_aliases_voice():
    cfg = ProviderConfig(
        id="kokoro",
        kind="http",
        base_url="http://127.0.0.1:8783",
        health_path="/health",
        port=8783,
        models=["kokoro"],
        estimate_gb=3,
        options={
            "stream_path": "/v1/audio/speech-stream",
            "stream_content_type": "audio/pcm",
            "voice_aliases": {"liam-default": "af_heart"},
        },
    )
    provider = HttpBackedProvider(cfg)
    req = TTSRequest(model="kokoro", text="hello", voice="liam-default", response_format="wav")

    payload = provider.speech_payload(req)

    assert payload["model"] == "kokoro"
    assert payload["input"] == "hello"
    assert payload["voice"] == "af_heart"
    assert provider.stream_path() == "/v1/audio/speech-stream"


def test_kokoro_sidecar_lang_inference_and_aliases(monkeypatch):
    module = _load_kokoro_server_module()
    monkeypatch.setenv("KOKORO_VOICE_ALIASES", '{"liam-default": "af_heart"}')

    assert module._resolve_voice("liam-default") == "af_heart"
    assert module._lang_for_voice("af_heart") == "a"
    assert module._lang_for_voice("bf_emma") == "b"
    assert module._lang_for_voice("zf_xiaobei") == "z"
    assert module._lang_for_voice("jf_alpha") == "j"


def test_kokoro_sidecar_realtime_segment_splitter():
    module = _load_kokoro_server_module()
    text = "Sentence one is short. Sentence two is also short. " + ("word " * 80)
    segments = module._split_realtime_segments(text, max_chars=80)

    assert len(segments) >= 3
    assert all(seg.strip() for seg in segments)
    assert all(len(seg) <= 90 for seg in segments)  # allow comma/space boundary slack
    assert segments[0].startswith("Sentence one")


def test_kokoro_float32_to_pcm16_not_silent():
    module = _load_kokoro_server_module()
    samples = np.array([-1.2, -0.5, 0.0, 0.25, 0.75, 1.2], dtype=np.float32)
    pcm = module._to_int16(samples)

    assert pcm.dtype == np.int16
    assert int(pcm[0]) == -32768
    assert int(pcm[-1]) == 32767
    assert int((pcm != 0).sum()) == 5
    assert int(np.max(np.abs(pcm))) == 32767
