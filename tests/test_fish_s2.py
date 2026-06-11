"""Tests for Fish Audio S2 Pro sidecar helpers/config wiring."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from universal_tts.config import ProviderConfig
from universal_tts.providers.http_backed import HttpBackedProvider


REPO_ROOT = Path(__file__).resolve().parents[1]
SIDE_CAR = REPO_ROOT / "sidecars" / "fish_s2_server.py"


def load_sidecar_module():
    spec = importlib.util.spec_from_file_location("fish_s2_server_test", SIDE_CAR)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_float_audio_scales_to_pcm16_without_silence():
    mod = load_sidecar_module()
    audio = np.array([-0.75, -0.25, 0.0, 0.25, 0.75], dtype=np.float32)
    pcm = mod._to_int16(audio)
    assert pcm.dtype == np.int16
    assert pcm.tolist() == [-24575, -8192, 0, 8192, 24575]
    assert int(np.count_nonzero(pcm)) == 4


def test_pcm16_clips_out_of_range_values():
    mod = load_sidecar_module()
    audio = np.array([-2.0, -1.0, 1.0, 2.0], dtype=np.float32)
    pcm = mod._to_int16(audio)
    assert pcm.tolist() == [-32768, -32768, 32767, 32767]


def test_wav_bytes_roundtrip():
    mod = load_sidecar_module()
    audio = np.sin(np.linspace(0, 6.28, 4410, dtype=np.float32)) * 0.3
    blob = mod._wav_bytes(audio, 44100)
    assert blob.startswith(b"RIFF")
    assert b"WAVE" in blob[:16]
    assert len(blob) > 44


def test_http_provider_accepts_fish_s2_stream_config():
    cfg = ProviderConfig(
        id="fish-s2",
        kind="http",
        models=["fish-s2", "mlx-community/fish-audio-s2-pro-8bit"],
        estimate_gb=12,
        category="tts",
        base_url="http://127.0.0.1:8784",
        health_path="/health",
        options={
            "stream_path": "/v1/audio/speech-stream",
            "supports_true_streaming": True,
            "streaming_kind": "pcm16",
            "streaming_mode": "prefix-incremental-pcm",
            "streaming_implementation": "mlx-audio-fish-s2-prefix-code-decode",
            "stream_sample_rate": 44100,
            "stream_channels": 1,
            "stream_sample_format": "pcm16",
            "stream_frame_ms": 5,
            "realtime_pacing": False,
        },
    )
    provider = HttpBackedProvider(cfg)
    assert provider.cfg.options["supports_true_streaming"] is True
    assert provider.cfg.options["streaming_kind"] == "pcm16"
    assert provider.cfg.options["streaming_mode"] == "prefix-incremental-pcm"
    assert provider.cfg.options["streaming_implementation"] == "mlx-audio-fish-s2-prefix-code-decode"
    assert provider.stream_path() == "/v1/audio/speech-stream"
