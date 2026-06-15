from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from universal_tts.config import load_config
from universal_tts.providers.http_backed import HttpBackedProvider
from universal_tts.providers.base import TTSRequest


def load_sidecar_module():
    path = Path(__file__).resolve().parents[1] / "sidecars" / "higgs_v3_mlx_server.py"
    spec = importlib.util.spec_from_file_location("higgs_v3_mlx_server_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_higgs_mlx_health_imports_without_mlx_runtime_load():
    server = load_sidecar_module()
    data = server.health(load=False)
    assert data["runtime"] == "mlx-audio"
    assert data["sample_rate"] == 24000
    assert data["supports_true_streaming"] is True
    assert data["streaming_mode"] == "prefix-incremental-codec-decode"


def test_higgs_mlx_config_and_payload_voice_alias():
    runtime = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    cfg = runtime.providers["higgs-audio-v3-mlx"]
    assert runtime.model_to_provider["higgs-audio-v3"] == "higgs-audio-v3-mlx"
    assert runtime.model_to_provider["higgs-audio-v3-torch"] == "higgs-audio-v3"
    assert cfg.port == 8806
    assert cfg.options["streaming_implementation"] == "mlx-audio-higgs-v3-prefix-decode"
    assert cfg.options["supports_true_streaming"] is True
    assert cfg.estimate_gb <= 11
    provider = HttpBackedProvider(cfg)
    req = TTSRequest(
        model="higgs-audio-v3-mlx",
        text="hello",
        voice="liam-default",
        response_format="pcm",
        speed=None,
        options={"stream_commit_tokens": 24},
    )
    payload = provider.speech_payload(req)
    assert payload["voice"] == "voice01"
    assert payload["stream_commit_tokens"] == 24
