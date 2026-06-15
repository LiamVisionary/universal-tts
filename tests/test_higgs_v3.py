from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from universal_tts.config import ProviderConfig
from universal_tts.providers.http_backed import HttpBackedProvider
from universal_tts.providers.base import TTSRequest


def load_sidecar_module():
    path = Path(__file__).resolve().parents[1] / "sidecars" / "higgs_v3_server.py"
    spec = importlib.util.spec_from_file_location("higgs_v3_server_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_higgs_health_contract_without_loading():
    server = load_sidecar_module()
    data = server.health(load=False)
    assert data["model"] == "higgs-audio-v3-tts-4b"
    assert data["sample_rate"] == 24000
    assert data["streaming_mode"] == "prefix-incremental-codec-decode"


def test_higgs_http_payload_passes_clone_and_stream_fields():
    cfg = ProviderConfig(
        id="higgs-audio-v3",
        kind="http",
        models=["higgs-audio-v3"],
        estimate_gb=18,
        category="tts",
        base_url="http://127.0.0.1:8786",
        options={"voice_aliases": {"voice01": "clone"}},
    )
    provider = HttpBackedProvider(cfg)
    req = TTSRequest(
        model="higgs-audio-v3",
        text="Realtime Higgs voice clone test.",
        voice="voice01",
        response_format="pcm",
        options={
            "ref_audio": "/tmp/voice01.wav",
            "ref_text": "reference words",
            "max_new_tokens": 256,
            "stream_commit_tokens": 8,
        },
    )
    payload = provider.stream_payload(req)
    assert payload["voice"] == "clone"
    assert payload["ref_audio"] == "/tmp/voice01.wav"
    assert payload["ref_text"] == "reference words"
    assert payload["stream_commit_tokens"] == 8
