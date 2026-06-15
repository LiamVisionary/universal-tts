from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from universal_tts.config import ProviderConfig
from universal_tts.providers.http_backed import HttpBackedProvider
from universal_tts.providers.base import TTSRequest


def load_sidecar_module():
    path = Path(__file__).resolve().parents[1] / "sidecars" / "cosyvoice3_server.py"
    spec = importlib.util.spec_from_file_location("cosyvoice3_server_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cosyvoice3_special_tag_conversion():
    server = load_sidecar_module()
    assert server.convert_special_tags("Hello <breath> friend <laughter>") == "Hello [breath] friend [laughter]"
    assert server.convert_special_tags("This is <laughing>funny</laughing>") == "This is <laughter>funny</laughter>"
    assert server.convert_special_tags("keep <strong>this</strong>") == "keep <strong>this</strong>"


def test_cosyvoice3_health_reports_model_contract_without_loading():
    server = load_sidecar_module()
    data = server.health(load=False)
    assert data["model"] == "Fun-CosyVoice3-0.5B-RL"
    assert data["sample_rate"] == 24000
    assert data["supports_true_streaming"] is True


def test_generic_http_payload_passes_cosyvoice_clone_fields():
    cfg = ProviderConfig(
        id="cosyvoice3",
        kind="http",
        models=["cosyvoice3"],
        estimate_gb=7,
        category="tts",
        base_url="http://127.0.0.1:8785",
        options={"voice_aliases": {"liam-default": "clone"}},
    )
    provider = HttpBackedProvider(cfg)
    req = TTSRequest(
        model="cosyvoice3",
        text="Hello <breath> world",
        voice="liam-default",
        response_format="pcm",
        speed=1.05,
        options={"ref_audio": "/tmp/ref.wav", "ref_text": "reference", "mode": "zero_shot", "instruct": "warm"},
    )
    payload = provider.speech_payload(req)
    assert payload["voice"] == "clone"
    assert payload["ref_audio"] == "/tmp/ref.wav"
    assert payload["ref_text"] == "reference"
    assert payload["mode"] == "zero_shot"
    assert payload["instruct"] == "warm"
