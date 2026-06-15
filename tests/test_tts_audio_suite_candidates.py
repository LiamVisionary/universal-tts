from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from universal_tts.app import app
from universal_tts.config import load_config
from universal_tts.providers.base import TTSRequest
from universal_tts.providers.http_backed import HttpBackedProvider
from universal_tts.srt import build_srt_cues, format_srt_timestamp, text_to_srt


def load_catalog_module():
    path = Path(__file__).resolve().parents[1] / "sidecars" / "tts_audio_suite_catalog_server.py"
    spec = importlib.util.spec_from_file_location("suite_catalog_server_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_suite_candidate_config_is_cataloged_but_not_true_streaming():
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    for provider_id in ["moss-tts", "indextts2", "step-audio-editx", "granite-asr", "rvc"]:
        provider = cfg.providers[provider_id]
        assert provider.kind == "http"
        assert provider.options["supports_true_streaming"] is False
        assert "catalog" in provider.options["streaming_implementation"]


def test_indextts2_payload_preserves_clone_and_emotion_fields():
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml").providers["indextts2"]
    provider = HttpBackedProvider(cfg)
    req = TTSRequest(
        model="IndexTTS-2",
        text="Testing expressive clone routing.",
        voice="voice01",
        response_format="wav",
        speed=None,
        options={
            "ref_audio": "/tmp/voice.wav",
            "emotion_vector": [0, 0, 0, 0, 0, 0, 0, 1],
            "emotion_alpha": 0.7,
        },
    )
    payload = provider.speech_payload(req)
    assert payload["voice"] == "clone"
    assert payload["ref_audio"] == "/tmp/voice.wav"
    assert payload["emotion_alpha"] == 0.7


def test_suite_catalog_health_can_switch_provider(monkeypatch):
    server = load_catalog_module()
    monkeypatch.setenv("SUITE_PROVIDER_ID", "step-audio-editx")
    data = server.health(load=False)
    assert data["provider"] == "step-audio-editx"
    assert data["runtime_enabled"] is False
    assert data["supports_true_streaming"] is False
    assert "[Laughter]" in data["paralinguistics"]


def test_srt_builder_formats_cues_and_endpoint():
    assert format_srt_timestamp(3723004) == "01:02:03,004"
    srt = text_to_srt("Hello world. This is a second cue.", words_per_minute=120)
    assert "00:00:00,000 -->" in srt
    assert "Hello world." in srt
    cues = build_srt_cues("One two three four five six.", total_duration_ms=3000, max_chars=12)
    assert len(cues) >= 2
    assert cues[0].end_ms <= cues[1].start_ms

    client = TestClient(app)
    response = client.post("/v1/audio/srt", json={"text": "Hello world. Another line.", "max_chars": 20})
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "audio.srt"
    assert body["cue_count"] == 2
    assert "Another line." in body["srt"]
