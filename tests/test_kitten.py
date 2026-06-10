"""Tests for the KittenTTS provider (HTTP-backed sidecar)."""

import pytest

from universal_tts.config import ProviderConfig
from universal_tts.providers.base import TTSRequest
from universal_tts.providers.kitten import KittenTTSProvider


@pytest.fixture
def cfg() -> ProviderConfig:
    return ProviderConfig(
        id="kitten",
        kind="kitten",
        base_url="http://127.0.0.1:8782",
        health_path="/health",
        port=8782,
        models=["KittenML/kitten-tts-nano-0.8"],
        estimate_gb=1,
        options={
            "default_model": "KittenML/kitten-tts-nano-0.8",
            "default_voice": "Bella",
            "default_clean_text": True,
            "voice_aliases": {"liam-default": "Bella", "bruno": "Bruno"},
            "voices": ["Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo"],
            "stream_path": "/v1/audio/speech-stream",
            "stream_content_type": "audio/pcm",
        },
    )


def test_kitten_provider_is_http_backed(cfg):
    from universal_tts.providers.http_backed import HttpBackedProvider

    p = KittenTTSProvider(cfg)
    assert isinstance(p, HttpBackedProvider)


def test_kitten_stream_path(cfg):
    p = KittenTTSProvider(cfg)
    assert p.stream_path() == "/v1/audio/speech-stream"


def test_kitten_voice_alias_resolution(cfg):
    p = KittenTTSProvider(cfg)
    payload = p.speech_payload(
        TTSRequest(model="kitten-tts-nano", text="hello", voice="liam-default")
    )
    assert payload["voice"] == "Bella"
    payload = p.speech_payload(
        TTSRequest(model="kitten-tts-nano", text="hello", voice="bruno")
    )
    assert payload["voice"] == "Bruno"
    payload = p.speech_payload(
        TTSRequest(model="kitten-tts-nano", text="hello", voice=None)
    )
    assert payload["voice"] == "Bella"
    payload = p.speech_payload(
        TTSRequest(model="kitten-tts-nano", text="hello", voice="Jasper")
    )
    assert payload["voice"] == "Jasper"


def test_kitten_default_clean_text_is_true(cfg):
    p = KittenTTSProvider(cfg)
    payload = p.speech_payload(
        TTSRequest(model="kitten-tts-nano", text="hi", voice="Bella")
    )
    assert payload["clean_text"] is True


def test_kitten_clean_text_request_override(cfg):
    p = KittenTTSProvider(cfg)
    payload = p.speech_payload(
        TTSRequest(
            model="kitten-tts-nano", text="hi", voice="Bella", options={"clean_text": False}
        )
    )
    assert payload["clean_text"] is False


def test_kitten_stream_payload_matches_speech(cfg):
    p = KittenTTSProvider(cfg)
    req = TTSRequest(model="kitten-tts-nano", text="hi", voice="bruno", speed=1.1)
    assert p.stream_payload(req) == p.speech_payload(req)
