"""Tests for the KittenTTS in-process provider.

Pure stdlib test — no numpy / no httpx mocks beyond what's needed to stub
``kittentts`` at import time.
"""

import asyncio
import io
import sys
import wave
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from universal_tts.config import ProviderConfig, RuntimeConfig
from universal_tts.memory import MemorySnapshot
from universal_tts.providers.base import TTSRequest
from universal_tts.providers.kitten import (
    AVAILABLE_VOICES,
    CHANNELS,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    KittenTTSProvider,
    _audio_to_pcm_bytes,
    _audio_to_wav_bytes,
    _flatten_samples,
)
from universal_tts.registry import ProviderRegistry


def _pcm_chunk_list(num_samples: int, value: int) -> list[int]:
    return [value] * num_samples


def _wav_samples(wav_bytes: bytes) -> list[int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        assert w.getframerate() == SAMPLE_RATE
        assert w.getnchannels() == CHANNELS
        assert w.getsampwidth() == SAMPLE_WIDTH
        data = w.readframes(w.getnframes())
    return list(wave.struct.unpack("<%dh" % (len(data) // 2), data))


def _seq_to_int16_bytes(values):
    import struct as _struct

    return _struct.pack("<%dh" % len(values), *values)


class _FakeKittenTTS:
    """Stand-in for kittentts.KittenTTS used only for tests."""

    last_instance: "_FakeKittenTTS | None" = None

    def __init__(self, model_name, backend="cpu", **kwargs):
        self.model_name = model_name
        self.backend = backend
        self.kwargs = kwargs
        # Pre-canned streaming output: 3 chunks of varying length
        self._chunks = [
            _pcm_chunk_list(1200, 100),
            _pcm_chunk_list(2400, -100),
            _pcm_chunk_list(1200, 200),
        ]
        self.last_kwargs: dict | None = None
        _FakeKittenTTS.last_instance = self

    def generate_stream(self, text, voice, speed=1.0, clean_text=False):
        self.last_kwargs = {"text": text, "voice": voice, "speed": speed, "clean_text": clean_text}
        for c in self._chunks:
            yield c


@pytest.fixture
def patched_kittentts(monkeypatch):
    """Install _FakeKittenTTS in sys.modules under 'kittentts'."""
    fake_mod = SimpleNamespace(KittenTTS=_FakeKittenTTS)
    monkeypatch.setitem(sys.modules, "kittentts", fake_mod)
    yield fake_mod
    _FakeKittenTTS.last_instance = None


@pytest.fixture
def cfg() -> ProviderConfig:
    return ProviderConfig(
        id="kitten",
        kind="kitten",
        models=["KittenML/kitten-tts-mini-0.8"],
        estimate_gb=1,
        options={
            "default_model": "KittenML/kitten-tts-mini-0.8",
            "default_voice": "Bella",
            "default_speed": 1.0,
            "default_clean_text": False,
            "voice_aliases": {"liam-default": "Bella", "bruno": "Bruno"},
            "voices": AVAILABLE_VOICES,
        },
    )


def test_flatten_samples_handles_sequences_and_clipping():
    flat = _flatten_samples([0, 1000, -1000, 999999, -999999, 0])
    assert list(flat) == [0, 1000, -1000, 32767, -32768, 0]


def test_flatten_samples_handles_nested():
    flat = _flatten_samples([[1, 2], [3, 4]])
    assert list(flat) == [1, 2, 3, 4]


def test_audio_helpers_round_trip():
    samples = [0, 1000, -1000, 32767, -32768]
    pcm = _audio_to_pcm_bytes(samples)
    assert pcm == _seq_to_int16_bytes(samples)
    wav = _audio_to_wav_bytes(samples)
    decoded = _wav_samples(wav)
    assert decoded == samples


@pytest.mark.asyncio
async def test_status_reports_unloaded_until_load(patched_kittentts, cfg):
    provider = KittenTTSProvider(cfg)
    status = await provider.status()
    assert status.loaded is False
    assert status.details["supports_true_streaming"] is True
    assert status.details["sample_rate"] == 24000
    assert status.details["voices"] == AVAILABLE_VOICES


@pytest.mark.asyncio
async def test_load_initializes_model(patched_kittentts, cfg):
    provider = KittenTTSProvider(cfg)
    status = await provider.load()
    assert status.loaded is True
    assert status.healthy is True
    assert status.details["model_name"] == "KittenML/kitten-tts-mini-0.8"
    assert status.details["backend"] == "cpu"


@pytest.mark.asyncio
async def test_synthesize_returns_wav(patched_kittentts, cfg):
    provider = KittenTTSProvider(cfg)
    await provider.load()
    audio, content_type = await provider.synthesize(
        TTSRequest(model="kitten-tts-mini", text="hello world", voice="Bella")
    )
    assert content_type == "audio/wav"
    samples = _wav_samples(audio)
    expected = (
        _pcm_chunk_list(1200, 100) + _pcm_chunk_list(2400, -100) + _pcm_chunk_list(1200, 200)
    )
    assert samples == expected


@pytest.mark.asyncio
async def test_stream_synthesize_yields_pcm_chunks(patched_kittentts, cfg):
    provider = KittenTTSProvider(cfg)
    await provider.load()
    chunks_iter, content_type = await provider.stream_synthesize(
        TTSRequest(model="kitten-tts-mini", text="hi", voice="Bruno", speed=1.1)
    )
    assert content_type == "audio/pcm"
    received = bytearray()
    async for chunk in chunks_iter:
        received.extend(chunk)
    expected = (
        _pcm_chunk_list(1200, 100) + _pcm_chunk_list(2400, -100) + _pcm_chunk_list(1200, 200)
    )
    assert bytes(received) == _seq_to_int16_bytes(expected)
    assert _FakeKittenTTS.last_instance is not None
    assert _FakeKittenTTS.last_instance.last_kwargs == {
        "text": "hi",
        "voice": "Bruno",
        "speed": 1.1,
        "clean_text": False,
    }


@pytest.mark.asyncio
async def test_stream_passes_request_options(patched_kittentts, cfg):
    provider = KittenTTSProvider(cfg)
    await provider.load()
    captured: dict = {}

    def fake_generate_stream(text, voice, speed=1.0, clean_text=False):
        captured.update({"text": text, "voice": voice, "speed": speed, "clean_text": clean_text})
        # Yield nothing — synthesize path also calls this with the same kwargs.
        return
        yield  # pragma: no cover  (makes this a generator)

    provider._state.model.generate_stream = fake_generate_stream  # type: ignore[assignment]

    req = TTSRequest(
        model="kitten-tts-mini",
        text="hi there",
        voice="bruno",  # alias -> Bruno
        speed=1.25,
        options={"clean_text": True},
    )
    chunks_iter, _ = await provider.stream_synthesize(req)
    async for _ in chunks_iter:
        pass
    assert captured == {"text": "hi there", "voice": "Bruno", "speed": 1.25, "clean_text": True}


@pytest.mark.asyncio
async def test_voice_aliases_resolve(patched_kittentts, cfg):
    provider = KittenTTSProvider(cfg)
    await provider.load()
    kwargs = provider._synth_kwargs(
        TTSRequest(model="kitten-tts-mini", text="x", voice="liam-default")
    )
    assert kwargs["voice"] == "Bella"
    kwargs = provider._synth_kwargs(
        TTSRequest(model="kitten-tts-mini", text="x", voice="bruno")
    )
    assert kwargs["voice"] == "Bruno"
    kwargs = provider._synth_kwargs(
        TTSRequest(model="kitten-tts-mini", text="x", voice="Bella")
    )
    assert kwargs["voice"] == "Bella"
    kwargs = provider._synth_kwargs(
        TTSRequest(model="kitten-tts-mini", text="x", voice=None)
    )
    assert kwargs["voice"] == "Bella"


@pytest.mark.asyncio
async def test_voices_endpoint(patched_kittentts, cfg):
    provider = KittenTTSProvider(cfg)
    data = await provider.voices()
    assert data["object"] == "list"
    assert {v["id"] for v in data["data"]} == set(AVAILABLE_VOICES)
    assert all(v["provider"] == "kitten" for v in data["data"])


@pytest.mark.asyncio
async def test_paralinguistics_is_empty(patched_kittentts, cfg):
    provider = KittenTTSProvider(cfg)
    data = await provider.paralinguistics()
    assert data == {"object": "audio.paralinguistics", "provider": "kitten", "data": []}


@pytest.mark.asyncio
async def test_unload_clears_state(patched_kittentts, cfg):
    provider = KittenTTSProvider(cfg)
    await provider.load()
    status = await provider.unload()
    assert status.loaded is False
    assert provider._state.model is None


def test_registry_wires_kitten_factory(patched_kittentts):
    runtime = RuntimeConfig(
        server={"host": "127.0.0.1", "port": 8899},
        memory={"reserve_gb": 0},
        providers={
            "kitten": ProviderConfig(
                id="kitten",
                kind="kitten",
                models=["KittenML/kitten-tts-mini-0.8"],
                estimate_gb=1,
                options={"default_model": "KittenML/kitten-tts-mini-0.8"},
            ),
        },
    )
    registry = ProviderRegistry(
        runtime,
        memory_snapshot=lambda: MemorySnapshot(128, 100, 100),
    )
    assert isinstance(registry.providers["kitten"], KittenTTSProvider)


@pytest.mark.asyncio
async def test_load_failure_when_kittentts_missing(monkeypatch):
    # If kittentts isn't importable, load should return an error status.
    monkeypatch.setitem(sys.modules, "kittentts", None)
    runtime = RuntimeConfig(
        server={"host": "127.0.0.1", "port": 8899},
        memory={"reserve_gb": 0},
        providers={
            "kitten": ProviderConfig(
                id="kitten",
                kind="kitten",
                models=["KittenML/kitten-tts-mini-0.8"],
                estimate_gb=1,
            ),
        },
    )
    registry = ProviderRegistry(
        runtime,
        memory_snapshot=lambda: MemorySnapshot(128, 100, 100),
    )
    provider = registry.providers["kitten"]
    status = await provider.load()
    assert status.loaded is False
    assert "kittentts" in status.details.get("error", "")
