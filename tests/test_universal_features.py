import asyncio

import pytest
from fastapi.testclient import TestClient

from universal_tts import app as app_module
from universal_tts.audio import convert_audio_format
from universal_tts.config import ProviderConfig, RuntimeConfig
from universal_tts.memory import MemorySnapshot
from universal_tts.providers.base import ProviderStatus, TTSRequest
from universal_tts.registry import ProviderRegistry


class FeatureFakeProvider:
    def __init__(self, cfg):
        self.cfg = cfg
        self.loaded = False
        self.voices_called = False
        self.cancelled = False
        self.batch_requests = []

    async def status(self):
        return ProviderStatus(id=self.cfg.id, loaded=self.loaded, healthy=self.loaded, details={})

    async def load(self):
        self.loaded = True
        return await self.status()

    async def unload(self):
        self.loaded = False
        return await self.status()

    async def synthesize(self, request: TTSRequest):
        await asyncio.sleep(float(request.options.get("delay", 0)))
        if request.options.get("cancel_event") and request.options["cancel_event"].is_set():
            self.cancelled = True
            raise asyncio.CancelledError()
        return b"RIFF$\x00\x00\x00WAVEfmt ", "audio/wav"

    async def batch_synthesize(self, requests):
        self.batch_requests.append(list(requests))
        return [(b"RIFF$\x00\x00\x00WAVEfmt ", "audio/wav") for _ in requests]

    async def voices(self):
        self.voices_called = True
        return {"object": "list", "data": [{"id": "voice-a", "provider": self.cfg.id}]}

    async def stream_synthesize(self, request: TTSRequest):
        async def chunks():
            yield b"RIFF"
            yield b"low-latency"
        return chunks(), "audio/wav"


def make_registry():
    runtime = RuntimeConfig(
        server={"host": "127.0.0.1", "port": 8899},
        memory={"reserve_gb": 0},
        providers={
            "fake": ProviderConfig(
                id="fake",
                kind="feature-fake",
                models=["fake-model"],
                estimate_gb=1,
                options={"supports_true_streaming": True, "batch_window_ms": 25, "max_batch_size": 4},
            ),
        },
    )
    return ProviderRegistry(runtime, provider_factories={"feature-fake": FeatureFakeProvider}, memory_snapshot=lambda: MemorySnapshot(128, 100, 100))


@pytest.mark.asyncio
async def test_registry_exposes_provider_scoped_voices_without_loading_every_provider():
    registry = make_registry()

    voices = await registry.voices("fake")

    assert voices["data"][0]["id"] == "voice-a"
    assert registry.providers["fake"].voices_called is True


def test_app_exposes_provider_scoped_voices(monkeypatch):
    async def fake_voices(provider_id):
        assert provider_id == "fake"
        return {"object": "list", "data": [{"id": "voice-a"}]}

    monkeypatch.setattr(app_module.registry, "voices", fake_voices)
    client = TestClient(app_module.app)

    response = client.get("/providers/fake/voices")

    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "voice-a"


def test_app_redirects_global_voices_to_first_loaded_provider(monkeypatch):
    async def fake_first_loaded_voice_provider():
        return "qwen3"

    async def fake_voices(provider_id):
        return {"object": "list", "data": [{"id": "qwen-voice", "provider": provider_id}]}

    monkeypatch.setattr(app_module.registry, "first_loaded_voice_provider", fake_first_loaded_voice_provider)
    monkeypatch.setattr(app_module.registry, "voices", fake_voices)
    client = TestClient(app_module.app)

    response = client.get("/v1/voices", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/providers/qwen3/voices"

    followed = client.get("/v1/voices")
    assert followed.status_code == 200
    assert followed.json()["data"][0]["provider"] == "qwen3"


@pytest.mark.asyncio
async def test_registry_stream_capabilities_report_true_decoder_streaming_flag():
    registry = make_registry()

    caps = await registry.capabilities()

    assert caps["providers"]["fake"]["supports_true_streaming"] is True
    assert caps["providers"]["fake"]["supports_batching"] is True


@pytest.mark.asyncio
async def test_registry_capabilities_distinguish_api_fallbacks_from_native_provider_features():
    runtime = RuntimeConfig(
        server={"host": "127.0.0.1", "port": 8899},
        memory={"reserve_gb": 0},
        providers={
            "compat": ProviderConfig(
                id="compat",
                kind="feature-fake",
                models=["compat-model"],
                estimate_gb=1,
                options={
                    "supports_true_streaming": False,
                    "supports_native_batching": False,
                    "max_batch_size": 4,
                    "batch_window_ms": 25,
                    "streaming_mode": "full-generate-then-chunk",
                },
            ),
            "native": ProviderConfig(
                id="native",
                kind="feature-fake",
                models=["native-model"],
                estimate_gb=1,
                options={
                    "supports_true_streaming": True,
                    "supports_native_batching": True,
                    "max_batch_size": 4,
                    "batch_window_ms": 25,
                    "streaming_mode": "true-decoder-pcm",
                    "streaming_implementation": "resident-coreml",
                    "streaming_kind": "pcm16",
                    "stream_sample_rate": 24000,
                    "stream_channels": 1,
                    "stream_sample_format": "pcm16",
                },
            ),
        },
    )
    registry = ProviderRegistry(runtime, provider_factories={"feature-fake": FeatureFakeProvider}, memory_snapshot=lambda: MemorySnapshot(128, 100, 100))

    caps = await registry.capabilities()

    compat = caps["providers"]["compat"]
    assert compat["supports_streaming_api"] is True
    assert compat["supports_batching_api"] is True
    assert compat["supports_microbatch_scheduler"] is True
    assert compat["supports_true_streaming"] is False
    assert compat["supports_native_batching"] is False
    assert compat["streaming_kind"] == "compatibility"
    assert compat["batching_kind"] == "universal-microbatch-sequential-fallback"

    native = caps["providers"]["native"]
    assert native["supports_true_streaming"] is True
    assert native["supports_native_batching"] is True
    assert native["streaming_kind"] == "pcm16"
    assert native["sample_rate"] == 24000
    assert native["channels"] == 1
    assert native["sample_format"] == "pcm16"
    assert native["batching_kind"] == "native-provider"
    assert native["streaming_implementation"] == "resident-coreml"


def test_audio_format_conversion_wav_to_mp3():
    wav = b"RIFF$\x00\x00\x00WAVEfmt " + b"\x10\x00\x00\x00\x01\x00\x01\x00\x40\x1f\x00\x00\x80>\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"

    converted, content_type = convert_audio_format(wav, "wav", "mp3")

    assert content_type == "audio/mpeg"
    assert converted.startswith(b"ID3") or converted.startswith(b"\xff")


@pytest.mark.asyncio
async def test_registry_can_batch_multiple_requests_for_batch_capable_provider():
    registry = make_registry()

    results = await registry.batch_synthesize([
        {"model": "fake-model", "input": "one"},
        {"model": "fake-model", "input": "two"},
    ])

    assert len(results) == 2
    assert len(registry.providers["fake"].batch_requests[0]) == 2


def test_job_queue_supports_submission_status_and_cancellation(monkeypatch):
    registry = make_registry()
    monkeypatch.setattr(app_module, "registry", registry)
    app_module.job_queue.registry = registry
    client = TestClient(app_module.app)

    created = client.post("/v1/audio/jobs", json={"model": "fake-model", "input": "slow", "delay": 2})
    assert created.status_code == 202
    job_id = created.json()["id"]

    cancelled = client.delete(f"/v1/audio/jobs/{job_id}")

    assert cancelled.status_code == 200
    assert cancelled.json()["status"] in {"cancelling", "cancelled"}
