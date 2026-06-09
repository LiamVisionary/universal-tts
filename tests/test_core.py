import pytest

from universal_tts.config import RuntimeConfig, ProviderConfig, load_config_dict
from universal_tts.memory import MemorySnapshot, MemoryGuard
from universal_tts.registry import ProviderRegistry
from universal_tts.providers.base import ProviderStatus, TTSRequest
from universal_tts.providers.http_backed import ChatterboxTurboProvider


class FakeProvider:
    def __init__(self, cfg):
        self.cfg = cfg
        self.loaded = False
        self.requests = []

    async def status(self):
        return ProviderStatus(id=self.cfg.id, loaded=self.loaded, healthy=self.loaded, details={})

    async def load(self):
        self.loaded = True
        return await self.status()

    async def unload(self):
        self.loaded = False
        return await self.status()

    async def synthesize(self, request: TTSRequest):
        self.requests.append(request)
        return b"RIFFfakewav", "audio/wav"


def test_config_loader_keeps_only_tts_providers_and_model_aliases():
    cfg = load_config_dict({
        "server": {"host": "127.0.0.1", "port": 8899},
        "memory": {"reserve_gb": 10},
        "providers": {
            "qwen3": {"kind": "qwen3", "models": ["qwen3-tts-0.6b-base-clone"], "estimate_gb": 7},
            "bad-image": {"kind": "comfy", "category": "image", "models": ["zimage"], "estimate_gb": 20},
        },
    })

    assert list(cfg.providers) == ["qwen3"]
    assert cfg.model_to_provider["qwen3-tts-0.6b-base-clone"] == "qwen3"


def test_memory_guard_refuses_load_when_reserve_would_be_crossed():
    guard = MemoryGuard(reserve_gb=12)
    snap = MemorySnapshot(total_gb=128, available_gb=8, available_plus_reclaimable_gb=10)

    with pytest.raises(RuntimeError) as exc:
        guard.assert_can_load(provider_id="miso", estimate_gb=4, snapshot=snap, force=False)

    assert "memory guard refused" in str(exc.value).lower()


def test_memory_guard_allows_force_with_low_memory():
    guard = MemoryGuard(reserve_gb=12)
    snap = MemorySnapshot(total_gb=128, available_gb=1, available_plus_reclaimable_gb=1)

    decision = guard.assert_can_load(provider_id="miso", estimate_gb=40, snapshot=snap, force=True)

    assert decision["forced"] is True


@pytest.mark.asyncio
async def test_registry_load_exclusive_unloads_other_loaded_providers():
    runtime = RuntimeConfig(
        server={"host": "127.0.0.1", "port": 8899},
        memory={"reserve_gb": 0},
        providers={
            "a": ProviderConfig(id="a", kind="fake", models=["a-model"], estimate_gb=1),
            "b": ProviderConfig(id="b", kind="fake", models=["b-model"], estimate_gb=1),
        },
    )
    registry = ProviderRegistry(runtime, provider_factories={"fake": FakeProvider}, memory_snapshot=lambda: MemorySnapshot(128, 100, 100))

    await registry.load("a")
    await registry.load("b", mode="exclusive")

    statuses = await registry.statuses()
    assert statuses["a"].loaded is False
    assert statuses["b"].loaded is True


@pytest.mark.asyncio
async def test_registry_routes_model_alias_to_provider_and_normalizes_request():
    runtime = RuntimeConfig(
        server={"host": "127.0.0.1", "port": 8899},
        memory={"reserve_gb": 0},
        providers={
            "qwen": ProviderConfig(id="qwen", kind="fake", models=["qwen-model", "tts-1"], estimate_gb=1),
        },
    )
    registry = ProviderRegistry(runtime, provider_factories={"fake": FakeProvider}, memory_snapshot=lambda: MemorySnapshot(128, 100, 100))

    audio, content_type = await registry.synthesize({"model": "qwen-model", "voice": "voice1", "input": "hello", "response_format": "wav"})

    provider = registry.providers["qwen"]
    assert audio.startswith(b"RIFF")
    assert content_type == "audio/wav"
    assert provider.requests[0].model == "qwen-model"
    assert provider.requests[0].voice == "voice1"
    assert provider.requests[0].text == "hello"


def test_chatterbox_stream_payload_passes_true_stream_quality_control():
    cfg = ProviderConfig(
        id="chatterbox-turbo",
        kind="http",
        models=["chatterbox-turbo"],
        estimate_gb=1,
        options={"voice_aliases": {"voice01": "voice1-all-samples-10s.wav"}},
    )
    provider = ChatterboxTurboProvider(cfg)
    request = TTSRequest(
        model="chatterbox-turbo",
        text="hello",
        voice="voice01",
        response_format="pcm",
        options={"true_stream_quality": True},
    )

    payload = provider.stream_payload(request)

    assert payload["predefined_voice_id"] == "voice1-all-samples-10s.wav"
    assert payload["true_stream_quality"] is True
