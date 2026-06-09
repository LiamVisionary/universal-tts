import pytest

from universal_tts.config import RuntimeConfig, ProviderConfig
from universal_tts.memory import MemorySnapshot
from universal_tts.providers.base import ProviderStatus, TTSRequest
from universal_tts.registry import ProviderRegistry


class FakeStreamingProvider:
    def __init__(self, cfg):
        self.cfg = cfg
        self.loaded = False
        self.stream_requests = []

    async def status(self):
        return ProviderStatus(id=self.cfg.id, loaded=self.loaded, healthy=self.loaded, details={})

    async def load(self):
        self.loaded = True
        return await self.status()

    async def unload(self):
        self.loaded = False
        return await self.status()

    async def synthesize(self, request: TTSRequest):
        return b"RIFFfull", "audio/wav"

    async def stream_synthesize(self, request: TTSRequest):
        self.stream_requests.append(request)
        async def chunks():
            yield b"RIFF"
            yield b"chunk"
        return chunks(), "audio/wav"


@pytest.mark.asyncio
async def test_registry_stream_routes_model_to_provider_and_returns_chunks():
    runtime = RuntimeConfig(
        server={"host": "127.0.0.1", "port": 8899},
        memory={"reserve_gb": 0},
        providers={
            "qwen": ProviderConfig(id="qwen", kind="fake-stream", models=["qwen-model"], estimate_gb=1),
        },
    )
    registry = ProviderRegistry(runtime, provider_factories={"fake-stream": FakeStreamingProvider}, memory_snapshot=lambda: MemorySnapshot(128, 100, 100))

    iterator, content_type, headers = await registry.stream_synthesize({"model": "qwen-model", "voice": "Ryan", "input": "hello", "response_format": "wav"})
    data = b"".join([chunk async for chunk in iterator])

    provider = registry.providers["qwen"]
    assert content_type == "audio/wav"
    assert headers == {}
    assert data == b"RIFFchunk"
    assert provider.stream_requests[0].model == "qwen-model"
    assert provider.stream_requests[0].voice == "Ryan"
    assert provider.stream_requests[0].text == "hello"
