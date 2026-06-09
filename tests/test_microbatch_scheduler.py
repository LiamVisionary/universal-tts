import asyncio

import pytest

from universal_tts.config import ProviderConfig, RuntimeConfig
from universal_tts.memory import MemorySnapshot
from universal_tts.providers.base import ProviderStatus, TTSRequest
from universal_tts.registry import ProviderRegistry


class MicroBatchProvider:
    def __init__(self, cfg):
        self.cfg = cfg
        self.loaded = False
        self.synth_calls = 0
        self.batch_calls = []

    async def status(self):
        return ProviderStatus(id=self.cfg.id, loaded=self.loaded, healthy=self.loaded, details={})

    async def load(self):
        self.loaded = True
        return await self.status()

    async def unload(self):
        self.loaded = False
        return await self.status()

    async def synthesize(self, request: TTSRequest):
        self.synth_calls += 1
        return f"RIFF-single-{request.text}".encode(), "audio/wav"

    async def batch_synthesize(self, requests):
        self.batch_calls.append([request.text for request in requests])
        return [(f"RIFF-batch-{request.text}".encode(), "audio/wav") for request in requests]


def registry():
    runtime = RuntimeConfig(
        server={"host": "127.0.0.1", "port": 8899},
        memory={"reserve_gb": 0},
        providers={
            "p": ProviderConfig(
                id="p",
                kind="micro",
                models=["m"],
                estimate_gb=1,
                options={"max_batch_size": 4, "batch_window_ms": 25, "supports_native_batching": True},
            ),
        },
    )
    return ProviderRegistry(runtime, provider_factories={"micro": MicroBatchProvider}, memory_snapshot=lambda: MemorySnapshot(128, 100, 100))


@pytest.mark.asyncio
async def test_concurrent_speech_requests_are_coalesced_into_microbatch():
    r = registry()

    results = await asyncio.gather(
        r.synthesize({"model": "m", "input": "one"}),
        r.synthesize({"model": "m", "input": "two"}),
    )

    provider = r.providers["p"]
    assert provider.batch_calls == [["one", "two"]]
    assert provider.synth_calls == 0
    assert results[0][0].startswith(b"RIFF-batch-one")
    assert results[1][0].startswith(b"RIFF-batch-two")


@pytest.mark.asyncio
async def test_capabilities_distinguish_generic_api_batching_from_native_batching():
    caps = await registry().capabilities()

    provider_caps = caps["providers"]["p"]
    assert provider_caps["supports_batching_api"] is True
    assert provider_caps["supports_native_batching"] is True
    assert provider_caps["supports_microbatch_scheduler"] is True
