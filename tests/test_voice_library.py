from __future__ import annotations

import wave

import numpy as np
import pytest

from universal_tts.config import ProviderConfig, RuntimeConfig
from universal_tts.memory import MemorySnapshot
from universal_tts.providers.base import ProviderStatus, TTSRequest
from universal_tts.registry import ProviderRegistry


class FakeProvider:
    def __init__(self, cfg):
        self.cfg = cfg
        self.loaded = False
        self.requests: list[TTSRequest] = []

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

    async def voices(self):
        return {"object": "list", "data": [{"id": "default", "provider": self.cfg.id}]}


def write_wav(path):
    audio = (np.sin(np.linspace(0, 1, 8000)) * 12000).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(audio.tobytes())


def make_registry(tmp_path, provider_id="fish-s2"):
    runtime = RuntimeConfig(
        server={"host": "127.0.0.1", "port": 8899, "data_dir": str(tmp_path / "data")},
        memory={"reserve_gb": 0},
        providers={
            provider_id: ProviderConfig(
                id=provider_id,
                kind="fake",
                models=[provider_id],
                estimate_gb=1,
                options={"supports_voice_cloning": True, "formats": ["wav", "pcm"]},
            )
        },
    )
    return ProviderRegistry(runtime, provider_factories={"fake": FakeProvider}, memory_snapshot=lambda: MemorySnapshot(128, 100, 100))


def test_create_voice_persists_and_appears_in_provider_voice_list(tmp_path):
    ref = tmp_path / "ref.wav"
    write_wav(ref)
    registry = make_registry(tmp_path)

    result = registry.create_voice({
        "provider": "fish-s2",
        "voice_id": "voice01",
        "name": "Voice 01",
        "ref_audio": str(ref),
        "ref_text": "Reference transcript.",
    })

    assert result["status"] == "ready"
    assert result["added_to_voice_library"] is True
    profile = registry.voice_library.get("fish-s2", "voice01")
    assert profile is not None
    assert profile.ref_text == "Reference transcript."
    assert profile.ref_audio.endswith("voice01.wav")


@pytest.mark.asyncio
async def test_saved_fish_voice_injects_clone_reference_options(tmp_path):
    ref = tmp_path / "ref.wav"
    write_wav(ref)
    registry = make_registry(tmp_path)
    registry.create_voice({
        "provider": "fish-s2",
        "voice_id": "voice01",
        "ref_audio": str(ref),
        "ref_text": "Reference transcript.",
    })

    await registry.synthesize({"model": "fish-s2", "voice": "voice01", "input": "hello", "response_format": "wav"})

    provider = registry.providers["fish-s2"]
    req = provider.requests[0]
    assert req.voice == "clone"
    assert req.options["ref_text"] == "Reference transcript."
    assert req.options["ref_audio"].endswith("voice01.wav")


@pytest.mark.asyncio
async def test_saved_miso_voice_uses_inline_reference_fields(tmp_path):
    ref = tmp_path / "ref.wav"
    write_wav(ref)
    registry = make_registry(tmp_path, provider_id="miso")
    registry.create_voice({
        "provider": "miso",
        "voice_id": "voice01",
        "ref_audio": str(ref),
        "ref_text": "Reference transcript.",
    })

    await registry.synthesize({"model": "miso", "voice": "voice01", "input": "hello", "response_format": "wav"})

    req = registry.providers["miso"].requests[0]
    assert req.voice == "voice01"
    assert req.options["reference_transcript"] == "Reference transcript."
    assert req.options["reference_audio_path"].endswith("voice01.wav")
