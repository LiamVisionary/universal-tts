import pytest

from universal_tts.config import RuntimeConfig, ProviderConfig, load_config_dict
from universal_tts.memory import MemorySnapshot, MemoryGuard
from universal_tts.registry import ProviderRegistry
from universal_tts.providers.base import ProviderStatus, TTSRequest
from universal_tts.providers.http_backed import AudioCppProvider, ChatterboxTurboProvider


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


def test_audiocpp_provider_maps_reference_voice_payload():
    cfg = ProviderConfig(
        id="audiocpp-qwen3",
        kind="audiocpp",
        models=["audiocpp-qwen3-tts-0.6b-base"],
        estimate_gb=4,
        options={
            "upstream_model": "audiocpp-qwen3-tts-0.6b-base",
            "voice_mode": "reference",
            "default_ref_audio": "/tmp/ref.wav",
            "default_ref_text": "reference transcript",
            "default_request_options": {"text_chunk_size": 512},
        },
    )
    provider = AudioCppProvider(cfg)
    request = TTSRequest(
        model="audiocpp-qwen3-tts-0.6b-base",
        text="hello",
        voice="voice01",
        response_format="pcm",
        options={"max_new_tokens": 64, "temperature": 0.5, "instruct": "calm", "foo": "bar"},
    )

    payload = provider.speech_payload(request)

    assert payload["model"] == "audiocpp-qwen3-tts-0.6b-base"
    assert payload["input"] == "hello"
    assert payload["response_format"] == "wav"
    assert "voice" not in payload
    assert payload["voice_ref"] == "/tmp/ref.wav"
    assert payload["reference_text"] == "reference transcript"
    assert payload["max_tokens"] == 64
    assert payload["temperature"] == 0.5
    assert payload["instructions"] == "calm"
    assert payload["options"] == {"text_chunk_size": 512, "foo": "bar"}


def _routing_runtime() -> RuntimeConfig:
    return RuntimeConfig(
        server={"host": "127.0.0.1", "port": 8899},
        memory={"reserve_gb": 0},
        providers={
            "mlx-fast": ProviderConfig(
                id="mlx-fast", kind="fake", models=["mlx-model"], estimate_gb=1,
                options={"platforms": ["darwin-arm64"], "auto_route_ttfb_ms": 220},
            ),
            "audiocpp-anywhere": ProviderConfig(
                id="audiocpp-anywhere", kind="fake", models=["audiocpp-model"], estimate_gb=1,
                options={"platforms": ["any"], "auto_route_ttfb_ms": 2200},
            ),
        },
    )


def _routing_registry(machine_platform: str) -> ProviderRegistry:
    return ProviderRegistry(
        _routing_runtime(),
        provider_factories={"fake": FakeProvider},
        memory_snapshot=lambda: MemorySnapshot(128, 100, 100),
        machine_platform=machine_platform,
    )


def test_auto_routes_to_fastest_supported_provider_per_machine():
    assert _routing_registry("darwin-arm64").provider_for_model("auto") == "mlx-fast"
    assert _routing_registry("windows-x86_64").provider_for_model("auto") == "audiocpp-anywhere"
    assert _routing_registry("linux-x86_64").provider_for_model("tts-1") == "audiocpp-anywhere"


def test_platform_unsupported_model_raises_with_auto_hint():
    registry = _routing_registry("windows-x86_64")
    with pytest.raises(KeyError) as excinfo:
        registry.provider_for_model("mlx-model")
    message = str(excinfo.value)
    assert "not supported on this machine" in message
    assert "audiocpp-anywhere" in message


@pytest.mark.asyncio
async def test_auto_alias_resolves_to_provider_default_model():
    registry = _routing_registry("windows-x86_64")
    await registry.synthesize({"model": "auto", "input": "hello", "disable_microbatch": True})
    provider = registry.providers["audiocpp-anywhere"]
    assert provider.requests[0].model == "audiocpp-model"


def test_routing_info_reports_platform_and_recommendation():
    info = _routing_registry("windows-x86_64").routing_info()
    assert info["platform"] == "windows-x86_64"
    assert info["recommended"] == {"provider": "audiocpp-anywhere", "model": "audiocpp-model"}
    assert info["providers"]["mlx-fast"]["supported"] is False
    assert info["providers"]["audiocpp-anywhere"]["supported"] is True


from universal_tts.config import _resolve_command
from universal_tts.provisioning import EngineProvisioner


def _provisioned_cfg(tmp_path, auto=True) -> ProviderConfig:
    return ProviderConfig(
        id="audiocpp-fake", kind="fake", models=["audiocpp-model"], estimate_gb=1,
        cwd=str(tmp_path),
        options={
            "platforms": ["any"],
            "provision": {
                "auto": auto,
                "engine": {
                    "check_paths": {"any": "bin/engine"},
                    "build": {"any": "mkdir -p bin && touch bin/engine"},
                },
                "models": [{
                    "repo_id": "fake/model",
                    "target_dir": "models/fake",
                    "required_files": ["config.json"],
                }],
            },
        },
    )


def test_provision_state_detects_missing_engine_and_models(tmp_path):
    provisioner = EngineProvisioner(machine_platform="windows-x86_64", log_dir=tmp_path / "logs")
    cfg = _provisioned_cfg(tmp_path)
    state = provisioner.state(cfg)
    assert state["provisioned"] is False
    assert state["engine_ok"] is False and state["models_ok"] is False
    assert len(state["missing"]) == 2

    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "engine").touch()
    (tmp_path / "models" / "fake").mkdir(parents=True)
    (tmp_path / "models" / "fake" / "config.json").touch()
    assert provisioner.state(cfg)["provisioned"] is True


@pytest.mark.asyncio
async def test_provisioner_builds_missing_engine(tmp_path):
    provisioner = EngineProvisioner(machine_platform="linux-x86_64", log_dir=tmp_path / "logs")
    cfg = _provisioned_cfg(tmp_path)
    # Model files already present, engine missing: only the build step should run.
    (tmp_path / "models" / "fake").mkdir(parents=True)
    (tmp_path / "models" / "fake" / "config.json").touch()

    provisioner.start(cfg)
    await provisioner._tasks[cfg.id]

    assert (tmp_path / "bin" / "engine").exists()
    state = provisioner.state(cfg)
    assert state["provisioned"] is True
    assert state["status"]["status"] == "done"


@pytest.mark.asyncio
async def test_load_kicks_auto_provisioning_instead_of_failing(tmp_path):
    cfg = _provisioned_cfg(tmp_path)
    (tmp_path / "models" / "fake").mkdir(parents=True)
    (tmp_path / "models" / "fake" / "config.json").touch()
    runtime = RuntimeConfig(server={}, memory={"reserve_gb": 0}, providers={cfg.id: cfg})
    registry = ProviderRegistry(
        runtime, provider_factories={"fake": FakeProvider},
        memory_snapshot=lambda: MemorySnapshot(128, 100, 100),
        machine_platform="windows-x86_64",
    )
    registry.provisioner.log_dir = tmp_path / "logs"

    status = await registry.load(cfg.id)

    assert status.loaded is False
    assert status.details["provisioning"]["provisioned"] is False
    await registry.provisioner._tasks[cfg.id]
    assert registry.provisioner.state(cfg)["provisioned"] is True
    # Next load proceeds normally now that the engine exists.
    status = await registry.load(cfg.id)
    assert status.loaded is True


def test_command_by_platform_prefers_machine_key_and_falls_back_to_any():
    from universal_tts.platforms import current_platform
    machine_os = current_platform().split("-")[0]
    item = {"command": "generic", "command_by_platform": {machine_os: "os-specific", "any": "portable"}}
    assert _resolve_command(item) == "os-specific"
    item = {"command": "generic", "command_by_platform": {"solaris": "weird", "any": "portable"}}
    assert _resolve_command(item) == "portable"
    assert _resolve_command({"command": "generic"}) == "generic"
