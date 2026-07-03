import pytest

from universal_tts.config import ProviderConfig
from universal_tts.lifecycle import ProcessLifecycle
from universal_tts.providers.base import ProviderStatus
from universal_tts.providers.http_backed import HttpBackedProvider


@pytest.mark.asyncio
async def test_http_provider_load_exposes_restart_metadata_and_starts_monitor(monkeypatch):
    cfg = ProviderConfig(
        id="p",
        kind="http",
        models=["m"],
        estimate_gb=1,
        base_url="http://127.0.0.1:1",
        command="fake",
        port=12345,
    )
    provider = HttpBackedProvider(cfg)
    calls = {"start": 0, "monitor": 0}

    def fake_start(**kwargs):
        calls["start"] += 1
        provider.lifecycle.last_start_result = {"method": "command", "adopted": True, "pids": ["111"]}
        return provider.lifecycle.last_start_result

    async def fake_wait(timeout, **kwargs):
        return ProviderStatus(id="p", loaded=True, healthy=True, details=provider.metadata())

    def fake_monitor():
        calls["monitor"] += 1

    monkeypatch.setattr(provider.lifecycle, "start", fake_start)
    monkeypatch.setattr(provider, "_wait_until_loaded", fake_wait)
    monkeypatch.setattr(provider, "_start_monitor", fake_monitor)

    status = await provider.load()

    assert calls == {"start": 1, "monitor": 1}
    assert status.details["restart_count"] == 0
    assert status.details["lifecycle"]["adopted"] is True


@pytest.mark.asyncio
async def test_http_provider_restart_records_count_and_reason(monkeypatch):
    cfg = ProviderConfig(id="p", kind="http", models=["m"], estimate_gb=1, base_url="http://127.0.0.1:1")
    provider = HttpBackedProvider(cfg)
    calls = {"kill": 0, "start": 0}

    def fake_kill():
        calls["kill"] += 1
        return {"killed": ["111"]}

    def fake_start(**kwargs):
        calls["start"] += 1
        return {"pid": 222}

    async def fake_wait(timeout, **kwargs):
        return ProviderStatus(id="p", loaded=True, healthy=True, details={})

    async def no_sleep(_):
        return None

    monkeypatch.setattr(provider.lifecycle, "kill", fake_kill)
    monkeypatch.setattr(provider.lifecycle, "start", fake_start)
    monkeypatch.setattr(provider, "_wait_until_loaded", fake_wait)
    monkeypatch.setattr("universal_tts.providers.http_backed.asyncio.sleep", no_sleep)

    await provider._restart_sidecar("stream watchdog: no bytes", force=True)

    assert calls == {"kill": 1, "start": 1}
    assert provider.restart_count == 1
    assert provider.last_restart_at is not None
    assert provider.last_restart_reason == "stream watchdog: no bytes"


def test_process_lifecycle_adopts_existing_port_listener(monkeypatch, tmp_path):
    monkeypatch.setattr("universal_tts.lifecycle.port_pids", lambda port: [123] if port == 8766 else [])
    lifecycle = ProcessLifecycle(provider_id="qwen3", command="fake", cwd=str(tmp_path), port=8766, log_dir=tmp_path)

    result = lifecycle.start()

    assert result == {"method": "command", "adopted": True, "pids": ["123"]}
    assert lifecycle.pid == 123
    assert lifecycle.adopted_pids == {123}


def test_process_lifecycle_replace_kills_existing_listener(monkeypatch, tmp_path):
    calls = {"terminate": 0, "popen": 0}
    port_checks = iter([[123], [], []])
    monkeypatch.setattr("universal_tts.lifecycle.port_pids", lambda port: next(port_checks, []))

    class FakeProc:
        pid = 456

        def poll(self):
            return None

    def fake_popen(*args, **kwargs):
        calls["popen"] += 1
        return FakeProc()

    lifecycle = ProcessLifecycle(provider_id="qwen3", command="fake", cwd=str(tmp_path), port=8766, log_dir=tmp_path)
    monkeypatch.setattr(lifecycle, "terminate", lambda **kwargs: calls.__setitem__("terminate", calls["terminate"] + 1) or {"killed": ["123"]})
    monkeypatch.setattr("universal_tts.lifecycle.subprocess.Popen", fake_popen)

    result = lifecycle.start(port_conflict_policy="replace")

    assert calls == {"terminate": 1, "popen": 1}
    assert result["pid"] == 456
