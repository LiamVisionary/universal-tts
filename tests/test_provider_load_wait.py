import pytest

from universal_tts.config import ProviderConfig
from universal_tts.providers.base import ProviderStatus
from universal_tts.providers.http_backed import HttpBackedProvider


@pytest.mark.asyncio
async def test_http_backed_provider_load_waits_until_status_healthy(monkeypatch):
    cfg = ProviderConfig(id="p", kind="http", models=["m"], estimate_gb=1, base_url="http://127.0.0.1:1")
    p = HttpBackedProvider(cfg)
    calls = {"n": 0, "started": False}

    def fake_start():
        calls["started"] = True
        return {"ok": True}

    async def fake_status():
        calls["n"] += 1
        return ProviderStatus(id="p", loaded=calls["n"] >= 3, healthy=calls["n"] >= 3, details={})

    async def fake_sleep(_):
        return None

    monkeypatch.setattr(p.lifecycle, "start", fake_start)
    monkeypatch.setattr(p, "status", fake_status)
    monkeypatch.setattr("universal_tts.providers.http_backed.asyncio.sleep", fake_sleep)

    status = await p.load()

    assert calls["started"] is True
    assert calls["n"] == 3
    assert status.loaded is True
