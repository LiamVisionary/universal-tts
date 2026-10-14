"""A page on hivemindos.app may list voices and speak through this engine, and nothing else."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from universal_tts.web_access import install_web_access

SITE = "https://hivemindos.app"


def _client() -> TestClient:
    app = FastAPI()

    @app.get("/v1/models")
    def models():
        return {"data": [{"id": "kokoro", "provider": "kokoro"}]}

    @app.post("/v1/audio/speech")
    def speech():
        return {"ok": True}

    @app.get("/v1/audio/history")
    def history():
        return {"data": ["an earlier take"]}

    install_web_access(app)
    return TestClient(app)


def _preflight(client: TestClient, path: str, method: str, origin: str = SITE):
    return client.options(path, headers={
        "Origin": origin,
        "Access-Control-Request-Method": method,
        "Access-Control-Request-Headers": "content-type",
        "Access-Control-Request-Private-Network": "true",
    })


def test_the_site_may_speak_and_list_models():
    client = _client()
    preflight = _preflight(client, "/v1/audio/speech", "POST")
    assert preflight.status_code == 204
    assert preflight.headers["access-control-allow-origin"] == SITE
    assert preflight.headers["access-control-allow-private-network"] == "true"
    listed = client.get("/v1/models", headers={"Origin": SITE})
    assert listed.headers["access-control-allow-origin"] == SITE


def test_history_and_other_sites_stay_closed():
    client = _client()
    assert "access-control-allow-origin" not in client.get("/v1/audio/history", headers={"Origin": SITE}).headers
    assert _preflight(client, "/v1/audio/history", "GET").status_code == 403
    assert _preflight(client, "/v1/audio/speech", "POST", origin="https://evil.example").status_code == 403
    assert "access-control-allow-origin" not in client.get("/v1/models", headers={"Origin": "https://evil.example"}).headers


def test_a_page_on_this_computer_is_allowed(monkeypatch):
    monkeypatch.setenv("UNIVERSAL_TTS_WEB_ORIGINS", "https://studio.example.test")
    client = _client()
    assert _preflight(client, "/v1/audio/speech", "POST", origin="http://localhost:5273").status_code == 204
    assert _preflight(client, "/v1/audio/speech", "POST", origin="https://studio.example.test").status_code == 204
    # Overriding the list replaces the default site rather than adding to it.
    assert _preflight(client, "/v1/audio/speech", "POST").status_code == 403
