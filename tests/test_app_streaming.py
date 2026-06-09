from fastapi.testclient import TestClient

from universal_tts import app as app_module


def test_speech_stream_endpoint_returns_streaming_audio(monkeypatch):
    async def fake_stream_synthesize(payload):
        async def chunks():
            yield b"RIFF"
            yield b"stream"
        return chunks(), "audio/wav", {"X-Audio-Sample-Rate": "24000", "X-Audio-Channels": "1", "X-Audio-Sample-Format": "pcm16"}

    monkeypatch.setattr(app_module.registry, "stream_synthesize", fake_stream_synthesize)
    client = TestClient(app_module.app)

    response = client.post("/v1/audio/speech-stream", json={"model": "tts-1", "input": "hi", "response_format": "wav"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.headers["x-audio-sample-rate"] == "24000"
    assert response.headers["x-audio-channels"] == "1"
    assert response.headers["x-audio-sample-format"] == "pcm16"
    assert response.content == b"RIFFstream"


def test_full_generate_is_rejected_from_stream_endpoint(monkeypatch):
    async def fake_stream_synthesize(payload):
        raise AssertionError("full generation must not be routed through stream_synthesize")

    monkeypatch.setattr(app_module.registry, "stream_synthesize", fake_stream_synthesize)
    client = TestClient(app_module.app)

    response = client.post(
        "/v1/audio/speech-stream",
        json={"model": "chatterbox-turbo", "input": "hi", "generation_mode": "full_generate"},
    )

    assert response.status_code == 409
    assert "not streaming" in response.text


def test_provider_paralinguistics_endpoint(monkeypatch):
    async def fake_paralinguistics(provider_id):
        assert provider_id == "chatterbox-turbo"
        return {"object": "audio.paralinguistics", "provider": provider_id, "data": [{"token": "[laugh]"}]}

    monkeypatch.setattr(app_module.registry, "paralinguistics", fake_paralinguistics)
    client = TestClient(app_module.app)

    response = client.get("/providers/chatterbox-turbo/paralinguistics")

    assert response.status_code == 200
    assert response.json()["data"] == [{"token": "[laugh]"}]


def test_default_paralinguistics_endpoint_uses_chatterbox(monkeypatch):
    async def fake_paralinguistics(provider_id):
        assert provider_id == "chatterbox-turbo"
        return {"object": "audio.paralinguistics", "provider": provider_id, "data": [{"token": "[laugh]"}]}

    monkeypatch.setattr(app_module.registry, "paralinguistics", fake_paralinguistics)
    client = TestClient(app_module.app)

    response = client.get("/v1/audio/paralinguistics")

    assert response.status_code == 200
    assert response.json()["provider"] == "chatterbox-turbo"
