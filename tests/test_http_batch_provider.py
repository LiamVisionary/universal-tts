import base64

import pytest

from universal_tts.config import ProviderConfig
from universal_tts.providers.base import TTSRequest
from universal_tts.providers.http_backed import HttpBackedProvider


class DummyResponse:
    status_code = 200
    text = "ok"

    def json(self):
        return {
            "object": "audio.batch",
            "data": [
                {"audio_base64": base64.b64encode(b"RIFFone").decode("ascii"), "content_type": "audio/wav"},
                {"audio": base64.b64encode(b"RIFFtwo").decode("ascii"), "encoding": "base64", "content_type": "audio/wav"},
            ],
        }


class DummyClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def post(self, url, json):
        assert url.endswith("/v1/audio/batches")
        assert len(json["items"]) == 2
        return DummyResponse()


@pytest.mark.asyncio
async def test_http_batch_path_decodes_base64_audio_items(monkeypatch):
    import universal_tts.providers.http_backed as http_backed

    monkeypatch.setattr(http_backed.httpx, "AsyncClient", DummyClient)
    provider = HttpBackedProvider(
        ProviderConfig(
            id="qwen3",
            kind="http",
            models=["qwen"],
            estimate_gb=1,
            base_url="http://example.test",
            options={"batch_path": "/v1/audio/batches"},
        )
    )

    results = await provider.batch_synthesize([
        TTSRequest(model="qwen", text="one"),
        TTSRequest(model="qwen", text="two"),
    ])

    assert results == [(b"RIFFone", "audio/wav"), (b"RIFFtwo", "audio/wav")]
