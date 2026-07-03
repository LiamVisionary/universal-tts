"""Universal streaming stall-watchdog tests.

Every http-backed provider streams through HttpBackedProvider.stream_synthesize /
upstream_chunks, so these tests (against a raw hanging HTTP stub) prove the
watchdog protects ANY voice model: a provider that never produces a first byte,
or stalls mid-stream, is aborted with an error (never an indefinite hang), the
stall-abort counter increments, and normal / 5xx upstreams behave correctly.
"""
import asyncio

import pytest

from universal_tts.config import ProviderConfig
from universal_tts.providers.base import TTSRequest
from universal_tts.providers.http_backed import HttpBackedProvider


async def _serve(mode: str):
    """Raw HTTP/1.1 stub on an ephemeral port. Returns (server, port)."""
    async def handle(reader, writer):
        try:
            await asyncio.wait_for(reader.read(65536), timeout=0.5)
        except Exception:
            pass
        try:
            if mode == "err":
                writer.write(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 4\r\n\r\nnope")
                await writer.drain()
                writer.close()
                return
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: audio/pcm\r\nConnection: close\r\n\r\n")
            await writer.drain()
            if mode in ("normal", "hang_mid"):
                writer.write(b"PCM0" * 64)  # one real frame
                await writer.drain()
            if mode == "normal":
                writer.close()
                return
            # hang_first / hang_mid: stop sending; keep the connection open so the
            # client blocks waiting for (more) bytes until its watchdog fires.
            await asyncio.sleep(2.0)
            writer.close()
        except Exception:
            pass

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, port


def _provider(port: int) -> HttpBackedProvider:
    cfg = ProviderConfig(
        id="stub", kind="http", models=["m"], estimate_gb=1,
        base_url=f"http://127.0.0.1:{port}",
        options={
            "stream_path": "/v1/audio/speech-stream",
            "stream_content_type": "audio/pcm",
            # Read small pieces so buffered frames surface as they arrive (like a
            # real sidecar emitting small PCM frames), letting the mid-stream gap
            # be observed rather than hidden behind a 4096-byte read buffer.
            "stream_read_bytes": 64,
        },
    )
    p = HttpBackedProvider(cfg)

    async def _no_restart(reason, force=False):  # never touch a real sidecar in tests
        p.last_restart_reason = reason
        p.restart_count += 1

    p._restart_sidecar = _no_restart
    return p


def _req(**opts) -> TTSRequest:
    return TTSRequest(model="m", text="hi", response_format="pcm", options=opts)


@pytest.mark.asyncio
async def test_watchdog_aborts_before_first_byte():
    server, port = await _serve("hang_first")
    try:
        p = _provider(port)
        chunks, _ct = await p.stream_synthesize(
            _req(stream_first_byte_timeout_sec=0.5, stream_no_bytes_timeout_sec=0.5)
        )
        with pytest.raises(RuntimeError, match="before first byte"):
            async for _chunk in chunks:
                pass
        assert p.stream_abort_count == 1
        assert p.last_stream_abort_at is not None
        assert p.restart_count == 1  # watchdog force-restarts the stuck sidecar
    finally:
        server.close()


@pytest.mark.asyncio
async def test_watchdog_aborts_mid_stream():
    server, port = await _serve("hang_mid")
    try:
        p = _provider(port)
        chunks, _ct = await p.stream_synthesize(
            _req(stream_first_byte_timeout_sec=5, stream_no_bytes_timeout_sec=0.5)
        )
        got = b""
        with pytest.raises(RuntimeError, match="mid-stream"):
            async for chunk in chunks:
                got += chunk
        assert got == b"PCM0" * 64  # first frame delivered before the stall
        assert p.stream_abort_count == 1
    finally:
        server.close()


@pytest.mark.asyncio
async def test_normal_stream_not_aborted():
    server, port = await _serve("normal")
    try:
        p = _provider(port)
        chunks, _ct = await p.stream_synthesize(
            _req(stream_first_byte_timeout_sec=5, stream_no_bytes_timeout_sec=5)
        )
        got = b""
        async for chunk in chunks:
            got += chunk
        assert got == b"PCM0" * 64
        assert p.stream_abort_count == 0
        assert p.restart_count == 0
    finally:
        server.close()


@pytest.mark.asyncio
async def test_upstream_5xx_raises_without_abort_counter():
    server, port = await _serve("err")
    try:
        p = _provider(port)
        chunks, _ct = await p.stream_synthesize(_req())
        with pytest.raises(RuntimeError, match="streaming failed: 503"):
            async for _chunk in chunks:
                pass
        assert p.stream_abort_count == 0  # a 5xx isn't a stall-abort
    finally:
        server.close()
