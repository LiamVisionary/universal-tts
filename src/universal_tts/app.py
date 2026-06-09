from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel

from universal_tts.config import load_config
from universal_tts.memory import get_memory_snapshot
from universal_tts.queue import JobQueue
from universal_tts.registry import ProviderRegistry

CONFIG_PATH = Path(os.environ.get("UNIVERSAL_TTS_CONFIG", Path(__file__).resolve().parents[2] / "config.yaml"))


def build_registry() -> ProviderRegistry:
    return ProviderRegistry(load_config(CONFIG_PATH))


runtime = load_config(CONFIG_PATH)
registry = ProviderRegistry(runtime)
job_queue = JobQueue(registry)
app = FastAPI(title="Universal TTS", version="0.1.0")


async def _pcm_playback_chunks(
    chunks: AsyncIterator[bytes],
    *,
    sample_rate: int = 24000,
    channels: int = 1,
    sample_width: int = 2,
    frame_ms: int = 20,
    realtime_pacing: bool = True,
    declick_ms: float = 0.0,
    declick_threshold: int = 300,
) -> AsyncIterator[bytes]:
    """Coalesce and optionally pace raw PCM for browser/call playback.

    Provider sidecars often emit very small PCM chunks or generate faster than
    realtime. Some browser clients crackle when fed 5 ms fragments or a burst of
    many seconds of audio at once. This keeps TTFB low while yielding stable
    ~20 ms PCM frames and throttling to the audio clock when requested.

    Some decoder-streaming backends also have a small DC/sample discontinuity at
    model chunk boundaries. For 16-bit PCM, ``declick_ms`` optionally corrects
    only the start of each new upstream chunk by adding a decaying offset from
    the last sample of the previous chunk. It preserves duration and avoids
    crossfading away real audio.
    """
    bytes_per_second = sample_rate * channels * sample_width
    frame_bytes = max(sample_width * channels, int(bytes_per_second * frame_ms / 1000))
    # Keep frames sample-aligned.
    frame_bytes -= frame_bytes % (sample_width * channels)
    frame_bytes = max(sample_width * channels, frame_bytes)
    declick_samples = max(0, int(sample_rate * declick_ms / 1000))

    buf = bytearray()
    sent = 0
    start: float | None = None
    last_sample: int | None = None

    def maybe_declick(raw: bytes) -> bytes:
        nonlocal last_sample
        if sample_width != 2 or channels != 1 or len(raw) < 2:
            return raw
        data = bytearray(raw)
        first = int.from_bytes(data[0:2], "little", signed=True)
        if last_sample is not None and declick_samples > 0:
            offset = last_sample - first
            if abs(offset) >= declick_threshold:
                n = min(declick_samples, len(data) // 2)
                for i in range(n):
                    j = i * 2
                    sample = int.from_bytes(data[j:j + 2], "little", signed=True)
                    correction = int(round(offset * (1.0 - (i / n))))
                    sample = max(-32768, min(32767, sample + correction))
                    data[j:j + 2] = int(sample).to_bytes(2, "little", signed=True)
        last_sample = int.from_bytes(data[-2:], "little", signed=True)
        return bytes(data)

    async for chunk in chunks:
        if not chunk:
            continue
        buf.extend(maybe_declick(chunk))
        while len(buf) >= frame_bytes:
            out = bytes(buf[:frame_bytes])
            del buf[:frame_bytes]
            if start is None:
                start = time.perf_counter()
            yield out
            sent += len(out)
            if realtime_pacing and start is not None:
                target_elapsed = sent / bytes_per_second
                delay = target_elapsed - (time.perf_counter() - start)
                if delay > 0:
                    await asyncio.sleep(delay)

    if buf:
        if start is None:
            start = time.perf_counter()
        out = bytes(buf)
        yield out
        sent += len(out)
        if realtime_pacing:
            target_elapsed = sent / bytes_per_second
            delay = target_elapsed - (time.perf_counter() - start)
            if delay > 0:
                await asyncio.sleep(delay)


class LoadRequest(BaseModel):
    mode: str = "multi"
    force: bool = False


@app.get("/health")
async def health():
    return {
        "ok": True,
        "service": "universal-tts",
        "providers": list(runtime.providers),
        "memory": get_memory_snapshot().__dict__,
    }


@app.get("/memory")
async def memory():
    return get_memory_snapshot().__dict__


@app.get("/providers")
@app.get("/runtimes")
async def providers():
    statuses = await registry.statuses()
    return {
        "ok": True,
        "providers": {
            provider_id: {
                "id": provider_id,
                "kind": runtime.providers[provider_id].kind,
                "models": runtime.providers[provider_id].models,
                "estimate_gb": runtime.providers[provider_id].estimate_gb,
                "loaded": status.loaded,
                "healthy": status.healthy,
                "details": status.details,
                "notes": runtime.providers[provider_id].notes,
            }
            for provider_id, status in statuses.items()
        },
        "memory": get_memory_snapshot().__dict__,
    }


@app.post("/providers/{provider_id}/load")
@app.post("/runtimes/{provider_id}/load")
async def load_provider(provider_id: str, req: LoadRequest):
    try:
        status = await registry.load(provider_id, mode=req.mode, force=req.force)
        return {"ok": True, "provider": provider_id, "status": status.__dict__, "memory": get_memory_snapshot().__dict__}
    except KeyError as e:
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/providers/{provider_id}/unload")
@app.post("/runtimes/{provider_id}/unload")
async def unload_provider(provider_id: str):
    try:
        status = await registry.unload(provider_id)
        return {"ok": True, "provider": provider_id, "status": status.__dict__, "memory": get_memory_snapshot().__dict__}
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.get("/v1/models")
async def v1_models():
    statuses = await registry.statuses()
    data = []
    for provider_id, provider in runtime.providers.items():
        for model in provider.models:
            data.append({
                "id": model,
                "object": "model",
                "owned_by": "universal-tts",
                "provider": provider_id,
                "loaded": statuses[provider_id].loaded,
            })
    return {"object": "list", "data": data}


@app.get("/capabilities")
@app.get("/v1/audio/capabilities")
async def capabilities():
    return await registry.capabilities()


@app.get("/providers/{provider_id}/voices")
@app.get("/runtimes/{provider_id}/voices")
@app.get("/v1/audio/{provider_id}/voices")
async def provider_voices(provider_id: str):
    try:
        return await registry.voices(provider_id)
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.get("/providers/{provider_id}/paralinguistics")
@app.get("/runtimes/{provider_id}/paralinguistics")
@app.get("/v1/audio/{provider_id}/paralinguistics")
async def provider_paralinguistics(provider_id: str):
    try:
        return await registry.paralinguistics(provider_id)
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.get("/v1/audio/paralinguistics")
async def default_paralinguistics(provider: str | None = None, model: str | None = None):
    try:
        provider_id = provider or (registry.provider_for_model(model) if model else "chatterbox-turbo")
        return await registry.paralinguistics(provider_id)
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.get("/voices")
@app.get("/v1/voices")
async def default_voices():
    provider_id = await registry.first_loaded_voice_provider()
    if provider_id is None:
        raise HTTPException(503, "no loaded TTS provider has a voices endpoint; load a provider first")
    return RedirectResponse(url=f"/providers/{provider_id}/voices", status_code=307)


@app.post("/v1/audio/batches")
async def audio_batch(request: Request):
    try:
        payload = await request.json()
        items = payload.get("items") or payload.get("requests")
        if not isinstance(items, list):
            raise ValueError("items must be a list of speech request payloads")
        results = await registry.batch_synthesize(items)
        return {
            "object": "audio.batch",
            "data": [
                {"index": index, "content_type": content_type, "bytes": len(audio)}
                for index, (audio, content_type) in enumerate(results)
            ],
        }
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.post("/v1/audio/jobs", status_code=202)
async def create_audio_job(request: Request):
    try:
        payload = await request.json()
        job = await job_queue.submit(payload)
        return job.public()
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/v1/audio/jobs/{job_id}")
async def get_audio_job(job_id: str):
    try:
        return job_queue.get(job_id).public(include_audio=True)
    except KeyError:
        raise HTTPException(404, "unknown job")


@app.delete("/v1/audio/jobs/{job_id}")
@app.post("/v1/audio/jobs/{job_id}/cancel")
async def cancel_audio_job(job_id: str):
    try:
        job = await job_queue.cancel(job_id)
        return job.public(include_audio=True)
    except KeyError:
        raise HTTPException(404, "unknown job")


@app.get("/v1/audio/jobs/{job_id}/content")
async def get_audio_job_content(job_id: str):
    try:
        job = job_queue.get(job_id)
    except KeyError:
        raise HTTPException(404, "unknown job")
    if job.status != "completed" or job.audio is None:
        raise HTTPException(409, f"job is {job.status}")
    return Response(content=job.audio, media_type=job.content_type or "audio/wav")


@app.post("/v1/audio/speech")

async def speech(request: Request):
    try:
        payload = await request.json()
        audio, content_type = await registry.synthesize(payload)
        return Response(content=audio, media_type=content_type)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/v1/audio/speech-stream")
async def speech_stream(request: Request):
    try:
        payload = await request.json()
        delivery_mode = str(
            payload.get("generation_mode")
            or payload.get("delivery_mode")
            or payload.get("chatterbox_delivery_mode")
            or ""
        ).lower().replace("-", "_")
        if delivery_mode in {"full_generate", "full_generation", "quality", "quality_reference"}:
            raise HTTPException(409, "full_generate is not streaming; use /v1/audio/speech")
        stream_result = await registry.stream_synthesize(payload)
        if len(stream_result) == 2:
            chunks, content_type = stream_result
            stream_headers = {}
        else:
            chunks, content_type, stream_headers = stream_result
        headers = {"X-Universal-TTS-Streaming": "true", **stream_headers}
        if content_type.startswith("audio/pcm"):
            pacing_value = payload.get("realtime_pacing", True)
            if isinstance(pacing_value, str):
                realtime_pacing = pacing_value.lower() not in {"0", "false", "no", "off"}
            else:
                realtime_pacing = bool(pacing_value)
            frame_ms = int(payload.get("stream_frame_ms", payload.get("frame_ms", 20)))
            sample_rate = int(stream_headers.get("X-Audio-Sample-Rate", payload.get("sample_rate", 24000)))
            channels = int(stream_headers.get("X-Audio-Channels", 1))
            # Keep Universal de-click opt-in only. The adapter may read upstream
            # PCM in transport-sized pieces (for example 5 ms), which are not
            # necessarily model chunk boundaries; smoothing every HTTP read can
            # introduce new artifacts. Provider sidecars should smooth real
            # decoder chunk joins before Universal coalesces/paces frames.
            declick_ms = float(payload.get("pcm_declick_ms", 0.0))
            declick_threshold = int(payload.get("pcm_declick_threshold", 300))
            chunks = _pcm_playback_chunks(
                chunks,
                sample_rate=sample_rate,
                channels=channels,
                frame_ms=frame_ms,
                realtime_pacing=realtime_pacing,
                declick_ms=declick_ms,
                declick_threshold=declick_threshold,
            )
            headers.update({
                "X-Universal-TTS-PCM-Frame-MS": str(frame_ms),
                "X-Universal-TTS-Realtime-Pacing": str(realtime_pacing).lower(),
                "X-Universal-TTS-PCM-Declick-MS": str(declick_ms),
            })
        return StreamingResponse(chunks, media_type=content_type, headers=headers)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.api_route("/proxy/{provider_id}/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def proxy(provider_id: str, path: str, request: Request):
    if provider_id not in runtime.providers:
        raise HTTPException(404, "unknown provider")
    cfg = runtime.providers[provider_id]
    if not cfg.base_url:
        raise HTTPException(400, "provider has no base_url")
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in {"host", "content-length"}}
    # Privacy: do not log request bodies or prompts.
    async with httpx.AsyncClient(timeout=None) as client:
        upstream = await client.request(request.method, cfg.base_url.rstrip("/") + "/" + path, params=request.query_params, content=body, headers=headers)
    excluded = {"content-encoding", "transfer-encoding", "connection"}
    resp_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in excluded}
    return Response(content=upstream.content, status_code=upstream.status_code, headers=resp_headers, media_type=upstream.headers.get("content-type"))


@app.post("/load-profile/voice-all")
async def voice_all(force: bool = False):
    results = {}
    for provider_id in runtime.providers:
        try:
            status = await registry.load(provider_id, mode="multi", force=force)
            results[provider_id] = status.__dict__
        except Exception as e:
            results[provider_id] = {"error": str(e)}
    return {"ok": True, "results": results, "memory": get_memory_snapshot().__dict__}
