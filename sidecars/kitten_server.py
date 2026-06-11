"""Standalone KittenTTS sidecar HTTP server.

The actual KittenTTS runtime (torch + ONNX + misaki) lives in a dedicated
venv and is started by the sidecar launcher at ``sidecars/kitten_sidecar.sh``.
This is the FastAPI/uvicorn server the launcher execs.

Endpoints
---------
- ``GET  /health``             – liveness + model status
- ``GET  /v1/voices``          – static voice list
- ``GET  /v1/audio/voices``    – alias of /v1/voices
- ``POST /v1/audio/speech``    – synthesize full audio (WAV)
- ``POST /v1/audio/speech-stream`` – stream raw 24 kHz mono PCM chunks
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import threading
import time
import wave
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

SAMPLE_RATE = 24000
CHANNELS = 1
SAMPLE_WIDTH = 2
AVAILABLE_VOICES = ["Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo"]
DEFAULT_VOICE = "Bella"
DEFAULT_MODEL = os.environ.get("KITTEN_MODEL", "KittenML/kitten-tts-mini-0.8")

app = FastAPI(title="Universal TTS — KittenTTS sidecar", version="0.1.0")

_model_lock = threading.Lock()
_model: Any = None
_model_name: str | None = None
_loaded_at: float | None = None
_load_lock = threading.Lock()
_loaded_event = threading.Event()


def _ensure_loaded() -> Any:
    global _model, _model_name, _loaded_at
    if _model is not None:
        return _model
    with _load_lock:
        if _model is not None:
            return _model
        from kittentts import KittenTTS  # imported lazily

        t0 = time.perf_counter()
        m = KittenTTS(DEFAULT_MODEL)
        elapsed = time.perf_counter() - t0
        with _model_lock:
            _model = m
            _model_name = DEFAULT_MODEL
            _loaded_at = elapsed
        _loaded_event.set()
        return _model


def _to_int16(samples: np.ndarray) -> np.ndarray:
    """Convert KittenTTS float32 audio in roughly [-1, 1] to PCM int16.

    KittenTTS returns float32 numpy arrays with peak around 0.5-1.0. Casting
    those directly to int16 truncates every value to 0 (silence). Scale by
    32767 with clipping to int16 range.
    """
    arr = np.asarray(samples, dtype=np.float32).squeeze()
    if arr.ndim > 1:
        arr = arr.reshape(-1)  # flatten (samples, 1) -> (samples,)
    clipped = np.clip(arr, -1.0, 1.0)
    return np.round(clipped * 32767.0).astype(np.int16)


def _to_pcm_bytes(samples: np.ndarray) -> bytes:
    return _to_int16(samples).tobytes()


def _to_wav_bytes(samples: np.ndarray) -> bytes:
    pcm = _to_int16(samples)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _synth_kwargs(payload: dict) -> dict[str, Any]:
    voice = payload.get("voice") or DEFAULT_VOICE
    speed = float(payload.get("speed", 1.0))
    clean_text = bool(payload.get("clean_text", True))
    text = payload.get("input") or payload.get("text") or ""
    if not text:
        raise ValueError("input is required")
    return {"text": text, "voice": voice, "speed": speed, "clean_text": clean_text}


def _resolve_voice(voice: str | None) -> str:
    aliases = json.loads(os.environ.get("KITTEN_VOICE_ALIASES", "{}"))
    if voice and voice in aliases:
        return aliases[voice]
    return voice or DEFAULT_VOICE


@app.on_event("startup")
async def _startup_warm() -> None:
    # Preload on a background thread so first request doesn't pay the cost.
    threading.Thread(target=_ensure_loaded, name="kitten-warm", daemon=True).start()


@app.get("/health")
async def health():
    if _model is None:
        return JSONResponse(
            {
                "ok": False,
                "loaded": False,
                "model_name": DEFAULT_MODEL,
                "voices": AVAILABLE_VOICES,
                "supports_true_streaming": True,
                "streaming_kind": "pcm16",
                "sample_rate": SAMPLE_RATE,
                "channels": CHANNELS,
                "hint": "warming up model",
            },
            status_code=503,
        )
    return {
        "ok": True,
        "loaded": True,
        "model_name": _model_name,
        "load_seconds": _loaded_at,
        "voices": _model.available_voices,
        "supports_true_streaming": True,
        "streaming_kind": "pcm16",
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "sample_format": "pcm16",
    }


@app.get("/v1/voices")
@app.get("/v1/audio/voices")
async def voices() -> dict:
    return {"object": "list", "data": [{"id": v, "provider": "kitten"} for v in AVAILABLE_VOICES]}


@app.post("/v1/audio/speech")
async def speech(request: Request) -> Response:
    payload = await request.json()
    kwargs = _synth_kwargs(payload)
    kwargs["voice"] = _resolve_voice(kwargs["voice"])
    model = _ensure_loaded()
    chunks: list[np.ndarray] = []
    for chunk in model.generate_stream(**kwargs):
        chunks.append(np.asarray(chunk).squeeze())
    if not chunks:
        raise HTTPException(500, "no audio produced")
    full = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    return Response(content=_to_wav_bytes(full), media_type="audio/wav")


@app.post("/v1/audio/speech-stream")
async def speech_stream(request: Request) -> StreamingResponse:
    payload = await request.json()
    kwargs = _synth_kwargs(payload)
    kwargs["voice"] = _resolve_voice(kwargs["voice"])
    model = _ensure_loaded()
    queue: asyncio.Queue = asyncio.Queue(maxsize=64)
    error_holder: dict[str, Any] = {}

    loop = asyncio.get_running_loop()

    def producer() -> None:
        try:
            for chunk in model.generate_stream(**kwargs):
                pcm = _to_pcm_bytes(chunk)
                if pcm:
                    asyncio.run_coroutine_threadsafe(queue.put(pcm), loop).result()
        except Exception as e:  # noqa: BLE001
            error_holder["error"] = f"{type(e).__name__}: {e}"
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

    threading.Thread(target=producer, name="kitten-stream", daemon=True).start()

    async def iterator():
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item
        if error_holder.get("error"):
            raise HTTPException(500, error_holder["error"])

    return StreamingResponse(
        iterator(),
        media_type="audio/pcm",
        headers={
            "X-Audio-Sample-Rate": str(SAMPLE_RATE),
            "X-Audio-Channels": str(CHANNELS),
            "X-Audio-Sample-Format": "pcm16",
            "X-Kitten-Model": _model_name or DEFAULT_MODEL,
        },
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8782)
    p.add_argument("--model", default=DEFAULT_MODEL)
    args = p.parse_args()
    os.environ["KITTEN_MODEL"] = args.model
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
