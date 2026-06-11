"""Standalone Fish Audio S2 Pro MLX sidecar.

Runs the MLX-Audio conversion of Fish Audio S2 Pro in a dedicated
`.fish-s2-venv`. The official Fish S2 Pro SGLang/vLLM path supports native
low-latency serving on large NVIDIA GPUs. On Apple Silicon, mlx-audio currently
exposes a generator that yields completed audio per text chunk and explicitly
raises NotImplementedError for decoder-frame `stream=True`; this sidecar
therefore provides segment-incremental PCM streaming, not full-WAV chunking.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import os
import queue as thread_queue
import threading
import time
import wave
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

DEFAULT_MODEL = os.environ.get("FISH_S2_MODEL", "mlx-community/fish-audio-s2-pro-8bit")
DEFAULT_VOICE = os.environ.get("FISH_S2_VOICE", "default")
DEFAULT_LANG = os.environ.get("FISH_S2_LANG", "en")
SAMPLE_WIDTH = 2
CHANNELS = 1
DEFAULT_MAX_TOKENS = int(os.environ.get("FISH_S2_MAX_TOKENS", "1024"))
DEFAULT_CHUNK_LENGTH = int(os.environ.get("FISH_S2_CHUNK_LENGTH", "80"))

app = FastAPI(title="Universal TTS — Fish S2 Pro MLX sidecar", version="0.1.0")


@dataclass
class ModelEntry:
    model: Any
    load_seconds: float
    model_name: str
    sample_rate: int


_model_lock = threading.Lock()
_model_entry: ModelEntry | None = None
_first_load_error: str | None = None
_work_queue: thread_queue.Queue = thread_queue.Queue()
_worker_started = threading.Event()


def _ensure_model() -> ModelEntry:
    global _model_entry, _first_load_error
    if _model_entry is not None:
        return _model_entry
    with _model_lock:
        if _model_entry is not None:
            return _model_entry
        try:
            from mlx_audio.tts.utils import load_model

            t0 = time.perf_counter()
            model = load_model(DEFAULT_MODEL)
            entry = ModelEntry(
                model=model,
                load_seconds=time.perf_counter() - t0,
                model_name=DEFAULT_MODEL,
                sample_rate=int(getattr(model, "sample_rate", 44100)),
            )
            _model_entry = entry
            return entry
        except Exception as e:  # noqa: BLE001
            _first_load_error = f"{type(e).__name__}: {e}"
            raise


def _worker_loop() -> None:
    """Own the MLX model and run all MLX work on one thread.

    MLX streams are thread-local. Loading the model on one thread and
    generating on another raises: "There is no Stream(gpu, 0) in current
    thread." Keep load, reference preprocessing, and generation on this
    dedicated worker thread.
    """
    _worker_started.set()
    try:
        _ensure_model()
    except Exception:
        pass
    while True:
        fn, fut = _work_queue.get()
        try:
            result = fn()
            if fut is not None and not fut.done():
                fut.set_result(result)
        except Exception as e:  # noqa: BLE001
            if fut is not None and not fut.done():
                fut.set_exception(e)
        finally:
            _work_queue.task_done()


def _start_worker() -> None:
    if _worker_started.is_set():
        return
    threading.Thread(target=_worker_loop, name="fish-s2-mlx-worker", daemon=True).start()
    _worker_started.wait(timeout=5)


def _run_on_worker(fn):
    _start_worker()
    fut: Future = Future()
    _work_queue.put((fn, fut))
    return fut.result()


def _submit_to_worker(fn) -> None:
    _start_worker()
    _work_queue.put((fn, None))


def _to_numpy(audio: Any) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32).squeeze()
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return arr


def _to_int16(audio: Any) -> np.ndarray:
    arr = _to_numpy(audio)
    clipped = np.clip(arr, -1.0, 1.0)
    return np.where(clipped <= -1.0, -32768, np.round(clipped * 32767.0)).astype(np.int16)


def _pcm_bytes(audio: Any) -> bytes:
    return _to_int16(audio).tobytes()


def _wav_bytes(audio: Any, sample_rate: int) -> bytes:
    pcm = _to_int16(audio)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _load_ref_audio(ref_audio: Any, sample_rate: int) -> Any:
    if not ref_audio:
        return None
    if isinstance(ref_audio, list):
        if len(ref_audio) != 1:
            raise ValueError("Fish S2 sidecar currently accepts one ref_audio path per request")
        ref_audio = ref_audio[0]
    if isinstance(ref_audio, str):
        if not os.path.exists(ref_audio):
            raise ValueError(f"ref_audio file not found: {ref_audio}")
        from mlx_audio.tts.generate import load_audio

        return load_audio(ref_audio, sample_rate=sample_rate, volume_normalize=False)
    return ref_audio


def _collapse_ref_text(ref_text: Any) -> str | None:
    if ref_text is None:
        return None
    if isinstance(ref_text, list):
        if len(ref_text) != 1:
            raise ValueError("Fish S2 sidecar currently accepts one ref_text value per request")
        ref_text = ref_text[0]
    return str(ref_text)


def _payload_kwargs(payload: dict[str, Any], sample_rate: int) -> dict[str, Any]:
    text = payload.get("input") or payload.get("text") or ""
    if not str(text).strip():
        raise ValueError("input is required")
    ref_audio_raw = payload.get("ref_audio") or payload.get("reference_audio")
    ref_text = payload.get("ref_text") or payload.get("reference_text")
    ref_audio = _load_ref_audio(ref_audio_raw, sample_rate) if ref_audio_raw else None
    if ref_audio is not None and not ref_text:
        raise ValueError("ref_text is required when ref_audio is supplied")
    return {
        "text": str(text),
        "voice": payload.get("voice") or DEFAULT_VOICE,
        "ref_audio": ref_audio,
        "ref_text": _collapse_ref_text(ref_text),
        "instruct": payload.get("instruct"),
        "speed": float(payload.get("speed", 1.0)),
        "max_tokens": int(payload.get("max_tokens", DEFAULT_MAX_TOKENS)),
        "temperature": float(payload.get("temperature", 0.7)),
        "top_p": float(payload.get("top_p", 0.7)),
        "top_k": int(payload.get("top_k", 30)),
        "chunk_length": int(payload.get("chunk_length", payload.get("max_segment_chars", DEFAULT_CHUNK_LENGTH))),
        "verbose": bool(payload.get("verbose", False)),
    }


def _iter_results(entry: ModelEntry, kwargs: dict[str, Any]):
    yield from entry.model.generate(**kwargs)


@app.on_event("startup")
async def _startup_warm() -> None:
    _start_worker()


@app.get("/health", response_model=None)
async def health():
    if _model_entry is None:
        body = {
            "ok": False,
            "loaded": False,
            "model_name": DEFAULT_MODEL,
            "license": "Fish Audio Research License (research/non-commercial; commercial use requires separate license)",
            "sample_rate": 44100,
            "channels": CHANNELS,
            "sample_format": "pcm16",
            "supports_true_streaming": True,
            "streaming_kind": "pcm16",
            "streaming_mode": "segment-incremental-pcm",
            "streaming_implementation": "mlx-audio-fish-s2-generate-chunks",
            "voice_cloning": True,
            "hint": "warming up model",
        }
        if _first_load_error:
            body["error"] = _first_load_error
        return JSONResponse(body, status_code=503)
    return {
        "ok": True,
        "loaded": True,
        "model_name": _model_entry.model_name,
        "load_seconds": _model_entry.load_seconds,
        "sample_rate": _model_entry.sample_rate,
        "channels": CHANNELS,
        "sample_format": "pcm16",
        "license": "Fish Audio Research License (research/non-commercial; commercial use requires separate license)",
        "supports_true_streaming": True,
        "streaming_kind": "pcm16",
        "streaming_mode": "segment-incremental-pcm",
        "streaming_implementation": "mlx-audio-fish-s2-generate-chunks",
        "voice_cloning": True,
        "voice_cloning_requires": ["ref_audio", "ref_text"],
        "notes": "MLX-Audio Fish S2 Pro raises NotImplementedError for decoder-frame stream=True; this sidecar streams generated chunks as they are produced.",
    }


@app.get("/v1/voices")
@app.get("/v1/audio/voices")
async def voices() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {"id": "default", "provider": "fish-s2", "description": "Base/no-reference generation"},
            {"id": "clone", "provider": "fish-s2", "description": "Use ref_audio + ref_text for zero-shot clone"},
        ],
    }


@app.post("/v1/audio/speech")
async def speech(request: Request) -> Response:
    payload = await request.json()
    entry = _ensure_model()
    try:
        def generate_all():
            worker_entry = _ensure_model()
            kwargs = _payload_kwargs(payload, worker_entry.sample_rate)
            return worker_entry.sample_rate, [_to_numpy(result.audio) for result in _iter_results(worker_entry, kwargs)]

        sample_rate, chunks = _run_on_worker(generate_all)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"{type(e).__name__}: {e}") from e
    if not chunks:
        raise HTTPException(500, "no audio produced")
    full = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    if str(payload.get("response_format", "wav")).lower() == "pcm":
        return Response(content=_pcm_bytes(full), media_type="audio/pcm")
    return Response(content=_wav_bytes(full, sample_rate), media_type="audio/wav")


@app.post("/v1/audio/speech-stream")
async def speech_stream(request: Request) -> StreamingResponse:
    payload = await request.json()
    entry = _ensure_model()

    queue: asyncio.Queue = asyncio.Queue(maxsize=8)
    error_holder: dict[str, str] = {}
    loop = asyncio.get_running_loop()

    def producer_work() -> None:
        try:
            worker_entry = _ensure_model()
            kwargs = _payload_kwargs(payload, worker_entry.sample_rate)
            for result in _iter_results(worker_entry, kwargs):
                pcm = _pcm_bytes(result.audio)
                if pcm:
                    asyncio.run_coroutine_threadsafe(queue.put(pcm), loop).result()
        except Exception as e:  # noqa: BLE001
            error_holder["error"] = f"{type(e).__name__}: {e}"
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

    _submit_to_worker(producer_work)

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
            "X-Audio-Sample-Rate": str(entry.sample_rate),
            "X-Audio-Channels": str(CHANNELS),
            "X-Audio-Sample-Format": "pcm16",
            "X-Fish-S2-Model": entry.model_name,
            "X-Fish-S2-License": "research-noncommercial",
            "X-Fish-S2-Streaming-Mode": "segment-incremental-pcm",
            "X-Fish-S2-Streaming-Implementation": "mlx-audio-fish-s2-generate-chunks",
        },
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8784)
    p.add_argument("--model", default=DEFAULT_MODEL)
    args = p.parse_args()
    os.environ["FISH_S2_MODEL"] = args.model
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

