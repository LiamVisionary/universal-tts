"""Standalone Kokoro TTS sidecar HTTP server.

Kokoro-82M runs in a dedicated `.kokoro-venv` with Torch/Kokoro/Misaki deps.
The streaming endpoint emits raw 24 kHz mono PCM16 as each Kokoro pipeline
segment is produced. Kokoro's public API does not expose decoder-frame
callbacks, so this is true segment-incremental streaming: audio for earlier
text segments is emitted before later segments are synthesized, not a chunked
full-utterance WAV.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import re
import threading
import time
import wave
from dataclasses import dataclass
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

SAMPLE_RATE = 24000
CHANNELS = 1
SAMPLE_WIDTH = 2
DEFAULT_MODEL = os.environ.get("KOKORO_MODEL", "hexgrad/Kokoro-82M")
DEFAULT_VOICE = os.environ.get("KOKORO_VOICE", "af_heart")
DEFAULT_LANG = os.environ.get("KOKORO_LANG", "a")
DEFAULT_DEVICE = os.environ.get("KOKORO_DEVICE", "auto")
MAX_SEGMENT_CHARS = int(os.environ.get("KOKORO_MAX_SEGMENT_CHARS", "220"))

VOICES = [
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
    "am_onyx", "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    "ef_dora", "em_alex", "em_santa",
    "ff_siwis",
    "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
    "if_sara", "im_nicola",
    "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
    "pf_dora", "pm_alex", "pm_santa",
    "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
    "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
]

LANG_BY_PREFIX = {
    "af": "a", "am": "a",
    "bf": "b", "bm": "b",
    "ef": "e", "em": "e",
    "ff": "f",
    "hf": "h", "hm": "h",
    "if": "i", "im": "i",
    "jf": "j", "jm": "j",
    "pf": "p", "pm": "p",
    "zf": "z", "zm": "z",
}

VOICE_GRADES = {
    "af_heart": "A", "af_bella": "A-", "af_nicole": "B-", "bf_emma": "B-",
    "ff_siwis": "B-", "af_aoede": "C+", "af_kore": "C+", "af_sarah": "C+",
    "am_fenrir": "C+", "am_michael": "C+", "bm_fable": "C", "bm_george": "C",
}

app = FastAPI(title="Universal TTS — Kokoro sidecar", version="0.1.0")


@dataclass
class PipelineEntry:
    pipeline: Any
    load_seconds: float
    lang_code: str
    device: str


_pipelines: dict[str, PipelineEntry] = {}
_pipeline_lock = threading.Lock()
_loaded_event = threading.Event()
_first_load_error: str | None = None


def _pick_device() -> str | None:
    if DEFAULT_DEVICE and DEFAULT_DEVICE != "auto":
        return DEFAULT_DEVICE
    try:
        import torch
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _lang_for_voice(voice: str | None, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    voice = voice or DEFAULT_VOICE
    return LANG_BY_PREFIX.get(voice[:2], DEFAULT_LANG)


def _voice_aliases() -> dict[str, str]:
    try:
        return json.loads(os.environ.get("KOKORO_VOICE_ALIASES", "{}"))
    except Exception:
        return {}


def _resolve_voice(voice: str | None) -> str:
    aliases = _voice_aliases()
    if voice and voice in aliases:
        voice = aliases[voice]
    return voice or DEFAULT_VOICE


def _ensure_pipeline(lang_code: str) -> PipelineEntry:
    global _first_load_error
    if lang_code in _pipelines:
        return _pipelines[lang_code]
    with _pipeline_lock:
        if lang_code in _pipelines:
            return _pipelines[lang_code]
        try:
            from kokoro import KPipeline
            device = _pick_device()
            t0 = time.perf_counter()
            # repo_id points at the pre-downloaded Hugging Face snapshot cache.
            pipeline = KPipeline(lang_code=lang_code, repo_id=DEFAULT_MODEL, device=device)
            entry = PipelineEntry(
                pipeline=pipeline,
                load_seconds=time.perf_counter() - t0,
                lang_code=lang_code,
                device=device or "auto",
            )
            _pipelines[lang_code] = entry
            _loaded_event.set()
            return entry
        except Exception as e:  # noqa: BLE001
            _first_load_error = f"{type(e).__name__}: {e}"
            raise


def _to_int16(samples: Any) -> np.ndarray:
    if hasattr(samples, "detach"):
        samples = samples.detach().cpu().numpy()
    arr = np.asarray(samples, dtype=np.float32).squeeze()
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    clipped = np.clip(arr, -1.0, 1.0)
    # Use int16_min for exactly -1.0 and 32767 for +1.0.
    return np.where(clipped <= -1.0, -32768, np.round(clipped * 32767.0)).astype(np.int16)


def _pcm_bytes(samples: Any) -> bytes:
    return _to_int16(samples).tobytes()


def _wav_bytes(samples: Any) -> bytes:
    pcm = _to_int16(samples)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _split_realtime_segments(text: str, max_chars: int = MAX_SEGMENT_CHARS) -> list[str]:
    """Split input into small synthesis units for incremental streaming.

    Kokoro yields completed audio per pipeline segment. This splitter keeps
    segments short enough that long requests can start audio before the whole
    request is generated, while preserving sentence/phrase boundaries when
    possible.
    """
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?。！？;；:：])\s+", text)
    out: list[str] = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        while len(piece) > max_chars:
            cut = max(piece.rfind(",", 0, max_chars), piece.rfind("，", 0, max_chars), piece.rfind(" ", 0, max_chars))
            if cut < max_chars // 2:
                cut = max_chars
            out.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        if piece:
            out.append(piece)
    return out


def _payload_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    text = payload.get("input") or payload.get("text") or ""
    if not str(text).strip():
        raise ValueError("input is required")
    voice = _resolve_voice(payload.get("voice"))
    if voice not in VOICES:
        raise ValueError(f"voice '{voice}' is not a known Kokoro voice")
    lang_code = _lang_for_voice(voice, payload.get("lang_code") or payload.get("language"))
    speed = float(payload.get("speed", 1.0))
    split_pattern = payload.get("split_pattern")
    max_segment_chars = int(payload.get("max_segment_chars", MAX_SEGMENT_CHARS))
    return {
        "text": str(text),
        "voice": voice,
        "lang_code": lang_code,
        "speed": speed,
        "split_pattern": split_pattern,
        "max_segment_chars": max_segment_chars,
    }


def _iter_audio_chunks(kwargs: dict[str, Any]):
    entry = _ensure_pipeline(kwargs["lang_code"])
    segments = _split_realtime_segments(kwargs["text"], kwargs["max_segment_chars"])
    if not segments:
        return
    # Passing a list avoids regex split ambiguities and forces incremental
    # synthesis units for realtime streaming.
    generator = entry.pipeline(
        segments,
        voice=kwargs["voice"],
        speed=kwargs["speed"],
        split_pattern=None,
    )
    for result in generator:
        audio = getattr(result, "audio", None)
        if audio is not None:
            yield result, audio


@app.on_event("startup")
async def _startup_warm() -> None:
    # Warm the default English pipeline asynchronously. Other languages lazy-load.
    def warm() -> None:
        try:
            _ensure_pipeline(_lang_for_voice(DEFAULT_VOICE))
        except Exception:
            pass
    threading.Thread(target=warm, name="kokoro-warm", daemon=True).start()


@app.get("/health", response_model=None)
async def health():
    if not _pipelines:
        body = {
            "ok": False,
            "loaded": False,
            "model_name": DEFAULT_MODEL,
            "default_voice": DEFAULT_VOICE,
            "voices": VOICES,
            "voices_loaded": len(VOICES),
            "supports_true_streaming": True,
            "streaming_kind": "pcm16",
            "streaming_mode": "segment-incremental-pcm",
            "streaming_implementation": "kokoro-kpipeline-segment-generator",
            "sample_rate": SAMPLE_RATE,
            "channels": CHANNELS,
            "sample_format": "pcm16",
            "hint": "warming up default pipeline",
        }
        if _first_load_error:
            body["error"] = _first_load_error
        return JSONResponse(body, status_code=503)
    return {
        "ok": True,
        "loaded": True,
        "model_name": DEFAULT_MODEL,
        "loaded_langs": sorted(_pipelines),
        "load_seconds": {k: v.load_seconds for k, v in _pipelines.items()},
        "device": {k: v.device for k, v in _pipelines.items()},
        "default_voice": DEFAULT_VOICE,
        "voices": VOICES,
        "voices_loaded": len(VOICES),
        "supports_true_streaming": True,
        "streaming_kind": "pcm16",
        "streaming_mode": "segment-incremental-pcm",
        "streaming_implementation": "kokoro-kpipeline-segment-generator",
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "sample_format": "pcm16",
    }


@app.get("/v1/voices")
@app.get("/v1/audio/voices")
async def voices() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": voice,
                "provider": "kokoro",
                "lang_code": _lang_for_voice(voice),
                "grade": VOICE_GRADES.get(voice),
            }
            for voice in VOICES
        ],
    }


@app.post("/v1/audio/speech")
async def speech(request: Request) -> Response:
    try:
        payload = await request.json()
        kwargs = _payload_kwargs(payload)
        chunks = [np.asarray(audio.detach().cpu().numpy() if hasattr(audio, "detach") else audio).squeeze() for _result, audio in _iter_audio_chunks(kwargs)]
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"{type(e).__name__}: {e}") from e
    if not chunks:
        raise HTTPException(500, "no audio produced")
    full = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    response_format = str(payload.get("response_format", "wav")).lower()
    if response_format == "pcm":
        return Response(content=_pcm_bytes(full), media_type="audio/pcm")
    return Response(content=_wav_bytes(full), media_type="audio/wav")


@app.post("/v1/audio/speech-stream")
async def speech_stream(request: Request) -> StreamingResponse:
    try:
        payload = await request.json()
        kwargs = _payload_kwargs(payload)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    queue: asyncio.Queue = asyncio.Queue(maxsize=16)
    error_holder: dict[str, str] = {}
    loop = asyncio.get_running_loop()

    def producer() -> None:
        try:
            for _result, audio in _iter_audio_chunks(kwargs):
                pcm = _pcm_bytes(audio)
                if pcm:
                    asyncio.run_coroutine_threadsafe(queue.put(pcm), loop).result()
        except Exception as e:  # noqa: BLE001
            error_holder["error"] = f"{type(e).__name__}: {e}"
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

    threading.Thread(target=producer, name="kokoro-stream", daemon=True).start()

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
            "X-Kokoro-Model": DEFAULT_MODEL,
            "X-Kokoro-Streaming-Implementation": "kokoro-kpipeline-segment-generator",
            "X-Kokoro-Streaming-Mode": "segment-incremental-pcm",
        },
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8783)
    p.add_argument("--model", default=DEFAULT_MODEL)
    args = p.parse_args()
    os.environ["KOKORO_MODEL"] = args.model
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
