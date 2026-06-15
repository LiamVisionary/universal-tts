from __future__ import annotations

import io
import os
import time
import wave
import threading
from pathlib import Path
from typing import Iterator

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

SAMPLE_RATE = 24000
MODEL_DIR = Path(os.environ.get("HIGGS_MLX_MODEL_DIR", "/Users/liam/voice-lab/models/TTS/higgs_audio_v3/higgs-audio-v3-tts-4b"))
MODEL_NAME = os.environ.get("HIGGS_MLX_MODEL_NAME", "higgs-audio-v3-tts-4b-mlx")
DEFAULT_REF_AUDIO = os.environ.get("HIGGS_MLX_VOICE01_REF_AUDIO", "/Users/liam/voice-lab/Chatterbox-TTS-Server/voices/voice1-all-samples-10s.wav")
DEFAULT_REF_TEXT = os.environ.get("HIGGS_MLX_VOICE01_REF_TEXT", "They let me pick. Did I ever tell you that? Choose whichever Spartan I wanted. You know me. I did my research. Watched as you became the soldier we needed you to be.")

app = FastAPI(title="Universal TTS Higgs Audio v3 MLX Sidecar", version="0.1.0")
_model = None
_lock = threading.Lock()
_last_error: str | None = None


class SpeechRequest(BaseModel):
    model: str | None = None
    input: str = Field(..., min_length=1)
    voice: str | None = None
    response_format: str = "wav"
    speed: float = 1.0
    ref_audio: str | None = None
    reference_audio: str | None = None
    reference_audio_path: str | None = None
    ref_text: str | None = None
    reference_text: str | None = None
    reference_transcript: str | None = None
    transcript: str | None = None
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    seed: int | None = 0
    stream_commit_tokens: int = 24
    stream_overlap_tokens: int = 8
    stream_declick_ms: float = 8.0
    stream_declick_threshold: int = 300


def _missing() -> list[str]:
    required = ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"]
    return [name for name in required if not (MODEL_DIR / name).exists()]


def _load():
    global _model, _last_error
    if _model is not None:
        return _model
    with _lock:
        if _model is not None:
            return _model
        try:
            if _missing():
                raise FileNotFoundError(f"missing Higgs MLX model files: {_missing()}")
            from mlx_audio.tts.utils import load
            # strict=False tolerates tied/head keys that the MLX implementation intentionally derives/skips.
            _model = load(MODEL_DIR, lazy=False, strict=False)
            _last_error = None
            return _model
        except Exception as e:
            _last_error = f"{type(e).__name__}: {e}"
            raise


def _ref_fields(req: SpeechRequest) -> tuple[str | None, str | None]:
    ref_audio = req.ref_audio or req.reference_audio or req.reference_audio_path
    ref_text = req.ref_text or req.reference_text or req.reference_transcript or req.transcript
    if not ref_audio and (req.voice or "").lower() in {"voice01", "liam-default", "clone"}:
        ref_audio = DEFAULT_REF_AUDIO
        ref_text = ref_text or DEFAULT_REF_TEXT
    return ref_audio, ref_text


def _float_to_pcm16(audio) -> bytes:
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    return np.round(np.clip(arr, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def _wav_bytes(audio, sample_rate: int = SAMPLE_RATE) -> bytes:
    bio = io.BytesIO()
    with wave.open(bio, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(_float_to_pcm16(audio))
    return bio.getvalue()


def _declick_segment_start(segment: np.ndarray, prev_last: float | None, ms: float, threshold_i16: int) -> np.ndarray:
    seg = np.asarray(segment, dtype=np.float32).reshape(-1).copy()
    if prev_last is None or seg.size == 0 or ms <= 0:
        return seg
    jump_i16 = abs(int(round((float(seg[0]) - float(prev_last)) * 32767.0)))
    if jump_i16 < int(threshold_i16):
        return seg
    n = min(seg.size, max(1, int(SAMPLE_RATE * float(ms) / 1000.0)))
    correction = float(prev_last) - float(seg[0])
    seg[:n] += correction * np.linspace(1.0, 0.0, n, endpoint=False, dtype=np.float32)
    return np.clip(seg, -1.0, 1.0)


def _generate_full(req: SpeechRequest) -> tuple[np.ndarray, int]:
    model = _load()
    ref_audio, ref_text = _ref_fields(req)
    result = next(model.generate(
        text=req.input,
        ref_audio=ref_audio,
        ref_text=ref_text,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=None if req.top_p <= 0 or req.top_p >= 1 else req.top_p,
        top_k=None if req.top_k <= 0 else req.top_k,
        seed=req.seed,
        stream=False,
    ))
    audio = np.array(result.audio).astype(np.float32, copy=False).reshape(-1)
    return audio, int(result.sample_rate or SAMPLE_RATE)


def _stream_prefix(req: SpeechRequest) -> Iterator[bytes]:
    """MLX prefix-incremental stream.

    The upstream mlx-audio Higgs v3 public generate() currently returns one final
    chunk. For Universal realtime, run the same loop with periodic prefix decode
    and emit only newly committed PCM. Label remains prefix-incremental, not
    native decoder-frame streaming.
    """
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_audio.tts.models.higgs_audio_v3.generation import HiggsSamplerState, step, reverse_delay_pattern

    model = _load()
    if req.seed is not None:
        mx.random.seed(int(req.seed))
    ref_audio, ref_text = _ref_fields(req)
    refs = model._normalize_references(ref_audio=ref_audio, ref_text=ref_text)
    prompt_embeds, _prompt_tokens = model._build_prompt_embeddings(req.input.strip(), refs)
    mx.eval(prompt_embeds)
    cache = make_prompt_cache(model)
    dummy = mx.zeros((1, prompt_embeds.shape[1]), dtype=mx.int32)
    hidden = model.backbone(dummy, cache=cache, input_embeddings=prompt_embeds)
    last_hidden = hidden[:, -1, :]
    state = HiggsSamplerState(num_codebooks=model.config.audio_num_codebooks)
    delayed_rows = []
    emitted_samples = 0
    prev_last: float | None = None
    commit = max(1, int(req.stream_commit_tokens))
    overlap = max(0, int(req.stream_overlap_tokens))
    for _ in range(int(req.max_new_tokens)):
        logits = model._audio_logits(last_hidden)[0]
        codes = step(
            logits,
            state,
            temperature=float(req.temperature),
            top_p=None if req.top_p <= 0 or req.top_p >= 1 else float(req.top_p),
            top_k=None if req.top_k <= 0 else int(req.top_k),
            boc_id=model.config.audio_boc_token_id,
            eoc_id=model.config.audio_eoc_token_id,
        )
        delayed_rows.append(codes)
        ready = len(delayed_rows) >= model.config.audio_num_codebooks + commit
        on_boundary = (len(delayed_rows) - model.config.audio_num_codebooks) % commit == 0
        if (ready and on_boundary) or state.generation_done:
            rows = delayed_rows
            if not state.generation_done and overlap > 0 and len(rows) > model.config.audio_num_codebooks + overlap:
                rows = rows[:-overlap]
            try:
                delayed = mx.stack(rows, axis=0).astype(mx.int32)
                raw_codes = reverse_delay_pattern(delayed)
                audio = model.codec.decode(raw_codes).astype(mx.float32).reshape(-1)
                mx.eval(audio)
                audio_np = np.array(audio).astype(np.float32, copy=False)
                if audio_np.shape[0] > emitted_samples:
                    segment = audio_np[emitted_samples:]
                    segment = _declick_segment_start(segment, prev_last, req.stream_declick_ms, req.stream_declick_threshold)
                    if segment.size:
                        prev_last = float(segment[-1])
                        emitted_samples = audio_np.shape[0]
                        yield _float_to_pcm16(segment)
            except Exception:
                pass
        if state.generation_done:
            break
        next_embed = model._embed_audio_codes(codes)[None]
        decode_dummy = mx.zeros((1, 1), dtype=mx.int32)
        hidden = model.backbone(decode_dummy, cache=cache, input_embeddings=next_embed)
        last_hidden = hidden[:, -1, :]


@app.get("/health")
def health(load: bool = False):
    data = {
        "ok": not _missing(),
        "loaded": _model is not None,
        "model": MODEL_NAME,
        "model_dir": str(MODEL_DIR),
        "missing_files": _missing(),
        "last_load_error": _last_error,
        "sample_rate": SAMPLE_RATE,
        "runtime": "mlx-audio",
        "supports_true_streaming": True,
        "streaming_mode": "prefix-incremental-codec-decode",
    }
    if load:
        try:
            _load()
            data.update({"ok": True, "loaded": True, "last_load_error": None})
        except Exception as e:
            data.update({"ok": False, "last_load_error": f"{type(e).__name__}: {e}"})
    return data


@app.get("/v1/voices")
def voices():
    return {"object": "list", "data": [{"id": "voice01", "provider": "higgs-audio-v3-mlx"}, {"id": "clone", "provider": "higgs-audio-v3-mlx"}]}


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    try:
        audio, sr = _generate_full(req)
        if req.response_format.lower() == "pcm":
            return Response(_float_to_pcm16(audio), media_type="audio/pcm")
        return Response(_wav_bytes(audio, sr), media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Higgs v3 MLX synthesis failed: {type(e).__name__}: {e}")


@app.post("/v1/audio/speech-stream")
def speech_stream(req: SpeechRequest):
    return StreamingResponse(
        _stream_prefix(req),
        media_type="audio/pcm",
        headers={
            "X-Audio-Sample-Rate": str(SAMPLE_RATE),
            "X-Audio-Channels": "1",
            "X-Audio-Sample-Format": "pcm16",
            "X-Universal-TTS-Streaming-Mode": "prefix-incremental-codec-decode",
            "X-Universal-TTS-Streaming-Implementation": "mlx-audio-higgs-v3-prefix-decode",
        },
    )
