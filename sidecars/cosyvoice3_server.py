from __future__ import annotations

import io
import os
import re
import sys
import wave
import time
import shutil
import threading
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

SAMPLE_RATE = 24000
CHANNELS = 1
MODEL_REPO = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
MODEL_DIR = Path(os.environ.get("COSYVOICE_MODEL_DIR", "/Users/liam/voice-lab/models/TTS/CosyVoice/Fun-CosyVoice3-0.5B"))
TTS_AUDIO_SUITE = Path(os.environ.get("TTS_AUDIO_SUITE_REPO", "/Users/liam/voice-lab/TTS-Audio-Suite"))
COSY_IMPL = TTS_AUDIO_SUITE / "engines" / "cosyvoice" / "impl"
MATCHA_IMPL = COSY_IMPL / "third_party" / "Matcha-TTS"
DEFAULT_MODEL = os.environ.get("COSYVOICE_MODEL", "Fun-CosyVoice3-0.5B-RL")
DEFAULT_VOICE = os.environ.get("COSYVOICE_DEFAULT_VOICE", "clone")

# User-friendly tags copied from TTS-Audio-Suite's CosyVoice3 support. Users can
# send <breath>/<laughter> while Universal keeps [Character] free for clients.
PARALINGUISTIC_TAGS = {
    "breath", "quick_breath", "laughter", "cough", "sigh", "gasp",
    "noise", "hissing", "vocalized-noise", "lipsmack", "mn", "clucking", "accent",
}
LANGUAGE_TAGS = {"en": "<|en|>", "zh": "<|zh|>", "ja": "<|ja|>", "ko": "<|ko|>"}
REQUIRED_SHARED = [
    "cosyvoice3.yaml", "campplus.onnx", "flow.pt", "hift.pt", "speech_tokenizer_v3.onnx",
    "CosyVoice-BlankEN/config.json", "CosyVoice-BlankEN/generation_config.json",
    "CosyVoice-BlankEN/merges.txt", "CosyVoice-BlankEN/model.safetensors",
    "CosyVoice-BlankEN/tokenizer_config.json", "CosyVoice-BlankEN/vocab.json",
]

app = FastAPI(title="Universal TTS CosyVoice3 Sidecar", version="0.1.0")
_engine: Any | None = None
_engine_model_name: str | None = None
_engine_lock = threading.Lock()
_last_load_error: str | None = None

class SpeechRequest(BaseModel):
    model: str | None = None
    input: str = Field(..., min_length=1)
    voice: str | None = None
    response_format: str = "wav"
    speed: float = 1.0
    mode: str | None = None
    ref_audio: str | None = None
    reference_audio: str | None = None
    reference_audio_path: str | None = None
    ref_text: str | None = None
    reference_text: str | None = None
    reference_transcript: str | None = None
    transcript: str | None = None
    instruct: str | None = None
    instruct_text: str | None = None
    language: str | None = None
    text_frontend: bool | None = None
    seed: int | None = None
    stream: bool | None = None

class VoiceConversionRequest(BaseModel):
    source_audio: str
    ref_audio: str
    response_format: str = "wav"
    speed: float = 1.0
    stream: bool = False

def _patch_yaml() -> None:
    try:
        import yaml
        for loader in (yaml.Loader, yaml.FullLoader, yaml.SafeLoader):
            if not hasattr(loader, "max_depth"):
                loader.max_depth = 100
    except Exception:
        pass

def _ensure_paths() -> None:
    for path in (str(COSY_IMPL), str(MATCHA_IMPL)):
        if path not in sys.path:
            sys.path.insert(0, path)

def _variant_llm(model_name: str | None) -> str:
    return "llm.rl.pt" if (model_name or DEFAULT_MODEL).lower().endswith("-rl") or "rl" in (model_name or DEFAULT_MODEL).lower() else "llm.pt"

def _missing_files(model_name: str | None = None) -> list[str]:
    required = list(REQUIRED_SHARED) + [_variant_llm(model_name)]
    return [rel for rel in required if not (MODEL_DIR / rel).exists()]

def _download_model(model_name: str | None = None) -> None:
    missing = _missing_files(model_name)
    if not missing:
        return
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        raise RuntimeError(f"huggingface_hub is not installed and model files are missing: {missing[:4]}...") from e
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for rel in missing:
        path = hf_hub_download(repo_id=MODEL_REPO, filename=rel, local_dir=str(MODEL_DIR), local_dir_use_symlinks=False)
        if not Path(path).exists():
            raise RuntimeError(f"download did not produce {rel}")

def _load_engine(model_name: str | None = None) -> Any:
    global _engine, _engine_model_name, _last_load_error
    chosen = model_name or DEFAULT_MODEL
    if _engine is not None and _engine_model_name == chosen:
        return _engine
    with _engine_lock:
        if _engine is not None and _engine_model_name == chosen:
            return _engine
        try:
            _download_model(chosen)
            _patch_yaml(); _ensure_paths()
            from cosyvoice.cli.cosyvoice import CosyVoice3
            engine = CosyVoice3(str(MODEL_DIR), load_trt=False, load_vllm=False, fp16=False, llm_filename=_variant_llm(chosen))
            # On Apple Silicon/CPU the downloaded Qwen2 LLM weights may load as
            # bfloat16 while token/embedding inputs are float32. CUDA autocast is
            # disabled here, so force eager modules to float32 to avoid
            # `mat1 and mat2 must have the same dtype` during first generation.
            try:
                for component in (engine.model.llm, engine.model.flow, engine.model.hift):
                    if hasattr(component, "float"):
                        component.float()
            except Exception:
                pass
            _engine = engine
            _engine_model_name = chosen
            _last_load_error = None
            return _engine
        except Exception as e:
            _last_load_error = f"{type(e).__name__}: {e}"
            raise

def _format_prompt_text(ref_text: str | None) -> str | None:
    if not ref_text:
        return ref_text
    if ref_text.startswith("You are a helpful assistant."):
        return ref_text if "<|endofprompt|>" in ref_text else ref_text.replace("You are a helpful assistant.", "You are a helpful assistant.<|endofprompt|>")
    return f"You are a helpful assistant.<|endofprompt|>{ref_text}"

def _format_instruct(instruct: str | None) -> str | None:
    if not instruct:
        return instruct
    if not instruct.endswith("<|endofprompt|>"):
        return instruct + "<|endofprompt|>" if instruct.startswith("You are a helpful assistant") else f"You are a helpful assistant. {instruct}<|endofprompt|>"
    return instruct if instruct.startswith("You are a helpful assistant") else f"You are a helpful assistant. {instruct}"

def convert_special_tags(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        tag = match.group(1)
        return f"[{tag}]" if tag.lower() in PARALINGUISTIC_TAGS else match.group(0)
    text = re.sub(r"(?<!</)(?<!\w)<([a-zA-Z_-]+)>(?!\w)", replace, text)
    return re.sub(r"<laughing>(.*?)</laughing>", r"<laughter>\1</laughter>", text, flags=re.I | re.S)

def _prepare_text(req: SpeechRequest) -> tuple[str, bool]:
    text = convert_special_tags(req.input)
    if req.language and req.language.lower() in LANGUAGE_TAGS and not text.startswith("<|"):
        text = LANGUAGE_TAGS[req.language.lower()] + text
    tags = ["[breath]", "[quick_breath]", "[laughter]", "[cough]", "[sigh]", "[gasp]", "[noise]", "[hissing]", "[vocalized-noise]", "[lipsmack]", "[mn]", "[clucking]", "[accent]", "<strong>", "</strong>", "<laughter>", "</laughter>", "<|en|>", "<|zh|>", "<|ja|>", "<|ko|>"]
    text_frontend = req.text_frontend if req.text_frontend is not None else not any(t in text for t in tags)
    return text, bool(text_frontend)

def _tensor_to_np(audio: Any) -> np.ndarray:
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().float().numpy()
    arr = np.asarray(audio, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return np.clip(arr, -1.0, 1.0)

def _float_to_pcm16(audio: Any) -> bytes:
    arr = _tensor_to_np(audio)
    return np.round(arr * 32767.0).astype("<i2").tobytes()

def _wav_bytes(audio: Any) -> bytes:
    pcm = _float_to_pcm16(audio)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE); wf.writeframes(pcm)
    return buf.getvalue()

def _generate_chunks(req: SpeechRequest, stream: bool) -> Iterator[Any]:
    engine = _load_engine(req.model)
    text, text_frontend = _prepare_text(req)
    ref_audio = req.ref_audio or req.reference_audio or req.reference_audio_path
    ref_text = req.ref_text or req.reference_text or req.reference_transcript or req.transcript
    instruct = req.instruct_text or req.instruct
    mode = req.mode or ("instruct" if instruct else "zero_shot" if ref_text else "cross_lingual")
    if not ref_audio:
        raise HTTPException(status_code=400, detail="CosyVoice3 requires ref_audio/reference_audio_path for zero_shot, instruct, and cross_lingual modes")
    if req.seed is not None:
        try:
            import torch, random
            import numpy as _np
            torch.manual_seed(req.seed); random.seed(req.seed); _np.random.seed(req.seed)
        except Exception:
            pass
    with _engine_lock:
        if mode == "instruct":
            for out in engine.inference_instruct2(text, _format_instruct(instruct or "Speak naturally."), ref_audio, stream=stream, speed=req.speed, text_frontend=text_frontend):
                yield out["tts_speech"]
        elif mode == "zero_shot":
            if not ref_text:
                raise HTTPException(status_code=400, detail="zero_shot mode requires ref_text/reference_text")
            for out in engine.inference_zero_shot(text, _format_prompt_text(ref_text), ref_audio, stream=stream, speed=req.speed, text_frontend=text_frontend):
                yield out["tts_speech"]
        elif mode == "cross_lingual":
            formatted = text if text.startswith("You are a helpful assistant.") else f"You are a helpful assistant.<|endofprompt|>{text}"
            for out in engine.inference_cross_lingual(formatted, ref_audio, stream=stream, speed=req.speed, text_frontend=text_frontend):
                yield out["tts_speech"]
        else:
            raise HTTPException(status_code=400, detail=f"unsupported CosyVoice3 mode: {mode}")

@app.get("/health")
def health(load: bool = False) -> dict[str, Any]:
    missing = _missing_files(DEFAULT_MODEL)
    details = {
        "ok": not missing and COSY_IMPL.exists(),
        "loaded": _engine is not None,
        "model": DEFAULT_MODEL,
        "model_dir": str(MODEL_DIR),
        "repo": MODEL_REPO,
        "tts_audio_suite_repo": str(TTS_AUDIO_SUITE),
        "missing_files": missing,
        "last_load_error": _last_load_error,
        "sample_rate": SAMPLE_RATE,
        "supports_true_streaming": True,
        "streaming_mode": "cosyvoice3-model-tts-stream",
    }
    if load:
        try:
            _load_engine(DEFAULT_MODEL)
            details.update({"ok": True, "loaded": True, "missing_files": [], "last_load_error": _last_load_error})
        except Exception as e:
            details.update({"ok": False, "last_load_error": f"{type(e).__name__}: {e}"})
    return details

@app.get("/v1/audio/paralinguistics")
def paralinguistics() -> dict[str, Any]:
    return {"object": "audio.paralinguistics", "provider": "cosyvoice3", "data": [{"token": f"<{t}>", "model_token": f"[{t}]", "kind": "nonverbal", "supported": True} for t in sorted(PARALINGUISTIC_TAGS)] + [{"token": "<laughing>...</laughing>", "model_token": "<laughter>...</laughter>", "kind": "style", "supported": True}]}

@app.get("/v1/voices")
def voices() -> dict[str, Any]:
    return {"object": "list", "data": [{"id": DEFAULT_VOICE, "provider": "cosyvoice3", "requires": ["ref_audio"], "recommended": ["ref_text"]}]}

@app.post("/v1/audio/speech")
def speech(req: SpeechRequest) -> Response:
    try:
        chunks = list(_generate_chunks(req, stream=False))
        if not chunks:
            raise RuntimeError("CosyVoice3 returned no audio chunks")
        audio = np.concatenate([_tensor_to_np(c) for c in chunks])
        fmt = (req.response_format or "wav").lower()
        if fmt == "pcm":
            return Response(_float_to_pcm16(audio), media_type="audio/pcm", headers={"X-Audio-Sample-Rate": str(SAMPLE_RATE), "X-Audio-Channels": "1", "X-Audio-Sample-Format": "pcm16"})
        return Response(_wav_bytes(audio), media_type="audio/wav")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CosyVoice3 synthesis failed: {type(e).__name__}: {e}")

@app.post("/v1/audio/speech-stream")
def speech_stream(req: SpeechRequest) -> StreamingResponse:
    def iterator() -> Iterator[bytes]:
        for chunk in _generate_chunks(req, stream=True):
            pcm = _float_to_pcm16(chunk)
            if pcm:
                yield pcm
    return StreamingResponse(iterator(), media_type="audio/pcm", headers={"X-Audio-Sample-Rate": str(SAMPLE_RATE), "X-Audio-Channels": "1", "X-Audio-Sample-Format": "pcm16", "X-Universal-TTS-Streaming-Mode": "cosyvoice3-model-tts-stream"})

@app.post("/v1/audio/voice-conversion")
def voice_conversion(req: VoiceConversionRequest) -> Response:
    try:
        engine = _load_engine(DEFAULT_MODEL)
        with _engine_lock:
            chunks = [out["tts_speech"] for out in engine.inference_vc(req.source_audio, req.ref_audio, stream=req.stream, speed=req.speed)]
        audio = np.concatenate([_tensor_to_np(c) for c in chunks]) if chunks else np.zeros(SAMPLE_RATE, dtype=np.float32)
        if req.response_format.lower() == "pcm":
            return Response(_float_to_pcm16(audio), media_type="audio/pcm")
        return Response(_wav_bytes(audio), media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CosyVoice3 VC failed: {type(e).__name__}: {e}")
