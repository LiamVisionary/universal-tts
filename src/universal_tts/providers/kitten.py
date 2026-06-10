"""KittenTTS in-process provider.

KittenTTS (https://github.com/KittenML/KittenTTS) is a tiny ONNX-based CPU
TTS model (15-80M parameters) with a built-in ``generate_stream`` method
that yields audio chunks. This adapter wraps it so it can be registered in
Universal TTS like the other providers.

Design notes:
- The model is lazy-imported so Universal TTS can still boot in test/CI
  environments that don't have ``kittentts`` installed.
- All synthesis runs on a single dedicated worker thread because ONNX
  inference is not generally safe to interleave from multiple Python threads.
- Streaming emits raw 24 kHz mono signed-16-bit PCM frames as they come off
  ``model.generate_stream`` so the upstream ``/v1/audio/speech-stream``
  endpoint can advertise real ``supports_true_streaming: true``.
- No numpy dependency: we use ``array.array`` and ``struct`` so the
  universal-tts core dependency set stays tiny.
"""

from __future__ import annotations

import array
import asyncio
import io
import queue
import struct
import threading
import wave
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from universal_tts.config import ProviderConfig
from universal_tts.providers.base import ProviderStatus, TTSRequest

SAMPLE_RATE = 24000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit PCM
DEFAULT_VOICE = "Bella"
AVAILABLE_VOICES = ["Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo"]


def _flatten_samples(samples: Any) -> array.array:
    """Flatten a numpy array / nested sequence into a flat signed-16 array.

    Accepts:
      - a numpy ndarray (any shape) -- flattened in C order
      - a sequence of ints
      - a sequence of sequences of ints (flattened one level)
      - bytes / bytearray of int16 little-endian samples
    """
    # numpy path
    if hasattr(samples, "shape") and hasattr(samples, "dtype"):
        try:
            return array.array("h", samples.reshape(-1).astype("int16").tolist())
        except Exception:  # noqa: BLE001
            pass
    if isinstance(samples, (bytes, bytearray)):
        return array.array("h")
    out = array.array("h")
    if hasattr(samples, "squeeze"):
        try:
            samples = samples.squeeze()
        except Exception:  # noqa: BLE001
            pass
    for item in samples:
        if isinstance(item, (int, float)):
            v = int(item)
            if v > 32767:
                v = 32767
            elif v < -32768:
                v = -32768
            out.append(v)
        else:
            for sub in item:
                v = int(sub)
                if v > 32767:
                    v = 32767
                elif v < -32768:
                    v = -32768
                out.append(v)
    return out


def _audio_to_wav_bytes(samples: Any) -> bytes:
    flat = _flatten_samples(samples)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(flat.tobytes())
    return buf.getvalue()


def _audio_to_pcm_bytes(samples: Any) -> bytes:
    return _flatten_samples(samples).tobytes()


def _concat_arrays(parts: list[array.array]) -> array.array:
    out = array.array("h")
    for p in parts:
        out.extend(p)
    return out


@dataclass
class _KittenState:
    model: Any = None
    model_name: str | None = None
    backend: str = "cpu"
    lock: threading.Lock = field(default_factory=threading.Lock)


class _Sentinel:
    pass


_SENTINEL = _Sentinel()


class KittenTTSProvider:
    """Universal TTS adapter that runs KittenTTS in-process on a worker thread.

    Behaves like the HTTP-backed providers from the registry's perspective:
    ``synthesize`` returns a complete audio buffer, ``stream_synthesize``
    returns an async iterator that yields raw PCM chunks as the model emits
    them, ``voices`` returns the static voice list, and ``status`` reports
    health based on whether the model is loadable.
    """

    cfg: ProviderConfig

    def __init__(self, cfg: ProviderConfig):
        self.cfg = cfg
        self._state = _KittenState()
        self._queue: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._worker_lock = threading.Lock()

    # --- worker thread for model-bound work -------------------------------

    def _ensure_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        with self._worker_lock:
            if self._worker is not None and self._worker.is_alive():
                return
            t = threading.Thread(target=self._worker_loop, name=f"kitten-{self.cfg.id}", daemon=True)
            self._worker = t
            t.start()

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            kind, payload, response_q = item
            try:
                if kind == "load":
                    response_q.put(self._do_load(payload))
                elif kind == "synth":
                    response_q.put(self._do_synth(payload))
                else:
                    response_q.put({"error": f"unknown task: {kind}"})
            except Exception as e:  # noqa: BLE001
                response_q.put({"error": f"{type(e).__name__}: {e}"})
            finally:
                self._queue.task_done()

    def _do_load(self, payload: dict) -> dict:
        import importlib

        model_name = payload.get("model_name") or self.cfg.options.get(
            "default_model", "KittenML/kitten-tts-mini-0.8"
        )
        backend = payload.get("backend") or self.cfg.options.get("backend", "cpu")
        cache_dir = payload.get("cache_dir") or self.cfg.options.get("cache_dir")
        try:
            mod = importlib.import_module("kittentts")
        except Exception as e:  # noqa: BLE001
            return {"error": f"kittentts not installed: {e}"}
        try:
            model_cls = getattr(mod, "KittenTTS", None)
            if model_cls is None:
                return {"error": "kittentts.KittenTTS not available"}
            kwargs: dict[str, Any] = {}
            if cache_dir:
                kwargs["cache_dir"] = cache_dir
            try:
                model = model_cls(model_name, backend=backend, **kwargs)
            except TypeError:
                # Older builds may not accept ``backend``.
                model = model_cls(model_name, **kwargs)
        except Exception as e:  # noqa: BLE001
            return {"error": f"failed to load {model_name}: {e}"}
        with self._state.lock:
            self._state.model = model
            self._state.model_name = model_name
            self._state.backend = backend
        return {"model_name": model_name, "backend": backend, "voices": list(self.available_voices())}

    def _do_synth(self, payload: dict) -> dict:
        with self._state.lock:
            model = self._state.model
        if model is None:
            return {"error": "KittenTTS model is not loaded"}
        chunks: list[array.array] = []
        try:
            for chunk in model.generate_stream(**payload):
                chunks.append(_flatten_samples(chunk))
        except Exception as e:  # noqa: BLE001
            return {"error": f"{type(e).__name__}: {e}"}
        if not chunks:
            return {"error": "no audio returned from KittenTTS"}
        full = _concat_arrays(chunks) if len(chunks) > 1 else chunks[0]
        return {"wav": _audio_to_wav_bytes(full)}

    # --- Universal TTS provider surface -----------------------------------

    async def status(self) -> ProviderStatus:
        details: dict[str, Any] = {
            "model_name": self._state.model_name,
            "backend": self._state.backend,
            "voices": list(self.available_voices()),
            "supports_true_streaming": True,
            "streaming_kind": "pcm16",
            "sample_rate": SAMPLE_RATE,
            "channels": CHANNELS,
        }
        if self._state.model is not None:
            return ProviderStatus(id=self.cfg.id, loaded=True, healthy=True, details=details)
        return ProviderStatus(
            id=self.cfg.id,
            loaded=False,
            healthy=False,
            details={**details, "hint": "call /load to initialize"},
        )

    async def load(self) -> ProviderStatus:
        self._ensure_worker()
        response_q: queue.Queue = queue.Queue()
        payload: dict[str, Any] = {}
        model_name = self.cfg.options.get("default_model")
        if model_name:
            payload["model_name"] = model_name
        self._queue.put(("load", payload, response_q))
        result = await asyncio.get_running_loop().run_in_executor(None, response_q.get)
        if "error" in result:
            return ProviderStatus(id=self.cfg.id, loaded=False, healthy=False, details={"error": result["error"]})
        return await self.status()

    async def unload(self) -> ProviderStatus:
        with self._state.lock:
            self._state.model = None
            self._state.model_name = None
        return await self.status()

    def _resolve_voice(self, voice: str | None) -> str:
        aliases = self.cfg.options.get("voice_aliases", {})
        if voice and voice in aliases:
            return aliases[voice]
        return voice or self.cfg.options.get("default_voice", DEFAULT_VOICE)

    def _synth_kwargs(self, request: TTSRequest) -> dict[str, Any]:
        voice = self._resolve_voice(request.voice)
        speed = float(request.speed) if request.speed is not None else float(
            self.cfg.options.get("default_speed", 1.0)
        )
        clean_text = bool(
            request.options.get("clean_text", self.cfg.options.get("default_clean_text", False))
        )
        return {"text": request.text, "voice": voice, "speed": speed, "clean_text": clean_text}

    async def synthesize(self, request: TTSRequest) -> tuple[bytes, str]:
        if self._state.model is None:
            await self.load()
        response_q: queue.Queue = queue.Queue()
        self._queue.put(("synth", self._synth_kwargs(request), response_q))
        result = await asyncio.get_running_loop().run_in_executor(None, response_q.get)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result["wav"], "audio/wav"

    async def stream_synthesize(self, request: TTSRequest) -> tuple[AsyncIterator[bytes], str]:
        if self._state.model is None:
            await self.load()
        kwargs = self._synth_kwargs(request)

        chunk_q: queue.Queue = queue.Queue(maxsize=64)
        error_holder: dict[str, Any] = {}

        def producer() -> None:
            try:
                with self._state.lock:
                    model = self._state.model
                if model is None:
                    error_holder["error"] = "KittenTTS model is not loaded"
                    return
                for chunk in model.generate_stream(**kwargs):
                    pcm = _audio_to_pcm_bytes(chunk)
                    if pcm:
                        chunk_q.put(pcm)
            except Exception as e:  # noqa: BLE001
                error_holder["error"] = f"{type(e).__name__}: {e}"
            finally:
                chunk_q.put(_SENTINEL)

        threading.Thread(
            target=producer, name=f"kitten-stream-{self.cfg.id}", daemon=True
        ).start()

        async def iterator() -> AsyncIterator[bytes]:
            loop = asyncio.get_running_loop()
            while True:
                item = await loop.run_in_executor(None, chunk_q.get)
                if isinstance(item, _Sentinel):
                    break
                yield item
            if error_holder.get("error"):
                raise RuntimeError(error_holder["error"])

        return iterator(), "audio/pcm"

    async def voices(self) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": v, "provider": self.cfg.id} for v in self.available_voices()],
        }

    async def paralinguistics(self) -> dict[str, Any]:
        # KittenTTS does not expose native paralinguistic tokens; return an
        # empty list explicitly so the endpoint contract is honored.
        return {"object": "audio.paralinguistics", "provider": self.cfg.id, "data": []}

    def available_voices(self) -> list[str]:
        configured = self.cfg.options.get("voices")
        if configured:
            return list(configured)
        return list(AVAILABLE_VOICES)
