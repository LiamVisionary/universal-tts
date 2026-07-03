from __future__ import annotations

import asyncio
import base64
import contextlib
import time
from typing import Any, AsyncIterator

import httpx

from universal_tts.config import ProviderConfig
from universal_tts.lifecycle import ProcessLifecycle
from universal_tts.providers.base import ProviderStatus, TTSRequest


class HttpBackedProvider:
    """Provider adapter that owns lifecycle plus common OpenAI-compatible HTTP forwarding.

    This keeps Universal TTS as one project while allowing provider-specific request shaping.
    Heavy model dependencies can stay isolated per provider until an inline adapter is selected.
    """

    def __init__(self, cfg: ProviderConfig):
        self.cfg = cfg
        self.lifecycle = ProcessLifecycle(
            provider_id=cfg.id,
            launchd_label=cfg.launchd_label,
            plist=cfg.plist,
            command=cfg.command,
            cwd=cfg.cwd,
            port=cfg.port,
            log_dir=cfg.options.get("log_dir", "logs"),
        )
        self.restart_count = 0
        self.last_restart_at: float | None = None
        self.last_restart_reason: str | None = None
        self.last_successful_synth: float | None = None
        self.last_error: str | None = None
        self._monitor_task: asyncio.Task | None = None
        self._monitor_stop: asyncio.Event | None = None
        self._probe_due_at = 0.0
        self._restart_lock = asyncio.Lock()

    def metadata(self) -> dict[str, Any]:
        return {
            "restart_count": self.restart_count,
            "last_restart_at": self.last_restart_at,
            "last_restart_reason": self.last_restart_reason,
            "last_successful_synth": self.last_successful_synth,
            "last_error": self.last_error,
            "lifecycle": self.lifecycle.last_start_result or {},
        }

    def _start_monitor(self) -> None:
        if self._monitor_task and not self._monitor_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._monitor_stop = asyncio.Event()
        self._monitor_task = loop.create_task(self._monitor_loop())

    async def shutdown(self) -> None:
        if self._monitor_stop is not None:
            self._monitor_stop.set()
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            with contextlib.suppress(BaseException):
                await self._monitor_task
        self.lifecycle.stop()

    async def _restart_sidecar(self, reason: str, *, force: bool = False) -> ProviderStatus:
        async with self._restart_lock:
            self.last_error = reason
            if force:
                self.lifecycle.kill()
            else:
                self.lifecycle.stop()
            base_delay = float(self.cfg.options.get("restart_backoff_base_sec", 1.0))
            max_delay = float(self.cfg.options.get("restart_backoff_max_sec", 30.0))
            delay = min(max_delay, base_delay * (2 ** min(self.restart_count, 5)))
            if delay > 0:
                await asyncio.sleep(delay)
            self._lifecycle_start()
            self.restart_count += 1
            self.last_restart_at = time.time()
            self.last_restart_reason = reason
            timeout = float(self.cfg.options.get("startup_timeout_sec", 120))
            load_path = self.cfg.options.get("load_path")
            last = await self._wait_until_loaded(timeout, require_loaded=not bool(load_path))
            if last.healthy and load_path and self.cfg.base_url:
                try:
                    async with httpx.AsyncClient(timeout=float(self.cfg.options.get("load_timeout_sec", timeout))) as client:
                        response = await client.post(self.cfg.base_url.rstrip("/") + str(load_path), json={})
                    if response.status_code >= 400:
                        self.last_error = f"load_path failed after restart: {response.status_code} {response.text[:200]}"
                    last = await self._wait_until_loaded(timeout, require_loaded=True)
                except Exception as e:
                    self.last_error = f"load_path failed after restart: {e!r}"
            return last

    async def _wait_until_loaded(self, timeout: float, *, require_loaded: bool = True) -> ProviderStatus:
        deadline = asyncio.get_event_loop().time() + timeout
        last = await self.status()
        while not (last.loaded if require_loaded else last.healthy) and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(1)
            last = await self.status()
        return last

    async def _probe_once(self) -> None:
        if not self.cfg.base_url:
            return
        timeout = float(self.cfg.options.get("synth_probe_timeout_sec", 60.0))
        path = self.cfg.options.get("synth_probe_path") or self.stream_path() or "/v1/audio/speech"
        payload = {
            "model": self.cfg.options.get("default_model", self.cfg.models[0] if self.cfg.models else self.cfg.id),
            "voice": self.cfg.options.get("default_voice", "voice01"),
            "input": self.cfg.options.get("synth_probe_text", "Yes."),
            "response_format": "pcm" if str(path).endswith("speech-stream") else "wav",
            "max_tokens": int(self.cfg.options.get("synth_probe_max_tokens", 32)),
            "max_new_tokens": int(self.cfg.options.get("synth_probe_max_tokens", 32)),
            "realtime_pacing": False,
            "stream_frame_ms": 20,
        }
        payload.update(dict(self.cfg.options.get("synth_probe_payload") or {}))
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                if str(path).endswith("speech-stream"):
                    async with client.stream("POST", self.cfg.base_url.rstrip("/") + str(path), json=payload) as response:
                        if response.status_code >= 400:
                            body = await response.aread()
                            raise RuntimeError(f"probe failed: {response.status_code} {body[:200]!r}")
                        ait = response.aiter_bytes(chunk_size=512).__aiter__()
                        chunk = await asyncio.wait_for(ait.__anext__(), timeout=timeout)
                        if not chunk:
                            raise RuntimeError("probe returned empty first chunk")
                else:
                    response = await client.post(self.cfg.base_url.rstrip("/") + str(path), json=payload)
                    if response.status_code >= 400 or not response.content:
                        raise RuntimeError(f"probe failed: {response.status_code} {response.text[:200]}")
            self.last_successful_synth = time.time()
            self.last_error = None
        except Exception as e:
            await self._restart_sidecar(f"synth probe timeout/failure: {e!r}", force=True)

    async def _monitor_loop(self) -> None:
        interval = float(self.cfg.options.get("sidecar_watch_interval_sec", 5.0))
        probe_interval = float(self.cfg.options.get("synth_probe_interval_sec", 30.0))
        self._probe_due_at = time.time() + probe_interval if probe_interval > 0 else 0.0
        assert self._monitor_stop is not None
        while not self._monitor_stop.is_set():
            try:
                status = await self.status()
                if not status.loaded and self.lifecycle.is_running():
                    await self._restart_sidecar(f"health check failed: {status.details.get('error') or status.details}", force=True)
                elif not self.lifecycle.is_running():
                    await self._restart_sidecar("sidecar process/listener exited", force=False)
                elif probe_interval > 0 and time.time() >= self._probe_due_at:
                    await self._probe_once()
                    self._probe_due_at = time.time() + probe_interval
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.last_error = repr(e)
            try:
                await asyncio.wait_for(self._monitor_stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass

    def _lifecycle_start(self) -> dict[str, Any]:
        try:
            return self.lifecycle.start(port_conflict_policy=str(self.cfg.options.get("port_conflict_policy", "adopt")))
        except TypeError:
            # Keeps older tests/fakes compatible while real ProcessLifecycle accepts
            # the ownership policy.
            return self.lifecycle.start()

    def ensure_managed(self) -> None:
        self._lifecycle_start()
        self._start_monitor()

    def speech_payload(self, request: TTSRequest) -> dict[str, Any]:
        payload = {
            "model": request.model,
            "input": request.text,
            "response_format": request.response_format,
        }
        if request.voice:
            voice_aliases = self.cfg.options.get("voice_aliases", {})
            payload["voice"] = voice_aliases.get(request.voice, request.voice)
        if request.speed is not None:
            payload["speed"] = request.speed
        payload.update({k: v for k, v in request.options.items() if k != "cancel_event"})
        return payload

    def stream_path(self) -> str | None:
        return self.cfg.options.get("stream_path")

    def stream_payload(self, request: TTSRequest) -> dict[str, Any]:
        return self.speech_payload(request)

    async def status(self) -> ProviderStatus:
        if not self.cfg.base_url:
            return ProviderStatus(id=self.cfg.id, loaded=False, healthy=False, details={"error": "missing base_url"})
        url = self.cfg.base_url.rstrip("/") + self.cfg.health_path
        try:
            async with httpx.AsyncClient(timeout=2.5) as client:
                response = await client.get(url)
            try:
                body: Any = response.json()
            except Exception:
                body = response.text[:500]
            healthy = 200 <= response.status_code < 400
            loaded = healthy
            if isinstance(body, dict) and "loaded" in body:
                loaded = bool(body.get("loaded") or body.get("last_successful_synth"))
            if healthy and loaded:
                self.last_error = None
            details = {"status_code": response.status_code, "url": url, "body": body, **self.metadata()}
            return ProviderStatus(id=self.cfg.id, loaded=loaded, healthy=healthy, details=details)
        except Exception as e:
            self.last_error = str(e)
            return ProviderStatus(id=self.cfg.id, loaded=False, healthy=False, details={"url": url, "error": str(e), **self.metadata()})

    async def load(self) -> ProviderStatus:
        self._lifecycle_start()
        timeout = float(self.cfg.options.get("startup_timeout_sec", 120))
        load_path = self.cfg.options.get("load_path")
        last = await self._wait_until_loaded(timeout, require_loaded=not bool(load_path))
        if last.healthy and load_path and self.cfg.base_url:
            try:
                async with httpx.AsyncClient(timeout=float(self.cfg.options.get("load_timeout_sec", timeout))) as client:
                    response = await client.post(self.cfg.base_url.rstrip("/") + str(load_path), json={})
                if response.status_code >= 400:
                    self.last_error = f"load_path failed: {response.status_code} {response.text[:200]}"
                last = await self._wait_until_loaded(timeout, require_loaded=True)
            except Exception as e:
                self.last_error = f"load_path failed: {e!r}"
        if last.loaded:
            self._start_monitor()
        return last

    async def unload(self) -> ProviderStatus:
        if self._monitor_stop is not None:
            self._monitor_stop.set()
        self.lifecycle.stop()
        return await self.status()

    async def synthesize(self, request: TTSRequest) -> tuple[bytes, str]:
        if not self.cfg.base_url:
            raise RuntimeError(f"{self.cfg.id} has no base_url")
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(self.cfg.base_url.rstrip("/") + "/v1/audio/speech", json=self.speech_payload(request))
            if response.status_code >= 400:
                raise RuntimeError(f"{self.cfg.id} synthesis failed: {response.status_code} {response.text[:500]}")
            self.last_successful_synth = time.time()
            self.last_error = None
            return response.content, response.headers.get("content-type", "audio/wav")
        except Exception as e:
            self.last_error = repr(e)
            raise

    async def batch_synthesize(self, requests: list[TTSRequest]) -> list[tuple[bytes, str]]:
        # Default compatibility path. Providers with native GPU microbatching
        # should override this method or expose a batch_path option.
        batch_path = self.cfg.options.get("batch_path")
        if batch_path and self.cfg.base_url:
            payload = {"items": [self.speech_payload(request) for request in requests]}
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(self.cfg.base_url.rstrip("/") + batch_path, json=payload)
            if response.status_code >= 400:
                raise RuntimeError(f"{self.cfg.id} batch synthesis failed: {response.status_code} {response.text[:500]}")
            data = response.json()
            out: list[tuple[bytes, str]] = []
            for item in data.get("data", []):
                if "audio_base64" in item:
                    audio = base64.b64decode(item["audio_base64"])
                elif item.get("encoding") == "base64" and "audio" in item:
                    audio = base64.b64decode(item["audio"])
                else:
                    raw = item.get("audio", b"")
                    audio = raw if isinstance(raw, bytes) else str(raw).encode()
                out.append((audio, item.get("content_type", "audio/wav")))
            return out
        return [await self.synthesize(request) for request in requests]

    async def voices(self) -> dict[str, Any]:
        configured = self.cfg.options.get("voices")
        if configured:
            return {"object": "list", "data": [{"id": voice, "provider": self.cfg.id} for voice in configured]}
        if not self.cfg.base_url:
            return {"object": "list", "data": []}
        async with httpx.AsyncClient(timeout=10) as client:
            for path in self.cfg.options.get("voices_paths", ["/v1/voices", "/v1/audio/voices", "/voices"]):
                try:
                    response = await client.get(self.cfg.base_url.rstrip("/") + path)
                except Exception:
                    continue
                if response.status_code < 400:
                    try:
                        return response.json()
                    except Exception:
                        return {"object": "list", "data": response.text[:1000]}
        return {"object": "list", "data": []}

    async def paralinguistics(self) -> dict[str, Any]:
        configured = self.cfg.options.get("paralinguistics")
        if configured:
            return {"object": "audio.paralinguistics", "provider": self.cfg.id, "data": configured}
        if not self.cfg.base_url:
            return {"object": "audio.paralinguistics", "provider": self.cfg.id, "data": []}
        async with httpx.AsyncClient(timeout=10) as client:
            for path in self.cfg.options.get("paralinguistics_paths", ["/v1/audio/paralinguistics", "/v1/paralinguistics", "/paralinguistics"]):
                try:
                    response = await client.get(self.cfg.base_url.rstrip("/") + path)
                except Exception:
                    continue
                if response.status_code < 400:
                    try:
                        data = response.json()
                    except Exception:
                        return {"object": "audio.paralinguistics", "provider": self.cfg.id, "data": [], "raw": response.text[:1000]}
                    data.setdefault("provider", self.cfg.id)
                    return data
        return {"object": "audio.paralinguistics", "provider": self.cfg.id, "data": []}

    async def stream_synthesize(self, request: TTSRequest) -> tuple[AsyncIterator[bytes], str]:
        path = self.stream_path()
        if not path:
            async def fallback_chunks():
                audio, _content_type = await self.synthesize(request)
                chunk_size = int(request.options.get("chunk_bytes", 65536))
                for i in range(0, len(audio), chunk_size):
                    yield audio[i:i + chunk_size]
            return fallback_chunks(), "audio/wav"

        if not self.cfg.base_url:
            raise RuntimeError(f"{self.cfg.id} has no base_url")

        async def upstream_chunks():
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", self.cfg.base_url.rstrip("/") + path, json=self.stream_payload(request)) as response:
                    if response.status_code >= 400:
                        body = await response.aread()
                        raise RuntimeError(f"{self.cfg.id} streaming failed: {response.status_code} {body[:500]!r}")
                    stream_read_bytes = int(self.cfg.options.get("stream_read_bytes", 4096))
                    ait = response.aiter_bytes(chunk_size=stream_read_bytes).__aiter__()
                    no_bytes_timeout = float(request.options.get("stream_no_bytes_timeout_sec", self.cfg.options.get("stream_no_bytes_timeout_sec", 60.0)))
                    while True:
                        try:
                            chunk = await asyncio.wait_for(ait.__anext__(), timeout=no_bytes_timeout)
                        except StopAsyncIteration:
                            break
                        except asyncio.TimeoutError as e:
                            await self._restart_sidecar(f"stream watchdog: no bytes for {no_bytes_timeout}s", force=True)
                            raise RuntimeError(f"{self.cfg.id} stream watchdog: no bytes for {no_bytes_timeout}s") from e
                        if chunk:
                            self.last_successful_synth = time.time()
                            self.last_error = None
                            yield chunk

        return upstream_chunks(), self.cfg.options.get("stream_content_type", "audio/wav")


class Qwen3Provider(HttpBackedProvider):
    def speech_payload(self, request: TTSRequest) -> dict[str, Any]:
        payload = super().speech_payload(request)
        # Qwen uses instruction control; keep common fields but pass through instruct/language when supplied.
        if "instruct" in request.options:
            payload["instruct"] = request.options["instruct"]
        if "language" in request.options:
            payload["language"] = request.options["language"]
        if self.cfg.options.get("default_seed") is not None:
            payload.setdefault("seed", int(self.cfg.options["default_seed"]))
        if self.cfg.options.get("default_max_new_tokens") is not None:
            payload.setdefault("max_new_tokens", int(self.cfg.options["default_max_new_tokens"]))
        return payload

    def stream_path(self) -> str | None:
        return self.cfg.options.get("stream_path", "/v1/audio/speech-stream")


class MisoProvider(HttpBackedProvider):
    def speech_payload(self, request: TTSRequest) -> dict[str, Any]:
        payload = super().speech_payload(request)
        # Miso is very sensitive in the live path. Prefer the verified cloned
        # voice/profile when callers do not specify a voice, and use conservative
        # sampling defaults that passed Whisper on short calling utterances.
        payload.setdefault("voice", self.cfg.options.get("default_voice", "speaker-0"))
        payload.setdefault("temperature", float(self.cfg.options.get("default_temperature", 0.55)))
        payload.setdefault("topk", int(self.cfg.options.get("default_topk", 20)))
        payload.setdefault("chunk_frames", int(self.cfg.options.get("default_chunk_frames", 1)))
        if self.cfg.options.get("default_seed") is not None:
            payload.setdefault("seed", int(self.cfg.options["default_seed"]))
        payload.setdefault("asr_verify", False)
        payload.setdefault("auto_duration", False)
        payload.setdefault("tail_silence_ms", 700)
        payload.setdefault("max_audio_length_ms", 2500)
        return payload

    def stream_path(self) -> str | None:
        return self.cfg.options.get("stream_path", "/v1/audio/speech-stream")


class ChatterboxTurboProvider(HttpBackedProvider):
    def _resolve_voice(self, voice: str | None) -> str | None:
        if not voice:
            return voice
        voice_aliases = self.cfg.options.get("voice_aliases", {})
        return voice_aliases.get(voice, voice)

    def speech_payload(self, request: TTSRequest) -> dict[str, Any]:
        payload = super().speech_payload(request)
        # Chatterbox Turbo exposes exaggeration/cfg_weight/seed/speed_factor knobs.
        if request.voice:
            payload["voice"] = self._resolve_voice(request.voice)
        if request.speed is not None:
            payload.pop("speed", None)
            payload["speed_factor"] = request.speed
        return payload

    def stream_path(self) -> str | None:
        # Chatterbox base has /tts/stream, but the installed Turbo class lacks
        # generate_stream. Default Turbo streaming to full-generate-then-chunk;
        # allow config override if a future Turbo build adds true streaming.
        return self.cfg.options.get("stream_path")

    def stream_payload(self, request: TTSRequest) -> dict[str, Any]:
        resolved_voice = self._resolve_voice(request.voice)
        payload: dict[str, Any] = {
            "text": request.text,
            "voice_mode": "predefined",
            "predefined_voice_id": resolved_voice,
            "chunk_size": int(request.options.get("stream_chunk_size", request.options.get("chunk_size", self.cfg.options.get("default_stream_chunk_size", 50)))),
            "response_format": request.options.get("response_format", request.response_format if request.response_format == "pcm" else "pcm"),
            "print_metrics": bool(request.options.get("print_metrics", False)),
        }
        if request.voice and request.options.get("voice_mode") == "clone":
            payload.pop("predefined_voice_id", None)
            payload["voice_mode"] = "clone"
            payload["reference_audio_filename"] = resolved_voice
        for key in [
            "temperature",
            "exaggeration",
            "cfg_weight",
            "seed",
            "speed_factor",
            "language",
            # MLX sidecar generation/decoder-stream controls. These must pass through
            # the public Universal endpoint; otherwise direct sidecar tests can
            # sound different from the actual API path.
            "top_p",
            "top_k",
            "topk",
            "repetition_penalty",
            "max_tokens",
            "smooth_join_ms",
            "lowpass_hz",
            "experimental_commit_stream",
            "commit_holdback_samples",
            "commit_search_samples",
            "true_stream_quality",
            "true_stream_quality_mode",
            "generation_mode",
            "delivery_mode",
            "chatterbox_delivery_mode",
            "full_generate_chunk_bytes",
        ]:
            if key in request.options:
                payload[key] = request.options[key]
        if request.speed is not None:
            payload["speed_factor"] = request.speed
        return payload


class AudioCppProvider(HttpBackedProvider):
    """Adapter for audio.cpp's OpenAI-ish offline server endpoint.

    audio.cpp currently returns a complete WAV from /v1/audio/speech. Universal
    can still expose it through the common speech-stream route, but first byte
    from this provider means full synthesis completed upstream.
    """

    _TOP_LEVEL_OPTIONS = {
        "seed",
        "temperature",
        "top_k",
        "top_p",
        "max_tokens",
        "max_steps",
        "repetition_penalty",
        "guidance_scale",
        "num_inference_steps",
    }

    def speech_payload(self, request: TTSRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.cfg.options.get("upstream_model", request.model),
            "input": request.text,
            # The audio.cpp server only returns WAV or JSON-wrapped WAV today.
            "response_format": "wav",
        }

        voice_mode = self.cfg.options.get("voice_mode", "named")
        voice_aliases = self.cfg.options.get("voice_aliases", {})
        resolved_voice = voice_aliases.get(request.voice, request.voice) if request.voice else None
        passthrough_options: dict[str, Any] = dict(self.cfg.options.get("default_request_options") or {})

        if voice_mode == "reference":
            ref_audio = (
                request.options.get("voice_ref")
                or request.options.get("ref_audio")
                or request.options.get("reference_audio")
                or request.options.get("reference_audio_path")
                or self.cfg.options.get("default_ref_audio")
            )
            ref_text = (
                request.options.get("reference_text")
                or request.options.get("ref_text")
                or request.options.get("reference_transcript")
                or self.cfg.options.get("default_ref_text")
            )
            if ref_audio:
                payload["voice_ref"] = ref_audio
            if ref_text:
                payload["reference_text"] = ref_text
        elif resolved_voice:
            payload["voice"] = resolved_voice

        if request.speed is not None:
            passthrough_options["speed"] = request.speed
        if "instruct" in request.options:
            payload["instructions"] = request.options["instruct"]
        if "instructions" in request.options:
            payload["instructions"] = request.options["instructions"]
        if "language" in request.options:
            payload["language"] = request.options["language"]
        if "max_new_tokens" in request.options:
            payload["max_tokens"] = request.options["max_new_tokens"]

        reserved = {
            "cancel_event",
            "voice_ref",
            "ref_audio",
            "reference_audio",
            "reference_audio_path",
            "reference_text",
            "ref_text",
            "reference_transcript",
            "instruct",
            "instructions",
            "language",
            "max_new_tokens",
        }
        for key, value in request.options.items():
            if key in reserved:
                continue
            if key in self._TOP_LEVEL_OPTIONS:
                payload[key] = value
            else:
                passthrough_options[key] = value
        if passthrough_options:
            payload["options"] = passthrough_options
        return payload


PROVIDER_FACTORIES = {
    "qwen3": Qwen3Provider,
    "miso": MisoProvider,
    "chatterbox-turbo": ChatterboxTurboProvider,
    "audiocpp": AudioCppProvider,
    "http": HttpBackedProvider,
}
