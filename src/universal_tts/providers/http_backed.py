from __future__ import annotations

import asyncio
import base64
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
            return ProviderStatus(id=self.cfg.id, loaded=healthy, healthy=healthy, details={"status_code": response.status_code, "url": url, "body": body})
        except Exception as e:
            return ProviderStatus(id=self.cfg.id, loaded=False, healthy=False, details={"url": url, "error": str(e)})

    async def load(self) -> ProviderStatus:
        self.lifecycle.start()
        timeout = float(self.cfg.options.get("startup_timeout_sec", 120))
        deadline = asyncio.get_event_loop().time() + timeout
        last = await self.status()
        while not last.loaded and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(1)
            last = await self.status()
        return last

    async def unload(self) -> ProviderStatus:
        self.lifecycle.stop()
        return await self.status()

    async def synthesize(self, request: TTSRequest) -> tuple[bytes, str]:
        if not self.cfg.base_url:
            raise RuntimeError(f"{self.cfg.id} has no base_url")
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(self.cfg.base_url.rstrip("/") + "/v1/audio/speech", json=self.speech_payload(request))
        if response.status_code >= 400:
            raise RuntimeError(f"{self.cfg.id} synthesis failed: {response.status_code} {response.text[:500]}")
        return response.content, response.headers.get("content-type", "audio/wav")

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
                    async for chunk in response.aiter_bytes(chunk_size=stream_read_bytes):
                        if chunk:
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


PROVIDER_FACTORIES = {
    "qwen3": Qwen3Provider,
    "miso": MisoProvider,
    "chatterbox-turbo": ChatterboxTurboProvider,
    "http": HttpBackedProvider,
}
