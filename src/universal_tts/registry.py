from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable
import asyncio

from universal_tts.audio import convert_audio_format, normalize_format
from universal_tts.config import RuntimeConfig
from universal_tts.memory import MemoryGuard, MemorySnapshot, get_memory_snapshot
from universal_tts.providers.base import ProviderStatus, TTSRequest, TTSProvider
from universal_tts.providers.http_backed import PROVIDER_FACTORIES
from universal_tts.providers.kitten import KittenTTSProvider
from universal_tts.voice_library import VoiceLibrary


class ProviderRegistry:
    def __init__(self, runtime: RuntimeConfig, provider_factories: dict | None = None, memory_snapshot: Callable[[], MemorySnapshot] = get_memory_snapshot):
        self.runtime = runtime
        self.provider_factories = {**PROVIDER_FACTORIES, "kitten": KittenTTSProvider, **(provider_factories or {})}
        self.memory_snapshot = memory_snapshot
        self.memory_guard = MemoryGuard(runtime.memory.get("reserve_gb", 12))
        data_dir = Path(runtime.server.get("data_dir", Path(__file__).resolve().parents[2] / "data"))
        self.voice_library = VoiceLibrary(data_dir / "voice_library")
        self.providers: dict[str, TTSProvider] = {}
        self._microbatch_queues: dict[str, list[tuple[TTSRequest, asyncio.Future]]] = {}
        self._microbatch_tasks: dict[str, asyncio.Task] = {}
        for provider_id, cfg in runtime.providers.items():
            factory = self.provider_factories[cfg.kind]
            self.providers[provider_id] = factory(cfg)

    async def statuses(self) -> dict[str, ProviderStatus]:
        out: dict[str, ProviderStatus] = {}
        for provider_id, provider in self.providers.items():
            out[provider_id] = await provider.status()
        return out

    def provider_for_model(self, model: str) -> str:
        if model in self.runtime.model_to_provider:
            return self.runtime.model_to_provider[model]
        if model in self.providers:
            return model
        # OpenAI generic tts-1 maps to first configured provider.
        if model in {"tts-1", "tts-1-hd", "gpt-4o-mini-tts"} and self.providers:
            return next(iter(self.providers))
        raise KeyError(f"unknown model/provider: {model}")

    async def load(self, provider_id: str, mode: str = "multi", force: bool = False) -> ProviderStatus:
        if provider_id not in self.providers:
            raise KeyError(f"unknown provider: {provider_id}")
        current = await self.providers[provider_id].status()
        if current.loaded:
            return current
        cfg = self.runtime.providers[provider_id]
        self.memory_guard.assert_can_load(provider_id, cfg.estimate_gb, self.memory_snapshot(), force=force)
        if mode == "exclusive":
            statuses = await self.statuses()
            for other_id, status in statuses.items():
                if other_id != provider_id and status.loaded:
                    await self.providers[other_id].unload()
        elif mode != "multi":
            raise ValueError("mode must be 'multi' or 'exclusive'")
        return await self.providers[provider_id].load()

    async def unload(self, provider_id: str) -> ProviderStatus:
        if provider_id not in self.providers:
            raise KeyError(f"unknown provider: {provider_id}")
        return await self.providers[provider_id].unload()

    def normalize_request(self, payload: dict) -> TTSRequest:
        if not payload.get("input"):
            raise ValueError("input is required")
        reserved = {"model", "input", "voice", "response_format", "speed"}
        return TTSRequest(
            model=payload.get("model") or next(iter(self.providers)),
            text=str(payload["input"]),
            voice=payload.get("voice"),
            response_format=payload.get("response_format", "wav"),
            speed=payload.get("speed"),
            options={k: v for k, v in payload.items() if k not in reserved},
        )

    async def _synthesize_direct(self, request: TTSRequest) -> tuple[bytes, str]:
        provider_id = self.provider_for_model(request.model)
        await self.load(provider_id, mode="multi")
        request = self.apply_saved_voice(provider_id, request)
        audio, content_type = await self.providers[provider_id].synthesize(request)
        target_format = normalize_format(request.response_format)
        source_format = "wav" if "wav" in content_type else content_type.split("/")[-1].split(";")[0]
        return convert_audio_format(audio, source_format, target_format)

    async def synthesize(self, payload: dict) -> tuple[bytes, str]:
        request = self.normalize_request(payload)
        provider_id = self.provider_for_model(request.model)
        cfg = self.runtime.providers[provider_id]
        max_batch_size = int(cfg.options.get("max_batch_size", cfg.options.get("api_max_batch_size", 1)))
        if payload.get("disable_microbatch") or max_batch_size <= 1:
            return await self._synthesize_direct(request)
        return await self._synthesize_microbatched(provider_id, request)

    async def _synthesize_microbatched(self, provider_id: str, request: TTSRequest) -> tuple[bytes, str]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        queue = self._microbatch_queues.setdefault(provider_id, [])
        queue.append((request, future))
        cfg = self.runtime.providers[provider_id]
        max_batch_size = int(cfg.options.get("max_batch_size", 1))
        if len(queue) >= max_batch_size:
            task = self._microbatch_tasks.pop(provider_id, None)
            if task and not task.done():
                task.cancel()
            loop.create_task(self._flush_microbatch(provider_id))
        elif provider_id not in self._microbatch_tasks or self._microbatch_tasks[provider_id].done():
            self._microbatch_tasks[provider_id] = loop.create_task(self._delayed_flush_microbatch(provider_id))
        return await future

    async def _delayed_flush_microbatch(self, provider_id: str) -> None:
        cfg = self.runtime.providers[provider_id]
        await asyncio.sleep(float(cfg.options.get("batch_window_ms", 10)) / 1000.0)
        await self._flush_microbatch(provider_id)

    async def _flush_microbatch(self, provider_id: str) -> None:
        queue = self._microbatch_queues.get(provider_id, [])
        if not queue:
            return
        cfg = self.runtime.providers[provider_id]
        max_batch_size = int(cfg.options.get("max_batch_size", 1))
        batch = queue[:max_batch_size]
        del queue[:max_batch_size]
        requests = [item[0] for item in batch]
        futures = [item[1] for item in batch]
        try:
            results = await self._batch_synthesize_requests(provider_id, requests)
            for fut, result in zip(futures, results):
                if not fut.done():
                    fut.set_result(result)
        except Exception as e:
            for fut in futures:
                if not fut.done():
                    fut.set_exception(e)
        if queue:
            asyncio.get_running_loop().create_task(self._flush_microbatch(provider_id))

    async def _batch_synthesize_requests(self, provider_id: str, requests: list[TTSRequest]) -> list[tuple[bytes, str]]:
        await self.load(provider_id, mode="multi")
        provider = self.providers[provider_id]
        requests = [self.apply_saved_voice(provider_id, req) for req in requests]
        if hasattr(provider, "batch_synthesize"):
            raw_results = await provider.batch_synthesize(requests)  # type: ignore[attr-defined]
        else:
            raw_results = [await provider.synthesize(req) for req in requests]
        converted: list[tuple[bytes, str]] = []
        for req, (audio, content_type) in zip(requests, raw_results):
            source_format = "wav" if "wav" in content_type else content_type.split("/")[-1].split(";")[0]
            converted.append(convert_audio_format(audio, source_format, normalize_format(req.response_format)))
        return converted

    async def batch_synthesize(self, payloads: list[dict]) -> list[tuple[bytes, str]]:
        if not payloads:
            return []
        requests = [self.normalize_request(payload) for payload in payloads]
        provider_ids = {self.provider_for_model(req.model) for req in requests}
        if len(provider_ids) != 1:
            raise ValueError("batch_synthesize currently requires all requests to route to the same provider")
        provider_id = next(iter(provider_ids))
        return await self._batch_synthesize_requests(provider_id, requests)

    async def voices(self, provider_id: str):
        if provider_id not in self.providers:
            raise KeyError(f"unknown provider: {provider_id}")
        provider = self.providers[provider_id]
        if hasattr(provider, "voices"):
            data = await provider.voices()  # type: ignore[attr-defined]
        else:
            cfg = self.runtime.providers[provider_id]
            data = {"object": "list", "data": [{"id": voice, "provider": provider_id} for voice in cfg.options.get("voices", [])]}
        return self._merge_library_voices(provider_id, data)

    def _merge_library_voices(self, provider_id: str, data: dict[str, Any]) -> dict[str, Any]:
        existing = {str(item.get("id")) for item in data.get("data", []) if isinstance(item, dict)}
        merged = list(data.get("data", []))
        for profile in self.voice_library.list(provider_id):
            if profile.voice_id not in existing:
                merged.append(profile.public())
        out = dict(data)
        out["data"] = merged
        return out

    def provider_supports_voice_cloning(self, provider_id: str) -> bool:
        cfg = self.runtime.providers[provider_id]
        return bool(cfg.options.get("supports_voice_cloning"))

    def clone_profile_defaults(self, provider_id: str) -> tuple[str | None, dict[str, Any]]:
        provider_voice: str | None = None
        options: dict[str, Any] = {}
        if provider_id == "fish-s2":
            provider_voice = "clone"
        elif provider_id == "chatterbox-turbo":
            options["voice_mode"] = "clone"
        return provider_voice, options

    def apply_saved_voice(self, provider_id: str, request: TTSRequest) -> TTSRequest:
        if not request.voice:
            return request
        profile = self.voice_library.get(provider_id, request.voice)
        if profile is None:
            return request
        options = {**profile.options, **request.options}
        voice = profile.provider_voice
        if provider_id == "fish-s2":
            voice = voice or "clone"
            options.setdefault("ref_audio", profile.ref_audio)
            options.setdefault("ref_text", profile.ref_text)
        elif provider_id == "miso":
            voice = voice or profile.voice_id
            options.setdefault("reference_audio_path", profile.ref_audio)
            options.setdefault("reference_transcript", profile.ref_text)
        elif provider_id == "chatterbox-turbo":
            voice = voice or profile.ref_audio
            options.setdefault("voice_mode", "clone")
            options.setdefault("reference_audio_filename", profile.ref_audio)
        elif provider_id == "qwen3":
            voice = voice or profile.voice_id
            options.setdefault("ref_audio", profile.ref_audio)
            options.setdefault("ref_text", profile.ref_text)
        else:
            options.setdefault("ref_audio", profile.ref_audio)
            options.setdefault("ref_text", profile.ref_text)
        return replace(request, voice=voice, options=options)

    def create_voice(self, payload: dict[str, Any], provider_id: str | None = None) -> dict[str, Any]:
        model = payload.get("model")
        provider_id = provider_id or payload.get("provider") or (self.provider_for_model(model) if model else None)
        if not provider_id:
            raise ValueError("provider or model is required")
        if provider_id not in self.providers:
            raise KeyError(f"unknown provider: {provider_id}")
        if not self.provider_supports_voice_cloning(provider_id):
            raise ValueError(f"provider does not support saved voice cloning: {provider_id}")
        voice_id = payload.get("voice_id") or payload.get("id") or payload.get("name")
        if not voice_id:
            raise ValueError("voice_id or name is required")
        ref_audio = payload.get("ref_audio") or payload.get("reference_audio") or payload.get("reference_audio_path") or payload.get("audio_path")
        ref_text = payload.get("ref_text") or payload.get("reference_text") or payload.get("reference_transcript") or payload.get("transcript")
        provider_voice, default_options = self.clone_profile_defaults(provider_id)
        required = set(self.runtime.providers[provider_id].options.get("voice_cloning_requires", ["ref_audio", "ref_text"]))
        profile = self.voice_library.create(
            provider=provider_id,
            voice_id=str(voice_id),
            name=payload.get("name") or str(voice_id),
            ref_audio=str(ref_audio or ""),
            ref_text=str(ref_text or ""),
            description=payload.get("description"),
            provider_voice=payload.get("provider_voice") or provider_voice,
            options={**default_options, **dict(payload.get("options") or {})},
            overwrite=bool(payload.get("overwrite", False)),
            require_ref_text="ref_text" in required,
        )
        return {
            "ok": True,
            "object": "voice.registration",
            "status": "ready",
            "added_to_voice_library": True,
            "voice": profile.public(),
            "voices_endpoint": f"/providers/{provider_id}/voices",
            "usage": {
                "model": self.runtime.providers[provider_id].models[0] if self.runtime.providers[provider_id].models else provider_id,
                "voice": profile.voice_id,
                "input": "Text to speak",
            },
        }

    async def paralinguistics(self, provider_id: str):
        if provider_id not in self.providers:
            raise KeyError(f"unknown provider: {provider_id}")
        provider = self.providers[provider_id]
        if hasattr(provider, "paralinguistics"):
            return await provider.paralinguistics()  # type: ignore[attr-defined]
        return {"object": "audio.paralinguistics", "provider": provider_id, "data": []}

    async def first_loaded_voice_provider(self) -> str | None:
        statuses = await self.statuses()
        for provider_id in self.runtime.providers:
            status = statuses[provider_id]
            if status.loaded and status.healthy:
                return provider_id
        return None

    def stream_headers(self, provider_id: str) -> dict[str, str]:
        cfg = self.runtime.providers[provider_id]
        headers = {}
        if cfg.options.get("stream_sample_rate"):
            headers["X-Audio-Sample-Rate"] = str(cfg.options["stream_sample_rate"])
        if cfg.options.get("stream_channels"):
            headers["X-Audio-Channels"] = str(cfg.options["stream_channels"])
        if cfg.options.get("stream_sample_format"):
            headers["X-Audio-Sample-Format"] = str(cfg.options["stream_sample_format"])
        if cfg.options.get("streaming_implementation"):
            headers["X-Universal-TTS-Streaming-Implementation"] = str(cfg.options["streaming_implementation"])
        if cfg.options.get("stream_frame_ms"):
            headers["X-Universal-TTS-Default-PCM-Frame-MS"] = str(cfg.options["stream_frame_ms"])
        if "realtime_pacing" in cfg.options:
            headers["X-Universal-TTS-Default-Realtime-Pacing"] = str(cfg.options["realtime_pacing"]).lower()
        return headers

    async def capabilities(self) -> dict:
        statuses = await self.statuses()
        return {
            "object": "universal_tts.capabilities",
            "providers": {
                provider_id: {
                    "models": cfg.models,
                    "loaded": statuses[provider_id].loaded,
                    "supports_streaming_api": True,
                    "supports_true_streaming": bool(cfg.options.get("supports_true_streaming", False)),
                    "streaming_kind": cfg.options.get("streaming_kind", "true-decoder" if bool(cfg.options.get("supports_true_streaming", False)) else "compatibility"),
                    "streaming_mode": cfg.options.get("streaming_mode", "full-generate-then-chunk"),
                    "streaming_implementation": cfg.options.get("streaming_implementation", "provider" if bool(cfg.options.get("supports_true_streaming", False)) else "full-generate-then-chunk"),
                    "sample_rate": cfg.options.get("stream_sample_rate"),
                    "channels": cfg.options.get("stream_channels"),
                    "sample_format": cfg.options.get("stream_sample_format"),
                    "supports_batching_api": True,
                    "supports_microbatch_scheduler": int(cfg.options.get("max_batch_size", 1)) > 1,
                    "supports_native_batching": bool(cfg.options.get("supports_native_batching", cfg.options.get("batch_path"))),
                    "batching_kind": "native-provider" if bool(cfg.options.get("supports_native_batching", cfg.options.get("batch_path"))) else ("universal-microbatch-sequential-fallback" if int(cfg.options.get("max_batch_size", 1)) > 1 else "sequential"),
                    "supports_batching": True,
                    "max_batch_size": int(cfg.options.get("max_batch_size", 1)),
                    "supports_cancellation": True,
                    "supports_voice_cloning": bool(cfg.options.get("supports_voice_cloning", False)),
                    "voice_cloning_requires": cfg.options.get("voice_cloning_requires", []),
                    "voice_registration_endpoint": f"/providers/{provider_id}/voices" if bool(cfg.options.get("supports_voice_cloning", False)) else None,
                    "formats": cfg.options.get("formats", ["wav", "mp3", "opus", "flac", "aac", "pcm"]),
                    "voices_endpoint": f"/providers/{provider_id}/voices",
                }
                for provider_id, cfg in self.runtime.providers.items()
            },
        }

    async def stream_synthesize(self, payload: dict):
        request = self.normalize_request(payload)
        provider_id = self.provider_for_model(request.model)
        await self.load(provider_id, mode="multi")
        provider = self.providers[provider_id]
        request = self.apply_saved_voice(provider_id, request)
        headers = self.stream_headers(provider_id)
        if hasattr(provider, "stream_synthesize"):
            chunks, content_type = await provider.stream_synthesize(request)
            return chunks, content_type, headers

        async def one_chunk():
            audio, _content_type = await provider.synthesize(request)
            yield audio

        return one_chunk(), "audio/wav", headers
