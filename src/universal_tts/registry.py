from __future__ import annotations

from typing import Callable
import asyncio

from universal_tts.audio import convert_audio_format, normalize_format
from universal_tts.config import RuntimeConfig
from universal_tts.memory import MemoryGuard, MemorySnapshot, get_memory_snapshot
from universal_tts.providers.base import ProviderStatus, TTSRequest, TTSProvider
from universal_tts.providers.http_backed import PROVIDER_FACTORIES
from universal_tts.providers.kitten import KittenTTSProvider


class ProviderRegistry:
    def __init__(self, runtime: RuntimeConfig, provider_factories: dict | None = None, memory_snapshot: Callable[[], MemorySnapshot] = get_memory_snapshot):
        self.runtime = runtime
        self.provider_factories = {**PROVIDER_FACTORIES, "kitten": KittenTTSProvider, **(provider_factories or {})}
        self.memory_snapshot = memory_snapshot
        self.memory_guard = MemoryGuard(runtime.memory.get("reserve_gb", 12))
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
            return await provider.voices()  # type: ignore[attr-defined]
        cfg = self.runtime.providers[provider_id]
        return {"object": "list", "data": [{"id": voice, "provider": provider_id} for voice in cfg.options.get("voices", [])]}

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
        headers = self.stream_headers(provider_id)
        if hasattr(provider, "stream_synthesize"):
            chunks, content_type = await provider.stream_synthesize(request)
            return chunks, content_type, headers

        async def one_chunk():
            audio, _content_type = await provider.synthesize(request)
            yield audio

        return one_chunk(), "audio/wav", headers
