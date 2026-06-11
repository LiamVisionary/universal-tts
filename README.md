# Universal TTS

Universal TTS is the single local TTS control surface for this machine. It exposes one OpenAI-compatible API on `127.0.0.1:8799`, handles shared runtime concerns once, and routes requests to isolated model/provider sidecars where the heavy or incompatible ML dependencies live.

The current repo covers:

- Qwen3-TTS via the local MLX sidecar
- MisoTTS
- Chatterbox Turbo via the local MLX sidecar
- VibeVoice CoreML
- VibeVoice.cpp
- KittenTTS (dedicated ONNX sidecar on `.kitten-venv`)
- Kokoro-82M (dedicated Torch/Misaki sidecar on `.kokoro-venv`)
- Fish Audio S2 Pro 8-bit MLX (dedicated MLX-Audio sidecar on `.fish-s2-venv`; Fish Audio Research License)

Provider sidecars are owned by Universal TTS and should normally not be called directly by clients. Clients should use the Universal TTS endpoints below so request normalization, model routing, memory checks, lifecycle control, audio format conversion, batching, streaming metadata, and provider quirks stay centralized.

### Sidecar install status

- `qwen3` — MLX sidecar, true streaming, port `8766`
- `miso` — MisoTTS, port `8775`
- `chatterbox-turbo` — MLX sidecar, port `8777`
- `vibevoice-coreml` — CoreML resident, port `8781`
- `vibevoice-cpp` — ggml/Metal, port `8780`
- `kitten` — ONNX sidecar, port `8782`, dedicated `.kitten-venv`
- `kokoro` — Kokoro-82M sidecar, port `8783`, dedicated `.kokoro-venv`, 54 voices
- `fish-s2` — Fish Audio S2 Pro 8-bit MLX sidecar, port `8784`, dedicated `.fish-s2-venv`, zero-shot voice cloning with `ref_audio` + `ref_text`

## Service address

- Universal API: `http://127.0.0.1:8799`
- Config: `config.yaml`
- Entrypoint: `./run.sh`
- FastAPI app: `src/universal_tts/app.py`

Run locally:

```bash
cd /Users/liam/voice-lab/universal-tts
./run.sh
```

Tests:

```bash
cd /Users/liam/voice-lab/universal-tts
.venv/bin/python -m pytest -q
```

## Architecture

```text
universal_tts.app FastAPI API
  ↓
ProviderRegistry
  ↓
Shared runtime layer:
  - config loading
  - model → provider routing
  - OpenAI request normalization
  - Apple unified-memory guard
  - provider lifecycle start/stop/status
  - provider-scoped voices/paralinguistics discovery
  - microbatch queue and async jobs
  - streaming headers, PCM frame coalescing, realtime pacing
  - ffmpeg-backed audio format conversion
  - privacy-preserving proxy path; request bodies/prompts are not logged
  ↓
Provider adapters:
  - Qwen3Provider
  - MisoProvider
  - ChatterboxTurboProvider
  - KittenTTSProvider (HTTP-backed ONNX sidecar client)
  - generic HttpBackedProvider for Kokoro / Fish S2 / VibeVoice CoreML / VibeVoice.cpp and future HTTP providers
  ↓
Isolated sidecars:
  - qwen3 MLX service on :8766
  - MisoTTS service on :8775
  - Chatterbox Turbo MLX service on :8777
  - VibeVoice CoreML service on :8781
  - VibeVoice.cpp service on :8780
  - KittenTTS ONNX service on :8782 (dedicated `.kitten-venv`)
  - Kokoro-82M Torch/Misaki service on :8783 (dedicated `.kokoro-venv`)
  - Fish Audio S2 Pro MLX service on :8784 (dedicated `.fish-s2-venv`)
```

Common code lives in `src/universal_tts/`:

- `app.py` — FastAPI routes and streaming response wrapper.
- `config.py` — YAML config schema and model/provider mapping.
- `registry.py` — routing, loading, synthesis, streaming, capabilities, voices, batching, and microbatch scheduling.
- `providers/base.py` — provider protocol plus `TTSRequest` / `ProviderStatus` dataclasses.
- `providers/http_backed.py` — HTTP/lifecycle-backed provider adapters and provider-specific payload shaping.
- `lifecycle.py` — command/launchd lifecycle helpers.
- `memory.py` — Apple unified-memory snapshot and OOM guard.
- `audio.py` — `response_format` normalization and ffmpeg conversion.
- `queue.py` — cancellable async audio jobs.

## Core features

- **Single API for all local TTS models**: all clients can use `/v1/audio/speech` and `/v1/audio/speech-stream` with OpenAI-ish request fields.
- **Model alias routing**: `model` values from `config.yaml` route to the correct provider; OpenAI-style generic names such as `tts-1` are also handled.
- **Provider lifecycle management**: `/providers/{provider}/load` starts sidecars on demand, waits for health, and supports multi-provider or exclusive loading.
- **Unified-memory safety**: before loading a provider, Universal checks the configured `estimate_gb` against available memory and `memory.reserve_gb`.
- **Provider isolation**: Torch, MLX, CoreML, ggml, and other incompatible dependency stacks stay in sidecar projects instead of being imported into one Python process.
- **OpenAI-compatible speech endpoint**: accepts `model`, `voice`, `input`, `response_format`, and `speed`, plus provider-specific passthrough options.
- **Streaming endpoint for every provider**: `/v1/audio/speech-stream` is always present. Capabilities distinguish true decoder streaming from compatibility/fallback behavior.
- **Raw PCM calling path support**: true-streaming providers emit `audio/pcm` with sample-rate/channel/sample-format headers. Universal can coalesce raw PCM into stable frames and pace delivery in realtime.
- **Audio format conversion**: non-streaming `/v1/audio/speech` supports `wav`, `mp3`, `opus`, `flac`, `aac`, and raw `pcm` when ffmpeg is available.
- **Provider-scoped voices**: `/providers/{provider}/voices` queries only that provider instead of loading every model to aggregate voices.
- **Global voices redirect**: `/voices` and `/v1/voices` redirect to the first already-loaded healthy provider with a voices endpoint.
- **Paralinguistics discovery**: Chatterbox and other providers can expose supported nonverbal/style tokens through provider-specific or default paralinguistics endpoints.
- **Batch APIs**: `/v1/audio/batches` handles grouped requests. The registry also microbatches concurrent `/v1/audio/speech` calls per provider.
- **Async jobs and cancellation**: `/v1/audio/jobs` creates cancellable background synthesis jobs; job content can be fetched after completion.
- **Privacy-preserving proxy**: `/proxy/{provider}/{path}` forwards to sidecars without logging request bodies or prompts.

## Request model

Common request fields accepted by Universal:

- `model` — model ID or provider ID. If omitted, Universal uses the first configured provider.
- `input` — required text to synthesize.
- `voice` — optional voice/profile/preset ID. Some providers apply `voice_aliases`.
- `response_format` — `wav`, `mp3`, `opus`, `flac`, `aac`, or `pcm`. Defaults to `wav`.
- `speed` — OpenAI-style speed scalar. Chatterbox maps this to `speed_factor`.

Everything else in the JSON payload is preserved in `TTSRequest.options` and passed through to the provider adapter unless the adapter intentionally rewrites it. That is how model-specific knobs such as `instruct`, `language`, `temperature`, `cfg_weight`, `smooth_join_ms`, and `stream_frame_ms` reach the correct sidecar.

Minimal non-streaming example:

```bash
curl -X POST http://127.0.0.1:8799/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -o out.wav \
  -d '{"model":"miso-tts-8b","voice":"voice1-all-samples","input":"Universal TTS is online.","response_format":"wav"}'
```

Minimal streaming example:

```bash
curl -X POST http://127.0.0.1:8799/v1/audio/speech-stream \
  -H 'Content-Type: application/json' \
  -o stream.pcm \
  -d '{"model":"qwen3-tts-0.6b-base-clone","voice":"voice01","input":"Low latency streaming is online.","response_format":"pcm"}'
```

## Public endpoints

Health, discovery, and lifecycle:

- `GET /health` — service health, configured provider IDs, and memory snapshot.
- `GET /memory` — current memory snapshot.
- `GET /providers` — provider status map. Alias: `GET /runtimes`.
- `POST /providers/{provider_id}/load` — start a provider sidecar. Alias: `POST /runtimes/{provider_id}/load`.
- `POST /providers/{provider_id}/unload` — stop a provider sidecar. Alias: `POST /runtimes/{provider_id}/unload`.
- `POST /load-profile/voice-all` — attempt to load every configured provider in `multi` mode.
- `GET /v1/models` — OpenAI-style model list with provider and loaded state.
- `GET /capabilities` — full capability report. Alias: `GET /v1/audio/capabilities`.

Voices and paralinguistics:

- `GET /providers/{provider_id}/voices` — provider-scoped voice list. Aliases: `GET /runtimes/{provider_id}/voices`, `GET /v1/audio/{provider_id}/voices`.
- `GET /voices` — redirect to the first loaded provider's voice list.
- `GET /v1/voices` — redirect to the first loaded provider's voice list.
- `GET /providers/{provider_id}/paralinguistics` — provider-scoped paralinguistic/style token list. Aliases: `GET /runtimes/{provider_id}/paralinguistics`, `GET /v1/audio/{provider_id}/paralinguistics`.
- `GET /v1/audio/paralinguistics` — default paralinguistics endpoint. Defaults to Chatterbox Turbo, or accepts `provider=` / `model=` query params.

Speech, streaming, batching, and jobs:

- `POST /v1/audio/speech` — non-streaming synthesis. Returns audio bytes in the requested format.
- `POST /v1/audio/speech-stream` — streaming synthesis. Returns a `StreamingResponse`; raw PCM streams are framed/paced by Universal.
- `POST /v1/audio/batches` — grouped synthesis; payload uses `items` or `requests` as a list of normal speech payloads.
- `POST /v1/audio/jobs` — create an async synthesis job. Returns `202` with an `audio.job` object.
- `GET /v1/audio/jobs/{job_id}` — read job status; includes byte count after completion.
- `GET /v1/audio/jobs/{job_id}/content` — fetch completed job audio bytes.
- `DELETE /v1/audio/jobs/{job_id}` — cancel a queued/running job.
- `POST /v1/audio/jobs/{job_id}/cancel` — cancellation alias.

Proxy:

- `GET|POST|PUT|PATCH|DELETE /proxy/{provider_id}/{path}` — forward to a provider sidecar. Request bodies and prompts are not logged by Universal.

## Lifecycle loading

Load request body:

```json
{"mode":"multi","force":false}
```

- `mode: "multi"` — load the provider without unloading other providers.
- `mode: "exclusive"` — unload every other loaded provider first.
- `force: true` — bypass the memory guard if you intentionally want to load despite the reserve check.

Examples:

```bash
curl -X POST http://127.0.0.1:8799/providers/qwen3/load \
  -H 'Content-Type: application/json' \
  -d '{"mode":"multi"}'

curl -X POST http://127.0.0.1:8799/providers/vibevoice-coreml/load \
  -H 'Content-Type: application/json' \
  -d '{"mode":"exclusive"}'

curl -X POST http://127.0.0.1:8799/providers/chatterbox-turbo/unload
```

## Streaming behavior and headers

Universal always exposes `POST /v1/audio/speech-stream`, but **streaming API support is not the same as true realtime decoder streaming**. Check `/v1/audio/capabilities` before using a provider for calls.

True raw PCM streaming responses include headers such as:

- `X-Universal-TTS-Streaming: true`
- `X-Audio-Sample-Rate: 24000`
- `X-Audio-Channels: 1`
- `X-Audio-Sample-Format: pcm16`
- `X-Universal-TTS-Streaming-Implementation: ...`
- `X-Universal-TTS-PCM-Frame-MS: ...`
- `X-Universal-TTS-Realtime-Pacing: true|false`
- `X-Universal-TTS-PCM-Declick-MS: ...`

Streaming request options handled by Universal itself:

- `stream_frame_ms` or `frame_ms` — output PCM frame size. Defaults to `20` ms.
- `realtime_pacing` — whether Universal throttles PCM chunks to the audio clock. Defaults to `true`.
- `pcm_declick_ms` — optional Universal-level PCM de-clicking. Defaults to `0.0`; keep opt-in because HTTP read boundaries are not always model chunk boundaries.
- `pcm_declick_threshold` — threshold for optional de-clicking. Defaults to `300`.
- `chunk_bytes` — fallback chunk size when a provider has no stream path and Universal chunks a full generated file.

The stream endpoint rejects `generation_mode`, `delivery_mode`, or `chatterbox_delivery_mode` values like `full_generate`, `quality`, or `quality_reference` with HTTP `409`, because full generation is not streaming. Use `/v1/audio/speech` for quality-reference/full-generate output.

## Audio formats

`response_format` supports:

- `wav` → `audio/wav`
- `mp3` / `mpeg` → `audio/mpeg`
- `opus` → `audio/ogg; codecs=opus`
- `ogg` → `audio/ogg`
- `flac` → `audio/flac`
- `aac` → `audio/aac`
- `pcm` → `audio/pcm`, converted to 24 kHz mono signed 16-bit little-endian PCM for non-streaming conversions

Format conversion is ffmpeg-backed. Universal looks for `ffmpeg` on `PATH`, then `/opt/homebrew/bin/ffmpeg`.

## Batching and jobs

Batch request:

```bash
curl -X POST http://127.0.0.1:8799/v1/audio/batches \
  -H 'Content-Type: application/json' \
  -d '{"items":[{"model":"miso-tts-8b","input":"one"},{"model":"miso-tts-8b","input":"two"}]}'
```

Notes:

- All items in a batch must route to the same provider.
- `max_batch_size` and `batch_window_ms` come from provider `options`.
- Providers with `batch_path` and `supports_native_batching: true` can use native sidecar batching.
- Other providers use Universal's microbatch scheduler and sequential provider execution.
- Per-request `disable_microbatch: true` bypasses the scheduler for `/v1/audio/speech`.

Async job example:

```bash
job=$(curl -s -X POST http://127.0.0.1:8799/v1/audio/jobs \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-tts-0.6b-base-clone","voice":"voice01","input":"Async synthesis."}')

curl http://127.0.0.1:8799/v1/audio/jobs/$JOB_ID
curl -o job.wav http://127.0.0.1:8799/v1/audio/jobs/$JOB_ID/content
curl -X DELETE http://127.0.0.1:8799/v1/audio/jobs/$JOB_ID
```

## Configuration reference

`config.yaml` has three top-level sections:

```yaml
server:
  host: 127.0.0.1
  port: 8799
memory:
  reserve_gb: 12
providers:
  <provider_id>:
    kind: <adapter kind>
    category: tts
    cwd: <sidecar working directory>
    command: <command Universal runs to start sidecar>
    port: <sidecar port>
    base_url: http://127.0.0.1:<port>
    health_path: /health
    estimate_gb: <memory estimate for guard>
    models: [<model ids routed to this provider>]
    notes: <operator note>
    options: {...}
```

Provider config fields:

- `kind` — adapter factory key. Current values: `qwen3`, `miso`, `chatterbox-turbo`, and generic `http`.
- `category` — only `tts` providers are loaded by this repo.
- `cwd` — provider sidecar working directory.
- `command` — command used by `ProcessLifecycle` to start the sidecar.
- `port` — sidecar port used for process checks.
- `base_url` — upstream HTTP base URL.
- `health_path` — provider health endpoint. Defaults to `/health`.
- `estimate_gb` — memory estimate used by the guard before load.
- `models` — model IDs that route to the provider.
- `notes` — human-readable provider note exposed by `/providers`.
- `launchd_label` / `label` and `plist` — optional launchd-backed lifecycle fields if a provider uses launchd instead of `command`.
- `options` — provider/runtime features and adapter-specific defaults.

Common `options` keys:

- `stream_path` — upstream provider streaming path.
- `stream_content_type` — media type returned by the stream path, usually `audio/pcm`.
- `supports_true_streaming` — whether provider emits audio before full utterance completion.
- `streaming_kind` — machine-readable stream kind, e.g. `pcm16` or `compatibility`.
- `streaming_mode` — short implementation mode string.
- `streaming_implementation` — concrete runtime/backend identifier.
- `stream_sample_rate` — stream sample rate, currently `24000` for true PCM providers.
- `stream_channels` — stream channel count, currently `1`.
- `stream_sample_format` — stream sample format, currently `pcm16`.
- `supports_batching` — whether the Universal API should advertise batching support.
- `supports_native_batching` — whether the sidecar itself has a native batch path.
- `max_batch_size` — Universal/native batch size cap.
- `batch_window_ms` — microbatch collection window.
- `batch_path` — optional native sidecar batch endpoint.
- `formats` — advertised response formats.
- `voices` — static provider voice list.
- `voices_paths` — upstream paths to try when discovering voices.
- `voice_aliases` — public voice IDs mapped to provider-specific voice IDs/files.
- `paralinguistics` — static paralinguistic token list.
- `paralinguistics_paths` — upstream paths to try when discovering token lists.
- `startup_timeout_sec` — max seconds to wait for sidecar health after start. Defaults to `120`.
- `stream_read_bytes` — upstream streaming read size in bytes.

## Current providers and models

### `qwen3`

Purpose: Qwen3-TTS MLX adapter for Apple Silicon true streaming via `mlx-audio`, raw PCM chunks, and the resident 0.6B Base 8-bit `voice01` reference-clone profile.

Runtime:

- Adapter kind: `qwen3`
- Sidecar cwd: `/Users/liam/voice-lab/qwen3-mlx`
- Command: `/Users/liam/voice-lab/mlx-chatterbox/.venv/bin/python -m uvicorn server:app --host 127.0.0.1 --port 8766`
- Base URL: `http://127.0.0.1:8766`
- Health: `/health`
- Memory estimate: `4` GB

Models routed here:

- `qwen3-tts-1.7b-custom`
- `qwen3-tts-1.7b-design`
- `qwen3-tts-1.7b-base-clone`
- `qwen3-tts-0.6b-custom`
- `qwen3-tts-0.6b-base-clone`
- `tts-1`

Configured capabilities/options:

- Stream path: `/v1/audio/speech-stream`
- Stream content type: `audio/pcm`
- True streaming: `true`
- Streaming kind: `pcm16`
- Streaming mode: `mlx-audio-streaming-step`
- Streaming implementation: `mlx-audio-qwen3-streaming-step`
- PCM stream: 24 kHz, mono, signed 16-bit
- `default_max_new_tokens: 512`
- Batching API: `true`
- Native batching: `true`
- Native batch path: `/v1/audio/batches`
- `max_batch_size: 4`
- `batch_window_ms: 25`
- Formats: `wav`, `mp3`, `opus`, `flac`, `aac`, `pcm`

Adapter/request behavior:

- Passes through `instruct` for natural-language style/emotion/prosody control.
- Passes through `language`.
- Applies `default_max_new_tokens` when the request does not set `max_new_tokens`.
- Applies `default_seed` if configured in the future.
- Preserves other custom request fields as provider options.

Useful request options:

- `instruct` — natural-language speaking instruction.
- `language` — language hint accepted by the sidecar.
- `max_new_tokens` — overrides `default_max_new_tokens`.
- `seed` — deterministic generation if supported by the sidecar.
- `voice` — e.g. `voice01`, `voice01-xvector`, `Ryan`, `Aiden`, depending on the sidecar.
- `clone_mode` — sidecar-specific clone mode such as `xvector` when supported.

### `miso`

Purpose: MisoTTS adapter with conservative defaults for the verified cloned voice/profile and anti-clipping tail behavior.

Runtime:

- Adapter kind: `miso`
- Sidecar cwd: `/Users/liam/voice-lab/MisoTTS`
- Command: `/Users/liam/.local/bin/misotts-server`
- Base URL: `http://127.0.0.1:8775`
- Health: `/health`
- Memory estimate: `10` GB

Models routed here:

- `miso-tts-8b`

Configured capabilities/options:

- Stream path: `/v1/audio/speech-stream-pcm`
- Stream content type: `audio/pcm`
- True streaming: `true`
- Streaming kind: `pcm16`
- Streaming mode: `true-decoder-pcm-experimental`
- Streaming implementation: `provider-generate-stream`
- PCM stream: 24 kHz, mono, signed 16-bit
- Default voice: `voice1-all-samples`
- Default temperature: `0.55`
- Default top-k: `20`
- Default chunk frames: `1`
- Default seed: `1003`
- Voice aliases: `liam-default → voice1-all-samples`
- Batching API: `true`
- Native batching: `false`
- `max_batch_size: 4`
- `batch_window_ms: 25`
- Formats: `wav`, `mp3`, `opus`, `flac`, `aac`, `pcm`

Adapter/request behavior:

- Defaults `voice` to `voice1-all-samples` when omitted.
- Defaults `temperature` to `0.55`.
- Defaults `topk` to `20`.
- Defaults `chunk_frames` to `1`.
- Defaults `seed` to `1003`.
- Defaults `asr_verify` to `false`.
- Defaults `auto_duration` to `false`.
- Defaults `tail_silence_ms` to `700`.
- Defaults `max_audio_length_ms` to `2500`.
- Preserves request overrides for these fields and other provider-specific options.

Useful request options:

- `voice` — `voice1-all-samples` or alias `liam-default`.
- `temperature`
- `topk`
- `chunk_frames`
- `seed`
- `asr_verify`
- `auto_duration`
- `tail_silence_ms`
- `max_audio_length_ms`

### `chatterbox-turbo`

Purpose: Chatterbox Turbo MLX adapter for Apple Silicon true streaming with resident `mlx-audio` model, cached voice conditionals, supported paralinguistic tokens, and raw PCM chunks.

Runtime:

- Adapter kind: `chatterbox-turbo`
- Sidecar cwd: `/Users/liam/voice-lab/mlx-chatterbox`
- Command: `/Users/liam/voice-lab/mlx-chatterbox/.venv/bin/python -m uvicorn server:app --host 127.0.0.1 --port 8777`
- Base URL: `http://127.0.0.1:8777`
- Health: `/health`
- Memory estimate: `5` GB

Models routed here:

- `chatterbox-turbo`
- `ResembleAI/chatterbox-turbo`

Configured capabilities/options:

- Stream path: `/v1/audio/speech-stream`
- Stream content type: `audio/pcm`
- True streaming: `true`
- Streaming kind: `pcm16`
- Streaming mode: `mlx-audio-stream-generate`
- Streaming implementation: `mlx-audio-plus-chatterbox-turbo-tts-fp16`
- PCM stream: 24 kHz, mono, signed 16-bit
- Default stream chunk size: `6`
- Upstream read size: `240` bytes
- Batching API: `true`
- Native batching: `false`
- `max_batch_size: 4`
- `batch_window_ms: 25`
- Formats: `wav`, `mp3`, `opus`, `flac`, `aac`, `pcm`

Configured voices:

- `voice01`
- `voice1-all-samples-10s.wav`
- `Thomas.wav`, `Taylor.wav`, `Ryan.wav`, `Olivia.wav`, `Miles.wav`, `Michael.wav`, `Leonardo.wav`, `Layla.wav`, `Julian.wav`, `Jordan.wav`, `Jeremiah.wav`, `Jade.wav`, `Ian.wav`, `Henry.wav`, `Gianna.wav`, `Gabriel.wav`, `Everett.wav`, `Emily.wav`, `Eli.wav`, `Elena.wav`, `Cora.wav`, `Connor.wav`, `Axel.wav`, `Austin.wav`, `Alice.wav`, `Alexander.wav`, `Adrian.wav`, `Abigail.wav`

Voice aliases:

- `voice01 → voice1-all-samples-10s.wav`
- `liam-default → voice1-all-samples-10s.wav`

Adapter/request behavior:

- Resolves public voice aliases before sending to the sidecar.
- Maps OpenAI `speed` to Chatterbox `speed_factor`.
- Streaming payload uses `text`, `voice_mode`, `predefined_voice_id` or `reference_audio_filename`, `chunk_size`, `response_format`, and provider knobs.
- If `voice_mode: "clone"`, Universal sends `reference_audio_filename` instead of `predefined_voice_id`.
- Full-generate/quality-reference delivery is intentionally rejected from `/v1/audio/speech-stream`; use `/v1/audio/speech` for that path.

Useful request options:

- `voice` — public voice ID or alias.
- `voice_mode` — `predefined` by default, or `clone` for reference audio filename mode.
- `speed` or `speed_factor`
- `temperature`
- `exaggeration`
- `cfg_weight`
- `seed`
- `language`
- `top_p`
- `top_k` / `topk`
- `repetition_penalty`
- `max_tokens`
- `stream_chunk_size` / `chunk_size`
- `print_metrics`
- `smooth_join_ms`
- `lowpass_hz`
- `experimental_commit_stream`
- `commit_holdback_samples`
- `commit_search_samples`
- `true_stream_quality`
- `true_stream_quality_mode`
- `generation_mode`
- `delivery_mode`
- `chatterbox_delivery_mode`
- `full_generate_chunk_bytes`

Paralinguistics:

- Use `GET /v1/audio/paralinguistics` or `GET /providers/chatterbox-turbo/paralinguistics` to discover supported bracket tokens dynamically.
- Do not hardcode unsupported variants; use the provider endpoint output.

### `vibevoice-coreml`

Purpose: VibeVoice CoreML 0.5B adapter for Apple Silicon true decoder PCM streaming with resident CoreML process and provider-scoped voices.

Runtime:

- Adapter kind: `http`
- Sidecar cwd: `/Users/liam/voice-lab/vibevoice-coreml`
- Command: `/Users/liam/.local/bin/vibevoice-coreml-server`
- Base URL: `http://127.0.0.1:8781`
- Health: `/health`
- Memory estimate: `4` GB

Models routed here:

- `vibevoice`
- `vibevoice-coreml-0.5b`
- `vibevoice-realtime-0.5b`
- `vibevoice-realtime-0.5B`

Configured capabilities/options:

- Stream path: `/v1/audio/speech-stream`
- Stream content type: `audio/pcm`
- True streaming: `true`
- Streaming kind: `pcm16`
- Streaming mode: `true-decoder-pcm`
- Streaming implementation: `resident-coreml`
- PCM stream: 24 kHz, mono, signed 16-bit
- Batching API: `true`
- Native batching: `false`
- `max_batch_size: 4`
- `batch_window_ms: 25`
- Voice discovery paths: `/v1/audio/voices`, `/v1/voices`
- Voice aliases: `liam-default → en-Emma_woman`
- Formats: `wav`, `mp3`, `opus`, `flac`, `aac`, `pcm`

Adapter/request behavior:

- Uses the generic HTTP-backed provider. Common fields and all extra request options pass through to the sidecar.
- Voice aliases are resolved by the common HTTP adapter before forwarding.
- Voices are discovered from the sidecar via the configured voice paths.

Useful request options:

- `voice` — sidecar voice preset such as `en-Emma_woman` or alias `liam-default`.
- `language` — accepted by Universal as passthrough if the sidecar supports it.
- `cfg_scale`, `eos_tail_frames`, `tail_silence_ms`, or other VibeVoice-sidecar fields when supported by the sidecar.

### `vibevoice-cpp`

Purpose: VibeVoice.cpp Metal/ggml runtime and GGUF voice presets. This is a fallback/alternate runtime; current CLI path is full-generation chunk streaming rather than true decoder streaming.

Runtime:

- Adapter kind: `http`
- Sidecar cwd: `/Users/liam/voice-lab/vibevoice.cpp`
- Command: `/Users/liam/.local/bin/vibevoice-cpp-server`
- Base URL: `http://127.0.0.1:8780`
- Health: `/health`
- Memory estimate: `4` GB

Models routed here:

- `vibevoice-cpp`

Configured capabilities/options:

- True streaming: `false`
- Streaming mode: `full-generate-then-chunk`
- Batching API: `true`
- Native batching: `false`
- `max_batch_size: 4`
- `batch_window_ms: 25`
- Voice discovery paths: `/v1/audio/voices`, `/v1/voices`
- Formats: `wav`, `mp3`, `opus`, `flac`, `aac`, `pcm`

Adapter/request behavior:

- Uses the generic HTTP-backed provider.
- Because `stream_path` is not configured, Universal's fallback stream path chunks a full generated audio file as `audio/wav` if streaming is requested.
- Do not treat this provider as call-ready true streaming unless the sidecar and config are upgraded to expose real incremental decoder audio.

### `kokoro`

Purpose: Kokoro-82M sidecar. Kokoro is an Apache-licensed open-weight 82M parameter TTS model. The full Hugging Face snapshot is pre-downloaded on this machine: `kokoro-v1_0.pth`, `config.json`, docs/sample metadata, and all 54 `voices/*.pt` voice tensors. Japanese and Chinese G2P extras (`misaki[ja,zh]`) are installed in the sidecar venv so non-English voice files are usable rather than merely present on disk.

Why a sidecar (not in-process): Kokoro pulls in Torch, Transformers, Misaki, spaCy, espeak, Japanese/Chinese G2P extras, and language-specific tokenizers. Those dependencies belong in a dedicated `.kokoro-venv`, not the universal-tts core venv.

Runtime:

- Adapter kind: `http`
- Sidecar cwd: `/Users/liam/voice-lab/universal-tts`
- Sidecar command: `sidecars/kokoro_sidecar.sh`
- Sidecar venv: `.kokoro-venv`
- Sidecar base URL: `http://127.0.0.1:8783`
- Sidecar health: `/health`
- Memory estimate: `3` GB
- Default model: `hexgrad/Kokoro-82M`
- Default voice: `af_heart`

Models routed here:

- `hexgrad/Kokoro-82M`
- `kokoro`
- `kokoro-82m`
- `kokoro-v1_0`

Voices:

- American English (`lang_code=a`): `af_alloy`, `af_aoede`, `af_bella`, `af_heart`, `af_jessica`, `af_kore`, `af_nicole`, `af_nova`, `af_river`, `af_sarah`, `af_sky`, `am_adam`, `am_echo`, `am_eric`, `am_fenrir`, `am_liam`, `am_michael`, `am_onyx`, `am_puck`, `am_santa`
- British English (`lang_code=b`): `bf_alice`, `bf_emma`, `bf_isabella`, `bf_lily`, `bm_daniel`, `bm_fable`, `bm_george`, `bm_lewis`
- Spanish (`lang_code=e`): `ef_dora`, `em_alex`, `em_santa`
- French (`lang_code=f`): `ff_siwis`
- Hindi (`lang_code=h`): `hf_alpha`, `hf_beta`, `hm_omega`, `hm_psi`
- Italian (`lang_code=i`): `if_sara`, `im_nicola`
- Japanese (`lang_code=j`): `jf_alpha`, `jf_gongitsune`, `jf_nezumi`, `jf_tebukuro`, `jm_kumo`
- Brazilian Portuguese (`lang_code=p`): `pf_dora`, `pm_alex`, `pm_santa`
- Mandarin Chinese (`lang_code=z`): `zf_xiaobei`, `zf_xiaoni`, `zf_xiaoxiao`, `zf_xiaoyi`, `zm_yunjian`, `zm_yunxi`, `zm_yunxia`, `zm_yunyang`

Configured capabilities/options:

- Stream path: `/v1/audio/speech-stream`
- Stream content type: `audio/pcm`
- Streaming kind: `pcm16`
- Streaming mode: `segment-incremental-pcm`
- Streaming implementation: `kokoro-kpipeline-segment-generator`
- PCM stream: 24 kHz, mono, signed 16-bit
- `max_segment_chars: 220` by default; callers can override `max_segment_chars` for finer streaming granularity.
- Voice aliases: `liam-default → af_heart`, `heart → af_heart`, `bella → af_bella`, `nicole → af_nicole`, `emma → bf_emma`, `liam → am_liam`
- Formats: `wav`, `mp3`, `opus`, `flac`, `aac`, `pcm`

Streaming truth:

Kokoro's public Python API does **not** expose decoder-frame callbacks. It exposes a `KPipeline(...)` generator that yields a completed audio tensor per text segment. The sidecar therefore implements real incremental streaming at the segment/phrase level: it force-splits input into short sentence/phrase units, starts synthesizing the first unit immediately, and emits raw PCM for each unit before later units are synthesized. This is not full-generate-then-chunk, but it is also not lower-level decoder-token streaming. The sidecar and capability metadata name this explicitly as `segment-incremental-pcm`.

Useful requests:

```bash
curl -X POST http://127.0.0.1:8799/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"kokoro","voice":"af_heart","input":"Hello from Kokoro.","response_format":"wav"}' \
  -o kokoro.wav

curl -N -X POST http://127.0.0.1:8799/v1/audio/speech-stream \
  -H 'Content-Type: application/json' \
  -d '{"model":"kokoro","voice":"af_heart","input":"Sentence one. Sentence two. Sentence three.","response_format":"pcm","max_segment_chars":80}' \
  -o kokoro.pcm
```

### `fish-s2`

Purpose: Fish Audio S2 Pro via the MLX-Audio 8-bit Apple Silicon conversion. This is a heavier, higher-quality multilingual voice-cloning TTS provider than Kokoro/Kitten. It is **not** permissively licensed: Fish S2 Pro is under the Fish Audio Research License, which permits research/non-commercial use and requires a separate Fish Audio commercial license for commercial use.

Why a sidecar (not in-process): Fish S2 Pro uses MLX/MLX-Audio and a large converted checkpoint. MLX streams are thread-local, so the sidecar owns a dedicated MLX worker thread that loads the model and runs all generation/reference-audio preprocessing on that same thread.

Runtime:

- Adapter kind: `http`
- Sidecar cwd: `/Users/liam/voice-lab/universal-tts`
- Sidecar command: `sidecars/fish_s2_sidecar.sh`
- Sidecar venv: `.fish-s2-venv`
- Sidecar base URL: `http://127.0.0.1:8784`
- Sidecar health: `/health`
- Memory estimate: `12` GB
- Default model: `mlx-community/fish-audio-s2-pro-8bit`
- Sample rate: 44.1 kHz mono PCM16

Models routed here:

- `fish-s2`
- `fish-s2-pro`
- `fishaudio/s2-pro`
- `mlx-community/fish-audio-s2-pro-8bit`

Voice / cloning behavior:

- `voice: default` — base/no-reference generation.
- `voice: clone` or alias `liam-default` — use zero-shot cloning when `ref_audio` and `ref_text` are supplied.
- Voice cloning requires both `ref_audio` and `ref_text`; the sidecar validates the reference WAV path and loads it at the model sample rate before generation.

Configured capabilities/options:

- Stream path: `/v1/audio/speech-stream`
- Stream content type: `audio/pcm`
- Streaming kind: `pcm16`
- Streaming mode: `prefix-incremental-pcm`
- Streaming implementation: `mlx-audio-fish-s2-prefix-code-decode`
- PCM stream: 44.1 kHz, mono, signed 16-bit
- Default stream commit size: `1` Fish semantic token, about 46 ms of audio at 44.1 kHz
- Universal default stream frame: `5` ms
- Universal default realtime pacing for Fish: `false` to avoid adding TTFA before the first frame; callers can opt back in with `"realtime_pacing": true`
- `default_chunk_length: 80`
- `default_max_tokens: 1024`
- Supports voice cloning: `true`
- Voice cloning requires: `ref_audio`, `ref_text`
- Formats: `wav`, `mp3`, `opus`, `flac`, `aac`, `pcm`

Streaming truth:

The official Fish S2 Pro SGLang/vLLM Omni server path supports native low-latency streaming on NVIDIA GPUs. The local Apple Silicon path here uses MLX-Audio. MLX-Audio Fish still raises `NotImplementedError` for its public decoder-frame `stream=True` flag, so the sidecar uses a lower-level prefix-code streaming path: it samples semantic/codebook tokens incrementally, periodically decodes the generated code prefix, and emits only the new PCM delta. This is not full-WAV chunking and no longer waits for a whole generated text segment. It is still not an official Fish decoder-frame callback, so keep the mode labeled `prefix-incremental-pcm` rather than claiming upstream-native SGLang/vLLM streaming.

Warm latency verification on this Mac through Universal TTS with `model: fish-s2`, `response_format: pcm`, no special request overrides:

- First cold-ish request after restart: around 175–235 ms while MLX kernels settle.
- Warm first byte / first non-silent PCM: typically 82–91 ms for the short `Yes.` benchmark.
- Response headers expose `X-Universal-TTS-Streaming-Implementation: mlx-audio-fish-s2-prefix-code-decode`, `X-Universal-TTS-PCM-Frame-MS: 5`, and `X-Universal-TTS-Realtime-Pacing: false`.

Useful requests:

```bash
curl -X POST http://127.0.0.1:8799/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"fish-s2","input":"Hello from Fish S2 Pro.","response_format":"wav","max_tokens":350,"chunk_length":80}' \
  -o fish-s2.wav

curl -X POST http://127.0.0.1:8799/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"fish-s2","voice":"clone","input":"This uses a reference voice.","ref_audio":"/path/to/reference.wav","ref_text":"Exact transcript of the reference clip.","response_format":"wav"}' \
  -o fish-s2-clone.wav

curl -N -X POST http://127.0.0.1:8799/v1/audio/speech-stream \
  -H 'Content-Type: application/json' \
  -d '{"model":"fish-s2","input":"Sentence one. Sentence two.","response_format":"pcm","chunk_length":45}' \
  -o fish-s2.pcm
```

### `kitten`

Purpose: KittenTTS ONNX sidecar. State-of-the-art tiny CPU TTS model (15M–80M parameters, 25–80 MB on disk) with 8 built-in voices and a real per-text-chunk `generate_stream` generator.

Why a sidecar (not in-process): KittenTTS needs `torch`, `onnxruntime`, `misaki`, and `spacy` to load — heavy incompatible deps that don't belong in the universal-tts core venv. The sidecar runs in its own dedicated `.kitten-venv` so the universal-tts venv stays small. The adapter is a thin HTTP-backed client over the existing `HttpBackedProvider` base, so all the other Universal TTS features (capabilities, voices, batching, streaming headers, model routing) work without any per-provider special-casing.

Runtime:

- Adapter kind: `kitten`
- Sidecar cwd: `/Users/liam/voice-lab/universal-tts`
- Sidecar command: `sidecars/kitten_sidecar.sh`
- Sidecar base URL: `http://127.0.0.1:8782`
- Sidecar health: `/health`
- Memory estimate: `1` GB (conservative; ONNX runtime + cached model is small)

Models routed here:

- `KittenML/kitten-tts-mini-0.8` (80M, highest quality)
- `KittenML/kitten-tts-micro-0.8` (40M)
- `KittenML/kitten-tts-nano-0.8` (15M, smallest and fastest)
- Friendly aliases: `kitten-tts-mini`, `kitten-tts-micro`, `kitten-tts-nano`
- `tts-1`

Built-in voices:

- `Bella`, `Jasper`, `Luna`, `Bruno`, `Rosie`, `Hugo`, `Kiki`, `Leo`
- Voice aliases: `liam-default → Bella`, plus lowercase aliases for each built-in voice.

Configured capabilities/options:

- Stream path: `/v1/audio/speech-stream`
- Stream content type: `audio/pcm`
- True streaming: `true`
- Streaming kind: `pcm16`
- Streaming mode: `true-decoder-pcm`
- Streaming implementation: `kitten-generate-stream`
- PCM stream: 24 kHz, mono, signed 16-bit
- `default_model: KittenML/kitten-tts-mini-0.8`
- `default_voice: Bella`
- `default_speed: 1.0`
- `default_clean_text: true`
- Batching API: `true`
- Native batching: `false`
- `max_batch_size: 4`
- `batch_window_ms: 25`
- Formats: `wav`, `mp3`, `opus`, `flac`, `aac`, `pcm`

Adapter/request behavior:

- HTTP-backed provider; `/providers/kitten/load` starts the sidecar via the configured command and waits for `/health` to return 200.
- Sidecar `POST /v1/audio/speech` returns a complete WAV file.
- Sidecar `POST /v1/audio/speech-stream` calls the ONNX model's `generate_stream(...)` and forwards each per-text-chunk audio as raw PCM frames over `Transfer-Encoding: chunked` with proper sample-rate/channels/format headers.
- Stream path is **true-decoder PCM**, not full-generate-then-chunk: audio arrives over wall-clock time as the model decodes.
- Exposes the static voice list at `/providers/kitten/voices`.
- Paralinguistics endpoint returns an empty list (KittenTTS does not expose native paralinguistic tokens).
- Voice aliases resolve on the client side (adapter) and in the sidecar's `KITTEN_VOICE_ALIASES` env passthrough, so any custom alias map works.

Useful request options:

- `voice` — built-in voice name or alias.
- `speed` — speech rate multiplier. Defaults to `1.0`.
- `clean_text` — whether to run the built-in text normalizer (numbers, currency, units, etc.) before synthesis. Defaults to `true`.

Install (KittenTTS venv):

```bash
cd /Users/liam/voice-lab/universal-tts
.venv/bin/python -m venv .kitten-venv
.kitten-venv/bin/pip install -U pip
# Install from main (the released 0.8.1 wheel ships without generate_stream)
.kitten-venv/bin/pip install -e /Users/liam/voice-lab/kitten-src/KittenTTS
# Or from the released wheel (note: 0.8.1 wheel lacks streaming — use main)
# .kitten-venv/bin/pip install 'https://github.com/KittenML/KittenTTS/releases/download/0.8.1/kittentts-0.8.1-py3-none-any.whl'
.kitten-venv/bin/pip install --no-deps 'fastapi' 'uvicorn' 'starlette' 'sniffio'
```

Run the sidecar:

```bash
./sidecars/kitten_sidecar.sh
# or with overrides
KITTEN_MODEL=KittenML/kitten-tts-mini-0.8 KITTEN_PORT=8782 ./sidecars/kitten_sidecar.sh
```

The model itself is downloaded from Hugging Face on first use. Cold start to first request is roughly `~1.1s` for nano and slightly longer for mini.

#### Verified end-to-end

Tested against the live `com.liam.universal-tts` LaunchAgent + KittenTTS sidecar with the real `KittenML/kitten-tts-nano-0.8` model:

- `/providers/kitten/load` → `loaded: true, healthy: true`, model load ≈ 1.15s.
- `/v1/audio/speech` full WAV (9.84s audio) returned in ≈ 0.31s wall-clock.
- `/v1/audio/speech-stream` returned `Transfer-Encoding: chunked`, `Content-Type: audio/pcm`, `X-Audio-Sample-Rate: 24000`, `X-Audio-Channels: 1`, `X-Audio-Sample-Format: pcm16`, `X-Universal-TTS-Streaming-Implementation: kitten-generate-stream`, `X-Universal-TTS-PCM-Frame-MS: 20`.
- Long-text streaming test (164s of audio) delivered across ~4.4s of wall-clock pacing — first byte at 148ms, last byte at 4.39s — proving true-decoder streaming rather than burst-then-finish.
- Voice alias `liam-default → Bella` and lowercase `bruno → Bruno` both resolve correctly.
- `tts-1` model alias routes to kitten via the model_to_provider map and synthesizes fine.

## Capabilities response fields

`GET /v1/audio/capabilities` returns `object: universal_tts.capabilities` and a provider map. Each provider includes:

- `models`
- `loaded`
- `supports_streaming_api`
- `supports_true_streaming`
- `streaming_kind`
- `streaming_mode`
- `streaming_implementation`
- `sample_rate`
- `channels`
- `sample_format`
- `supports_batching_api`
- `supports_microbatch_scheduler`
- `supports_native_batching`
- `batching_kind`
- `supports_batching`
- `max_batch_size`
- `supports_cancellation`
- `formats`
- `voices_endpoint`

Use these fields instead of assuming every `/v1/audio/speech-stream` response is true realtime.

## Provider-specific branching policy

Universal normalizes every speech request into `TTSRequest`, but adapters still branch where behavior differs:

- Qwen3 needs instruction/language/max-token controls and an MLX true-streaming path.
- Miso needs conservative voice/sampling/duration defaults to avoid clipping and maintain verified cloned-voice behavior.
- Chatterbox Turbo needs voice aliasing, OpenAI `speed` → `speed_factor`, paralinguistic/token controls, and true-stream quality/commit-stream knobs.
- VibeVoice CoreML and VibeVoice.cpp use the generic HTTP path but expose different capability truth: CoreML is true PCM streaming, cpp is currently full-generate-then-chunk.
- KittenTTS runs in-process on a dedicated worker thread and lazy-imports the `kittentts` package. The registry treats it like any other provider for routing, capabilities, voices, batching, and streaming.

Keep common behavior in Universal; keep model-specific runtime logic in adapters/sidecars.

## Operational notes and pitfalls

- Prefer Universal TTS on `:8799` for all client traffic.
- Do not call sidecars directly unless debugging provider-specific behavior.
- Do not advertise true realtime streaming from a provider unless `/v1/audio/capabilities` says `supports_true_streaming: true` and the sidecar emits audible PCM before full utterance completion.
- `StreamingResponse` plus quick HTTP first byte is not enough; WAV headers and transport chunking can be fake streaming.
- For raw PCM browser/calling playback, use the sample-rate/channel/format headers and frame/pacing headers.
- Keep `pcm_declick_ms` opt-in; Universal read chunks are not guaranteed model chunk boundaries.
- Use provider-scoped voices endpoints to avoid surprise-loading all providers.
- Restarting or unloading a provider unloads its model; the next request may pay cold-start/warmup cost.
- The proxy route deliberately avoids body/prompt logging; keep diagnostics privacy-aware.
- For exact audio quality disputes, verify the exact delivered artifact, not just an internal generation or a similar prompt.

---

## Want to build and manage an entire private swarm of agents?

Want to build and manage an entire private swarm of agents all with shared memory, skills, and single setup?

Check out: https://hivemindos.liamvisionary.com  
X: [@TheHivemindOS](https://x.com/TheHivemindOS)

<p align="center">
  <a href="https://hivemindos.liamvisionary.com">
    <img src="assets/hivemindos-icon.png" alt="HivemindOS icon" width="180">
  </a>
</p>
