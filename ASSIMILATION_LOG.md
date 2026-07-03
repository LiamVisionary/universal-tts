# GitHub Assimilation Log

## 2026-07-02T06:47:39.211535+00:00 - shared-brain

- Request: Add automatic engine provisioning to Universal TTS
- Source: shared-brain
- Decision: inspected

### Candidates
- Skills/universal-tts-models/SKILL.md
  - Decision: selected
  - Reason: authoritative provider-integration conventions for universal-tts
- learning/2026-07-02-universal-tts-audio-cpp-true-streaming
  - Decision: inspected
  - Reason: prior session context, TTFB numbers
## 2026-07-02T06:47:39.243891+00:00 - local-search

- Request: Add automatic engine provisioning to Universal TTS
- Source: local-project
- Decision: selected
- Selected backbone: local-project:audio.cpp scripts + universal-tts lifecycle

### Candidates
- audio.cpp/scripts/build_metal.sh
  - Decision: selected
  - Reason: darwin engine build entrypoint
- audio.cpp/scripts/build_linux.sh
  - Decision: selected-donor
  - Reason: linux build with cuda auto
- audio.cpp/scripts/build_windows.ps1
  - Decision: selected-donor
  - Reason: windows cuda preset build
- audio.cpp/tools/model_manager.py
  - Decision: selected
  - Reason: model package downloader into models/ layout
## 2026-07-02T06:48:34.003158+00:00 - local-search

- Request: TTS server automatic engine provisioning build binary download model weights on demand FastAPI provider lifecycle
- Source: local-index
- Query: `TTS server automatic engine provisioning build binary download model weights on demand FastAPI provider lifecycle`
- Decision: no-results
- Reason: No local index hits or local index unavailable.
- Note: Index not found: /Users/liam/.codex/hive-assimilate/index/chunks.jsonl. Run index_github_repos.py first.
## 2026-07-02T06:48:36.133947+00:00 - public-search

- Request: TTS server automatic engine provisioning build binary download model weights on demand FastAPI provider lifecycle
- Source: public-github
- Query: `TTS server automatic engine provisioning build binary download model weights on demand FastAPI provider lifecycle`
- Decision: retrieved
- Reason: Retrieved 7 public candidates from GitHub search.

### Candidates
- rahul2261999/multi-agent-chat (0 stars, TypeScript)
  - URL: https://github.com/rahul2261999/multi-agent-chat
  - Description: Next.js 15 + React 19 Chat & Voice assistants with realtime WebSocket streaming. Push‑to‑talk (ASR), smooth TTS with safe cancel, auto‑scroll/jump‑to‑bottom, stop‑generating, session + message caching, reconnect/backoff, and en‑US/en‑GB sup
- inktide-ai/inktide (0 stars, TypeScript)
  - URL: https://github.com/inktide-ai/inktide
  - Description: AI-powered streaming character - listens to Discord & Twitch chat, thinks, and responds in real time with voice. Built on a multi-stage LLM pipeline: RAG, emotion, TTS, and Live2D avatar. .NET 10 · React · Rust · Keycloak
- weshaan/SUSI-Web-Audio-API (0 stars)
  - URL: https://github.com/weshaan/SUSI-Web-Audio-API
  - Description: Implement an AudioWorklet within the eventyay-video player to extract raw PCM audio chunks directly from the active WebRTC stream.
- saislamb97/ai-demo (0 stars, TypeScript)
  - URL: https://github.com/saislamb97/ai-demo
  - Description: A full-stack voice-enabled AI chat demo built with FastAPI, OpenAI GPT-4o, ElevenLabs TTS, and a React (Vite + TypeScript) client. It supports real-time streaming replies, sentence-by-sentence speech synthesis, and slide JSON output for vis
- hidatara-ds/vertex-ai-websocket-gateway (1 stars, Python)
  - URL: https://github.com/hidatara-ds/vertex-ai-websocket-gateway
  - Description: A Go application that provides a WebSocket interface for communicating with Google Vertex AI. This app supports real-time text and audio input with Text-to-Speech (TTS) responses.
- LEOSOLAR8/webwaifu-ai-assistant (2 stars, JavaScript)
  - URL: https://github.com/LEOSOLAR8/webwaifu-ai-assistant
  - Description: WEBWAIFU — A browser-based AI VTuber platform with VRM avatar support, real-time Whisper speech recognition, multi-provider AI (OpenAI, Gemini, Ollama), Azure TTS voices, and Twitch chat integration. Create and stream your own AI companion 
- smartManual/stream-audio-player (1 stars, TypeScript, MIT License)
  - URL: https://github.com/smartManual/stream-audio-player
  - Description: 音频流式播放库，支持 PCM/MP3/WAV 格式的实时解码与播放。适用于 Web 音频应用开发
## 2026-07-02T06:48:36.164331+00:00 - prebuild-gate

- Request: TTS server automatic engine provisioning build binary download model weights on demand FastAPI provider lifecycle
- Source: public-github
- Query: `TTS server automatic engine provisioning build binary download model weights on demand FastAPI provider lifecycle`
- Decision: passed
- Reason: Public search returned candidates; choose and audit backbone/donors before implementation.
## 2026-07-02T06:49:23.211888+00:00 - public-search

- Request: Add automatic engine provisioning to Universal TTS
- Source: public-github
- Decision: rejected
- Note: prebuild gate: 7 weak candidates max score 5.5; backbone stays local audio.cpp scripts + model_manager package table

### Candidates
- eliranwong/webwaifu
  - Decision: rejected
  - Reason: VTuber chat platform, no engine provisioning logic
- smartManual/stream-audio-player
  - Decision: rejected
  - Reason: browser PCM player, unrelated to server provisioning
## 2026-07-02T06:55:07.640504+00:00 - assimilation-manifest

- Request: Add automatic engine provisioning to Universal TTS
- Source: selected-github-code
- Decision: assimilated
- Assimilated: 0xShug0/audio.cpp:tools/model_manager.py(qwen3_tts_0_6b_base ModelPackage+SnapshotSource) => src/universal_tts/provisioning.py, 0xShug0/audio.cpp:scripts/build_metal.sh => config.yaml(provision.engine.build.darwin), 0xShug0/audio.cpp:scripts/build_linux.sh => config.yaml(provision.engine.build.linux), 0xShug0/audio.cpp:scripts/build_windows.ps1 => config.yaml(provision.engine.build.windows), 0xShug0/audio.cpp:server-qwen3-metal.json => audio.cpp/server-qwen3-cpu.json+server-qwen3-cuda.json, universal-tts:src/universal_tts/lifecycle.py => src/universal_tts/provisioning.py
- Verification: Wrote ASSIMILATION.engine-provisioning.json with 6 entries and custom_code_assessment=balanced.
## 2026-07-02T06:55:07.701116+00:00 - verification

- Request: Add automatic engine provisioning to Universal TTS
- Source: verify-assimilation-manifest
- Decision: failed
- Reason: ASSIMILATION.engine-provisioning.json: Entry 2 target_path does not exist: /Users/liam/voice-lab/universal-tts/config.yaml(provision.engine.build.darwin)
## 2026-07-02T06:55:07.738783+00:00 - verification

- Request: Add automatic engine provisioning to Universal TTS
- Source: local-project
- Decision: selected
- Note: manifest verified; 64 pytest pass; live E2E: binary stashed -> load auto-provisioned rebuild in 15s -> load ok -> real PCM streamed; HF snapshot downloader live-tested with tiny repo
## 2026-07-02T06:55:23.876130+00:00 - assimilation-manifest

- Request: Add automatic engine provisioning to Universal TTS
- Source: selected-github-code
- Decision: assimilated
- Assimilated: 0xShug0/audio.cpp:tools/model_manager.py => src/universal_tts/provisioning.py, 0xShug0/audio.cpp:scripts/build_metal.sh => config.yaml, 0xShug0/audio.cpp:scripts/build_linux.sh => config.yaml, 0xShug0/audio.cpp:scripts/build_windows.ps1 => config.yaml, 0xShug0/audio.cpp:server-qwen3-metal.json => /Users/liam/voice-lab/audio.cpp/server-qwen3-cpu.json, universal-tts:src/universal_tts/lifecycle.py => src/universal_tts/provisioning.py
- Verification: Wrote ASSIMILATION.engine-provisioning.json with 6 entries and custom_code_assessment=balanced.
## 2026-07-02T06:55:23.930764+00:00 - verification

- Request: Add automatic engine provisioning to Universal TTS
- Source: verify-assimilation-manifest
- Decision: passed
- Reason: ASSIMILATION.engine-provisioning.json: 6 concrete reuse entries, 6 substantive
## 2026-07-02T06:55:23.964603+00:00 - verification

- Request: Add automatic engine provisioning to Universal TTS
- Source: local-project
- Decision: selected
- Note: manifest re-verified after path fix; result pass
## 2026-07-02T06:56:22.026464+00:00 - final

- Request: Add automatic engine provisioning to Universal TTS
- Source: local-project
- Decision: selected
- Note: shipped: provisioning.py + registry/app/config integration, provision block for audiocpp-qwen3, command_by_platform, 64 tests pass, darwin E2E verified
