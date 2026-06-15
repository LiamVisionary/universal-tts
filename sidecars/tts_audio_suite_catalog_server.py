from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

app = FastAPI(title="TTS-Audio-Suite candidate bridge", version="0.1.0")


CATALOG: dict[str, dict[str, Any]] = {
    "moss-tts": {
        "kind": "tts",
        "engine": "MOSS-TTS",
        "models": [
            "MOSS-TTS-Local-Transformer",
            "MOSS-TTS",
            "MOSS-TTSD-v1.0",
        ],
        "sample_rate": 24000,
        "status": "catalog_stub",
        "supports_voice_cloning": False,
        "supports_dialogue": True,
        "supports_true_streaming": False,
        "streaming_mode": "not-enabled-local-runtime-pending",
        "implementation": "tts-audio-suite-moss-tts-catalog",
        "voices": ["default"],
        "notes": (
            "Cataloged from diodiogod/TTS-Audio-Suite. Useful candidate for "
            "long-form narration and MOSS-TTSD multi-speaker dialogue. Native "
            "runtime requires large OpenMOSS model + MOSS-Audio-Tokenizer weights "
            "and transformers>=4.57; no verified Apple-Silicon realtime stream yet."
        ),
        "install": {
            "suite_path": "/Users/liam/voice-lab/TTS-Audio-Suite/engines/moss_tts",
            "model_repos": [
                "OpenMOSS-Team/MOSS-TTS-Local-Transformer",
                "OpenMOSS-Team/MOSS-TTS",
                "OpenMOSS-Team/MOSS-TTSD-v1.0",
                "OpenMOSS-Team/MOSS-Audio-Tokenizer",
            ],
        },
    },
    "indextts2": {
        "kind": "tts",
        "engine": "IndexTTS-2",
        "models": ["IndexTTS-2", "IndexTeam/IndexTTS-2"],
        "sample_rate": 24000,
        "status": "catalog_stub",
        "supports_voice_cloning": True,
        "voice_cloning_requires": ["ref_audio"],
        "supports_emotion_control": True,
        "supports_true_streaming": False,
        "streaming_mode": "full-generation-engine-cataloged-not-realtime",
        "implementation": "tts-audio-suite-indextts2-catalog",
        "voices": ["clone"],
        "emotion_labels": [
            "happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm",
        ],
        "notes": (
            "Cataloged from TTS-Audio-Suite. Best Universal benefit is expressive "
            "zero-shot cloning with separate speaker/emotion controls. Runtime pulls "
            "IndexTeam/IndexTTS-2 plus w2v-bert, MaskGCT, BigVGAN, CampPlus assets; "
            "streaming is not verified and should not be advertised as realtime."
        ),
    },
    "step-audio-editx": {
        "kind": "tts",
        "engine": "Step Audio EditX",
        "models": ["Step-Audio-EditX", "stepfun-ai/Step-Audio-EditX"],
        "sample_rate": 24000,
        "status": "catalog_stub",
        "supports_voice_cloning": True,
        "voice_cloning_requires": ["ref_audio", "ref_text"],
        "supports_audio_editing": True,
        "supports_true_streaming": False,
        "streaming_mode": "audio-edit/full-generation-cataloged-not-realtime",
        "implementation": "tts-audio-suite-step-audio-editx-catalog",
        "voices": ["clone"],
        "emotion_options": [
            "happy", "sad", "angry", "excited", "calm", "fearful", "surprised", "disgusted",
            "confusion", "empathy", "embarrass", "depressed", "coldness", "admiration",
        ],
        "style_options": [
            "whisper", "serious", "child", "older", "girl", "pure", "sister", "sweet",
            "exaggerated", "ethereal", "generous", "recite", "act_coy", "warm", "shy",
            "comfort", "authority", "chat", "radio", "soulful", "gentle", "story", "vivid",
            "program", "news", "advertising", "roar", "murmur", "shout", "deeply", "loudly",
            "arrogant", "friendly",
        ],
        "paralinguistics": [
            "[Breathing]", "[Laughter]", "[Surprise-oh]", "[Confirmation-en]", "[Uhm]",
            "[Surprise-ah]", "[Surprise-wa]", "[Sigh]", "[Question-ei]", "[Dissatisfaction-hnn]",
        ],
        "notes": (
            "Cataloged from TTS-Audio-Suite. Useful as a post-TTS edit/voice-clone "
            "engine for emotion, style, speed, and paralinguistic edits. It is not "
            "currently a realtime calling backend."
        ),
    },
    "granite-asr": {
        "kind": "asr",
        "engine": "Granite ASR",
        "models": ["granite-4.0-1b-speech", "ibm-granite/granite-4.0-1b-speech"],
        "sample_rate": None,
        "status": "catalog_stub",
        "supports_transcription": True,
        "supports_translation": True,
        "supports_voice_cloning": False,
        "supports_true_streaming": False,
        "streaming_mode": "asr-verification-candidate-not-tts",
        "implementation": "tts-audio-suite-granite-asr-catalog",
        "voices": [],
        "notes": (
            "Cataloged from TTS-Audio-Suite as a future closed-loop ASR verifier "
            "for generated clips. It is not a TTS backend; do not route speech "
            "synthesis to it. Runtime uses ibm-granite/granite-4.0-1b-speech."
        ),
    },
    "rvc": {
        "kind": "voice_conversion",
        "engine": "RVC",
        "models": ["rvc", "RVC"],
        "sample_rate": None,
        "status": "catalog_stub",
        "supports_voice_cloning": False,
        "supports_voice_conversion": True,
        "supports_true_streaming": False,
        "streaming_mode": "postprocess-conversion-not-tts",
        "implementation": "tts-audio-suite-rvc-catalog",
        "voices": [],
        "pitch_methods": ["rmvpe", "rmvpe+", "mangio-crepe", "crepe", "pm", "harvest"],
        "notes": (
            "Cataloged from TTS-Audio-Suite as a post-synthesis voice-conversion "
            "stage, not a text-to-speech provider. Universal can use this later as "
            "a postprocess pipeline after a realtime TTS backend, but it should not "
            "appear as a direct realtime TTS model."
        ),
    },
}


def provider_id() -> str:
    value = os.environ.get("SUITE_PROVIDER_ID") or os.environ.get("UNIVERSAL_TTS_SUITE_PROVIDER") or "moss-tts"
    if value not in CATALOG:
        return "moss-tts"
    return value


def meta() -> dict[str, Any]:
    provider = provider_id()
    data = dict(CATALOG[provider])
    data.update({
        "ok": True,
        "provider": provider,
        "service": "tts-audio-suite-candidate-bridge",
        "loaded": False,
        "healthy": True,
        "runtime_enabled": False,
    })
    return data


@app.get("/health")
def health(load: bool = False):
    data = meta()
    data["load_requested"] = bool(load)
    return data


@app.get("/v1/models")
def models():
    data = meta()
    return {
        "object": "list",
        "data": [
            {"id": model, "object": "model", "provider": data["provider"], "status": data["status"]}
            for model in data.get("models", [])
        ],
    }


@app.get("/v1/voices")
@app.get("/v1/audio/voices")
def voices():
    data = meta()
    return {
        "object": "list",
        "provider": data["provider"],
        "data": [{"id": voice, "provider": data["provider"]} for voice in data.get("voices", [])],
    }


@app.get("/v1/audio/paralinguistics")
def paralinguistics():
    data = meta()
    tokens = data.get("paralinguistics") or []
    return {
        "object": "audio.paralinguistics",
        "provider": data["provider"],
        "data": [
            {"token": token, "kind": "nonverbal", "supported": False, "source": data["engine"]}
            for token in tokens
        ],
    }


@app.get("/v1/audio/candidate")
def candidate():
    return meta()


@app.post("/v1/audio/speech")
async def speech(request: Request):
    # Consume JSON so malformed client requests still surface as normal validation failures.
    await request.json()
    data = meta()
    raise HTTPException(
        status_code=501,
        detail=(
            f"{data['provider']} is cataloged from TTS-Audio-Suite but its heavy runtime "
            "is not installed/enabled in Universal TTS yet. Use /v1/audio/candidate "
            "for metadata; do not select it for realtime calling."
        ),
    )


@app.post("/v1/audio/speech-stream")
async def speech_stream(request: Request):
    await request.json()
    data = meta()

    async def fail_fast():
        # Keep response generation lazy but explicit; Universal treats upstream >=400
        # as errors before reading this generator, so this is primarily direct-sidecar safety.
        if False:
            yield b""

    raise HTTPException(
        status_code=501,
        detail=f"{data['provider']} has no verified true streaming runtime enabled.",
    )
