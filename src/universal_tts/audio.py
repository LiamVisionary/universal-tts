from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

CONTENT_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "mpeg": "audio/mpeg",
    "opus": "audio/ogg; codecs=opus",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
    "aac": "audio/aac",
    "pcm": "audio/pcm",
}

FFMPEG_FORMATS = {
    "wav": "wav",
    "mp3": "mp3",
    "mpeg": "mp3",
    "opus": "opus",
    "ogg": "ogg",
    "flac": "flac",
    "aac": "adts",
    "pcm": "s16le",
}


def normalize_format(fmt: str | None) -> str:
    value = (fmt or "wav").lower().strip().lstrip(".")
    if value == "wave":
        value = "wav"
    if value not in CONTENT_TYPES:
        raise ValueError(f"unsupported audio response_format: {fmt}")
    return value


def content_type_for_format(fmt: str | None) -> str:
    return CONTENT_TYPES[normalize_format(fmt)]


def convert_audio_format(audio: bytes, source_format: str | None, target_format: str | None) -> tuple[bytes, str]:
    source = normalize_format(source_format)
    target = normalize_format(target_format)
    if target == source:
        return audio, content_type_for_format(target)

    with tempfile.TemporaryDirectory(prefix="universal-tts-audio-") as td:
        ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
        if not Path(ffmpeg).exists():
            raise RuntimeError("ffmpeg is required for audio format conversion")
        src = Path(td) / f"input.{source}"
        dst = Path(td) / f"output.{target}"
        src.write_bytes(audio)
        cmd = [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
        ]
        if target == "pcm":
            cmd += ["-f", FFMPEG_FORMATS[target], "-acodec", "pcm_s16le", "-ac", "1", "-ar", "24000"]
        elif target == "opus":
            cmd += ["-c:a", "libopus", "-b:a", "64k"]
        cmd.append(str(dst))
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg conversion failed: {proc.stderr.decode(errors='replace')[:500]}")
        return dst.read_bytes(), content_type_for_format(target)
