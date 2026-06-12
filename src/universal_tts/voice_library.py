from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class VoiceProfile:
    provider: str
    voice_id: str
    name: str
    ref_audio: str
    ref_text: str
    description: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    provider_voice: str | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def public(self) -> dict[str, Any]:
        return {
            "id": self.voice_id,
            "object": "voice",
            "provider": self.provider,
            "name": self.name,
            "description": self.description,
            "type": "saved_reference_clone",
            "requires_reference_audio": False,
            "has_reference_audio": True,
            "ref_audio": self.ref_audio,
            "ref_text": self.ref_text,
            "provider_voice": self.provider_voice,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "options": self.options,
            "status": "ready",
        }


class VoiceLibrary:
    """Persistent reference-clone voice registry for Universal TTS.

    Stores only local file paths/transcripts and copies reference audio into the
    project data dir so saved voices survive moving/deleting upload temp files.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def sanitize_id(value: str) -> str:
        clean = _SAFE_ID.sub("-", value.strip()).strip(".-_")[:80]
        return clean or f"voice-{int(time.time())}"

    def provider_dir(self, provider: str) -> Path:
        p = self.root / self.sanitize_id(provider)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def profile_path(self, provider: str, voice_id: str) -> Path:
        return self.provider_dir(provider) / f"{self.sanitize_id(voice_id)}.json"

    def list(self, provider: str | None = None) -> list[VoiceProfile]:
        providers = [self.sanitize_id(provider)] if provider else [p.name for p in self.root.iterdir() if p.is_dir()]
        out: list[VoiceProfile] = []
        for provider_id in providers:
            pdir = self.root / provider_id
            if not pdir.is_dir():
                continue
            for meta in sorted(pdir.glob("*.json")):
                try:
                    out.append(self._load(meta))
                except Exception:
                    continue
        return out

    def get(self, provider: str, voice_id: str) -> VoiceProfile | None:
        path = self.profile_path(provider, voice_id)
        if not path.exists():
            return None
        return self._load(path)

    def create(
        self,
        *,
        provider: str,
        voice_id: str,
        name: str | None,
        ref_audio: str,
        ref_text: str,
        description: str | None = None,
        provider_voice: str | None = None,
        options: dict[str, Any] | None = None,
        overwrite: bool = False,
        require_ref_text: bool = True,
    ) -> VoiceProfile:
        provider = self.sanitize_id(provider)
        voice_id = self.sanitize_id(voice_id)
        if require_ref_text and (not ref_text or not str(ref_text).strip()):
            raise ValueError("ref_text / transcript is required for voice cloning")
        src = Path(ref_audio).expanduser()
        if not src.exists() or not src.is_file():
            raise ValueError(f"ref_audio file not found: {src}")
        pdir = self.provider_dir(provider)
        profile_path = self.profile_path(provider, voice_id)
        if profile_path.exists() and not overwrite:
            raise ValueError(f"voice already exists for provider {provider}: {voice_id}")
        suffix = src.suffix or ".wav"
        dest = pdir / f"{voice_id}{suffix}"
        if src.resolve() != dest.resolve():
            shutil.copyfile(src, dest)
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        previous = self.get(provider, voice_id) if profile_path.exists() else None
        profile = VoiceProfile(
            provider=provider,
            voice_id=voice_id,
            name=name or voice_id,
            description=description,
            ref_audio=str(dest),
            ref_text=str(ref_text),
            provider_voice=provider_voice,
            options=options or {},
            created_at=previous.created_at if previous else now,
            updated_at=now,
        )
        tmp = profile_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(profile.public(), indent=2, sort_keys=True))
        tmp.replace(profile_path)
        return profile

    def delete(self, provider: str, voice_id: str) -> bool:
        profile = self.get(provider, voice_id)
        path = self.profile_path(provider, voice_id)
        if not path.exists():
            return False
        path.unlink()
        if profile:
            audio = Path(profile.ref_audio)
            try:
                if audio.is_file() and audio.parent == self.provider_dir(provider):
                    audio.unlink()
            except Exception:
                pass
        return True

    def _load(self, path: Path) -> VoiceProfile:
        data = json.loads(path.read_text())
        return VoiceProfile(
            provider=str(data["provider"]),
            voice_id=str(data.get("id") or data.get("voice_id") or path.stem),
            name=str(data.get("name") or data.get("id") or path.stem),
            description=data.get("description"),
            ref_audio=str(data["ref_audio"]),
            ref_text=str(data["ref_text"]),
            provider_voice=data.get("provider_voice"),
            options=dict(data.get("options") or {}),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )
