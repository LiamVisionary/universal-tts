from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol

from universal_tts.config import ProviderConfig


@dataclass(frozen=True)
class TTSRequest:
    model: str
    text: str
    voice: str | None = None
    response_format: str = "wav"
    speed: float | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderStatus:
    id: str
    loaded: bool
    healthy: bool
    details: dict[str, Any] = field(default_factory=dict)


class TTSProvider(Protocol):
    cfg: ProviderConfig

    async def status(self) -> ProviderStatus: ...
    async def load(self) -> ProviderStatus: ...
    async def unload(self) -> ProviderStatus: ...
    async def synthesize(self, request: TTSRequest) -> tuple[bytes, str]: ...
    async def stream_synthesize(self, request: TTSRequest) -> tuple[AsyncIterator[bytes], str]: ...
