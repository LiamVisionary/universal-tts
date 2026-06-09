from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProviderConfig:
    id: str
    kind: str
    models: list[str]
    estimate_gb: float
    category: str = "tts"
    base_url: str | None = None
    health_path: str = "/health"
    launchd_label: str | None = None
    plist: str | None = None
    cwd: str | None = None
    command: str | None = None
    port: int | None = None
    notes: str | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeConfig:
    server: dict[str, Any]
    memory: dict[str, Any]
    providers: dict[str, ProviderConfig]

    @property
    def model_to_provider(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for provider_id, provider in self.providers.items():
            for model in provider.models:
                out[model] = provider_id
        return out


def load_config_dict(raw: dict[str, Any]) -> RuntimeConfig:
    providers: dict[str, ProviderConfig] = {}
    for provider_id, item in (raw.get("providers") or {}).items():
        category = item.get("category", "tts")
        if category != "tts":
            continue
        providers[provider_id] = ProviderConfig(
            id=provider_id,
            kind=item["kind"],
            models=list(item.get("models") or []),
            estimate_gb=float(item.get("estimate_gb", 0)),
            category=category,
            base_url=item.get("base_url"),
            health_path=item.get("health_path", "/health"),
            launchd_label=item.get("launchd_label") or item.get("label"),
            plist=item.get("plist"),
            cwd=item.get("cwd"),
            command=item.get("command"),
            port=item.get("port"),
            notes=item.get("notes"),
            options=dict(item.get("options") or {}),
        )
    return RuntimeConfig(
        server=dict(raw.get("server") or {"host": "127.0.0.1", "port": 8799}),
        memory=dict(raw.get("memory") or {"reserve_gb": 12}),
        providers=providers,
    )


def load_config(path: str | Path) -> RuntimeConfig:
    return load_config_dict(yaml.safe_load(Path(path).read_text()))
