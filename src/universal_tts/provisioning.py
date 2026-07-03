from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from universal_tts.config import ProviderConfig
from universal_tts.platforms import current_platform


def _platform_os(machine_platform: str) -> str:
    return machine_platform.split("-")[0]


def _select_for_platform(mapping: dict[str, Any], machine_platform: str) -> Any:
    """Pick a value from a platform-keyed mapping ('darwin', 'windows-x86_64', 'any')."""
    if not isinstance(mapping, dict):
        return mapping
    for key in (machine_platform, _platform_os(machine_platform), "any"):
        if key in mapping:
            return mapping[key]
    return None


@dataclass
class ModelSpec:
    """One model package. Field shape adapted from audio.cpp tools/model_manager.py
    ModelPackage + SnapshotSource (repo_id, target_directory, required_files)."""
    repo_id: str
    target_dir: str
    required_files: list[str] = field(default_factory=list)


@dataclass
class ProvisionSpec:
    auto: bool
    engine_check_path: str | None
    engine_build_command: str | None
    models: list[ModelSpec]
    cwd: Path

    @staticmethod
    def from_provider(cfg: ProviderConfig, machine_platform: str) -> "ProvisionSpec | None":
        raw = cfg.options.get("provision")
        if not isinstance(raw, dict):
            return None
        engine = raw.get("engine") or {}
        models = [
            ModelSpec(
                repo_id=str(item["repo_id"]),
                target_dir=str(item["target_dir"]),
                required_files=[str(f) for f in item.get("required_files") or []],
            )
            for item in raw.get("models") or []
        ]
        check_path = _select_for_platform(engine.get("check_paths") or {}, machine_platform)
        build_command = _select_for_platform(engine.get("build") or {}, machine_platform)
        return ProvisionSpec(
            auto=bool(raw.get("auto", True)),
            engine_check_path=str(check_path) if check_path else None,
            engine_build_command=str(build_command) if build_command else None,
            models=models,
            cwd=Path(cfg.cwd or "."),
        )


class EngineProvisioner:
    """Detects missing engine binaries/model weights and provisions them.

    Engine builds shell out to the engine repo's own build scripts (e.g.
    audio.cpp scripts/build_metal.sh / build_linux.sh / build_windows.ps1);
    model weights come from Hugging Face snapshots, mirroring the package
    definitions in audio.cpp tools/model_manager.py.
    """

    def __init__(self, machine_platform: str | None = None, log_dir: str | Path = "logs"):
        self.machine_platform = machine_platform or current_platform()
        self.log_dir = Path(log_dir)
        self._status: dict[str, dict[str, Any]] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    def spec(self, cfg: ProviderConfig) -> ProvisionSpec | None:
        return ProvisionSpec.from_provider(cfg, self.machine_platform)

    def state(self, cfg: ProviderConfig) -> dict[str, Any] | None:
        spec = self.spec(cfg)
        if spec is None:
            return None
        missing: list[str] = []
        engine_ok = True
        if spec.engine_check_path:
            engine_path = spec.cwd / spec.engine_check_path
            engine_ok = engine_path.exists()
            if not engine_ok:
                missing.append(f"engine binary: {engine_path}")
        models_ok = True
        for model in spec.models:
            target = spec.cwd / model.target_dir
            required = model.required_files or ["config.json"]
            for rel in required:
                if not (target / rel).exists():
                    models_ok = False
                    missing.append(f"model file: {target / rel}")
        return {
            "provisioned": engine_ok and models_ok,
            "engine_ok": engine_ok,
            "models_ok": models_ok,
            "auto": spec.auto,
            "missing": missing,
            "status": self._status.get(cfg.id, {"status": "idle"}),
        }

    def is_running(self, provider_id: str) -> bool:
        task = self._tasks.get(provider_id)
        return task is not None and not task.done()

    def start(self, cfg: ProviderConfig, force: bool = False) -> dict[str, Any]:
        """Kick off provisioning in the background; returns current status."""
        if self.is_running(cfg.id):
            return self._status[cfg.id]
        state = self.state(cfg)
        if state is None:
            raise KeyError(f"provider '{cfg.id}' has no provision configuration")
        if state["provisioned"] and not force:
            self._status[cfg.id] = {"status": "done", "detail": "already provisioned"}
            return self._status[cfg.id]
        self._status[cfg.id] = {"status": "running", "started_at": time.time(), "steps": []}
        self._tasks[cfg.id] = asyncio.get_running_loop().create_task(self._provision(cfg))
        return self._status[cfg.id]

    async def _provision(self, cfg: ProviderConfig) -> None:
        spec = self.spec(cfg)
        status = self._status[cfg.id]
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"provision-{cfg.id}.log"
        status["log"] = str(log_path)
        try:
            assert spec is not None
            for model in spec.models:
                target = spec.cwd / model.target_dir
                required = model.required_files or ["config.json"]
                if all((target / rel).exists() for rel in required):
                    status["steps"].append({"step": f"model {model.repo_id}", "result": "present"})
                    continue
                await asyncio.to_thread(self._download_snapshot, model, target, log_path)
                missing = [rel for rel in required if not (target / rel).exists()]
                if missing:
                    raise RuntimeError(f"model {model.repo_id} incomplete after download; missing {missing}")
                status["steps"].append({"step": f"model {model.repo_id}", "result": "downloaded"})
            if spec.engine_check_path and not (spec.cwd / spec.engine_check_path).exists():
                if not spec.engine_build_command:
                    raise RuntimeError(
                        f"engine binary missing and no build command configured for {self.machine_platform}")
                await self._run_build(spec.engine_build_command, spec.cwd, log_path)
                if not (spec.cwd / spec.engine_check_path).exists():
                    raise RuntimeError(f"build finished but {spec.engine_check_path} still missing; see {log_path}")
                status["steps"].append({"step": "engine build", "result": "built"})
            elif spec.engine_check_path:
                status["steps"].append({"step": "engine build", "result": "present"})
            status["status"] = "done"
        except Exception as e:
            status["status"] = "failed"
            status["error"] = str(e)
        finally:
            status["finished_at"] = time.time()

    def _download_snapshot(self, model: ModelSpec, target: Path, log_path: Path) -> None:
        from huggingface_hub import snapshot_download

        with open(log_path, "a") as log:
            log.write(f"[provision] snapshot_download {model.repo_id} -> {target}\n")
        target.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=model.repo_id, local_dir=str(target))

    async def _run_build(self, command: str, cwd: Path, log_path: Path) -> None:
        with open(log_path, "a") as log:
            log.write(f"[provision] build: {command} (cwd={cwd})\n")
        with open(log_path, "ab") as log:
            proc = await asyncio.create_subprocess_shell(
                command, cwd=str(cwd), stdout=log, stderr=asyncio.subprocess.STDOUT)
            rc = await proc.wait()
        if rc != 0:
            raise RuntimeError(f"engine build failed with exit code {rc}; see {log_path}")
