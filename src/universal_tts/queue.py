from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AudioJob:
    id: str
    payload: dict[str, Any]
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    content_type: str | None = None
    audio: bytes | None = None
    error: str | None = None
    task: asyncio.Task | None = None
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)

    def public(self, include_audio: bool = False) -> dict[str, Any]:
        out = {
            "id": self.id,
            "object": "audio.job",
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "content_type": self.content_type,
            "error": self.error,
        }
        if include_audio and self.audio is not None:
            out["bytes"] = len(self.audio)
        return out


class JobQueue:
    def __init__(self, registry):
        self.registry = registry
        self.jobs: dict[str, AudioJob] = {}

    async def submit(self, payload: dict[str, Any]) -> AudioJob:
        job = AudioJob(id=str(uuid.uuid4()), payload=dict(payload))
        self.jobs[job.id] = job
        job.task = asyncio.create_task(self._run(job))
        return job

    async def _run(self, job: AudioJob) -> None:
        job.status = "running"
        job.updated_at = time.time()
        payload = dict(job.payload)
        payload["cancel_event"] = job.cancel_event
        try:
            audio, content_type = await self.registry.synthesize(payload)
            if job.cancel_event.is_set():
                job.status = "cancelled"
                return
            job.audio = audio
            job.content_type = content_type
            job.status = "completed"
        except asyncio.CancelledError:
            job.status = "cancelled"
        except Exception as e:
            if job.cancel_event.is_set():
                job.status = "cancelled"
            else:
                job.status = "failed"
                job.error = str(e)
        finally:
            job.updated_at = time.time()

    def get(self, job_id: str) -> AudioJob:
        if job_id not in self.jobs:
            raise KeyError(job_id)
        return self.jobs[job_id]

    async def cancel(self, job_id: str) -> AudioJob:
        job = self.get(job_id)
        if job.status in {"completed", "failed", "cancelled"}:
            return job
        job.status = "cancelling"
        job.updated_at = time.time()
        job.cancel_event.set()
        if job.task:
            job.task.cancel()
        await asyncio.sleep(0)
        if job.status == "cancelling":
            job.status = "cancelled"
            job.updated_at = time.time()
        return job
