from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import faulthandler
import json
import os
import queue
import signal
import threading
import time
from pathlib import Path
from typing import AsyncIterator

import mlx.core as mx
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

SERVER_BUILD = "watchdog-2"

ROOT = Path(os.environ.get("QWEN3_MLX_ROOT", str(Path(__file__).resolve().parent)))
MODEL_ID = os.environ.get("QWEN3_MLX_MODEL", "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit")
DEFAULT_VOICE = os.environ.get("QWEN3_MLX_DEFAULT_VOICE", "voice01")
DEFAULT_INSTRUCT = os.environ.get("QWEN3_MLX_DEFAULT_INSTRUCT", "Speak clearly, naturally, and warmly.")
DEFAULT_STREAMING_INTERVAL = float(os.environ.get("QWEN3_MLX_STREAMING_INTERVAL", "0.16"))
DEFAULT_STREAMING_CONTEXT = int(os.environ.get("QWEN3_MLX_STREAMING_CONTEXT", "25"))
DEFAULT_MAX_TOKENS = int(os.environ.get("QWEN3_MLX_MAX_TOKENS", "512"))
VOICE01_PROFILE = Path(os.environ.get("QWEN3_MLX_VOICE01_PROFILE", str(ROOT / "voice01_fast_profile.json")))

# --- Watchdog / recovery tunables ------------------------------------------
# The per-synth progress watchdog is the *guaranteed* fail-fast mechanism: if no
# new PCM frame is handed off within these budgets, the synth is aborted with an
# error instead of hanging forever. Kept strictly below UTTS's 60s stream/probe
# timeouts so the sidecar aborts cleanly *before* the keepalive force-restarts it.
FIRST_FRAME_TIMEOUT_SEC = float(os.environ.get("QWEN3_MLX_FIRST_FRAME_TIMEOUT_SEC", "25"))
STALL_WATCHDOG_SEC = float(os.environ.get("QWEN3_MLX_STALL_WATCHDOG_SEC", "20"))
ABORT_GRACE_SEC = float(os.environ.get("QWEN3_MLX_ABORT_GRACE_SEC", "2.5"))
PUT_TIMEOUT_SEC = float(os.environ.get("QWEN3_MLX_PUT_TIMEOUT_SEC", "10"))
QUEUE_MAX = int(os.environ.get("QWEN3_MLX_QUEUE_MAX", "8"))

# MLX footprint caps (hardening — bound our memory so MLX reclaims/returns buffers
# under system pressure instead of pushing the box deeper into swap). GiB.
MLX_MEM_LIMIT_GB = float(os.environ.get("QWEN3_MLX_MEM_LIMIT_GB", "16"))
MLX_CACHE_LIMIT_GB = float(os.environ.get("QWEN3_MLX_CACHE_LIMIT_GB", "2"))
MLX_WIRED_LIMIT_GB = float(os.environ.get("QWEN3_MLX_WIRED_LIMIT_GB", "2"))

# Test-only stall injection (see stream_iter). Disabled unless QWEN3_MLX_DEBUG=1
# so a stray client can never trigger it in production.
DEBUG_HOOKS = os.environ.get("QWEN3_MLX_DEBUG") == "1"

app = FastAPI(title="Qwen3 TTS MLX realtime sidecar", version="0.2.0")


# --- Diagnostics ------------------------------------------------------------
# SIGUSR1 dumps every OS thread's Python stack; SIGUSR2 dumps every asyncio
# task's coroutine/async-gen chain. macOS blocks py-spy from attaching without
# root, so these in-process dumpers are how a wedged synth gets introspected
# live. Kept in the production build; they never affect generation.
_FAULT_LOG_PATH = os.environ.get("QWEN3_MLX_FAULT_LOG", str(ROOT / "faultdump.log"))
try:
    _FAULT_LOG = open(_FAULT_LOG_PATH, "a", buffering=1)  # noqa: SIM115 (process-lifetime)
    faulthandler.enable(_FAULT_LOG)
    faulthandler.register(signal.SIGUSR1, file=_FAULT_LOG, all_threads=True, chain=False)
except Exception:  # pragma: no cover - diagnostics must never break startup
    _FAULT_LOG = None

_TASK_LOG_PATH = os.environ.get("QWEN3_MLX_TASK_LOG", str(ROOT / "taskdump.log"))


def _walk_coro_chain(coro) -> list[str]:
    out: list[str] = []
    seen: set[int] = set()
    while coro is not None and id(coro) not in seen:
        seen.add(id(coro))
        frame = getattr(coro, "cr_frame", None) or getattr(coro, "ag_frame", None) or getattr(coro, "gi_frame", None)
        if frame is not None:
            out.append(f"    {frame.f_code.co_filename}:{frame.f_lineno} in {frame.f_code.co_name}")
        coro = getattr(coro, "cr_await", None) or getattr(coro, "ag_await", None) or getattr(coro, "gi_yieldfrom", None)
    return out


def _dump_asyncio_tasks(loop) -> None:  # pragma: no cover - diagnostics
    try:
        tasks = asyncio.all_tasks(loop)
        with open(_TASK_LOG_PATH, "a") as f:
            f.write(f"\n===== asyncio tasks @ {time.time():.3f} (n={len(tasks)}) =====\n")
            for t in tasks:
                f.write(f"--- task={t.get_name()} done={t.done()}\n")
                for line in _walk_coro_chain(t.get_coro()):
                    f.write(line + "\n")
            f.flush()
    except Exception as e:
        with contextlib.suppress(Exception):
            with open(_TASK_LOG_PATH, "a") as f:
                f.write(f"task dump error: {e!r}\n")


class Qwen3SynthAborted(Exception):
    """A synth was aborted by the progress watchdog (no PCM frame in time)."""


class Qwen3SidecarWedged(Exception):
    """A prior synth left a worker stuck inside native MLX; the sidecar is
    degraded and fast-fails until it is restarted (by the UTTS keepalive)."""


# Sentinel put on the frame queue when the generation worker finishes normally.
_DONE = object()


class Qwen3MLXRuntime:
    def __init__(self) -> None:
        self.model = None
        self.loaded_at: float | None = None
        self.last_error: str | None = None
        self.last_successful_synth: float | None = None
        self.last_synth_started_at: float | None = None
        self.last_synth_finished_at: float | None = None
        self.synth_count = 0            # PCM frames delivered (historical meaning)
        self.completed_synth_count = 0  # synths that streamed real audio to DONE
        self.aborted_synth_count = 0    # synths killed by the watchdog
        self.empty_synth_count = 0      # synths that reached DONE with zero audio
        self.failed_synth_count = 0     # synths that raised a real error
        self.client_cancel_count = 0    # synths cut short by client/probe disconnect
        self.mlx_limits: dict[str, int] = {}

        # Degraded latch: set when a worker is stuck INSIDE native MLX and cannot be
        # cancelled. We must NOT run a fresh generate() concurrently (one shared
        # Metal context is not safe for concurrent use), so instead we fast-fail new
        # requests with 503 until the UTTS keepalive force-restarts this process.
        self._degraded = False
        self._degraded_reason: str | None = None

        # Async layer serializes whole requests. Safe to hold: the consumer ALWAYS
        # returns/raises within the watchdog budget, so the lock can never be held
        # forever (that was the old whole-sidecar wedge).
        self.lock = asyncio.Lock()
        # Single persistent generation worker: serializes all Metal work onto ONE
        # thread. We deliberately do NOT recycle/replace it — a fresh generate()
        # running concurrently with an orphaned Metal-stuck worker is undefined.
        self._gen_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen3-mlx-gen")
        # Small pool used only to do blocking, timeout-bounded queue reads off the
        # event loop — this is what implements the per-frame watchdog.
        self._reader_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="qwen3-mlx-read")

        self.voice_aliases = {
            "voice01": "voice01",
            "voice01-xvector": "voice01-xvector",
            "voice01-speaker": "voice01-xvector",
            "voice1-all-samples": "voice01",
            "liam-default": "voice01",
            "ryan": "Ryan",
            "aiden": "Aiden",
        }
        self.clone_profiles = self._load_clone_profiles()
        self.warmed_profiles: set[str] = set()

    def _load_clone_profiles(self) -> dict[str, dict[str, str]]:
        profiles: dict[str, dict[str, str]] = {}
        if VOICE01_PROFILE.is_file():
            data = json.loads(VOICE01_PROFILE.read_text())
            profiles["voice01"] = {
                "ref_audio": data["ref_audio"],
                "ref_text": data["ref_text"],
            }
            profiles["voice01-xvector"] = {
                "ref_audio": data["ref_audio"],
                "ref_text": "",
            }
        return profiles

    def resolve_voice(self, voice: str | None) -> str:
        return self.voice_aliases.get(voice or DEFAULT_VOICE, voice or DEFAULT_VOICE)

    def _apply_mlx_limits(self) -> None:
        """Bound MLX's footprint. Best-effort; failures never break load."""
        limits: dict[str, int] = {}
        try:
            mem = int(MLX_MEM_LIMIT_GB * 1024**3)
            if mem > 0:
                mx.set_memory_limit(mem)
                limits["memory_limit"] = mem
        except Exception as e:  # pragma: no cover
            self.last_error = f"set_memory_limit failed: {e!r}"
        try:
            cache = int(MLX_CACHE_LIMIT_GB * 1024**3)
            mx.set_cache_limit(cache)
            limits["cache_limit"] = cache
        except Exception:  # pragma: no cover
            pass
        try:
            wired = int(MLX_WIRED_LIMIT_GB * 1024**3)
            if wired > 0:
                mx.set_wired_limit(wired)
                limits["wired_limit"] = wired
        except Exception:  # pragma: no cover
            pass
        self.mlx_limits = limits

    def load_sync(self) -> None:
        if self.model is not None:
            return
        try:
            from mlx_audio.tts.utils import load_model

            self.model = load_model(MODEL_ID)
            self._apply_mlx_limits()
            self.loaded_at = time.time()
            self.last_error = None
        except Exception as e:
            self.last_error = repr(e)
            raise

    def warm_voice_sync(self, voice: str = "voice01") -> None:
        if self.model is None:
            self.load_sync()
        if voice in self.warmed_profiles:
            return
        profile = self.clone_profiles.get(voice)
        if not profile:
            return
        for idx, _result in enumerate(
            self.model.generate(
                text="Yes.",
                ref_audio=profile["ref_audio"],
                ref_text=profile.get("ref_text") or None,
                lang_code="English",
                stream=True,
                streaming_interval=DEFAULT_STREAMING_INTERVAL,
                max_tokens=48,
                verbose=False,
            )
        ):
            if idx >= 1:
                break
        self.warmed_profiles.add(voice)

    async def ensure(self) -> None:
        loop = asyncio.get_running_loop()
        if self.model is None:
            await loop.run_in_executor(self._gen_executor, self.load_sync)
        if "voice01" not in self.warmed_profiles:
            await loop.run_in_executor(self._gen_executor, self.warm_voice_sync, "voice01")

    def _float_audio_to_pcm16(self, audio) -> bytes:
        arr = np.asarray(audio, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return b""
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, -1.0, 1.0)
        return (arr * 32767.0).astype("<i2", copy=False).tobytes()

    # --- generation worker (runs in self._gen_executor, single thread) ------
    def stream_iter(self, payload: dict, cancel_event: threading.Event):
        text = payload.get("input") or payload.get("text")
        if not text:
            raise RuntimeError("input is required")
        self.load_sync()
        assert self.model is not None

        # Test-only: simulate a synth that never produces a PCM frame, to prove
        # the watchdog aborts and the process recovers. Gated behind QWEN3_MLX_DEBUG.
        if DEBUG_HOOKS and payload.get("__debug_stall"):
            mode = str(payload.get("__debug_stall"))
            secs = float(payload.get("__debug_stall_sec", 3600))
            if mode == "hard":
                # Uninterruptible: ignores cancel_event -> exercises the degraded
                # (fast-fail-until-restart) path for a truly-stuck worker.
                time.sleep(secs)
            else:
                deadline = time.time() + secs
                while time.time() < deadline and not cancel_event.is_set():
                    time.sleep(0.1)
            return

        voice = self.resolve_voice(payload.get("voice") or payload.get("speaker"))
        clone_mode = str(payload.get("clone_mode") or payload.get("voice_mode") or "icl").lower()
        if voice == "voice01" and clone_mode in {"xvector", "speaker", "speaker_embedding"}:
            voice = "voice01-xvector"
        clone_profile = self.clone_profiles.get(voice)
        if clone_profile is None and (payload.get("ref_audio") or payload.get("reference_audio") or payload.get("reference_audio_path")):
            clone_profile = {
                "ref_audio": payload.get("ref_audio") or payload.get("reference_audio") or payload.get("reference_audio_path"),
                "ref_text": payload.get("ref_text") or payload.get("reference_text") or payload.get("reference_transcript") or "",
            }
        instruct = payload.get("instruct") or payload.get("instruction") or DEFAULT_INSTRUCT
        language = payload.get("language") or payload.get("lang_code") or "English"
        streaming_interval = float(payload.get("streaming_interval", DEFAULT_STREAMING_INTERVAL))
        streaming_context = int(payload.get("streaming_context_size", DEFAULT_STREAMING_CONTEXT))
        max_tokens = int(payload.get("max_tokens", payload.get("max_new_tokens", DEFAULT_MAX_TOKENS)))
        temperature = float(payload.get("temperature", 0.9))
        top_k = int(payload.get("top_k", payload.get("topk", 50)))
        top_p = float(payload.get("top_p", 1.0))
        repetition_penalty = float(payload.get("repetition_penalty", 1.05))
        speed = float(payload.get("speed", payload.get("speed_factor", 1.0)))

        gen_kwargs = {
            "text": str(text),
            "instruct": str(instruct) if instruct else None,
            "lang_code": str(language),
            "stream": True,
            "streaming_interval": streaming_interval,
            "streaming_context_size": streaming_context,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "speed": speed,
            "verbose": False,
        }
        if clone_profile:
            # Base Qwen3-TTS uses ICL/reference cloning when ref_audio/ref_text
            # are provided. If ref_text is blank, this becomes speaker-embedding
            # only (x-vector) mode: faster, but usually less faithful.
            gen_kwargs["ref_audio"] = clone_profile["ref_audio"]
            if clone_profile.get("ref_text"):
                gen_kwargs["ref_text"] = clone_profile["ref_text"]
        else:
            gen_kwargs["voice"] = voice

        for result in self.model.generate(**gen_kwargs):
            if cancel_event.is_set():
                break
            pcm = self._float_audio_to_pcm16(result.audio)
            if pcm:
                yield pcm

    def _gen_worker(self, payload: dict, tq: "queue.Queue", cancel_event: threading.Event) -> None:
        """Producer: runs generation, pushes PCM frames onto a thread-safe queue.
        A stdlib queue.Queue (not asyncio.Queue) cannot deadlock across the
        thread boundary and supports real timeouts."""
        with contextlib.suppress(Exception):
            mx.clear_cache()  # start each synth from a reclaimed cache under pressure
        try:
            for pcm in self.stream_iter(payload, cancel_event):
                if cancel_event.is_set():
                    break
                try:
                    tq.put(pcm, timeout=PUT_TIMEOUT_SEC)
                except queue.Full:
                    # Consumer aborted/gone and is no longer draining -> stop.
                    break
            with contextlib.suppress(queue.Full):
                tq.put(_DONE, timeout=PUT_TIMEOUT_SEC)
        except BaseException as e:
            with contextlib.suppress(Exception):
                tq.put(e, timeout=1.0)

    async def _reap_worker(self, fut: "concurrent.futures.Future") -> None:
        """After a synth ends, confirm the worker released the single generation
        thread. A cooperative stall exits promptly on cancel_event -> the next
        synth runs cleanly, no restart needed. A worker still running after the
        grace window is stuck inside native MLX and cannot be safely interrupted
        or run alongside a fresh generate() -> latch DEGRADED so new requests
        fast-fail 503 and the keepalive restarts us. We never touch MLX from here."""
        try:
            # wait_for cancels only the asyncio wrapper on timeout; the underlying
            # thread keeps running (an already-started thread can't be cancelled) —
            # that IS the orphan. _gen_worker never re-raises, so the only outcome
            # other than clean completion is TimeoutError (worker still stuck).
            await asyncio.wait_for(asyncio.wrap_future(fut), timeout=ABORT_GRACE_SEC)
        except asyncio.TimeoutError:
            self._degraded = True
            self._degraded_reason = (
                "generation worker stuck inside native MLX after watchdog abort; "
                "fast-failing until restart"
            )
            self.last_error = self._degraded_reason
        # Any other exception (e.g. CancelledError from request teardown) propagates.

    async def stream(self, payload: dict) -> AsyncIterator[bytes]:
        if not (payload.get("input") or payload.get("text")):
            raise HTTPException(400, "input is required")
        # A prior synth left a worker stuck in native MLX: refuse new work fast
        # (503 at the endpoint) rather than start a second concurrent generate().
        if self._degraded:
            raise Qwen3SidecarWedged(self._degraded_reason or "sidecar degraded")
        await self.ensure()
        loop = asyncio.get_running_loop()
        async with self.lock:
            if self._degraded:  # may have latched while we awaited ensure()/lock
                raise Qwen3SidecarWedged(self._degraded_reason or "sidecar degraded")
            tq: "queue.Queue" = queue.Queue(maxsize=QUEUE_MAX)
            cancel_event = threading.Event()
            self.last_synth_started_at = time.time()
            fut = self._gen_executor.submit(self._gen_worker, payload, tq, cancel_event)
            produced_any = False
            outcome = "pending"  # -> completed | empty | aborted | failed | cancelled
            try:
                while True:
                    budget = STALL_WATCHDOG_SEC if produced_any else FIRST_FRAME_TIMEOUT_SEC
                    try:
                        # Blocking, timeout-bounded read off the loop == the watchdog.
                        item = await loop.run_in_executor(self._reader_executor, tq.get, True, budget)
                    except queue.Empty:
                        outcome = "aborted"
                        self.aborted_synth_count += 1
                        self.last_error = (
                            f"synth aborted by watchdog: no PCM frame for {budget:.0f}s "
                            f"({'mid-stream' if produced_any else 'before first frame'})"
                        )
                        cancel_event.set()
                        raise Qwen3SynthAborted(self.last_error)
                    if item is _DONE:
                        break
                    if isinstance(item, BaseException):
                        outcome = "failed"
                        self.failed_synth_count += 1
                        self.last_error = f"synth failed: {item!r}"
                        raise RuntimeError(self.last_error) from item
                    produced_any = True
                    self.synth_count += 1
                    self.last_successful_synth = time.time()
                    yield item
                # Generation reached DONE. Real audio => success; zero frames is a
                # no-audio failure the endpoint turns into a 503 (never a 200/0-bytes).
                if produced_any:
                    outcome = "completed"
                    self.last_synth_finished_at = time.time()
                    self.completed_synth_count += 1
                    self.last_error = None
                else:
                    outcome = "empty"
                    self.empty_synth_count += 1
                    self.last_error = "synth produced no audio (0 frames)"
            except (asyncio.CancelledError, GeneratorExit):
                # Client/probe disconnected mid-stream: not a synth failure. Count
                # it separately and do NOT poison last_error with a scary traceback.
                outcome = "cancelled"
                self.client_cancel_count += 1
                raise
            except Qwen3SynthAborted:
                raise
            except BaseException as e:
                if outcome == "pending":
                    outcome = "failed"
                    self.failed_synth_count += 1
                    self.last_error = f"synth failed/interrupted: {e!r}"
                raise
            finally:
                cancel_event.set()
                await self._reap_worker(fut)


runtime = Qwen3MLXRuntime()


@app.on_event("startup")
async def _install_task_dumper() -> None:  # pragma: no cover - diagnostics
    with contextlib.suppress(Exception):
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGUSR2, _dump_asyncio_tasks, loop)


def _mlx_mem_stats() -> dict:
    stats: dict = {"limits": runtime.mlx_limits}
    with contextlib.suppress(Exception):
        stats["active_bytes"] = int(mx.get_active_memory())
    with contextlib.suppress(Exception):
        stats["peak_bytes"] = int(mx.get_peak_memory())
    with contextlib.suppress(Exception):
        stats["cache_bytes"] = int(mx.get_cache_memory())
    return stats


@app.get("/health")
async def health():
    return {
        "ok": True,
        "service": "qwen3-tts-mlx",
        "server_build": SERVER_BUILD,
        "model": MODEL_ID,
        "loaded": runtime.model is not None or runtime.last_successful_synth is not None,
        "loaded_at": runtime.loaded_at,
        "last_successful_synth": runtime.last_successful_synth,
        "last_synth_started_at": runtime.last_synth_started_at,
        "last_synth_finished_at": runtime.last_synth_finished_at,
        "synth_count": runtime.synth_count,
        "completed_synth_count": runtime.completed_synth_count,
        "aborted_synth_count": runtime.aborted_synth_count,
        "empty_synth_count": runtime.empty_synth_count,
        "failed_synth_count": runtime.failed_synth_count,
        "client_cancel_count": runtime.client_cancel_count,
        "degraded": runtime._degraded,
        "degraded_reason": runtime._degraded_reason,
        "watchdog": {
            "first_frame_timeout_sec": FIRST_FRAME_TIMEOUT_SEC,
            "stall_watchdog_sec": STALL_WATCHDOG_SEC,
            "abort_grace_sec": ABORT_GRACE_SEC,
        },
        "mlx_memory": _mlx_mem_stats(),
        "last_error": runtime.last_error,
        "supports_true_streaming": True,
        "streaming_implementation": "mlx-audio-qwen3-streaming-step",
        "sample_rate": 24000,
        "channels": 1,
        "sample_format": "pcm16",
        "default_voice": DEFAULT_VOICE,
        "clone_voices": sorted(runtime.clone_profiles.keys()),
        "warmed_profiles": sorted(runtime.warmed_profiles),
    }


@app.post("/load")
async def load():
    await runtime.ensure()
    return await health()


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": "qwen3-tts-0.6b-custom-mlx", "object": "model", "owned_by": "local-mlx"}]}


@app.get("/v1/voices")
@app.get("/v1/audio/voices")
async def voices():
    return {
        "object": "list",
        "data": [
            {"id": "voice01", "object": "voice", "provider": "qwen3-mlx", "type": "reference_clone"},
            {"id": "voice01-xvector", "object": "voice", "provider": "qwen3-mlx", "type": "speaker_embedding_only"},
            {"id": "voice1-all-samples", "object": "voice", "provider": "qwen3-mlx", "alias_for": "voice01"},
            {"id": "Ryan", "object": "voice", "provider": "qwen3-mlx", "type": "predefined"},
            {"id": "Aiden", "object": "voice", "provider": "qwen3-mlx", "type": "predefined"},
        ],
    }


async def _safe_aclose(agen: AsyncIterator[bytes]) -> None:
    with contextlib.suppress(Exception):
        await agen.aclose()


@app.post("/v1/audio/speech-stream")
async def speech_stream(request: Request):
    payload = await request.json()
    if not (payload.get("input") or payload.get("text")):
        raise HTTPException(400, "input is required")

    agen = runtime.stream(payload)
    # Contract fix: pull the FIRST frame before committing the HTTP status. A
    # stalled/failed/empty synth must return a real 5xx, never a false-success
    # 200 with 0 bytes (StreamingResponse commits 200 before the first body byte).
    try:
        first = await agen.__anext__()
    except StopAsyncIteration:
        await _safe_aclose(agen)
        return JSONResponse(
            status_code=503,
            content={"error": "qwen3-mlx produced no audio", "detail": runtime.last_error},
        )
    except Qwen3SynthAborted as e:
        await _safe_aclose(agen)
        return JSONResponse(
            status_code=503,
            content={"error": "qwen3-mlx synth aborted before first frame", "detail": str(e)},
        )
    except Qwen3SidecarWedged as e:
        await _safe_aclose(agen)
        return JSONResponse(
            status_code=503,
            content={"error": "qwen3-mlx sidecar degraded (restart pending)", "detail": str(e)},
        )
    except HTTPException:
        await _safe_aclose(agen)
        raise
    except asyncio.CancelledError:
        await _safe_aclose(agen)
        raise  # client disconnected — propagate cancellation, don't mask as 5xx
    except BaseException as e:
        await _safe_aclose(agen)
        return JSONResponse(
            status_code=502,
            content={"error": "qwen3-mlx synth failed before first frame", "detail": repr(e)},
        )

    async def body() -> AsyncIterator[bytes]:
        try:
            yield first
            async for chunk in agen:
                yield chunk
        except Qwen3SynthAborted as e:
            # Mid-stream abort: the 200 headers are already sent, so we cannot
            # downgrade to 5xx. Raise so uvicorn drops the connection abnormally
            # (truncated stream / RemoteProtocolError) instead of a clean EOF that
            # a client would read as success. UTTS treats the truncation as failure.
            raise RuntimeError(str(e)) from e
        finally:
            await _safe_aclose(agen)

    return StreamingResponse(
        body(),
        media_type="audio/pcm",
        headers={
            "X-Qwen-Streaming": "true",
            "X-Qwen-Streaming-Mode": "mlx-audio-streaming-step",
            "X-Audio-Sample-Rate": "24000",
            "X-Audio-Channels": "1",
            "X-Audio-Sample-Format": "pcm16",
        },
    )
