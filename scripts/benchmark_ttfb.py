from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any

import httpx


def parse_json(value: str) -> dict[str, Any]:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("JSON value must be an object")
    return parsed


def measure_once(
    client: httpx.Client,
    *,
    base_url: str,
    payload: dict[str, Any],
    consume_full_stream: bool,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    with client.stream("POST", f"{base_url.rstrip('/')}/v1/audio/speech-stream", json=payload) as response:
        headers_ms = (time.perf_counter() - t0) * 1000.0
        first_chunk_ms: float | None = None
        first_chunk_bytes = 0
        total_bytes = 0
        if response.status_code >= 400:
            body = response.read().decode("utf-8", "replace")
            return {
                "status": response.status_code,
                "headers_ms": headers_ms,
                "content_type": response.headers.get("content-type"),
                "error": body[:1000],
            }
        for chunk in response.iter_bytes():
            if not chunk:
                continue
            total_bytes += len(chunk)
            if first_chunk_ms is None:
                first_chunk_ms = (time.perf_counter() - t0) * 1000.0
                first_chunk_bytes = len(chunk)
                if not consume_full_stream:
                    break
        return {
            "status": response.status_code,
            "headers_ms": headers_ms,
            "first_chunk_ms": first_chunk_ms,
            "first_chunk_bytes": first_chunk_bytes,
            "total_ms": (time.perf_counter() - t0) * 1000.0,
            "bytes": total_bytes,
            "content_type": response.headers.get("content-type"),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Universal TTS streaming TTFB.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8799")
    parser.add_argument("--provider", help="Provider id to load before benchmarking.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--voice")
    parser.add_argument("--input", default="Hello, this is a short latency benchmark.")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--consume-full-stream", action="store_true")
    parser.add_argument("--payload-json", type=parse_json, default={})
    args = parser.parse_args()

    payload: dict[str, Any] = {
        "model": args.model,
        "input": args.input,
        "response_format": "pcm",
        "realtime_pacing": False,
        "disable_microbatch": True,
        **args.payload_json,
    }
    if args.voice:
        payload["voice"] = args.voice

    timeout = httpx.Timeout(args.timeout, connect=10.0)
    results: list[dict[str, Any]] = []
    with httpx.Client(timeout=timeout) as client:
        if args.provider:
            response = client.post(
                f"{args.base_url.rstrip('/')}/providers/{args.provider}/load",
                json={"mode": "multi", "force": False},
            )
            print(json.dumps({"event": "load", "provider": args.provider, "status": response.status_code}))
            response.raise_for_status()
        for index in range(args.warmups):
            result = measure_once(
                client,
                base_url=args.base_url,
                payload=payload,
                consume_full_stream=args.consume_full_stream,
            )
            print(json.dumps({"event": "warmup", "index": index + 1, **result}))
        for index in range(args.runs):
            result = measure_once(
                client,
                base_url=args.base_url,
                payload=payload,
                consume_full_stream=args.consume_full_stream,
            )
            print(json.dumps({"event": "run", "index": index + 1, **result}))
            results.append(result)

    good = [item for item in results if isinstance(item.get("first_chunk_ms"), (int, float))]
    summary = {
        "model": args.model,
        "voice": args.voice,
        "runs": len(results),
        "ok_runs": len(good),
        "median_first_chunk_ms": statistics.median(item["first_chunk_ms"] for item in good) if good else None,
        "median_headers_ms": statistics.median(item["headers_ms"] for item in good) if good else None,
        "median_total_ms": statistics.median(item["total_ms"] for item in good) if good else None,
    }
    print(json.dumps({"event": "summary", **summary}, indent=2))
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
