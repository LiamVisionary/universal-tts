"""Let a web page on hivemindos.app speak through this machine's engine.

The Hivemind Content Studio's Voice page, opened on hivemindos.app, can use the
engine running on the visitor's own computer: the browser calls
http://127.0.0.1:8799 directly, the way hivemindos.app/chat reaches a model on
localhost, and the script and the audio never touch a server of ours. A
browser only allows that when the engine says so (CORS, plus the Private
Network Access preflight Chromium sends before a public page reaches
loopback).

It says so for as little as the page needs: listing models and voices, and
speaking. Everything else — history (earlier audio), saved voice profiles,
cloning, provisioning, loading — answers web pages exactly as before, which is
to say a browser will not hand them the response.

Origins come from UNIVERSAL_TTS_WEB_ORIGINS (comma-separated); the default is
hivemindos.app and pages served from this computer.
"""

from __future__ import annotations

import os
import re

from starlette.requests import Request
from starlette.responses import Response

DEFAULT_ORIGINS = ("https://hivemindos.app", "https://www.hivemindos.app")
# A page served from this computer (a dev server, the desktop studio).
_LOOPBACK_ORIGIN = re.compile(r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d{1,5})?$")

# (method, path pattern) a web page may use. Anything not here gets no CORS
# headers, so a browser keeps the response from the page.
_WEB_ROUTES = (
    ("GET", re.compile(r"^/v1/models$")),
    ("GET", re.compile(r"^/v1/voices$")),
    ("GET", re.compile(r"^/v1/audio/[A-Za-z0-9_.-]{1,80}/voices$")),
    ("POST", re.compile(r"^/v1/audio/speech$")),
)


def web_origins() -> tuple[str, ...]:
    raw = os.environ.get("UNIVERSAL_TTS_WEB_ORIGINS", "").strip()
    if not raw:
        return DEFAULT_ORIGINS
    return tuple(origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip())


def origin_allowed(origin: str) -> bool:
    clean = (origin or "").strip().rstrip("/")
    return bool(clean) and (clean in web_origins() or bool(_LOOPBACK_ORIGIN.match(clean)))


def route_open_to_web(method: str, path: str) -> bool:
    verb = method.upper()
    return any(verb == allowed and pattern.match(path) for allowed, pattern in _WEB_ROUTES)


def _cors_headers(origin: str) -> dict[str, str]:
    return {
        "Access-Control-Allow-Origin": origin,
        "Vary": "Origin",
    }


class WebAccessMiddleware:
    """Plain ASGI, so the streaming routes this engine also serves (the
    desktop's live voice) pass through untouched: only an allowed web route
    gets a header added, and only a preflight is answered here."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope)
        origin = request.headers.get("origin", "")
        path = scope.get("path", "")
        method = scope.get("method", "GET")
        if method == "OPTIONS" and request.headers.get("access-control-request-method"):
            wanted = request.headers.get("access-control-request-method", "")
            if not (origin_allowed(origin) and route_open_to_web(wanted, path)):
                await Response(status_code=403)(scope, receive, send)
                return
            headers = {
                **_cors_headers(origin),
                "Access-Control-Allow-Methods": wanted.upper(),
                "Access-Control-Allow-Headers": "Content-Type, Accept",
                "Access-Control-Max-Age": "600",
            }
            if request.headers.get("access-control-request-private-network", "").lower() == "true":
                headers["Access-Control-Allow-Private-Network"] = "true"
            await Response(status_code=204, headers=headers)(scope, receive, send)
            return
        if not (origin and origin_allowed(origin) and route_open_to_web(method, path)):
            await self.app(scope, receive, send)
            return
        extra = [(key.lower().encode("latin-1"), value.encode("latin-1")) for key, value in _cors_headers(origin).items()]

        async def send_with_cors(message):
            if message.get("type") == "http.response.start":
                message = {**message, "headers": [*message.get("headers", []), *extra]}
            await send(message)

        await self.app(scope, receive, send_with_cors)


def install_web_access(app) -> None:
    app.add_middleware(WebAccessMiddleware)
