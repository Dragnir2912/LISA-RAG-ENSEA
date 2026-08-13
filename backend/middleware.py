"""Protections HTTP communes à toutes les routes de l'API."""

from __future__ import annotations

import secrets
import threading
import time
from collections import defaultdict, deque

from fastapi import Request
from fastapi.responses import JSONResponse

from backend.configuration import ApplicationSettings


class SlidingWindowLimiter:
    """Limiteur léger ; PostgreSQL/Redis pourra le remplacer sans toucher aux routes."""

    def __init__(self) -> None:
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, *, limit: int, window_seconds: int) -> bool:
        now = time.monotonic()
        cutoff = now - window_seconds
        with self._lock:
            events = self._events[key]
            while events and events[0] < cutoff:
                events.popleft()
            if len(events) >= limit:
                return False
            events.append(now)
            return True


limiter = SlidingWindowLimiter()


def install_security_middleware(app, settings: ApplicationSettings) -> None:
    from starlette.middleware.cors import CORSMiddleware
    from starlette.middleware.trustedhost import TrustedHostMiddleware

    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.trusted_hosts))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-CSRF-Token", "X-Request-ID"],
    )

    @app.middleware("http")
    async def security_boundary(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", "")[:80] or secrets.token_hex(12)
        request.state.request_id = request_id

        content_length = request.headers.get("content-length")
        if content_length:
            try:
                declared_size = int(content_length)
            except ValueError:
                return JSONResponse(
                    {"detail": "En-tête Content-Length invalide.", "request_id": request_id},
                    status_code=400,
                )
            if declared_size < 0 or declared_size > settings.max_request_bytes:
                return JSONResponse(
                    {"detail": "Requête trop volumineuse.", "request_id": request_id},
                    status_code=413,
                )

        client = request.client.host if request.client else "unknown"
        auth_path = request.url.path.startswith("/api/auth/")
        limit = 30 if auth_path else 600
        if not limiter.allow(
            f"{client}:{'auth' if auth_path else 'api'}",
            limit=limit,
            window_seconds=60,
        ):
            return JSONResponse(
                {"detail": "Trop de requêtes. Réessayez dans un instant.", "request_id": request_id},
                status_code=429,
                headers={"Retry-After": "60"},
            )

        if request.method in {"POST", "PATCH", "PUT", "DELETE"}:
            fetch_site = request.headers.get("sec-fetch-site", "")
            if fetch_site == "cross-site":
                return JSONResponse(
                    {"detail": "Requête intersite refusée.", "request_id": request_id},
                    status_code=403,
                )

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
            "base-uri 'self'; form-action 'self'; frame-ancestors 'none'"
        )
        if settings.production:
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response
