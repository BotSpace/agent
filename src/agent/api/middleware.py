"""HTTP API middleware — per-IP rate limiting + access logging.

No external deps (KISS): an in-memory sliding-window limiter and a one-line
access logger. In-memory state is PER-POD — with multiple agent replicas the
effective limit is `limit × replicas`, which is fine as a coarse abuse guard.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger("agent.access")

# k8s liveness/readiness probes + Prometheus scrape + OAuth discovery — never
# rate-limited and never access-logged (pure noise).
_NOISE_PREFIXES = ("/health", "/metrics", "/.well-known")
# Additionally exempt from RATE LIMITING: the MCP constructor surface makes many
# legitimate high-frequency tool calls while a user builds a bot in the
# dashboard — limiting it would break the builder. It is still access-logged.
_RATE_EXEMPT_PREFIXES = (*_NOISE_PREFIXES, "/mcp")


def _client_ip(request: Request) -> str:
    """Real client IP behind the k8s ingress (X-Forwarded-For), else peer."""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _has_prefix(path: str, prefixes: tuple[str, ...]) -> bool:
    return any(path == p or path.startswith(p + "/") or path.startswith(p) for p in prefixes)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """In-memory per-IP sliding-window limiter: `limit` requests per `window_sec`."""

    def __init__(self, app, *, limit: int, window_sec: float = 60.0) -> None:
        super().__init__(app)
        self._limit = max(1, limit)
        self._window = window_sec
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if _has_prefix(path, _RATE_EXEMPT_PREFIXES):
            return await call_next(request)

        ip = _client_ip(request)
        now = time.monotonic()
        cutoff = now - self._window
        dq = self._hits[ip]
        while dq and dq[0] < cutoff:
            dq.popleft()

        if len(dq) >= self._limit:
            retry_after = int(self._window - (now - dq[0])) + 1
            logger.warning(
                "rate_limit_block ip=%s path=%s hits=%d limit=%d/%ds",
                ip, path, len(dq), self._limit, int(self._window),
            )
            return JSONResponse(
                {"detail": {"message": "Juda ko'p so'rov yuborildi. Biroz kutib qayta urinib ko'ring.",
                            "code": "rate_limited"}},
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )

        dq.append(now)
        # Bound memory: prune idle IPs when the table grows large.
        if len(self._hits) > 10_000:
            for stale in [k for k, v in self._hits.items() if not v or v[-1] < cutoff]:
                del self._hits[stale]
        return await call_next(request)


class AccessLogMiddleware(BaseHTTPMiddleware):
    """One structured log line per API request: method, path, ip, status, ms."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if _has_prefix(path, _NOISE_PREFIXES):
            return await call_next(request)

        ip = _client_ip(request)
        start = time.monotonic()
        try:
            response = await call_next(request)
        except Exception:
            dur_ms = (time.monotonic() - start) * 1000
            logger.exception("%s %s ip=%s -> ERROR (%.0fms)", request.method, path, ip, dur_ms)
            raise
        dur_ms = (time.monotonic() - start) * 1000
        logger.info(
            "%s %s ip=%s -> %d (%.0fms)",
            request.method, path, ip, response.status_code, dur_ms,
        )
        return response
