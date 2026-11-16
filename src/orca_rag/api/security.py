"""Authentication, tenant principal construction, rate limiting."""
from __future__ import annotations

import hmac
import time
from collections import defaultdict

from fastapi import Depends, HTTPException, Request, status

from ..store.qdrant_store import Principal


def parse_principal(spec: str) -> Principal:
    """api_keys values look like  'tenant|user|group1,group2[|admin]'."""
    parts = spec.split("|")
    tenant, user = parts[0], parts[1] if len(parts) > 1 else "user"
    groups = tuple(g for g in (parts[2].split(",") if len(parts) > 2 else []) if g)
    return Principal(id=user, tenant=tenant, groups=groups, is_admin="admin" in parts[3:])


def authenticate(request: Request) -> Principal:
    auth = request.app.state.env.auth
    key = request.headers.get("x-api-key") or ""
    if not key and (h := request.headers.get("authorization", "")).lower().startswith("bearer "):
        key = h[7:]
    for k, spec in auth.api_keys.items():  # constant-time compare, no early exit on first mismatch
        if hmac.compare_digest(k.encode(), key.encode()):
            return parse_principal(spec)
    if auth.allow_anonymous:
        return Principal("anonymous", "default")
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing or invalid API key", headers={"WWW-Authenticate": "Bearer"})


def require_admin(p: Principal = Depends(authenticate)) -> Principal:
    if not p.is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "admin only")
    return p


class RateLimiter:
    """Token bucket per principal (in-process). For several replicas enforce at the gateway or back with Redis."""

    def __init__(self, rate_per_min: int = 30, burst: int = 10):
        self.rate, self.burst = rate_per_min / 60.0, burst
        self._b: dict[str, tuple[float, float]] = defaultdict(lambda: (float(burst), time.monotonic()))

    def allow(self, key: str) -> bool:
        tokens, last = self._b[key]
        now = time.monotonic()
        tokens = min(self.burst, tokens + (now - last) * self.rate)
        if tokens < 1:
            self._b[key] = (tokens, now)
            return False
        self._b[key] = (tokens - 1, now)
        return True
