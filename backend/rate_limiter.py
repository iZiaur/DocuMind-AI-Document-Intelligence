"""
In-Memory Rate Limiting Module for DocuMind AI (SEC-04).

Design & Deployment Notes:
--------------------------
This module implements a thread-safe, bounded, in-memory sliding-window rate limiter
designed specifically for single-instance development and staging environments.

For multi-instance / distributed production clusters:
Replace this backend with a distributed Redis-based limiter or manage rate limiting
at the reverse proxy / API gateway layer (e.g. Nginx, Cloudflare, Kong).

Key Features:
1. Configurable limits via environment variables (per-minute sliding window).
2. Login endpoint (/api/auth/login): Keyed by validated client IP (protects against credential stuffing).
3. RAG endpoint (/api/rag/query): Keyed by authenticated user ID (ensures corporate NAT users get independent quotas).
4. Anti-Spoofing: Strictly ignores client-supplied X-Forwarded-For headers unless the immediate connection
   originates from a configured TRUSTED_PROXIES host.
5. Bounded Memory: Enforces an LRU key cap (default: 10,000 keys) with periodic expired key pruning
   to guarantee zero memory leaks or denial-of-memory vulnerabilities.
6. Returns standard HTTP 429 with a 'Retry-After' header indicating the wait time in seconds.
7. Zero Logging of Sensitive Data: Never logs credentials, query texts, or tokens.
"""

import os
import time
import math
import logging
import threading
from collections import OrderedDict
from typing import Dict, List, Optional, Set
from fastapi import Request, HTTPException, status

logger = logging.getLogger("documind.rate_limiter")

# Configuration with conservative defaults
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").strip().lower() in ("true", "1", "yes")
RATE_LIMIT_LOGIN_PER_MINUTE = int(os.getenv("RATE_LIMIT_LOGIN_PER_MINUTE", "5"))
RATE_LIMIT_RAG_PER_MINUTE = int(os.getenv("RATE_LIMIT_RAG_PER_MINUTE", "20"))
RATE_LIMITER_MAX_KEYS = int(os.getenv("RATE_LIMITER_MAX_KEYS", "10000"))

# Trusted reverse proxies allowed to set X-Forwarded-For
raw_proxies = os.getenv("TRUSTED_PROXIES", "127.0.0.1,::1,localhost").split(",")
TRUSTED_PROXIES: Set[str] = {p.strip().lower() for p in raw_proxies if p.strip()}


class SlidingWindowRateLimiter:
    """
    Thread-safe, bounded in-memory sliding window rate limiter.
    """

    def __init__(self, max_keys: int = RATE_LIMITER_MAX_KEYS):
        self.max_keys = max_keys
        self._lock = threading.Lock()
        # Maps key -> list of float timestamps within the active window
        self._cache: OrderedDict[str, List[float]] = OrderedDict()

    def _prune_expired_locked(self, now: float, window_seconds: float) -> None:
        """Evicts oldest keys when capacity is reached or keys are completely expired."""
        cutoff = now - window_seconds
        keys_to_delete = []

        for k, timestamps in self._cache.items():
            # Keep only timestamps within window
            valid_ts = [t for t in timestamps if t > cutoff]
            if not valid_ts:
                keys_to_delete.append(k)
            else:
                self._cache[k] = valid_ts

        for k in keys_to_delete:
            self._cache.pop(k, None)

        # Enforce hard capacity bound
        while len(self._cache) > self.max_keys:
            self._cache.popitem(last=False)

    def check_rate_limit(
        self,
        key: str,
        limit: int,
        window_seconds: float = 60.0,
    ) -> None:
        """
        Records a request against `key` and raises HTTP 429 if `limit` is exceeded
        within `window_seconds`.
        """
        if not RATE_LIMIT_ENABLED or limit <= 0:
            return

        now = time.time()
        cutoff = now - window_seconds

        with self._lock:
            # Periodic pruning when approaching capacity
            if len(self._cache) >= self.max_keys:
                self._prune_expired_locked(now, window_seconds)

            timestamps = self._cache.get(key, [])
            # Filter timestamps to current sliding window
            valid_timestamps = [t for t in timestamps if t > cutoff]

            if len(valid_timestamps) >= limit:
                # Limit exceeded: calculate retry-after based on oldest timestamp in window
                oldest_timestamp = valid_timestamps[0]
                retry_after = max(1, int(math.ceil(window_seconds - (now - oldest_timestamp))))
                self._cache[key] = valid_timestamps
                self._cache.move_to_end(key)

                logger.warning(
                    "Rate limit exceeded: key_prefix=%s limit=%d window=%.0fs retry_after=%ds",
                    key.split(":")[0],
                    limit,
                    window_seconds,
                    retry_after,
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. Maximum {limit} requests per minute allowed. Please retry in {retry_after} seconds.",
                    headers={"Retry-After": str(retry_after)},
                )

            # Record current request timestamp
            valid_timestamps.append(now)
            self._cache[key] = valid_timestamps
            self._cache.move_to_end(key)

            # Enforce hard capacity bound
            while len(self._cache) > self.max_keys:
                self._cache.popitem(last=False)

    def reset(self) -> None:
        """Clears all stored rate limit records (useful for test isolation)."""
        with self._lock:
            self._cache.clear()


# Global limiter instance for the FastAPI process
rate_limiter = SlidingWindowRateLimiter()


def resolve_client_ip(request: Request) -> str:
    """
    Securely resolves the client IP address.

    Anti-Spoofing Rule:
    Only inspects X-Forwarded-For if the direct peer IP (request.client.host)
    is an explicitly trusted reverse proxy (TRUSTED_PROXIES).
    Otherwise, strictly returns the direct peer IP.
    """
    direct_host = request.client.host if request.client else "unknown"
    direct_host_clean = direct_host.strip().lower()

    if direct_host_clean in TRUSTED_PROXIES:
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Client IP is the first entry in X-Forwarded-For
            client_ip = forwarded_for.split(",")[0].strip()
            if client_ip:
                return client_ip

    return direct_host


def check_login_rate_limit(request: Request) -> None:
    """
    Enforces IP-based rate limiting on the login endpoint (SEC-04).
    Protects against automated credential-stuffing and brute-force attacks.
    """
    client_ip = resolve_client_ip(request)
    limit = int(os.getenv("RATE_LIMIT_LOGIN_PER_MINUTE", str(RATE_LIMIT_LOGIN_PER_MINUTE)))
    rate_limiter.check_rate_limit(f"login_ip:{client_ip}", limit=limit, window_seconds=60.0)


def check_rag_rate_limit(user_id: int) -> None:
    """
    Enforces user-based rate limiting on the RAG query endpoint (SEC-04).
    Keyed by authenticated user ID to ensure users sharing corporate NATs
    receive independent quotas while preventing Denial-of-Wallet billing exhaustion.
    """
    limit = int(os.getenv("RATE_LIMIT_RAG_PER_MINUTE", str(RATE_LIMIT_RAG_PER_MINUTE)))
    rate_limiter.check_rate_limit(f"rag_user:{user_id}", limit=limit, window_seconds=60.0)
