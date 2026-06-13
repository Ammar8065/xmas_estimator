"""Transient in-memory job store (stateless product: nothing is persisted to disk/DB).

Holds the uploaded photo (as JPEG bytes) just long enough for the review→export round-trip,
keyed by an ephemeral job id. Oldest jobs are evicted past a small cap; everything is lost on
restart — by design (see the stateless/no-cloud architecture decision).
"""
from __future__ import annotations

import threading
import time
import uuid


class JobStore:
    def __init__(self, max_jobs: int = 32):
        self._d: dict[str, dict] = {}
        self._lock = threading.Lock()
        self.max_jobs = max_jobs

    def put(self, jpeg: bytes, width: int, height: int) -> str:
        jid = uuid.uuid4().hex
        with self._lock:
            self._d[jid] = {"jpeg": jpeg, "w": width, "h": height, "ts": time.time()}
            if len(self._d) > self.max_jobs:
                oldest = min(self._d, key=lambda k: self._d[k]["ts"])
                del self._d[oldest]
        return jid

    def get(self, jid: str) -> dict | None:
        with self._lock:
            return self._d.get(jid)
