"""ACL-safe semantic answer cache.

Key = (tenant, sorted groups, collection, data_version). An entry is only served to a principal with the SAME
visibility, so a cached answer can never leak documents across tenants/groups. Matching is by embedding cosine
(>= threshold) on the *standalone* question. Bumping `data_version` (after ingest) invalidates everything.
In-process LRU+TTL by design: a multi-replica deployment should back this with Redis/Qdrant (see docs/PRODUCTION.md)."""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from ..store.qdrant_store import Principal


@dataclass
class _Entry:
    vec: np.ndarray
    events: list[dict]
    at: float


class SemanticCache:
    def __init__(self, embed, threshold: float = 0.97, ttl_s: float = 3600, max_items: int = 512):
        self.embed, self.threshold, self.ttl, self.max = embed, threshold, ttl_s, max_items
        self._d: OrderedDict[tuple, list[_Entry]] = OrderedDict()
        self.data_version = 0
        self.hits = self.misses = 0

    def _key(self, p: Principal, collection: str) -> tuple:
        return (p.tenant, tuple(sorted(p.groups)), f"user:{p.id}" if p.is_admin else "", collection, self.data_version)

    def invalidate(self) -> None:
        self.data_version += 1
        self._d.clear()

    def get(self, p: Principal, collection: str, question: str) -> list[dict] | None:
        v = np.array(self.embed(question), dtype=np.float32)
        v /= np.linalg.norm(v) + 1e-9
        now = time.time()
        for e in self._d.get(self._key(p, collection), []):
            if now - e.at <= self.ttl and float(e.vec @ v) >= self.threshold:
                self.hits += 1
                self._d.move_to_end(self._key(p, collection))
                return e.events
        self.misses += 1
        return None

    def put(self, p: Principal, collection: str, question: str, events: list[dict]) -> None:
        v = np.array(self.embed(question), dtype=np.float32)
        v /= np.linalg.norm(v) + 1e-9
        k = self._key(p, collection)
        self._d.setdefault(k, []).append(_Entry(v, events, time.time()))
        self._d.move_to_end(k)
        while len(self._d) > self.max:
            self._d.popitem(last=False)
