"""ACL-safe semantic answer cache.

Key = (tenant, user id, sorted groups, collection, data_version). The USER ID is always part of the key: documents can be
granted to a single user (`user:<id>`), so two users in the same tenant/groups can legitimately see different sources,
and a cached answer must never cross that line. (A shared per-group cache is possible only if every cited source is
verified group-level or public; deliberately not done here.)

Matching is two-stage: embedding cosine >= threshold AND an exact match of the *identifiers and numbers* in the
question. Embeddings treat "well MB-104" and "well MB-140" as near-identical, so similarity alone would serve the wrong
well's answer. Bumping `data_version` (after ingest) invalidates everything.
In-process LRU+TTL by design: a multi-replica deployment should back this with Redis/Qdrant (see docs/PRODUCTION.md)."""
from __future__ import annotations

import re
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from ..store.qdrant_store import Principal


_FACTS = re.compile(r"[A-Za-z]*-?\d[\w\-.,/%]*")


def fact_signature(question: str) -> frozenset[str]:
    """Identifiers and numbers in the question (anything containing a digit), normalised. Two questions may share a
    cache entry only if these are identical."""
    return frozenset(t.lower().strip(".,;:?!") for t in _FACTS.findall(question))


@dataclass
class _Entry:
    vec: np.ndarray
    events: list[dict]
    at: float
    facts: frozenset[str]


class SemanticCache:
    def __init__(self, embed, threshold: float = 0.97, ttl_s: float = 3600, max_items: int = 512):
        self.embed, self.threshold, self.ttl, self.max = embed, threshold, ttl_s, max_items
        self._d: OrderedDict[tuple, list[_Entry]] = OrderedDict()
        self.data_version = 0
        self.hits = self.misses = 0

    def _key(self, p: Principal, collection: str) -> tuple:
        return (p.tenant, p.id, tuple(sorted(p.groups)), collection, self.data_version)

    def invalidate(self) -> None:
        self.data_version += 1
        self._d.clear()

    def get(self, p: Principal, collection: str, question: str) -> list[dict] | None:
        v = np.array(self.embed(question), dtype=np.float32)
        v /= np.linalg.norm(v) + 1e-9
        now = time.time()
        facts = fact_signature(question)
        for e in self._d.get(self._key(p, collection), []):
            if now - e.at <= self.ttl and e.facts == facts and float(e.vec @ v) >= self.threshold:
                self.hits += 1
                self._d.move_to_end(self._key(p, collection))
                return e.events
        self.misses += 1
        return None

    def put(self, p: Principal, collection: str, question: str, events: list[dict]) -> None:
        v = np.array(self.embed(question), dtype=np.float32)
        v /= np.linalg.norm(v) + 1e-9
        k = self._key(p, collection)
        self._d.setdefault(k, []).append(_Entry(v, events, time.time(), fact_signature(question)))
        self._d.move_to_end(k)
        while len(self._d) > self.max:
            self._d.popitem(last=False)
