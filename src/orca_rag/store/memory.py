"""In-process reference backend: numpy brute force, no server, no extra dependency. For tests, teaching and tiny corpora
(it scans every point on each query, so do not use it beyond ~50k chunks). Behaves like the Qdrant backend for everything
the contract tests check: hybrid = dense cosine + BM25 fused with RRF, tenant/ACL filter applied before ranking."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import numpy as np

from ..config.settings import Env, Profile
from .base import Hit, Principal, register_store
from .embedders import get_dense

_TOK = re.compile(r"[a-z0-9]+")


class MemoryStore:
    def __init__(self, env: Env, profile: Profile, *, collection: str | None = None, **_):
        self.env, self.profile = env, profile
        self.collection = collection or profile.collection_name()
        e = profile.embedding
        self.dense = get_dense(e.dense_model, e.doc_prefix, e.query_prefix)
        self._pts: dict[str, dict[str, Any]] = {}  # id -> {payload, vec|None, toks}

    def ensure_collection(self) -> None:
        return None

    def health(self) -> dict[str, Any]:
        return {"ok": True, "collection": self.collection, "exists": True, "points": len(self._pts)}

    def stats(self) -> dict[str, Any]:
        return {"points": len(self._pts), "status": "green"}

    def upsert_children(self, rows, batch: int = 64) -> None:
        for i in range(0, len(rows), batch):
            part = rows[i : i + batch]
            vecs = self.dense.embed_docs([r["embed_text"] for r in part])
            for r, v in zip(part, vecs):
                self._pts[r["id"]] = {"payload": {**r["payload"], "text": r["text"]}, "vec": np.asarray(v, dtype=np.float32),
                                      "toks": Counter(_TOK.findall(r["embed_text"].lower()))}

    def upsert_payload_only(self, rows) -> None:
        for r in rows:
            self._pts[r["id"]] = {"payload": {**r["payload"], "text": r.get("text", "")}, "vec": None, "toks": None}

    def _drop(self, pred) -> None:
        for k in [k for k, p in self._pts.items() if pred(p["payload"])]:
            del self._pts[k]

    def delete_doc(self, doc_id: str) -> None:
        self._drop(lambda pl: pl.get("doc_id") == doc_id)

    def delete_source(self, source_key: str, tenant: str) -> None:
        self._drop(lambda pl: pl.get("source_key") == source_key and pl.get("tenant") == tenant)

    def get_manifest(self, source_key: str, tenant: str):
        return next((p["payload"] for p in self._pts.values() if p["payload"].get("kind") == "doc"
                     and p["payload"].get("source_key") == source_key and p["payload"].get("tenant") == tenant), None)

    def _visible(self, principal: Principal, kind: str):
        acl = principal.visible_acl()
        return [(k, p) for k, p in self._pts.items() if p["payload"].get("kind") == kind
                and p["payload"].get("tenant") == principal.tenant and acl & set(p["payload"].get("acl", []))]

    def search(self, query, principal, *, mode="hybrid", top_k=40, fusion="rrf", extra=None, query_dense=None):
        cand = self._visible(principal, "child")  # filter FIRST, then rank
        if not cand:
            return []
        ranks: list[list[str]] = []
        if mode in ("dense", "hybrid"):
            q = np.asarray(query_dense or self.dense.embed_query(query), dtype=np.float32)
            sc = [(float(p["vec"] @ q / ((np.linalg.norm(p["vec"]) * np.linalg.norm(q)) or 1.0)), k) for k, p in cand]
            ranks.append([k for _, k in sorted(sc, reverse=True)[:top_k]])
        if mode in ("sparse", "hybrid"):
            qt = _TOK.findall(query.lower())
            n = len(cand)
            df = Counter(t for _, p in cand for t in set(p["toks"]))
            avg = sum(sum(p["toks"].values()) for _, p in cand) / n
            sc = []
            for k, p in cand:
                dl, s = sum(p["toks"].values()), 0.0
                for t in qt:
                    f = p["toks"].get(t, 0)
                    if f:
                        s += math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * dl / avg))
                sc.append((s, k))
            ranks.append([k for s, k in sorted(sc, reverse=True)[:top_k] if s > 0])
        fused: dict[str, float] = {}
        for lst in ranks:
            for r, k in enumerate(lst):
                fused[k] = fused.get(k, 0.0) + 1.0 / (60 + r + 1)
        out = sorted(fused.items(), key=lambda x: -x[1])[:top_k]
        return [Hit(k, s, self._pts[k]["payload"].get("text", ""), self._pts[k]["payload"]) for k, s in out]

    def fetch_parents(self, parent_ids, principal):
        want = set(parent_ids)
        return {p["payload"]["parent_id"]: Hit(k, 1.0, p["payload"].get("text", ""), p["payload"])
                for k, p in self._visible(principal, "parent") if p["payload"].get("parent_id") in want}


register_store("memory", lambda env, profile, **kw: MemoryStore(env, profile, **kw))
