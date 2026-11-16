"""Embedding backends. Local ONNX via fastembed (no torch, no keys) plus a deterministic hash embedder for tests.

Models are process singletons: loading one per request/task was a measured cost in the baseline (REVIEW S5/S7).
"""
from __future__ import annotations

import hashlib
import math
import re
import threading
from functools import lru_cache
from typing import Protocol

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")
_lock = threading.Lock()


class DenseEmbedder(Protocol):
    name: str
    dim: int

    def embed_docs(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class HashDense:
    """Feature-hashing bag-of-words (+bigrams), L2-normalised. Offline, instant, *lexically* meaningful -
    good enough to unit-test retrieval plumbing. Never use it for quality numbers."""

    def __init__(self, dim: int = 256):
        self.name, self.dim = f"hash:{dim}", dim

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = _TOKEN.findall(text.lower())
        for t in toks + [a + "_" + b for a, b in zip(toks, toks[1:])]:
            h = int(hashlib.md5(t.encode()).hexdigest()[:8], 16)
            v[h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
            v[(h >> 7) % self.dim] += 0.5
        n = float(np.linalg.norm(v)) or 1.0
        return (v / n).tolist()

    def embed_docs(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FastEmbedDense:
    def __init__(self, model: str, doc_prefix: str = "", query_prefix: str = ""):
        from fastembed import TextEmbedding

        self.name = model
        self._m = TextEmbedding(model_name=model)
        self.dim = len(next(iter(self._m.embed(["probe"]))))
        self._dp, self._qp = doc_prefix, query_prefix

    def embed_docs(self, texts):
        with _lock:
            return [v.tolist() for v in self._m.embed([self._dp + t for t in texts], batch_size=32)]

    def embed_query(self, text):
        with _lock:
            return next(iter(self._m.embed([self._qp + text]))).tolist()


@lru_cache(maxsize=8)
def get_dense(model: str, doc_prefix: str = "", query_prefix: str = "") -> DenseEmbedder:
    if model.startswith("hash:"):
        return HashDense(int(model.split(":")[1]))
    return FastEmbedDense(model, doc_prefix, query_prefix)


# ------------------------------------------------------------------------------------- sparse
class SparseVec(tuple):  # (indices, values)
    pass


class HashSparse:
    """Term-frequency sparse vectors with hashed vocabulary; Qdrant applies IDF (modifier=idf)."""

    def __init__(self):
        self.name = "hash-sparse"

    def _vec(self, text: str):
        counts: dict[int, float] = {}
        for t in _TOKEN.findall(text.lower()):
            i = int(hashlib.md5(t.encode()).hexdigest()[:7], 16)
            counts[i] = counts.get(i, 0.0) + 1.0
        idx = sorted(counts)
        return idx, [1.0 + math.log(counts[i]) for i in idx]

    def embed_docs(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FastEmbedSparse:
    def __init__(self, model: str):
        from fastembed import SparseTextEmbedding

        self.name = model
        self._m = SparseTextEmbedding(model_name=model)

    def embed_docs(self, texts):
        with _lock:
            return [(e.indices.tolist(), e.values.tolist()) for e in self._m.embed(texts, batch_size=32)]

    def embed_query(self, text):
        with _lock:
            e = next(iter(self._m.query_embed(text)))
        return e.indices.tolist(), e.values.tolist()


@lru_cache(maxsize=4)
def get_sparse(model: str):
    return HashSparse() if model.startswith("hash") else FastEmbedSparse(model)


# ------------------------------------------------------------------------------- late interaction
class FastEmbedLate:
    def __init__(self, model: str):
        from fastembed import LateInteractionTextEmbedding

        self.name = model
        self._m = LateInteractionTextEmbedding(model_name=model)
        self.dim = int(next(iter(self._m.embed(["probe"]))).shape[-1])

    def embed_docs(self, texts):
        with _lock:
            return [m.tolist() for m in self._m.embed(texts, batch_size=16)]

    def embed_query(self, text):
        with _lock:
            return next(iter(self._m.query_embed(text))).tolist()


@lru_cache(maxsize=2)
def get_late(model: str):
    return FastEmbedLate(model)
