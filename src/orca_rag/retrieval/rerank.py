from __future__ import annotations

import threading
from functools import lru_cache
from typing import Callable, Protocol

from ..store.base import Hit

_lock = threading.Lock()


class Reranker(Protocol):
    """Re-orders first-stage hits for one query. Scores are only comparable within a query."""
    name: str

    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]: ...


_FACTORIES: dict[str, Callable[[str], Reranker]] = {}


def register_reranker(model_prefix: str, factory: Callable[[str], Reranker]) -> None:
    """Route model ids starting with `model_prefix` to `factory(model_id)` (e.g. a hosted rerank API); the longest
    registered prefix wins and anything unregistered falls back to a local fastembed cross-encoder."""
    _FACTORIES[model_prefix] = factory


class NoopReranker:
    name = "none"

    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]:
        return hits[:top_n]


class LexicalReranker:
    """Offline stand-in for tests: scores by query-term coverage. Not a quality baseline."""
    name = "lexical"

    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]:
        import re
        q = set(re.findall(r"[a-z0-9]+", query.lower()))
        out = []
        for h in hits:
            t = set(re.findall(r"[a-z0-9]+", h.text.lower()))
            out.append(Hit(h.id, len(q & t) / (len(q) or 1), h.text, h.payload))
        return sorted(out, key=lambda x: -x.score)[:top_n]


class CrossEncoderReranker:
    """ONNX cross-encoder via fastembed (no torch). Scores are raw logits: comparable across hits for one query,
    NOT a calibrated probability, so abstention thresholds must be tuned per model (see docs/TUNING.md)."""

    def __init__(self, model: str):
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        self.name = model
        self._m = TextCrossEncoder(model_name=model)

    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]:
        if not hits:
            return []
        with _lock:
            scores = list(self._m.rerank(query, [h.text for h in hits]))
        out = [Hit(h.id, float(s), h.text, h.payload) for h, s in zip(hits, scores)]
        return sorted(out, key=lambda x: -x.score)[:top_n]


@lru_cache(maxsize=4)
def get_reranker(model: str | None):
    if not model or model == "none":
        return NoopReranker()
    if model == "lexical":
        return LexicalReranker()
    match = max((k for k in _FACTORIES if model.startswith(k)), key=len, default=None)
    return _FACTORIES[match](model) if match else CrossEncoderReranker(model)
