from __future__ import annotations

import numpy as np

from ..store.qdrant_store import Hit


def rrf(rankings: list[list[Hit]], k: int = 60, weights: list[float] | None = None) -> list[Hit]:
    """Reciprocal Rank Fusion across result lists (multi-query / multi-retriever). Ranks only; scores ignored."""
    weights = weights or [1.0] * len(rankings)
    score: dict[str, float] = {}
    best: dict[str, Hit] = {}
    for w, lst in zip(weights, rankings):
        for r, h in enumerate(lst):
            score[h.id] = score.get(h.id, 0.0) + w / (k + r + 1)
            best.setdefault(h.id, h)
    return [Hit(i, s, best[i].text, best[i].payload) for i, s in sorted(score.items(), key=lambda x: -x[1])]


def mmr(hits: list[Hit], vectors: np.ndarray, query_vec: np.ndarray | None, lam: float, k: int,
        relevance: np.ndarray | None = None) -> list[Hit]:
    """Maximal Marginal Relevance: trade relevance against redundancy with already-selected chunks.

    `relevance` should be the *ranker's own* scores (e.g. cross-encoder, min-max normalised). Recomputing relevance
    from dense cosine here would silently discard a better ranker's ordering (measured: it pushed the correct
    document revision from rank 1 to rank 6). Falls back to cosine only when no scores are supplied."""
    if not hits:
        return []
    v = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)
    if relevance is not None:
        rel = np.asarray(relevance, dtype=np.float32)
        span = float(rel.max() - rel.min())
        rel = (rel - rel.min()) / span if span > 1e-9 else np.ones_like(rel)
    else:
        q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        rel = v @ q
    chosen: list[int] = []
    cand = list(range(len(hits)))
    while cand and len(chosen) < k:
        if not chosen:
            best = max(cand, key=lambda i: rel[i])
        else:
            sim = v[cand] @ v[chosen].T
            mm = lam * rel[cand] - (1 - lam) * sim.max(axis=1)
            best = cand[int(np.argmax(mm))]
        chosen.append(best)
        cand.remove(best)
    return [hits[i] for i in chosen]
