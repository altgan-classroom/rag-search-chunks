"""Evaluation metrics. Retrieval metrics are computed on *matched child chunks* (precise pages, comparable across
chunking strategies); `answer_in_context` is computed on the final context the model sees (boundary-independent)."""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field

from ..store.qdrant_store import Hit
from .schema import GoldQuestion, GoldRef

ABSTAIN_PAT = re.compile(r"(could not find|cannot find|can't find|not (?:found|mentioned|provided|available|contained)|"
                         r"no (?:information|mention|data)|do(?:es)? not (?:contain|include|provide|specify|mention))", re.I)


def norm(s: str) -> str:
    s = s.lower().replace("–", "-").replace("−", "-")
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)  # 1,520 -> 1520
    return re.sub(r"\s+", " ", re.sub(r"[*_`|]", " ", s)).strip()


def contains_all(text: str, groups: list[list[str]]) -> bool:
    t = norm(text)
    return all(any(norm(alt) in t for alt in g) for g in groups) if groups else True


def hit_matches(h: Hit, ref: GoldRef) -> bool:
    if not ref.matches_doc(h.payload.get("doc_name", "")):
        return False
    if not ref.pages:
        return True
    lo, hi = h.payload.get("page_start"), h.payload.get("page_end") or h.payload.get("page_start")
    return lo is not None and any(lo <= p <= hi for p in ref.pages)


def first_rank(hits: list[Hit], ref: GoldRef) -> int | None:
    for i, h in enumerate(hits):
        if hit_matches(h, ref):
            return i + 1
    return None


@dataclass
class RetrievalScores:
    recall: dict[int, float] = field(default_factory=dict)  # k -> share of gold refs found in top-k
    hit: dict[int, float] = field(default_factory=dict)  # k -> any gold ref found
    doc_recall: dict[int, float] = field(default_factory=dict)  # doc-level (ignores pages)
    mrr: float = 0.0
    ndcg: dict[int, float] = field(default_factory=dict)


def retrieval_scores(hits: list[Hit], q: GoldQuestion, ks=(1, 3, 5, 10)) -> RetrievalScores:
    s = RetrievalScores()
    ranks = [first_rank(hits, r) for r in q.gold]
    doc_ranks = [first_rank(hits, GoldRef(doc=r.doc)) for r in q.gold]
    for k in ks:
        found = [r for r in ranks if r is not None and r <= k]
        s.recall[k] = len(found) / len(q.gold)
        s.hit[k] = 1.0 if found else 0.0
        s.doc_recall[k] = sum(1 for r in doc_ranks if r is not None and r <= k) / len(q.gold)
        # binary-relevance nDCG where each gold ref can be satisfied ONCE: extra chunks from an already-found
        # gold doc are redundant, not additional relevance (otherwise DCG can exceed the ideal, i.e. nDCG > 1)
        remaining = list(q.gold)
        rel = []
        for h in hits[:k]:
            m = next((r for r in remaining if hit_matches(h, r)), None)
            if m is not None:
                remaining.remove(m)
            rel.append(1.0 if m is not None else 0.0)
        dcg = sum(v / math.log2(i + 2) for i, v in enumerate(rel))
        ideal = sum(1 / math.log2(i + 2) for i in range(min(len(q.gold), k)))
        s.ndcg[k] = dcg / ideal if ideal else 0.0
    valid = [r for r in ranks if r is not None]
    s.mrr = 1.0 / min(valid) if valid else 0.0
    return s


def is_abstention(answer: str) -> bool:
    return bool(ABSTAIN_PAT.search(answer[:400]))


@dataclass
class AnswerScores:
    correct_kw: float | None  # None when the question has no must_contain
    abstained: bool
    abstain_correct: bool
    grounding: float | None
    citations_valid: bool | None
    cited_gold: float | None  # share of cited sources that are gold docs (citation precision)


def cited_source_precision(cited_docs: list[tuple[str, int | None]], q: GoldQuestion) -> float | None:
    if not cited_docs or not q.gold:
        return None
    ok = sum(1 for d, p in cited_docs if any(r.matches_doc(d) and (not r.pages or p in r.pages) for r in q.gold))
    return ok / len(cited_docs)


def bootstrap_ci(vals: list[float], n: int = 1000, seed: int = 7) -> tuple[float, float, float]:
    if not vals:
        return (float("nan"),) * 3
    rng = random.Random(seed)
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return m, m, m
    means = sorted(sum(rng.choices(vals, k=len(vals))) / len(vals) for _ in range(n))
    return m, means[int(0.025 * n)], means[int(0.975 * n) - 1]
