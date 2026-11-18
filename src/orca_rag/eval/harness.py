"""Run a Profile against a GoldSet and aggregate by category with bootstrap CIs.

Two modes:
  retrieval-only  (keyless): ranking metrics on matched children, answer_in_context on the final context, latency,
                  context size, ACL-leak count, abstention separability (AUROC of best rerank score).
  generate        : additionally runs the full Answerer (real LLM, or the offline extractive mock) and scores the
                    answer: keyword correctness, abstention correctness, grounding, citation precision.
Numbers from the offline mock generator are NOT quality evidence for a real LLM; reports are tagged by `llm_mode`."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

from ..config.settings import Profile
from ..generation.answer import Answerer
from ..generation.context import build_sources
from ..generation.verify import check_grounding
from ..ingestion.chunkers import count_tokens
from ..llm.gateway import LLMGateway
from ..retrieval.pipeline import Retriever
from ..store.base import Principal, VectorStore
from .metrics import (bootstrap_ci, cited_source_precision, contains_all, is_abstention, retrieval_scores)
from .schema import GoldQuestion, GoldSet

KS = (1, 3, 5, 10)


@dataclass
class QResult:
    id: str
    category: str
    answerable: bool
    metrics: dict[str, float]
    retrieved: list[dict]
    answer: str = ""
    error: str = ""


@dataclass
class EvalRun:
    profile: str
    goldset: str
    llm_mode: str  # none | mock | real
    results: list[QResult]
    llm_calls: int = 0
    wall_s: float = 0.0
    failures: int = 0
    meta: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- aggregation
    def _vals(self, metric: str, cat: str | None = None) -> list[float]:
        return [r.metrics[metric] for r in self.results if metric in r.metrics and (cat is None or r.category == cat)]

    def summary(self) -> dict:
        names = sorted({m for r in self.results for m in r.metrics})
        cats = sorted({r.category for r in self.results})
        out = {"profile": self.profile, "goldset": self.goldset, "llm_mode": self.llm_mode, "n": len(self.results),
               "failures": self.failures, "llm_calls": self.llm_calls, "wall_s": round(self.wall_s, 1),
               "overall": {}, "by_category": {c: {} for c in cats}, "counts": {c: sum(r.category == c for r in self.results) for c in cats}}
        for m in names:
            v = self._vals(m)
            if v:
                mu, lo, hi = bootstrap_ci(v)
                out["overall"][m] = {"mean": round(mu, 4), "lo": round(lo, 4), "hi": round(hi, 4), "n": len(v)}
            for c in cats:
                v = self._vals(m, c)
                if v:
                    out["by_category"][c][m] = {"mean": round(mean(v), 4), "n": len(v)}
        out["abstention_auroc"] = self.abstention_auroc()
        return out

    def abstention_auroc(self) -> float | None:
        """How well the best reranker score separates answerable from unanswerable questions (1.0 = perfect)."""
        pos = [r.metrics["best_score"] for r in self.results if r.answerable and "best_score" in r.metrics]
        neg = [r.metrics["best_score"] for r in self.results if not r.answerable and "best_score" in r.metrics]
        if not pos or not neg:
            return None
        wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
        return round(wins / (len(pos) * len(neg)), 4)

    def save(self, outdir: str | Path) -> Path:
        d = Path(outdir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(json.dumps(self.summary(), indent=1))
        with (d / "results.jsonl").open("w") as f:
            for r in self.results:
                f.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")
        return d


async def eval_question(q: GoldQuestion, retriever: Retriever, answerer: Answerer | None, ks=KS) -> QResult:
    p = Principal(f"eval-{q.id}", q.tenant, tuple(q.groups))
    m: dict[str, float] = {}
    t0 = time.perf_counter()
    ret = await retriever.retrieve(q.question, p, q.history or None)
    m["retrieval_ms"] = (time.perf_counter() - t0) * 1000
    ctx_text = "\n".join(h.text for h in ret.hits)
    m["context_tokens"] = float(count_tokens(ctx_text))
    if ret.best_score is not None:
        m["best_score"] = float(ret.best_score)
    elif ret.children:
        m["best_score"] = float(ret.children[0].score)
    if q.forbidden_docs:
        m["acl_leak"] = float(any(h.payload.get("doc_name") in q.forbidden_docs for h in ret.children + ret.hits))
    if q.answerable and q.gold:
        s = retrieval_scores(ret.children, q, ks)
        for k in ks:
            m[f"recall@{k}"], m[f"hit@{k}"], m[f"doc_recall@{k}"], m[f"ndcg@{k}"] = s.recall[k], s.hit[k], s.doc_recall[k], s.ndcg[k]
        m["mrr"] = s.mrr
        if q.must_contain:
            m["answer_in_context"] = float(contains_all(ctx_text, q.must_contain))
    res = QResult(q.id, q.category, q.answerable, m,
                  [{"doc": h.payload.get("doc_name"), "pages": list(h.pages), "score": round(h.score, 4)} for h in ret.children[:10]])
    if answerer is not None:
        events = [e async for e in answerer.run(q.question, p, q.history or None)]
        ans = next((e["text"] for e in reversed(events) if e["type"] == "answer"), "")
        res.answer = ans
        err = [e for e in events if e["type"] == "error"]
        if err:
            res.error = err[0]["text"]
        abst = is_abstention(ans)
        m["abstained"] = float(abst)
        m["abstain_correct"] = float(abst == (not q.answerable))
        if q.answerable and q.must_contain:
            m["answer_correct_kw"] = float(contains_all(ans, q.must_contain))
        srcs = build_sources(ret.hits, 10_000)
        if ans and not abst and srcs:
            g = check_grounding(ans, srcs)
            m["grounding"], m["citations_valid"] = g.score, float(not g.invalid_citations)
            import re
            used = sorted({int(n) for n in re.findall(r"\[(\d+)\]", ans)})
            cp = cited_source_precision([(srcs[n - 1].doc_name, srcs[n - 1].page) for n in used if 0 < n <= len(srcs)], q)
            if cp is not None:
                m["citation_precision"] = cp
    return res


async def evaluate(profile: Profile, gold: GoldSet, store: VectorStore, llm: LLMGateway, *, generate: bool = False,
                   limit: int | None = None, categories: set[str] | None = None) -> EvalRun:
    retriever = Retriever(store, profile, llm)
    answerer = Answerer(retriever, profile, llm) if generate else None
    qs = [q for q in gold.questions if not categories or q.category in categories][:limit]
    t0, results, fails = time.perf_counter(), [], 0
    calls0 = llm.usage.calls
    for q in qs:
        try:
            results.append(await eval_question(q, retriever, answerer))
        except Exception as e:  # a failed question is reported, not silently dropped or zero-filled
            fails += 1
            results.append(QResult(q.id, q.category, q.answerable, {}, [], error=f"{type(e).__name__}: {e}"))
    mode = "none" if not generate else ("mock" if llm.mock else "real")
    return EvalRun(profile.name, gold.name, mode, results, llm.usage.calls - calls0, time.perf_counter() - t0, fails)


# ------------------------------------------------------------------------------ reporting
def fmt(summary: dict, metric: str, ci: bool = False) -> str:
    v = summary["overall"].get(metric)
    if not v:
        return "-"
    return f"{v['mean']:.3f}" + (f" [{v['lo']:.2f},{v['hi']:.2f}]" if ci else "")


def comparison_table(summaries: list[dict], metrics: list[str], ci: bool = False) -> str:
    head = "| profile | n | " + " | ".join(metrics) + " |"
    sep = "|---|---:|" + "---:|" * len(metrics)
    rows = [f"| {s['profile']} | {s['n']} | " + " | ".join(fmt(s, m, ci) for m in metrics) + " |" for s in summaries]
    return "\n".join([head, sep, *rows])


def category_table(summaries: list[dict], metric: str) -> str:
    cats = sorted({c for s in summaries for c in s["by_category"] if metric in s["by_category"][c]})
    head = "| category (n) | " + " | ".join(s["profile"] for s in summaries) + " |"
    sep = "|---|" + "---:|" * len(summaries)
    rows = []
    for c in cats:
        n = summaries[0]["counts"].get(c, 0)
        cells = [f"{s['by_category'].get(c, {}).get(metric, {}).get('mean', float('nan')):.3f}" for s in summaries]
        rows.append(f"| {c} ({n}) | " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *rows])
