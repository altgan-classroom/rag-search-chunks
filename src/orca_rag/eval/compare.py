"""Paired comparison of two eval runs on the SAME questions.

Unpaired confidence intervals on two means are far too wide to detect a technique's effect; the per-question *difference*
is what matters. We bootstrap that difference. If the 95 % interval contains 0 we say so: 'no measurable effect' is a
result, not a failure, and is the honest reading of most small deltas at n ~ 100."""
from __future__ import annotations

import json
import random
from pathlib import Path


def load_results(run_dir: str | Path) -> dict[str, dict]:
    out = {}
    for line in (Path(run_dir) / "results.jsonl").read_text().splitlines():
        r = json.loads(line)
        out[r["id"]] = r
    return out


def paired_delta(a: dict[str, dict], b: dict[str, dict], metric: str, n_boot: int = 2000, seed: int = 11,
                 categories: set[str] | None = None) -> dict:
    ids = [i for i in a if i in b and metric in a[i]["metrics"] and metric in b[i]["metrics"]
           and (not categories or a[i]["category"] in categories)]
    d = [b[i]["metrics"][metric] - a[i]["metrics"][metric] for i in ids]
    if not d:
        return {"n": 0, "mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "verdict": "n/a", "wins": 0, "losses": 0}
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(d, k=len(d))) / len(d) for _ in range(n_boot))
    lo, hi = means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]
    verdict = "better" if lo > 0 else "worse" if hi < 0 else "no measurable effect"
    return {"n": len(d), "mean": sum(d) / len(d), "lo": lo, "hi": hi, "verdict": verdict,
            "wins": sum(x > 0 for x in d), "losses": sum(x < 0 for x in d)}


def ladder_markdown(root: str | Path, metrics=("recall@5", "mrr", "ndcg@10", "answer_in_context"), skip: tuple[str, ...] = ()) -> str:
    """Adjacent-rung paired deltas, in directory-name order (0-baseline, 1+..., 2+...)."""
    root = Path(root)
    rungs = sorted((d for d in root.iterdir() if d.is_dir() and (d / "results.jsonl").exists() and d.name not in skip),
                   key=lambda p: p.name)
    res = {r.name: load_results(r) for r in rungs}
    summ = {r.name: json.loads((r / "summary.json").read_text())["overall"] for r in rungs}
    lines = ["| rung | " + " | ".join(f"{m} (Δ vs previous, 95% CI)" for m in metrics) + " | ctx tokens | p50 ms |",
             "|---|" + "---|" * len(metrics) + "---:|---:|"]
    prev = None
    for r in rungs:
        cells = []
        for m in metrics:
            v = summ[r.name].get(m, {}).get("mean")
            if prev is None or v is None:
                cells.append(f"{v:.3f}" if v is not None else "-")
                continue
            pd = paired_delta(res[prev.name], res[r.name], m)
            tag = {"better": "▲", "worse": "▼", "no measurable effect": "≈", "n/a": ""}[pd["verdict"]]
            cells.append(f"{v:.3f} {tag} {pd['mean']:+.3f} [{pd['lo']:+.3f},{pd['hi']:+.3f}]")
        ctx = summ[r.name].get("context_tokens", {}).get("mean", float("nan"))
        lat = summ[r.name].get("retrieval_ms", {}).get("mean", float("nan"))
        lines.append(f"| {r.name} | " + " | ".join(cells) + f" | {ctx:.0f} | {lat:.0f} |")
        prev = r
    lines.append("\n▲ better / ▼ worse: paired-bootstrap 95% CI of the per-question difference excludes 0. "
                 "≈: CI includes 0, i.e. **no measurable effect** at this n.")
    return "\n".join(lines)


def reference_markdown(root: str | Path, ref: str, metrics=("recall@5", "mrr", "answer_in_context")) -> str:
    """Every variant against ONE reference run (not adjacent rungs): per-question paired-bootstrap delta + CI."""
    root = Path(root)
    runs = sorted(d for d in root.iterdir() if d.is_dir() and (d / "results.jsonl").exists())
    res = {r.name: load_results(r) for r in runs}
    summ = {r.name: json.loads((r / "summary.json").read_text())["overall"] for r in runs}
    lines = ["| variant | " + " | ".join(f"{m} (Δ vs {ref}, 95% CI)" for m in metrics) + " | ctx tokens | p50 ms |",
             "|---|" + "---|" * len(metrics) + "---:|---:|"]
    for r in runs:
        cells = []
        for m in metrics:
            v = summ[r.name].get(m, {}).get("mean")
            if r.name == ref or v is None:
                cells.append(f"{v:.3f}" if v is not None else "-")
                continue
            pd = paired_delta(res[ref], res[r.name], m)
            tag = {"better": "▲", "worse": "▼", "no measurable effect": "≈", "n/a": ""}[pd["verdict"]]
            cells.append(f"{v:.3f} {tag} {pd['mean']:+.3f} [{pd['lo']:+.3f},{pd['hi']:+.3f}]")
        ctx = summ[r.name].get("context_tokens", {}).get("mean", float("nan"))
        lat = summ[r.name].get("retrieval_ms", {}).get("mean", float("nan"))
        lines.append(f"| {r.name}{' (reference)' if r.name == ref else ''} | " + " | ".join(cells) + f" | {ctx:.0f} | {lat:.0f} |")
    lines.append("\n▲ better / ▼ worse: the paired-bootstrap 95% CI of the per-question difference excludes 0; ≈ = no measurable effect.")
    return "\n".join(lines)
