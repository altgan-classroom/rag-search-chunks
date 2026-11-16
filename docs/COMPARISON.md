# Comparison: template recipe vs v2, one technique at a time

**Status: partial.** Synthetic ladder complete. Real-corpus (CCS) ladder has finished rungs 0-2; rungs 3-7, leave-one-out,
sweeps and the Qdrant benchmark are still running and are marked *pending*. Nothing below is extrapolated.

## How to read this
* The template **cannot run** (`generate.py` missing, REVIEW C1), so the baseline is a *re-implementation of its recipe*
  (`configs/baseline.yaml`): fixed 2048-char word chunks, dense + BM42 hybrid, no rerank. It substitutes bge-small for
  mxbai-large and drops the Cohere reranker (no key / no torch). It is a fair stand-in, not the template byte for byte.
* Each rung adds one technique on top of the previous (`eval/ladders.py`). Deltas are **paired bootstrap** on per-question
  differences: ▲/▼ = 95% CI excludes 0, ≈ = **no measurable effect**.
* Retrieval-only, keyless runs: ranking metrics are on matched child chunks; `answer_in_context` = answer text present in
  the final context; `ctx tokens` = cost proxy. **LLM-dependent quality (answer correctness, faithfulness, abstention,
  HyDE, decomposition, contextual retrieval) is NOT benchmarked** (no key). Rung 8 (agentic generation) is identical to
  rung 7 in retrieval-only mode and is omitted.
* Gold questions are model-authored, machine-verified, `needs_review`.

## 1. Real documents (CCS corpus, 17 PDFs, 55 questions), *partial*
| rung | recall@5 (Δ vs previous, 95% CI) | mrr (Δ vs previous, 95% CI) | ndcg@10 (Δ vs previous, 95% CI) | answer_in_context (Δ vs previous, 95% CI) | ctx tokens | p50 ms |
|---|---|---|---|---|---:|---:|
| 0-baseline | 0.510 | 0.396 | 0.434 | 0.837 | 4430 | 167 |
| 1+structure-chunks | 0.786 ▲ +0.276 [+0.143,+0.408] | 0.674 ▲ +0.278 [+0.176,+0.384] | 0.718 ▲ +0.284 [+0.187,+0.389] | 0.857 ≈ +0.020 [-0.082,+0.122] | 2299 | 483 |
| 2+parent-child | 0.786 ≈ +0.000 [-0.061,+0.061] | 0.650 ≈ -0.024 [-0.065,+0.009] | 0.701 ≈ -0.018 [-0.053,+0.014] | 0.980 ▲ +0.122 [+0.041,+0.224] | 7701 | 444 |

▲ better / ▼ worse: paired-bootstrap 95% CI of the per-question difference excludes 0. ≈: CI includes 0, i.e. **no measurable effect** at this n.


Findings (rungs 0-2):
* **Structure-aware chunking is the big win on real documents**: recall@5 0.51 → 0.79, MRR 0.40 → 0.67 (CI excludes 0).
  Baseline word windows flatten tables/headings and attribute pages by first word.
* **Parent-child changes ranking not at all, but triples context** (2.3k → 7.7k tokens) while raising answer-in-context
  0.86 → 0.98 (▲). It buys answer coverage at ~3× token cost; worth it only where chunks need surrounding text.
* Indexing cost: 1.3k chunks (379 s) → 5.0k chunks (974 s) → 5.2k (1295 s) on contended CPU: smaller chunks cost index time.
* Pending: BM25 swap, cross-encoder, blend, MMR, query understanding, leave-one-out.

## 2. Constructed corpus (synthetic, 58 PDFs, 94 questions), complete
| rung | recall@5 (Δ vs previous, 95% CI) | mrr (Δ vs previous, 95% CI) | ndcg@10 (Δ vs previous, 95% CI) | answer_in_context (Δ vs previous, 95% CI) | ctx tokens | p50 ms |
|---|---|---|---|---|---:|---:|
| 0-baseline | 0.916 | 0.813 | 0.840 | 0.904 | 1729 | 14 |
| 1+structure-chunks | 0.952 ≈ +0.036 [+0.000,+0.084] | 0.894 ▲ +0.080 [+0.026,+0.136] | 0.906 ▲ +0.066 [+0.023,+0.110] | 0.940 ≈ +0.036 [+0.000,+0.084] | 1405 | 14 |
| 2+parent-child | 0.952 ≈ +0.000 [+0.000,+0.000] | 0.894 ≈ +0.000 [-0.026,+0.028] | 0.908 ≈ +0.003 [-0.017,+0.024] | 0.940 ≈ +0.000 [+0.000,+0.000] | 3405 | 17 |
| 3+bm25-sparse | 0.952 ≈ +0.000 [+0.000,+0.000] | 0.874 ≈ -0.020 [-0.058,+0.016] | 0.887 ≈ -0.021 [-0.050,+0.007] | 0.940 ≈ +0.000 [+0.000,+0.000] | 3171 | 22 |
| 4+cross-encoder | 0.940 ≈ -0.012 [-0.042,+0.018] | 0.890 ≈ +0.016 [-0.037,+0.071] | 0.888 ≈ +0.001 [-0.046,+0.048] | 0.904 ≈ -0.036 [-0.096,+0.012] | 2145 | 1459 |
| 5+rerank-blend | 0.940 ≈ +0.000 [+0.000,+0.000] | 0.893 ≈ +0.003 [-0.015,+0.020] | 0.890 ≈ +0.001 [-0.012,+0.015] | 0.904 ≈ +0.000 [+0.000,+0.000] | 2124 | 1470 |
| 6+mmr | 0.940 ≈ +0.000 [+0.000,+0.000] | 0.890 ≈ -0.003 [-0.020,+0.015] | 0.888 ≈ -0.001 [-0.015,+0.012] | 0.904 ≈ +0.000 [+0.000,+0.000] | 2074 | 1972 |
| 7+query-understanding | 0.976 ≈ +0.036 [+0.000,+0.084] | 0.920 ≈ +0.030 [+0.000,+0.072] | 0.920 ≈ +0.032 [+0.000,+0.072] | 0.916 ≈ +0.012 [+0.000,+0.036] | 2076 | 2427 |

▲ better / ▼ worse: paired-bootstrap 95% CI of the per-question difference excludes 0. ≈: CI includes 0, i.e. **no measurable effect** at this n.


Findings:
* Only **structure-aware chunking** is measurably better (MRR ▲ +0.08, nDCG ▲ +0.07). Query understanding (+0.036 recall@5,
  CI lower bound 0) fixed pronoun follow-ups via a keyless identifier carry-over, borderline at this n.
* **Cross-encoder, blend, MMR, BM25 vs BM42: no measurable effect**, and the cross-encoder adds ~1.4 s/query (14 ms → 1459 ms).
  This corpus is lexically easy (explicit ids), which flatters hybrid search; do not generalise to semantic-gap corpora.
* Parent-child: no ranking effect, 2.4× context tokens.

## 3. Bugs the evaluation found in my own pipeline (fixed, regression-tested)
| symptom | cause | fix |
|---|---|---|
| `conflicting` recall 1.0 → 0.0 with MMR | MMR re-ordered by raw cosine, discarding the reranker's order (Rev B fell from rank 1 to 6) | MMR now uses reranker scores |
| `multi_hop` 1.0 → 0.5 with rerank | cross-encoder scored the operator-directory chunk low (it never names the well) | `rerank.blend` option; real fix needs LLM decomposition (untested) |
| nDCG@10 = 1.11 | several chunks of one gold doc each counted relevant | each gold ref satisfied once |
| numeric tables dropped; 83/111 pages of a falloff report excluded | quality gate counted only alphabetic words | numbers/ids count as content; raw-page cache so re-gating needs no re-OCR |
| cache served MB-140 answer for MB-104; cross-user leak | cosine-only match; user id not in key | fact-signature guard + user-scoped key |

## 4. Pending (will be filled from `reports/`)
Leave-one-out on CCS; chunk size / overlap / strategy / embedding-model sweeps (on `datasets/ccs-mini`); rerank
latency mini-sweep; Qdrant HNSW/quantization/payload-index benchmark. `configs/advanced.yaml` will be set from these
results; until then it is a hypothesis, not a recommendation.

## 5. Caveats
Small n (CIs reported); one machine, CPU contention during runs inflates latency and index times; small ONNX models only;
retrieval-only metrics; model-authored gold needing human review.
