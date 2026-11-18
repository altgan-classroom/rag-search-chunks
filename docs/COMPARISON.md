# Comparison: template recipe vs v2, one technique at a time

**Status: complete for retrieval-only evidence.** Synthetic ladder, real-corpus (CCS) ladder, leave-one-out and the
ccs-mini sweeps are done (sections 1-4); the Qdrant parameter benchmark (synthetic vectors) is in section 5. LLM-dependent quality is not benchmarked
(see *How to read this*). Nothing below is extrapolated.

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

## 1. Real documents (CCS corpus, 17 PDFs, 55 questions)
| rung | recall@5 (Δ vs previous, 95% CI) | mrr (Δ vs previous, 95% CI) | ndcg@10 (Δ vs previous, 95% CI) | answer_in_context (Δ vs previous, 95% CI) | ctx tokens | p50 ms |
|---|---|---|---|---|---:|---:|
| 0-baseline | 0.510 | 0.396 | 0.434 | 0.837 | 4430 | 167 |
| 1+structure-chunks | 0.786 ▲ +0.276 [+0.143,+0.408] | 0.674 ▲ +0.278 [+0.176,+0.384] | 0.718 ▲ +0.284 [+0.187,+0.389] | 0.857 ≈ +0.020 [-0.082,+0.122] | 2299 | 483 |
| 2+parent-child | 0.786 ≈ +0.000 [-0.061,+0.061] | 0.650 ≈ -0.024 [-0.065,+0.009] | 0.701 ≈ -0.018 [-0.053,+0.014] | 0.980 ▲ +0.122 [+0.041,+0.224] | 7701 | 444 |
| 3+bm25-sparse | 0.878 ≈ +0.092 [+0.000,+0.194] | 0.713 ≈ +0.063 [-0.004,+0.130] | 0.751 ≈ +0.050 [-0.017,+0.117] | 0.959 ≈ -0.020 [-0.061,+0.000] | 7332 | 446 |
| 4+cross-encoder | 0.857 ≈ -0.020 [-0.122,+0.082] | 0.765 ≈ +0.052 [-0.068,+0.169] | 0.783 ≈ +0.032 [-0.076,+0.138] | 0.939 ≈ -0.020 [-0.082,+0.041] | 6052 | 2298 |
| 5+rerank-blend | 0.878 ≈ +0.020 [+0.000,+0.061] | 0.750 ≈ -0.015 [-0.048,+0.007] | 0.775 ≈ -0.008 [-0.029,+0.007] | 0.939 ≈ +0.000 [+0.000,+0.000] | 6034 | 5101 |
| 6+mmr | 0.857 ≈ -0.020 [-0.061,+0.000] | 0.764 ≈ +0.014 [-0.009,+0.048] | 0.781 ≈ +0.006 [-0.009,+0.028] | 0.939 ≈ +0.000 [+0.000,+0.000] | 6030 | 3265 |
| 7+query-understanding | 0.857 ≈ +0.000 [+0.000,+0.000] | 0.764 ≈ +0.000 [-0.031,+0.031] | 0.781 ≈ +0.000 [-0.023,+0.023] | 0.939 ≈ +0.000 [+0.000,+0.000] | 6040 | 3228 |

▲ better / ▼ worse: paired-bootstrap 95% CI of the per-question difference excludes 0. ≈: CI includes 0, i.e. **no measurable effect** at this n.

Findings:
* **Structure-aware chunking is the one clear win on real documents**: recall@5 0.51 → 0.79, MRR 0.40 → 0.67, both with intervals
  that exclude 0. The baseline's word windows flatten tables and headings.
* **Parent-child leaves ranking unchanged but raises answer-in-context** 0.86 → 0.98 (▲), at 3.4× the context tokens
  (2.3k → 7.7k). It buys answer coverage with tokens.
* **Everything after that is within noise for recall@5**: BM25 (+0.09, CI [0.00, +0.19]), cross-encoder (-0.02), blend, MMR, query
  understanding. The cross-encoder rung costs ~1.9 s more per query on this machine (446 → 2298 ms), and the blend rung 5.1 s
  (contended CPU: compare ratios, not absolutes).
* Per-category detail (`reports/ccs-real-ladder/report.md`): the cross-encoder helped `temporal` and `follow_up` questions
  (n=5 and 3) and hurt `table` (n=5); these cells are far too small to act on.

### 1.1 Leave-one-out from the full recipe (real corpus)
| profile | n | recall@1 | recall@5 | mrr | ndcg@10 | answer_in_context | context_tokens | retrieval_ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 55 | 0.622 | 0.857 | 0.764 | 0.781 | 0.939 | 6039.546 | 3257.350 |
| -structure/parent | 55 | 0.429 | 0.551 | 0.510 | 0.510 | 0.878 | 2960.564 | 3438.583 |
| -parent-child | 55 | 0.643 | 0.857 | 0.779 | 0.793 | 0.878 | 1772.491 | 3181.671 |
| -bm25 | 55 | 0.622 | 0.888 | 0.768 | 0.798 | 0.939 | 6029.673 | 3168.875 |
| -cross-encoder | 55 | 0.612 | 0.898 | 0.757 | 0.788 | 0.980 | 5645.709 | 472.360 |
| -blend | 55 | 0.622 | 0.857 | 0.764 | 0.781 | 0.939 | 6039.546 | 3124.723 |
| -mmr | 55 | 0.612 | 0.878 | 0.760 | 0.783 | 0.939 | 6017.891 | 2317.476 |
| -query-understanding | 55 | 0.622 | 0.857 | 0.764 | 0.781 | 0.939 | 6029.800 | 3156.013 |

* **Removing structure-aware/parent chunking is the only big drop**: recall@5 0.857 → 0.551.
* **Removing the cross-encoder** (this profile also drops MMR, which depends on it) did **not** lower recall (0.857 → 0.898) and cut
  retrieval from ~3.3 s to ~0.5 s. Intervals are not computed for leave-one-out; at n=55 a ±0.04 move is noise.
* Removing parent-child keeps recall and shrinks context 6.0k → 1.8k tokens but lowers answer-in-context 0.939 → 0.878.
* BM25, blend, MMR and query understanding: no clear effect.

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

## 4. Sweeps (ccs-mini: 13 PDFs of ≤30 pages, 46 questions of which 40 answerable, 225 chunks, `reports/sweep-ccs-mini-all.md`)
| experiment | dim | chunks | index s | vec MB | p50 ms | recall@1 | recall@5 | recall@10 | MRR | nDCG@10 | answer-in-ctx |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| model=bge-small-en-v1.5 | 384 | 225 | nan | 0.4 | 23 | 0.775 | 0.950 | 1.000 | 0.884 | 0.907 | 0.975 |
| model=bge-base-en-v1.5 | 768 | 225 | nan | 0.7 | 32 | 0.700 | 0.950 | 1.000 | 0.847 | 0.881 | 0.950 |
| model=snowflake-arctic-embed-s | 384 | 225 | nan | 0.4 | 24 | 0.775 | 1.000 | 1.000 | 0.892 | 0.914 | 0.975 |
| model=all-MiniLM-L6-v2 | 384 | 225 | nan | 0.4 | 21 | 0.650 | 0.887 | 0.950 | 0.769 | 0.802 | 0.900 |
| model=nomic-embed-text-v1.5 | 768 | 225 | nan | 0.7 | 40 | 0.800 | 0.950 | 1.000 | 0.900 | 0.917 | 0.975 |
| model=potion-retrieval-32M | 512 | 225 | 0 | 0.5 | 18 | 0.662 | 0.925 | 1.000 | 0.796 | 0.844 | 0.950 |
| model=bge-small+instruction | 384 | 225 | nan | 0.4 | 25 | 0.775 | 0.975 | 1.000 | 0.887 | 0.908 | 0.950 |
| size=128 | 384 | 637 | 29 | 1.0 | 46 | 0.713 | 0.950 | 0.975 | 0.841 | 0.861 | 0.975 |
| size=256 | 384 | 315 | 24 | 0.5 | 27 | 0.738 | 0.963 | 0.988 | 0.871 | 0.894 | 1.000 |
| size=350 | 384 | 226 | 22 | 0.4 | 23 | 0.725 | 0.950 | 1.000 | 0.858 | 0.887 | 0.975 |
| size=512 | 384 | 153 | 16 | 0.3 | 18 | 0.775 | 0.925 | 0.975 | 0.883 | 0.903 | 0.975 |
| size=1024 | 384 | 76 | 8 | 0.1 | 14 | 0.600 | 0.900 | 0.975 | 0.763 | 0.810 | 0.975 |
| overlap=0% | 384 | 220 | 22 | 0.4 | 23 | 0.725 | 0.975 | 1.000 | 0.856 | 0.888 | 1.000 |
| overlap=15% | 384 | 226 | nan | 0.4 | 23 | 0.725 | 0.950 | 1.000 | 0.858 | 0.887 | 0.975 |
| overlap=30% | 384 | 237 | 24 | 0.4 | 23 | 0.650 | 0.938 | 1.000 | 0.820 | 0.858 | 1.000 |
| strategy=fixed_words(2048ch) | 384 | 131 | 14 | 0.2 | 17 | 0.438 | 0.662 | 0.725 | 0.562 | 0.596 | 0.900 |
| strategy=structure | 384 | 225 | nan | 0.4 | 24 | 0.775 | 0.950 | 1.000 | 0.884 | 0.907 | 0.975 |
| strategy=semantic | 384 | 187 | 62 | 0.3 | 20 | 0.637 | 0.900 | 0.963 | 0.780 | 0.816 | 0.975 |
| mode=dense | 384 | 225 | nan | 0.4 | 9 | 0.625 | 0.838 | 0.975 | 0.770 | 0.812 | 0.875 |
| mode=sparse | 384 | 225 | nan | 0.4 | 14 | 0.675 | 0.950 | 1.000 | 0.825 | 0.860 | 0.975 |
| mode=hybrid | 384 | 225 | nan | 0.4 | 22 | 0.775 | 0.950 | 1.000 | 0.884 | 0.907 | 0.975 |

Single runs, no intervals; with 40 answerable questions a 0.025 difference is one question. Read ranks and gaps, not decimals.
* **Chunking strategy is the largest lever, again**: fixed 2048-char windows 0.66 recall@5 vs structure 0.95 vs semantic 0.90.
  Semantic chunking cost 62 s to index against ~22 s for structure and did not beat it.
* **Chunk size** (relative token budget): 128-512 all land within 0.04 on recall@5; 1024 is clearly worse (0.90, MRR 0.76).
  Smaller chunks mean more vectors and slower indexing (637 chunks at 128 vs 153 at 512).
* **Overlap**: no usable signal between 0% and 30% (0.975 / 0.950 / 0.938).
* **Embedding model**: every model except all-MiniLM-L6-v2 (0.89) is within 0.05 of the others on recall@5; bigger (768-d) was
  not better than bge-small (384-d) here, and costs more latency and memory. Use your own gold set to choose.
* **Retrieval mode**: dense-only 0.84, sparse (BM25) 0.95, hybrid 0.95 (best MRR 0.884). This corpus is lexically easy; do not
  generalise to semantic-gap corpora.

## 5. Qdrant parameter benchmark (100,000 vectors, 384-d, 200 queries, Qdrant v1.20.2 in Docker)
Recall@10 is measured against exact search on the same vectors. **The vectors are synthetic clustered Gaussians, not real embeddings**
(`eval/vector_bench.py`), on a CPU laptop, single run. Treat it as a demonstration of the method and of orders of magnitude;
re-run it on your own vectors (`python -m orca_rag.eval.vector_bench`) before choosing parameters.

| config | recall@10 | p50 ms | p95 ms | note |
|---|---:|---:|---:|---|
| HNSW m=8 ef=32 | 0.990 | 3.7 | 4.9 | build 39s |
| HNSW m=8 ef=64 | 1.000 | 3.4 | 4.3 |  |
| HNSW m=8 ef=128 | 1.000 | 3.6 | 4.3 |  |
| HNSW m=8 ef=256 | 1.000 | 4.8 | 6.4 |  |
| HNSW m=16 ef=32 | 0.994 | 4.2 | 5.3 | build 52s |
| HNSW m=16 ef=64 | 1.000 | 3.8 | 5.1 |  |
| HNSW m=16 ef=128 | 1.000 | 4.2 | 5.4 |  |
| HNSW m=16 ef=256 | 1.000 | 5.3 | 6.4 |  |
| exact (brute force) | 1.000 | 8.6 | 9.7 |  |
| HNSW m=32 ef=32 | 0.998 | 5.1 | 7.1 | build 63s |
| HNSW m=32 ef=64 | 1.000 | 4.3 | 5.9 |  |
| HNSW m=32 ef=128 | 1.000 | 4.3 | 5.2 |  |
| HNSW m=32 ef=256 | 1.000 | 5.5 | 6.6 |  |
| quantization=none ef=128 (rescore) | 1.000 | 5.2 | 8.1 | |
| quantization=scalar ef=128 (rescore) | 0.996 | 3.9 | 5.3 | |
| quantization=binary ef=128 (rescore) | 0.273 | 3.5 | 6.1 | |
| filter tenant=7 (1%), payload_index=True | 1.000 | 3.6 | 5.5 |  |
| filter tenant=7 (1%), payload_index=False | 1.000 | 21.8 | 24.3 |  |

* **HNSW `m` and `ef`**: recall@10 ≥ 0.99 everywhere and 1.00 from ef=64. Latency is flat (3-5 ms) until ef=256. Larger `m` raised
  build time (39 s → 52 s → 63 s for m=8/16/32) without a recall benefit here. Exact search was ~2x slower (8.6 ms) at this size;
  the gap widens with collection size.
* **Scalar quantization** (rescored) kept recall at 0.996 with lower p50 (3.9 vs 5.2 ms). **Binary quantization** collapsed to
  0.27 at 384 dimensions with 2x oversampling: it is generally recommended only for much higher-dimensional embeddings, so check
  your dimensionality and always measure recall before enabling it.
* **Payload index on the filter field**: a 1%-selectivity tenant filter ran at 3.6 ms with an index and 21.8 ms without
  (about 6x), with identical recall. This is why `tenant`, `acl` and `doc_id` are indexed in `store/qdrant_store.py`.
* The same container answered `/readyz` with "all shards are ready" (`data/qd_health.txt`), the probe the compose healthcheck uses.

## 6. File formats (`datasets/formats`, 10 files, 16 questions): does every parser keep its facts addressable?
Not a quality benchmark: ten tiny invented documents, one planted fact per format, gold questions naming document **and page**
(`docs/FORMATS.md`). `orca eval datasets/formats`, profiles `advanced` and `full`, bge-small + BM25, 15 answerable questions:
recall@1 = recall@5 = MRR = nDCG@10 = answer-in-context = **1.000** for both profiles (15 chunks in total). That is the expected
result for a correct pipeline on a tiny corpus, and it is what failed before the page-boundary fix: a 4-slide deck was a single chunk cited
as "pp. 1-4", so "Q3 revenue for APAC" could not cite slide 3. Fixed by treating slides and sheets as hard page boundaries
(`chunkers._pack(hard_pages)`, regression test `test_slides_and_sheets_are_never_merged_into_one_chunk`). The same ten files were also ingested through
the Celery worker and the admin API in the Docker stack, and an SSE answer cited `Q3_Board_Update` page 3. Without Tesseract the PNG fails with
an explicit "OCR is required" message while the other nine ingest.

## 7. Model refresh (real corpus: 17 PDFs, 5,198 chunks, 55 questions; everything else = `configs/advanced.yaml`)
Reference = bge-small-en-v1.5 + BM25 (the default). Each row changes ONE thing. Δ is the per-question paired difference vs the reference with a
bootstrap 95% CI; ▲ = CI excludes 0. `configs/experiments/models-2026.yaml`, `orca variants`, `orca compare --ref`.

| variant | recall@1 | recall@5 (primary) | MRR | nDCG@10 | retrieval ms (mean, CPU) |
|---|---|---|---|---|---:|
| **bge-small-en-v1.5** (reference) | 0.612 | 0.898 | 0.760 | 0.792 | 455 |
| **snowflake-arctic-embed-s** (33M, 384-d) | 0.745 ▲ +0.133 [+0.031,+0.245] | 0.929 ≈ +0.031 [0.000,+0.082] | 0.851 ▲ +0.091 [+0.026,+0.162] | 0.863 ▲ +0.071 [+0.023,+0.126] | 449 |
| granite-embedding-small-english-r2 (47M, 384-d) | 0.694 ≈ +0.082 [-0.041,+0.204] | 0.867 ≈ -0.031 [-0.082,0.000] | 0.808 ≈ +0.048 [-0.016,+0.118] | 0.831 ≈ +0.039 [-0.011,+0.092] | 457 |
| sparse = minicoil-v1 instead of BM25 | 0.541 ▼ -0.071 [-0.143,-0.010] | 0.857 ≈ -0.041 [-0.102,0.000] | 0.710 ▼ -0.050 [-0.098,-0.010] | 0.759 ▼ -0.033 [-0.070,-0.001] | 565 |
| + reranker **bge-reranker-v2-m3-int8** (24 candidates) | 0.755 ▲ +0.143 [+0.010,+0.276] | 0.888 ≈ -0.010 [-0.102,+0.082] | 0.858 ≈ +0.098 [-0.007,+0.203] | 0.874 ≈ +0.082 [-0.002,+0.169] | **8,933** |
| mxbai-embed-large-v1, embeddinggemma-300m, Qwen3-Embedding-0.6B-Q | not completed: see below | | | | |

How to read it:
* **arctic-embed-s ranks the right passage higher** (recall@1, MRR, nDCG@10 intervals exclude 0) at the same dimension and query cost, for a
  ~66 MB larger download. The primary metric recall@5 shows no measurable difference (lower bound 0.000). Shipped as `configs/quality.yaml`;
  `advanced.yaml` stays on bge-small so the lab, the ladder and the deck remain one consistent, reproducible recipe. The same direction
  appeared on ccs-mini (arctic 1.00 vs bge-small 0.95, §4).
* **A stronger reranker does help ranking, which refines the earlier "reranking did not help" result.** The MiniLM cross-encoder in the ladder (§1)
  showed nothing; bge-reranker-v2-m3 raised recall@1 by +0.14 (CI excludes 0) and MRR by +0.10 (CI just includes 0), but recall@5 did not move, and
  it cost ~9 s per query on a CPU laptop (some of that run overlapped other work on the machine). Worth testing where a GPU or hosted reranker is available.
* granite-r2-small and the learned sparse model minicoil did not beat the defaults; minicoil is clearly worse than BM25 here and slower to index.
* **Not completed:** mxbai-embed-large (335M), embeddinggemma-300m and Qwen3-Embedding-0.6B-Q. Indexing 5,198 chunks took ~9 min for a 33M-parameter model on this CPU, so
  300M-600M models need hours; they are in the spec file, in that order, to run on a GPU/overnight (`orca variants datasets/ccs configs/experiments/models-2026.yaml --only mxbai-embed-large-v1`).
  jina-embeddings-v3 was excluded on licence (CC-BY-NC).
* **Caveats:** one corpus, n=55, model-authored questions; five metrics were compared per variant, so an isolated ▲ is weaker evidence than one that repeats across
  metrics and corpora (arctic did both). Latencies are CPU laptop, single run, compare ratios not absolutes.

## 8. Caveats
Small n (CIs reported); one machine, CPU contention during runs inflates latency and index times; small ONNX models only;
retrieval-only metrics; model-authored gold needing human review; **abstention is not benchmarked, and with the reranker off
(`advanced.yaml`) there is no relevance-score gate: an unanswerable question relies on the grounded prompt and the grounding check
(with the offline mock, which is extractive, it simply returns the nearest passages)**; the real corpus is CO₂ storage, not hydrocarbon
production; the Qdrant benchmark uses synthetic vectors; sweeps are single runs without intervals.
