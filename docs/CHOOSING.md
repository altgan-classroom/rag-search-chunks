# Choosing a vector database, embedding model and chunking

Principle: **decide with your own eval set, one factor at a time, reading quality and cost together.** Sections 2-4 give
measured results from this repo's datasets; treat them as a worked example of the method, and re-run `orca sweep` on
your corpus (`docs/DOMAIN_EVAL.md`). Numbers here come from a CPU laptop with small ONNX models; absolute latencies will
differ on your hardware, rankings between options are the transferable part.

## 1. Vector database: decide by constraints, then benchmark the shortlist

Start from what you cannot compromise on. Characteristics below are well-known properties as of this writing; verify
against current documentation before committing.

| if you need… | favours | why / caveat |
|---|---|---|
| **metadata/ACL filtering inside the ANN query** (multi-tenant, per-document permissions) | Qdrant, Elasticsearch/OpenSearch, Weaviate, Milvus | filter-aware HNSW with payload indexes; post-filtering loses recall on selective filters (measured in §2.3) |
| **hybrid lexical + semantic in one call** | Qdrant (sparse + dense + RRF/DBSF), Elasticsearch/OpenSearch (BM25 native), Weaviate | otherwise you run two systems and fuse yourself |
| **already running Postgres, ≲ 10 M vectors, transactional consistency with app data** | pgvector | one datastore, SQL joins, RLS for tenancy; plan for index build/maintenance and lower filtered-ANN headroom at scale |
| **billions of vectors / horizontal scale-out** | Milvus, managed services (Pinecone, Qdrant Cloud, Zilliz) | operational weight or vendor lock-in is the trade |
| **no ops, embedded, prototyping** | Chroma, LanceDB, Qdrant local mode, FAISS (library) | FAISS has no filtering/persistence/ACL: a component, not a database |
| **strict on-prem / air-gapped** | Qdrant, Milvus, Weaviate, pgvector, OpenSearch | all self-hostable; managed-only products are out |
| **memory-bound** | any with quantization (scalar/binary/product) + on-disk vectors | see §2.2 for recall/latency/RAM trade-offs |

**This repo uses Qdrant** because the template did, and because the three requirements that dominate production RAG here,
filter-aware ANN for ACLs, native sparse+dense hybrid, and quantization, are first-class. The store is a small class
(`store/qdrant_store.py`); swapping databases means re-implementing `search/upsert/delete/get_manifest`, not rewriting the
pipeline. Nothing was benchmarked against other databases: only Qdrant was installed, so the table above is criteria,
not a head-to-head.

Questions to answer before benchmarking: expected vector count in 12 months, filter selectivity (how many tenants/ACL
groups), update rate vs read rate, p95 latency budget, RAM budget, who operates it, and data-residency rules.

## 2. Chunking strategy and size (biggest lever)
Measured on `datasets/ccs-mini` (details and caveats: `docs/COMPARISON.md` §4) and confirmed on the full real corpus (§1).

| decision | result | choose |
|---|---|---|
| strategy | fixed 2048-char windows recall@5 0.66; semantic 0.90 (62 s to index); **structure-aware 0.95** (~22 s) | structure-aware; keep tables whole |
| size (relative token budget) | 128-512 within 0.04 of each other; 1024 clearly worse (0.90, MRR 0.76) | 256-512; smaller = more vectors, slower indexing |
| overlap | 0% / 15% / 30% = 0.975 / 0.950 / 0.938, no usable signal | start at ~15% and spend your effort elsewhere |
| parent-child | recall unchanged; answer-in-context 0.86 → 0.98 on the real corpus, 3.4x context tokens | use when answers need surrounding text; drop to cut cost |

## 3. Embedding model
Same sweep: bge-small 0.95, bge-base 0.95, snowflake-arctic-embed-s 1.00, nomic-embed-text-v1.5 0.95, potion-retrieval-32M 0.925,
all-MiniLM-L6-v2 0.89 (recall@5, 40 answerable questions: one question is 0.025). Larger/768-d models were not better than
384-d ones here and cost more latency and memory. **Conclusion: differences among the good models are inside the noise of this
set; run `orca sweep --experiment models` on your own gold questions.** Check licence, language coverage, max input length,
and whether the model expects a query/document prefix (`embedders.py` has a prefix registry).

**Newer models, real corpus** (`docs/COMPARISON.md` §7, 55 questions, paired CIs vs bge-small): **snowflake-arctic-embed-s** ranked the right passage
higher (MRR +0.09, recall@1 +0.13, both intervals exclude 0) at the same 384-d size, recall@5 unchanged: shipped as `configs/quality.yaml`.
granite-embedding-small-english-r2 showed no measurable difference. Learned sparse **minicoil-v1 was worse than BM25**. The 300M-600M models
(mxbai-embed-large, embeddinggemma-300m, Qwen3-Embedding-0.6B) were not run to completion on a CPU laptop (hours each): spec in
`configs/experiments/models-2026.yaml`. Licences (from the fastembed model list): arctic, granite, mxbai, Qwen3: Apache-2.0; embeddinggemma: Gemma terms;
**jina-embeddings-v3: CC-BY-NC (non-commercial), excluded**. Hosted embeddings (`dense_model: litellm:openai/text-embedding-3-small`, Voyage, Cohere,
Gemini ...) are supported but unbenchmarked here; weigh per-token cost, data leaving your network and forced re-embedding when a provider retires a model.

## 4. Search mode and Qdrant parameters
* Mode (recall@5): dense 0.84, sparse BM25 0.95, hybrid 0.95 (best MRR). Hybrid is the safe default when ids and codes matter.
* HNSW: `m=16, ef=64..128` reached recall@10 1.00 at ~4 ms on 100k vectors; bigger `m` only cost build time.
* Scalar quantization kept recall (0.996) and was faster; binary quantization collapsed at 384-d (0.27), so measure before use.
* Index every field you filter on: a 1%-selectivity tenant filter was ~6x slower without a payload index (3.6 vs 21.8 ms).
* Vector-benchmark data are synthetic Gaussians; re-run `orca_rag.eval.vector_bench` on your embeddings.
