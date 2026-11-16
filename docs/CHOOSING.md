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

## 2-4. Measured results
(filled in from `reports/` after the experiments; see below)
