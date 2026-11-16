# Architecture

```
                         ┌──────────────────────── ingestion (idempotent, versioned) ────────────────────────┐
 files ──► parse (pymupdf4llm layout + OCR) ──► clean/gate pages ──► chunk (structure / parent-child) ──► enrich ──► embed ──► Qdrant
            │ parse cache (raw pages)           │ header/footer strip   │ tables whole, page ranges        │ doc year/type,   dense + sparse
            │ keyed by content sha              │ quality gate          │ heading path                     │ contextual (opt) (+ late, opt)
                         └───────────────────────────────────────────────────────────────────────────────────┘

 question ─► auth ─► Principal(tenant, groups) ─► semantic cache? ─► Answerer
                                                                       │
                                                     ┌─ Corrective workflow (LlamaIndex Workflows) ─────────────┐
                                                     │ retrieve ─► grade ──sufficient──► done                    │
                                                     │    ▲           └─insufficient─► better query ─┐           │
                                                     │    └──────────────────────────────────────────┘           │
                                                     └───────────────────────────────────────────────────────────┘
        Retriever:  rewrite(history) → route → decompose/HyDE → hybrid search (ACL filter inside Qdrant)
                    → RRF across queries → cross-encoder rerank (+blend) → MMR (on reranker scores) → parent expansion
        Generator:  numbered sources in a token budget → grounded prompt → stream → grounding check → related questions
        Events:     context ► tokens… ► answer ► grounding ► related… ► trace ► done          (SSE, [DONE])
```

## Packages (`src/orca_rag`)

| package | responsibility | notes |
|---|---|---|
| `config` | `Env` (deployment, secrets as `SecretStr`) and `Profile` (the RAG *recipe*, YAML) | recipe ≠ deployment: the same code runs `baseline` and `advanced` |
| `llm` | LiteLLM gateway, offline mock, usage accounting | model ids only come from config; no provider in code |
| `ingestion` | parsers, chunkers, enrichment, idempotent pipeline, Celery tasks | content-addressed ids, versioning, parse cache |
| `store` | Qdrant store + embedders (fastembed ONNX) | explicit collection, named vectors, payload indexes, ACL filter |
| `retrieval` | query understanding, fusion, rerank, MMR, pipeline | every step traced with timings |
| `generation` | prompts, context builder, grounding verifier, answerer | abstains instead of improvising |
| `agentic` | corrective retrieval as a LlamaIndex Workflow | bounded rounds, RRF-merged |
| `cache` | ACL-safe semantic cache | key includes tenant + groups + data version |
| `api` | FastAPI app, auth, rate limit, history, metrics | lifespan singletons, SSE |
| `eval` | schema, metrics, harness, ablation ladders, sweeps, vector bench | the measurement backbone |
| `datasets` | synthetic corpus with ground truth by construction | deterministic for a seed |

## Where LlamaIndex is used

LlamaIndex stays the backbone for **Workflows** (the corrective loop, `agentic/corrective.py`) and for its `litellm` LLM
integration if you want LlamaIndex components. The hot path (hybrid query, fusion, rerank, parent fetch) talks to
`qdrant-client`'s Query API directly: it exposes prefetch + RRF/DBSF fusion, named dense/sparse/multi-vectors and
server-side ACL filters, which the baseline's `CitationQueryEngine.from_args(...)` path silently dropped
(REVIEW R3). Wrapping those calls as LlamaIndex retrievers is straightforward (`BaseRetriever` returning `NodeWithScore`
from `Hit`); it was not done because it adds a conversion layer without changing behaviour.

## Data model (Qdrant payload)

`kind` ∈ `child` (searchable, vectors) | `parent` (context only, no vectors) | `doc` (manifest: sha256, recipe signature,
version, parser version). Point ids are `uuid5(doc_id, kind:ordinal)` ⇒ retries and re-ingests are upserts. Filterable,
indexed fields: `tenant, kind, acl, doc_id, source_key, doc_type, doc_year, page_start, is_table`.

## Security model

* API key → `Principal(tenant, groups)` server-side. **The request body can never widen visibility** (`metadata` is ignored
  for filtering). Every search carries `tenant == p.tenant AND acl ∈ {public, groups…, user:id}`.
* Cache entries are scoped to the same (tenant, groups) signature; ingest invalidates the cache.
* Admin ingest resolves paths under `<data_dir>/raw` and rejects traversal. CORS is an allow-list, never `*` with credentials.
* Prompts fence untrusted text (`<sources>`, `<question>`) and the system prompt forbids following instructions inside it;
  the synthetic set contains a document with an injection string to test it.
