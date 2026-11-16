# Production notes

## What is production-grade here, and what is deliberately not
| concern | implemented | still your job |
|---|---|---|
| Health | `/health` (liveness), `/ready` (Qdrant reachable), real container healthchecks | alerting on `/ready` flaps |
| Auth / tenancy | API key → principal; tenant + ACL filter enforced in the vector query; leakage tests | SSO/JWT at your gateway; key rotation; per-tenant quotas in a shared store |
| Streaming | SSE with explicit `error` + `[DONE]`; no mid-stream 500 surprises | proxy buffering (`X-Accel-Buffering: no` is set) |
| Rate limiting | in-process token bucket | multi-replica: enforce at the gateway or back with Redis |
| Cache | ACL-safe semantic cache, invalidated on ingest | multi-replica: back with Redis/Qdrant |
| Ingestion | idempotent, versioned, Celery with transient-only retries, `acks_late` | autoscaling workers, dead-letter handling, scheduled re-sync from connectors |
| Observability | request ids, Prometheus metrics (`/metrics`), per-request retrieval trace | traces to your APM (OpenTelemetry), dashboards, SLOs |
| Secrets | `SecretStr`, env only, nothing logged | a secret manager in front of the env |
| Data governance | no document text in logs; page images sent to an LLM only on opt-in | retention, PII redaction, DPA with your LLM provider |

## Sizing rules of thumb (measured numbers in `docs/CHOOSING.md`)
* Vector RAM ≈ `points × dim × 4 B` (float32) before quantization; ≈ ¼ with int8 scalar quantization (+ originals on disk).
* Cross-encoder reranking is the largest CPU latency item (linear in `rerank.candidates`); cap candidates, or serve the
  reranker on a GPU/remote endpoint.
* Parent-child indexing roughly doubles *context tokens* per answer for no retrieval gain on short documents; enable it
  where chunks lose meaning without their neighbourhood (long reports), not everywhere.

## Deploy
`cp deployment/.env.example deployment/.env`, set secrets, `make up`. Qdrant is pinned to the client's minor version;
upgrade both together. Run a snapshot/backup of the `qdrant_storage` volume on a schedule.

## Rollback and recipe changes
A recipe change (chunking, embedding model) creates a **new collection** (`orca_<recipe hash>`) next to the old one;
re-ingest, evaluate, then flip `ORCA_PROFILE`. Roll back by flipping it back. Nothing is overwritten in place.
