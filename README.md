# Orca RAG v2

A production-shaped reference for RAG over unstructured documents, **rebuilt from the course template with every
change measured**. The original template is preserved at `git tag baseline`; `git diff baseline..HEAD` is the
difference, and `docs/COMPARISON.md` says which of those differences actually improved retrieval, by how much, and at what cost.

* **Many file types**: PDF, DOCX, PPTX (slides, notes, charts), XLSX, CSV, HTML, e-mail, images (OCR): one registry, one verified fixture per format (`docs/FORMATS.md`).
* **Modular**: parsers, vector store, reranker and embedders are swappable behind small protocols with contract tests (`docs/TECHNIQUES.md`, "Extension points").
* **Advanced ingestion**: layout-aware parsing, OCR with a quality gate, structure-aware/parent-child chunking, idempotent
  versioned indexing.
* **Advanced retrieval**: hybrid dense + sparse, query understanding, cross-encoder reranking with blend, MMR, parent expansion,
  server-side ACL filtering.
* **Agentic generation**: corrective retrieval (LlamaIndex Workflows), grounded answers with citations, abstention,
  grounding verification, injection fencing.
* **Production**: FastAPI + SSE, auth/tenancy, semantic cache, Celery worker, healthchecked compose, metrics, CI.
* **Evaluation**: gold-set schema, retrieval/answer metrics with confidence intervals, ablation ladders, sweeps for
  embedding model / chunk size / overlap / strategy, a Qdrant parameter benchmark, and datasets with ground truth.

## Quick start (no API key, no GPU, no torch)
```bash
make install && make data          # venv, real corpus (sha256-verified), synthetic corpus + gold
make test                          # 130+ tests, fully offline
make parse                         # OCR/layout once (cached)
make eval D=datasets/synthetic     # baseline vs advanced vs full
make ladder D=datasets/ccs         # one technique at a time
ORCA_LLM__MOCK=true make serve     # API on :8080 (anonymous dev mode, offline extractive answers)
```
With a provider key (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, ... via LiteLLM) set `ORCA_LLM__MOCK=false`; model ids live in
`configs/*.yaml`. UI: `web/orca` (`npm ci && npm run dev`), set `VITE_API_URL`.

## Read next
`docs/REVIEW.md` (what was wrong with the template, with file:line) → `docs/ARCHITECTURE.md` → `docs/TECHNIQUES.md` →
`docs/FORMATS.md` → `docs/COMPARISON.md` (the evidence) → `docs/CHOOSING.md` (vector DB / embedding / chunk size) → `docs/DOMAIN_EVAL.md`
(build your own eval) → `docs/PRODUCTION.md` → `docs/DATASETS.md`.

## Honest limits
* No LLM key was available while building this. Retrieval and pipeline numbers are real; **LLM-dependent quality**
  (answer correctness, faithfulness, abstention behaviour, contextual retrieval, HyDE, decomposition) is implemented and
  unit-tested with a mock, but **not benchmarked**. Run `orca eval --generate --real-llm` to fill those in.
* Questions are model-authored and machine-verified against page text, but still need human review (`needs_review`).
* Small corpora make recall metrics saturate; see confidence intervals before reading small deltas.
