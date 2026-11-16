# Technique catalogue

Every technique is a switch in a `Profile` (`configs/*.yaml`), has a code location, a cost, and an **ablation row** in
`docs/COMPARISON.md` so you can see whether it earned its place on *your* data. "Needs LLM" techniques degrade to a no-op
(never to an error) without a key.

## Ingestion
| technique | what / why | code | knob | cost | needs LLM |
|---|---|---|---|---|---|
| Layout-aware parsing | markdown with headings and real tables (pymupdf4llm) instead of flattened text | `ingestion/parsers.py` | n/a | CPU | no |
| OCR with quality gate | OCR text-less pages; **exclude** garbage pages instead of indexing OCR noise; numbers/ids count as valid text | `parsers.py:text_quality` | `min quality` | CPU (Tesseract) | no |
| Running header/footer removal | recurring page furniture pollutes BM25/embeddings | `parsers.py:strip_running_lines` | `frac` | ~0 | no |
| Parse cache | OCR once per content hash; re-chunk/re-embed for free | `parsers.py:parse_cached` | cache dir | disk | no |
| Structure-aware chunking | respect headings/paragraphs/tables; page ranges; heading path | `chunkers.py:chunk_structure` | `size`, `overlap`, `keep_tables_whole` | ~0 | no |
| Table handling | keep tables whole, or split by rows **repeating the header** | `chunkers.py:_split_table` | `keep_tables_whole` | ~0 | no |
| Parent-child | index small chunks, return the larger parent for context | `chunkers.py`, `retrieval/pipeline.py` | `parent_child`, `parent_size`, `parent_expand` | **~2× context tokens** | no |
| Semantic chunking | split where adjacent-sentence similarity drops | `chunkers.py:chunk_semantic` | `strategy: semantic` | embeds every sentence | no |
| Title/section prefix | embed `Document: … Section: …` with each chunk so identical tables are distinguishable | `models.py:Chunk.embed_text` | `add_doc_title` | ~0 | no |
| Contextual retrieval | LLM writes a situating sentence per chunk before embedding | `enrich.py:contextualize` | `contextual: true` | 1 small-LLM call / chunk (one-off) | **yes** |
| Metadata extraction | `doc_year`, `doc_type` for filters/routing | `enrich.py:extract_doc_meta` | n/a | ~0 | no |
| Idempotency + versioning | content-hash ids, replace changed docs, skip unchanged | `pipeline.py` | `force` | saves re-embedding | no |

## Retrieval
| technique | what / why | code | knob | cost | needs LLM |
|---|---|---|---|---|---|
| Hybrid dense + sparse | semantic + exact-term match, fused server-side (RRF/DBSF) | `store/qdrant_store.py:search` | `mode`, `fusion`, `sparse_model` | 2 queries, 1 round trip | no |
| Multi-query RRF | fuse results of rewrite/sub-queries/HyDE | `retrieval/fusion.py:rrf` | n/a | N searches | no |
| History-aware rewrite | make follow-ups standalone (LLM; keyless identifier carry-over fallback) | `retrieval/query.py` | `rewrite_with_history` | 1 small call | optional |
| Routing | greeting → no retrieval; compound → decompose | `query.py:route` | `router` | 1 small call | optional |
| Decomposition | split compound questions into sub-queries | `query.py:decompose` | `decompose` | 1 small call | yes |
| HyDE | embed a hypothetical answer for vocabulary-gap queries | `query.py:hyde` | `hyde` | 1 small call | yes |
| Cross-encoder rerank | precise relevance on the shortlist | `retrieval/rerank.py` | `rerank.model`, `candidates`, `top_n` | **CPU latency, linear in candidates** | no |
| Rerank blend | fuse reranker and first-stage order so the reranker cannot veto evidence | `pipeline.py` | `rerank.blend` | ~0 | no |
| MMR | drop near-duplicates, *on the reranker's scores* | `fusion.py:mmr` | `mmr_lambda` | 1 embed pass | no |
| ACL filter in the query | tenant/group visibility enforced in Qdrant, never post-hoc | `store/qdrant_store.py:acl_filter` | principal | ~0 | no |
| Late interaction (ColBERT) | multi-vector rerank field (index-side support) | `qdrant_store.py` (`late_model`) | `embedding.late_model` | 10-50× vector size | no |

## Generation and agentic
| technique | what / why | code | knob | cost | needs LLM |
|---|---|---|---|---|---|
| Grounded prompt + abstention | answer only from numbered sources; refuse otherwise | `generation/prompts.py` | `abstain_below` | ~0 | yes |
| Token-budgeted context | rank-ordered sources, dedup, `max_context_tokens` | `generation/context.py` | `max_context_tokens` | saves cost | no |
| Citation + grounding check | valid citation numbers; lexical support per sentence (+ LLM claim check) | `generation/verify.py` | `verify_grounding` | ~0 (+1 call) | optional |
| Corrective retrieval | grade sufficiency; retry with a better query (LlamaIndex Workflow) | `agentic/corrective.py` | `corrective`, `max_corrective_rounds` | +1 call / round | yes |
| Prompt-injection fencing | untrusted text in delimiters + instruction hierarchy | `prompts.py` | n/a | 0 | n/a |
| Semantic cache (ACL-safe) | reuse answers for equivalent questions | `cache/semantic.py` | `semantic_cache` | memory | no |

## Where a technique does **not** pay (measured, see COMPARISON)
Parent-child adds context cost without retrieval gain on short documents; a cross-encoder can *hurt* multi-hop unless
blended; MMR hurts versioned corpora unless it respects the reranker. Treat each as a hypothesis the ablation must confirm.
