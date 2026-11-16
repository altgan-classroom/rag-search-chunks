# Building a domain-specific RAG evaluation (and choosing with it)

Generic benchmarks do not tell you whether *your* documents, *your* questions and *your* failure costs are served.
This is the procedure used for `datasets/ccs` (real corpus) and `datasets/synthetic` (constructed corpus), as a recipe.

## 1. Sample the questions your users actually ask
Collect 50–100 real questions (support tickets, search logs, interviews with SMEs). If you have none, write them from
documents *and then rewrite them in the user's vocabulary*: questions that copy document wording flatter lexical search
and hide the vocabulary gap (`paraphrase` category).

## 2. Cover failure modes, not just topics
| category | tests | why it matters |
|---|---|---|
| factoid / numeric / table | exact values, units, cell lookup | tables and numbers are where chunking and parsing break |
| keyword (identifiers) | `INC-2019-0447`, well names | dense embeddings blur identifiers; BM25 shines |
| paraphrase | doc says "pore space", user says "porosity" | where lexical-only retrieval fails |
| multi_hop / comparison | answer needs two documents | needs query decomposition or iterative retrieval |
| temporal / conflicting | revisions, 2012 vs 2013 | near-duplicate documents; latest must win |
| follow_up | pronouns + history | needs history-aware rewriting |
| unanswerable | answer is not in the corpus | measures abstention, the main hallucination guard |
| acl / tenant | restricted docs, same ids in another tenant | must be **zero leakage**, not "mostly" |
| adversarial | injection text inside a document | generation-side safety |

Aim for ≥ 8 questions per category you care about; below that, per-category numbers are anecdotes (see §5).

## 3. Label what retrieval must find, not just the answer
Each question stores `gold: [{doc, pages}]` and `must_contain` (AND of OR-groups, matched on normalised text). This
supports page-level recall@k, MRR, nDCG **and** `answer_in_context` (is the answer text anywhere in the context the model
sees?), which is independent of how a recipe chunks. Gold labels must be **machine-verified** against the parsed page
text (`datasets/ccs/build_gold.py` fails the build if any expected string is absent on its gold page).

## 4. Review protocol (humans stay in the loop)
Model-authored questions are `status: needs_review`. A reviewer checks: (a) the question is natural, (b) gold pages are
sufficient *and complete* (other pages may also answer it, so add them), (c) `must_contain` accepts every correct phrasing
(units, thousand separators, synonyms), (d) unanswerables are truly absent (search the whole corpus). Promote to
`verified`. Keep a held-out slice you never tune on.

## 5. How big, and how sure
Bootstrap 95 % intervals are reported (`--ci`). With n = 20 a recall of 0.80 has a CI of roughly ±0.18, so differences under
~0.1 between recipes are noise at that size. Use n ≥ 100 overall and paired comparisons (same questions) before
trusting a small delta. Report per-category numbers as diagnostics, not as headline claims.

## 6. Keyless vs LLM-dependent metrics
Runs without a key give retrieval metrics, `answer_in_context`, ACL-leak, latency, abstention separability (AUROC of the
reranker's best score). Anything that depends on a real generator (answer correctness, faithfulness, abstention
behaviour, injection resistance) needs `--generate --real-llm`; the offline mock produces **extractive** answers and is
only for exercising the pipeline. Reports are tagged `llm_mode: none|mock|real`; never compare across modes.

## 7. Using it to choose
* Change **one factor at a time** (`orca sweep`), same questions, same retrieval settings.
* Read quality **and cost**: index time, vector bytes, p50/p95 latency, context tokens (cost per answer).
* Prefer the smallest recipe within the confidence interval of the best one.
* Re-run on every change to chunking, embeddings, prompts or models (CI can gate on recall@5 and ACL-leak = 0).
