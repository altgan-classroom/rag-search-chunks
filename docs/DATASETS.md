# Datasets

| dataset | what | size | license / provenance | how to get it |
|---|---|---|---|---|
| `datasets/ccs` | **Real** CO₂-storage documents (Illinois Basin – Decatur Project): well sheets, falloff-test reports (2012 vs 2013), risk reports, fact sheets, a licence, a 72-page scanned completion report (OCR path) | 17 PDFs, 35 MB | Apache-2.0 Hugging Face dataset `Altgan/rag-for-unstructured-data`, pinned revision, sha256 per file (`sources.json`) | `python datasets/ccs/download.py` |
| `datasets/ccs/gold.json` | 55 questions authored from the documents, every expected string machine-verified on its gold page | 55 q | model-authored, `needs_review` | `python datasets/ccs/build_gold.py` |
| `datasets/synthetic` | **Constructed** "Meridian Basin" corpus: confusable ids, 20-page repetitive reports, Rev A/B, tables, glossary, ACL docs, a second tenant, a prompt-injection document | 58 PDFs, 94 q | generated, deterministic for `--seed` | `python -m orca_rag.datasets.synthetic` |
| `datasets/open` | Open papers (arXiv) for domain-shift checks: different vocabulary, equations, figures | see `datasets/open/README.md` | author licences retained; **downloaded, not redistributed** | `python datasets/open/download.py` |

Why both real and synthetic: real documents have real mess (scans, odd tables, internal inconsistencies such as the 2013
report quoting two transmissibility values) but expensive, uncertain labels. Synthetic documents have exact labels and
let us *construct* each hard case, but are cleaner than reality. Conclusions that hold on both are the ones to trust.

Known caveats: the CCS corpus is CO₂ storage rather than hydrocarbon production (the HF dataset is the course corpus);
`unanswerable` questions were checked against the 17-document subset only; OCR of the scanned report is noisy and about
half its pages are excluded by the quality gate on purpose.
