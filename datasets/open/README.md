# Open documents (domain-shift check)

Three arXiv papers on retrieval-augmented generation: RAG (Lewis et al., 2005.11401), Self-RAG (2310.11511), CRAG (2401.15884).
They differ from the CCS corpus in vocabulary, equations, two-column layout and figure/table density, so they show whether a
recipe tuned on one domain generalises.

* PDFs are **downloaded, not committed** (each paper keeps its own licence): `python datasets/open/download.py` (sha256 recorded in `sources.json`).
* `python datasets/open/build_gold.py` builds 20 questions (incl. 3 unanswerable); every expected string is verified on its gold page.
* Status: model-authored, `needs_review`.
