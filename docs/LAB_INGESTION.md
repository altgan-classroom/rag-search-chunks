# Lab: run your first ingestion

Goal: index 58 PDFs on your own laptop and search them, as two different users. **No API key, no Docker, no GPU.**
Verified from clean `git archive` exports of this repo (fresh venv, literal `make install`): install 28 s with a warm pip cache and 234-264 s cold;
first ingest 49 s including a one-off ~64 MB model download; re-ingest 1 s. Disk: about 0.9 GB (venv 0.8 GB, data 65 MB).

## 0. Prerequisites
* Python **3.11 or newer** (`python3 --version`). `make install` uses `python3`; if yours is older, install 3.11 first.
* `make` and `git`. Tesseract is not needed for this corpus (checked: all 58 files ingest with it off the PATH). Internet on the first run only (to fetch the embedding models from Hugging Face).

## 1. Install
```bash
git clone <repo URL from your instructor> && cd rag-for-unstructured-data-v2
make install
```

## 2. Ingest the bundled corpus
```bash
make ingest          # = python -m orca_rag.cli ingest datasets/synthetic
```
The corpus is a constructed oil-and-gas set: well summaries, annual reports, monthly production tables, two HR documents,
and one document from a **second company (tenant `globex`)** that reuses a well id on purpose. Each file's tenant and
access group come from `datasets/synthetic/manifest.json`, so the index is multi-tenant.

Expected (timings vary):
```
ingested  AnnualMonitoringReport_2021.pdf   pages=  6 skip=  0 chunks=  10
...                                          (58 files)
profile=advanced collection=rag_461716c917 total=20s stats={'points': 216, 'status': 'green'}
```
Parser notices such as "RapidOCR is missing / Using Tesseract" are printed per file and are harmless.

Run `make ingest` again: every file reports `unchanged` and the run takes about a second. Ingestion is idempotent: ids are
derived from content, so re-running never duplicates anything.

No `make` (Windows, or a minimal image)? Every make target is one command. After activating the venv
(`.venv/bin/python`, or `.venv\Scripts\python` on Windows, which I have not tested) run, from the repo root:
```bash
python -m orca_rag.cli ingest datasets/synthetic
python -m orca_rag.cli --tenant acme search "What is the total depth of well MB-104?"
python -m orca_rag.cli --tenant acme search "compensation bands" --groups group:hr
```

## 3. Search, as different users
```bash
make ask Q="What is the total depth of well MB-104?"            # tenant acme: WellSummary_MB-104 ranks first
make ask Q="compensation bands"                                   # acme, no groups: the HR document is NOT returned
make ask Q="compensation bands" G=group:hr                        # acme + group:hr: HR_Compensation_2023 ranks first
make ask Q="What is the total depth of well MB-104?" T=globex     # tenant globex: only its own MB-104 document
```
Columns: score, document, page range, section, start of the chunk text. Scores are hybrid (dense + BM25) rank-fusion
scores, so they are comparable within one query only.

## 4. Questions to answer in the session
1. Why did the second command not return the HR document, and which line of code applies that rule?
   (`acl_filter` in `src/orca_rag/store/qdrant_store.py`: the filter runs inside the vector query.)
2. Why did the first search return one `MB-104` document for `acme` and a different one for `globex`?
3. Open `datasets/synthetic/manifest.json` and add `"tenant": "acme"` to a file of your own; ingest it and find it.
4. Change `size` in `configs/advanced.yaml` from 350 to 128 and run `make ingest` again. The collection name changes
   (`rag_461716c917` becomes another id) and the points go from 216 to 301 (checked). Why does a new chunk size create a new
   collection instead of overwriting? (`Profile.collection_name` hashes the model and chunking settings.) Change it back afterwards.

## Stretch
`ORCA_LLM__MOCK=true make serve` starts the API on :8080 with an offline extractive answerer, and `web/orca`
(`npm ci && npm run dev`, set `VITE_API_URL`) is the streaming UI; see `README.md`. Real answers need a provider key
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, ...) and `ORCA_LLM__MOCK=false`.

## If something fails
| symptom | cause / fix |
|---|---|
| `requires-python >=3.11` error during install | your `python3` is older; install Python 3.11+ and re-run `make install` |
| download errors on first ingest | no internet or Hugging Face blocked; the models land in `data/models/`, copy that folder from a classmate |
| `No space left on device` | free about 1 GB; delete `.venv` and `data/` to start over |
| empty search results | you searched a tenant that has nothing: use `T=acme` (the default) or `T=globex` |
