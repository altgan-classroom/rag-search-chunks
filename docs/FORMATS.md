# File formats

`orca ingest <folder>` (and the worker, and the admin API) accept every format below. Each parser turns a file into
per-page markdown (headings as `#`, tables as markdown tables), so chunking, table handling, citations and the quality
gates are identical for every format. Parsers live in `src/orca_rag/ingestion/formats.py` and `parsers.py`.

| format | extensions | library | what a citation "page" is | notes |
|---|---|---|---|---|
| PDF | `.pdf` | pymupdf4llm | page number | layout-aware; OCR for text-less pages (Tesseract, optional); OCR-noise quality gate |
| Word | `.docx` | python-docx | page, if Word recorded page breaks; else 1 | headings, lists, **tables in document order**; section path carries the heading |
| PowerPoint | `.pptx` | python-pptx | **slide number** | slide title = heading; tables, chart data, picture alt-text and **speaker notes** are indexed; slides are hard page boundaries |
| Excel | `.xlsx`, `.xlsm` | openpyxl | **sheet index** | one markdown table per visible sheet, `## Sheet: <name>` heading; cached formula values; capped at 5,000 rows/sheet (warning if truncated); sheets are hard page boundaries |
| CSV / TSV | `.csv`, `.tsv` | stdlib `csv` | 1 | delimiter sniffed; large tables are split by rows with the header repeated; capped at 5,000 rows |
| Web page | `.html`, `.htm` | stdlib `html.parser` | 1 | tables kept as tables; `script/style/nav/footer` dropped |
| E-mail | `.eml` | stdlib `email` | 1 | subject + From/To/Cc/Date header lines + body; attachments are **listed in a warning, not indexed** |
| Image | `.png .jpg .jpeg .tif .tiff` | pymupdf + Tesseract | 1 | **needs OCR**: without Tesseract ingestion fails with a clear message instead of indexing nothing |
| Markdown / text | `.md`, `.txt` | n/a | 1 | |

Not supported (rejected up front with a clear error, not retried): legacy `.doc` / `.ppt` / `.xls`, `.rtf`, `.odt`, password-protected
files, and `.json` (it would swallow `manifest.json` and gold files; convert the records you want searchable to CSV or markdown).

## Why slides and sheets are hard page boundaries
Without it, a 4-slide deck became **one chunk cited as "pp. 1-4"** (found by the format corpus below). With `hard_pages` each
slide or sheet is chunked on its own, so the answer to "Q3 revenue for APAC?" cites slide 3 exactly.

## Add a format
```python
from orca_rag.ingestion.parsers import register
from orca_rag.ingestion.models import Page
from orca_rag.ingestion.parsers import _finish

class RtfParser:
    exts = (".rtf",)
    cache_raw = False                       # True only for expensive parsers (layout + OCR) whose raw output is cached
    def parse(self, path, **kw):
        text = ...                          # your extraction
        return _finish(path, [Page(1, text)], {"page_count": 1}, gate=False)   # gate=False: born-digital, no OCR noise
register(RtfParser())
```
The CLI, `ingest_dir`, the Celery task and the admin API all read the registry (`SUPPORTED`), so nothing else changes.
Add a planted-fact fixture to `src/orca_rag/datasets/formats.py` and a test: "it did not crash" is not verification.

## How this was verified
`python -m orca_rag.datasets.formats --out datasets/formats` writes ten files, one per format (`datasets/formats/corpus`), each
with a fact planted somewhere a naive text dump would lose it: a **DOCX table cell**, **PPTX slide 3** and its **speaker
notes**, a **PPTX chart**, the **second sheet of an XLSX**, a **CSV row**, an **HTML table behind navigation noise**, an
**e-mail body**, a **PDF**, and **text inside a PNG (OCR)**. Sixteen gold questions name the expected document *and page*.

1. `tests/unit/test_formats.py` (offline): each parser returns the planted fact on the right page; short slides are not dropped by
   the OCR gate; slides/sheets never merge; hybrid search over the whole corpus returns the right document and exact page.
2. `orca eval datasets/formats` (real embeddings): see the table in `docs/COMPARISON.md` §6.
3. Docker: the same files ingested through the Celery worker and the admin API inside the compose stack, then answered over SSE
   with the right slide cited.

**What this does and does not show.** It shows each parser extracts what is in the file and keeps it addressable (page, table,
heading). It is a *correctness* test of the format pipeline on invented, tiny documents, **not a retrieval-quality benchmark**: with
ten small documents, recall is easy by construction. Retrieval quality on messy real documents is measured on the PDF corpus
(`docs/COMPARISON.md` §1). Real Office files are messier than generated ones (merged cells, text boxes, SmartArt, embedded
objects, tracked changes): expect to add cases from your own documents.
