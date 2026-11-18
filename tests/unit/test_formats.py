"""Every file format: parse the generated mixed-format corpus and check that the *planted fact* is extracted from the
right page, then ingest it all (hash embedders, offline) and check retrieval returns the right document and page."""
import shutil

import pytest
from qdrant_client import QdrantClient

from orca_rag.config.settings import ChunkingCfg, EmbeddingCfg, Env, Profile
from orca_rag.datasets.formats import Q, build
from orca_rag.ingestion.parsers import PARSERS, SUPPORTED, get_parser, parse_file
from orca_rag.ingestion.pipeline import Ingestor
from orca_rag.store.qdrant_store import Principal, QdrantStore

HAS_OCR = shutil.which("tesseract") is not None


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    return build(tmp_path_factory.mktemp("formats")) / "corpus"


def page_text(corpus, name, page):
    d = parse_file(str(corpus / name))
    return next(p.markdown for p in d.pages if p.number == page and not p.skipped)


def test_registry_covers_documented_formats():
    assert {".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".html", ".eml", ".md", ".txt", ".png"} <= SUPPORTED
    assert get_parser("a.DOCX") is PARSERS[".docx"]
    with pytest.raises(ValueError, match="unsupported"):
        get_parser("a.rtf")


def test_docx_table_and_headings(corpus):
    t = page_text(corpus, "Employee_Handbook.docx", 1)
    assert "| International | 70 | No |" in t and "# Travel and expenses" in t and "1.75 days" in t


def test_pptx_slide_table_notes_and_chart(corpus):
    assert "| APAC | 3.1 | 27% |" in page_text(corpus, "Q3_Board_Update.pptx", 3)
    s4 = page_text(corpus, "Q3_Board_Update.pptx", 4)
    assert "14 March 2027" in s4 and "| Beta | 340 |" in s4


def test_short_slide_is_not_dropped_by_the_ocr_gate(corpus):
    d = parse_file(str(corpus / "Q3_Board_Update.pptx"))
    assert [p.skipped for p in d.pages] == [False] * 4 and len(d.pages[0].markdown) < 60


def test_xlsx_sheet_index_is_the_page_and_name_is_the_heading(corpus):
    d = parse_file(str(corpus / "Sales_Pipeline_2026.xlsx"))
    p2 = d.pages[1]
    assert p2.number == 2 and "## Sheet: Pipeline" in p2.markdown and "| Falcon | Arrowline Logistics | 480000 |" in p2.markdown


def test_csv_html_eml(corpus):
    assert "| Norden Fasteners | Ingrid Solberg |" in page_text(corpus, "Supplier_Contacts.csv", 1)
    h = page_text(corpus, "Release_Notes.html", 1)
    assert "| 3.5 | 2026-08-19 | Batch export API" in h and "do-not-index" not in h and "Pricing" not in h
    assert "expired TLS certificate" in page_text(corpus, "Incident_4471.eml", 1)


@pytest.mark.skipif(not HAS_OCR, reason="needs Tesseract")
def test_image_is_ocrd(corpus):
    assert "Priya Raman" in page_text(corpus, "Whiteboard_Photo.png", 1)


def test_image_without_ocr_says_so(corpus, monkeypatch):
    import orca_rag.ingestion.formats as f
    monkeypatch.setattr(f, "parse_pdf_raw", lambda *a, **k: ([f.Page(1, "")], {}))
    with pytest.raises(ValueError, match="OCR is required"):
        parse_file(str(corpus / "Whiteboard_Photo.png"))


def test_every_format_retrieves_the_right_document_and_page(corpus):
    """Offline (hash embedders, lexical): plumbing check across formats. Paraphrase and OCR questions need real embeddings
    / Tesseract and are covered by the `eval` run in docs/FORMATS.md."""
    prof = Profile(embedding=EmbeddingCfg(dense_model="hash:256", sparse_model="hash-sparse"),
                   chunking=ChunkingCfg(size=120, overlap=20, parent_size=400))
    store = QdrantStore(Env(), prof, client=QdrantClient(":memory:"), collection="t")
    ing = Ingestor(store, prof)
    reps = [ing.ingest_file(str(f)) for f in sorted(corpus.iterdir()) if f.suffix != ".png" or HAS_OCR]
    assert all(r.status == "ingested" for r in reps), [(r.source_key, r.error) for r in reps if r.status != "ingested"]
    me = Principal("u", "default")
    for qid, q, cat, doc, pages, must, _ in Q:
        if not doc or cat == "paraphrase" or (qid == "img-ocr" and not HAS_OCR):
            continue
        hits = store.search(q, me, mode="hybrid", top_k=5)
        ok = [h for h in hits if h.payload["doc_name"] == doc and h.payload["page_start"] <= pages[0] <= h.payload["page_end"]]
        if doc in ("Q3_Board_Update", "Sales_Pipeline_2026"):  # slides / sheets: the citation must be exactly that page
            ok = [h for h in ok if h.payload["page_start"] == h.payload["page_end"] == pages[0]]
        assert ok, f"{qid}: expected {doc} p.{pages}, got {[(h.payload['doc_name'], h.payload['page_start']) for h in hits]}"


def test_slides_and_sheets_are_never_merged_into_one_chunk(corpus):
    from orca_rag.ingestion.chunkers import chunk_document
    for name, n in (("Q3_Board_Update.pptx", 4), ("Sales_Pipeline_2026.xlsx", 3)):
        kids, parents = chunk_document(parse_file(str(corpus / name)), ChunkingCfg(size=400, parent_size=1500))
        assert sorted({c.page_start for c in kids}) == list(range(1, n + 1))
        assert all(c.page_start == c.page_end for c in kids + parents)
