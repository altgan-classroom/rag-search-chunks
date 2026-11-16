from orca_rag.config.settings import ChunkingCfg
from orca_rag.ingestion.chunkers import chunk_document, chunk_fixed_words, chunk_structure, count_tokens
from orca_rag.ingestion.models import Page, ParsedDocument
from orca_rag.ingestion.parsers import strip_running_lines, text_quality

TABLE = "|Depth|Pressure|\n|---|---|\n" + "\n".join(f"|{6000 + i}|{2900 + i}|" for i in range(60))


def mkdoc(*pages):
    return ParsedDocument("d", "d.pdf", "sha", [Page(i + 1, p) for i, p in enumerate(pages)])


def test_overlap_must_be_smaller_than_size():
    import pytest
    with pytest.raises(ValueError):
        ChunkingCfg(size=100, overlap=100)


def test_structure_tracks_heading_path_and_pages():
    d = mkdoc("# Intro\n\nAlpha para one. " * 1 + "\n\n## Methods\n\nBeta text here.", "More methods text on page two.")
    ch, _ = chunk_structure(d, ChunkingCfg(size=200, overlap=0, parent_child=False))
    assert ch[0].page_start == 1 and ch[0].page_end == 2
    assert ch[0].section == "Intro"  # chunk is labelled by its first content block
    assert "Methods" in ch[0].text   # ...and heading text stays in the body
    long = "word " * 25 + "."
    d2 = mkdoc(f"# Intro\n\n{long}\n\n## Methods\n\n{long}")
    ch2, _ = chunk_structure(d2, ChunkingCfg(size=40, overlap=0, parent_child=False))
    assert any(c.section == "Intro > Methods" for c in ch2)
    assert "Alpha" in ch[0].text and "More methods" in ch[0].text


def test_tables_kept_whole_when_they_fit_and_split_with_header_when_not():
    d = mkdoc("# Data\n\n" + TABLE)
    whole, _ = chunk_structure(d, ChunkingCfg(size=2000, overlap=0, parent_child=False))
    assert sum(c.is_table for c in whole) == 1 and "|6059|2959|" in whole[0].text
    split, _ = chunk_structure(d, ChunkingCfg(size=120, overlap=0, parent_child=False))
    tables = [c for c in split if c.is_table]
    assert len(tables) > 1
    assert all("|Depth|Pressure|\n|---|---|" in t.text for t in tables)  # header repeated on every part
    rows = {r for t in tables for r in t.text.split("\n") if r[1:5].isdigit()}
    assert len(rows) == 60  # nothing lost


def test_no_heading_left_stranded_at_chunk_end():
    para = " ".join(["word"] * 90) + "."
    d = mkdoc(f"{para}\n\n## Next Section\n\n{para}")
    ch, _ = chunk_structure(d, ChunkingCfg(size=100, overlap=0, parent_child=False))
    assert all(not c.text.rstrip().endswith("Next Section") for c in ch)
    assert any("Next Section" in c.text and "word" in c.text.split("Next Section")[-1] for c in ch)


def test_chunks_respect_budget():
    para = ". ".join(f"Sentence number {i} about wells" for i in range(300)) + "."
    ch, _ = chunk_structure(mkdoc(para), ChunkingCfg(size=80, overlap=10, parent_child=False))
    assert len(ch) > 3 and max(c.tokens for c in ch) <= 80 + 10 + 20


def test_parent_child_links_and_containment():
    para = "\n\n".join(f"Paragraph {i}. " + "filler " * 40 for i in range(20))
    kids, parents = chunk_structure(mkdoc(para), ChunkingCfg(size=120, overlap=0, parent_size=500, parent_child=True))
    assert parents and len(kids) > len(parents)
    for k in kids:
        assert k.parent_ordinal is not None and k.text.split(".")[0][:12] in parents[k.parent_ordinal].text
    assert max(p.tokens for p in parents) <= 500 + 5


def test_skipped_pages_are_excluded():
    d = mkdoc("Real content about injection wells.", "!! | ?? ~~ ..")
    d.pages[1].skipped = True
    ch, _ = chunk_document(d, ChunkingCfg(size=100, overlap=0, parent_child=False))
    assert all("??" not in c.text for c in ch)


def test_fixed_words_baseline_flattens_newlines_and_uses_first_page():
    d = mkdoc("line one\nline two " * 30, "page two words " * 30)
    ch = chunk_fixed_words(d, ChunkingCfg(strategy="fixed_words", size=200, overlap=20))
    assert all("\n" not in c.text for c in ch) and ch[0].page_start == 1


def test_running_header_footer_removed():
    pages = [Page(i + 1, f"ACME Project 1820-SH Page {i + 1}\n\nBody text {['alpha','bravo','charlie','delta','echo','foxtrot','golf','hotel'][i]} about wells.") for i in range(8)]
    n = strip_running_lines(pages)
    assert n >= 8 and all("ACME" not in p.markdown and "Body text" in p.markdown for p in pages)


def test_text_quality_flags_ocr_garbage():
    assert text_quality("The injection well was completed in the Mt. Simon Sandstone formation.") > 0.8
    assert text_quality("!| | aa fr | | 1 | ' {] [| [] [| 9 LJ [] f] ud") < 0.3


def test_numeric_table_is_not_mistaken_for_garbage():
    tbl = "|Month|MB-107|MB-104|\n|---|---|---|\n" + "\n".join(f"|Month{i}|{1200 + i},{300 + i}|{900 + i}|" for i in range(12))
    assert text_quality(tbl) > 0.8
    assert text_quality("$1,250 12.5% 7,000 ft 1,520 psi") > 0.8
