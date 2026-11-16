from orca_rag.generation.context import build_sources, render_sources, sources_event_payload
from orca_rag.generation.verify import check_grounding
from orca_rag.store.qdrant_store import Hit

import json


def hit(text, doc="d.pdf", page=3, section="S", id="1", **extra):
    return Hit(id, 1.0, text, {"doc_name": doc, "page_start": page, "page_end": page, "section": section, **extra})


def srcs():
    return build_sources([hit("Injection pressure reached 1,520 psi at 7,000 ft depth.", id="a"),
                          hit("The Mt. Simon sandstone has 18% porosity.", doc="e.pdf", page=9, id="b")], 1000)


def test_sources_numbered_deduped_budgeted_and_payload_matches_ui_contract():
    s = build_sources([hit("same text here", id="a"), hit("same text here", id="b"), hit("other " * 50, id="c")], 40)
    assert [x.n for x in s] == [1] and "[1] (d.pdf, p.3 | S)" in render_sources(s)
    p = json.loads(sources_event_payload(srcs()))
    assert p["0"]["doc_name"] == "d.pdf" and p["1"]["page_num"] == 9


def test_matched_pages_preferred_for_parent_expanded_hits():
    s = build_sources([hit("parent text", matched_pages=(12, 12))], 100)
    assert s[0].page == 12


def test_supported_cited_sentence_passes():
    g = check_grounding("Injection pressure reached 1,520 psi at 7,000 ft depth [1].", srcs())
    assert g.ok and g.score == 1.0


def test_fabricated_number_is_flagged_unsupported():
    g = check_grounding("The injection pressure reached 9,999 psi at 12,500 ft in the Eau Claire shale [1].", srcs())
    assert g.unsupported and not g.ok


def test_invalid_citation_detected():
    g = check_grounding("Injection pressure reached 1,520 psi at 7,000 ft depth [7].", srcs())
    assert g.invalid_citations == [7] and not g.ok


def test_uncited_but_supported_gets_partial_credit_and_is_reported():
    g = check_grounding("Injection pressure reached 1,520 psi at 7,000 ft depth.", srcs())
    assert g.uncited and 0 < g.score < 1


def test_abstention_is_always_grounded():
    assert check_grounding("I could not find this in the provided documents.", srcs()).ok


def test_citation_after_period_stays_with_its_sentence():
    ans = "Injection pressure reached 1,520 psi at 7,000 ft depth. [1] The Mt. Simon sandstone has 18% porosity. [2]"
    g = check_grounding(ans, srcs())
    assert g.ok and g.score == 1.0 and not g.uncited
