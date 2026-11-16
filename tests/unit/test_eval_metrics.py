import math

import pytest

from orca_rag.eval.metrics import bootstrap_ci, contains_all, is_abstention, retrieval_scores, hit_matches
from orca_rag.eval.schema import GoldQuestion, GoldRef, GoldSet
from orca_rag.store.qdrant_store import Hit


def H(doc, lo, hi=None):
    return Hit(f"{doc}{lo}", 1.0, "t", {"doc_name": doc, "page_start": lo, "page_end": hi or lo})


def Q(*refs, **kw):
    return GoldQuestion(id="q", question="x", category="factoid", gold=list(refs), **kw)


def test_recall_mrr_ndcg_hand_computed():
    q = Q(GoldRef(doc="a", pages=[3]), GoldRef(doc="b", pages=[1]))
    hits = [H("z", 1), H("a", 3), H("y", 2), H("b", 1)]
    s = retrieval_scores(hits, q, ks=(1, 2, 4))
    assert s.recall == {1: 0.0, 2: 0.5, 4: 1.0} and s.hit[1] == 0.0 and s.hit[2] == 1.0
    assert s.mrr == 0.5
    dcg = 1 / math.log2(3) + 1 / math.log2(5)  # relevant at ranks 2 and 4
    ideal = 1 + 1 / math.log2(3)
    assert s.ndcg[4] == pytest.approx(dcg / ideal)


def test_page_range_overlap_counts_and_wrong_page_does_not():
    ref = GoldRef(doc="a", pages=[5])
    assert hit_matches(H("a", 4, 6), ref) and not hit_matches(H("a", 6, 7), ref) and hit_matches(H("a", 1), GoldRef(doc="a"))


def test_prefix_doc_match():
    assert GoldRef(doc="GM1_*").matches_doc("GM1_Well_Summary") and not GoldRef(doc="GM1_*").matches_doc("VW1")


def test_normalisation_handles_commas_dashes_markdown():
    assert contains_all("The pressure was **1,520** psi", [["1520 psi"]])
    assert contains_all("depth 6,976–7,050 ft", [["6976-7050"]])
    assert contains_all("a b", [["a"], ["x", "b"]]) and not contains_all("a b", [["a"], ["x"]])


@pytest.mark.parametrize("t,exp", [("I could not find this in the provided documents.", True),
                                   ("The documents do not specify the porosity.", True),
                                   ("Pressure was 1520 psi [1].", False)])
def test_abstention_detector(t, exp):
    assert is_abstention(t) is exp


def test_bootstrap_ci_brackets_mean_and_is_deterministic():
    v = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1]
    m, lo, hi = bootstrap_ci(v)
    assert lo <= m <= hi and bootstrap_ci(v) == (m, lo, hi)


def test_goldset_validation_rejects_bad_sets():
    with pytest.raises(ValueError):
        GoldQuestion(id="a", question="q", category="factoid")  # answerable w/o gold
    with pytest.raises(ValueError):
        GoldQuestion(id="a", question="q", category="unanswerable", answerable=False, must_contain=[["x"]])
    ok = lambda i, t: GoldQuestion(id=i, question=t, category="factoid", gold=[GoldRef(doc="d")])
    with pytest.raises(ValueError):
        GoldSet(name="n", questions=[ok("1", "same?"), ok("2", "Same? ")])
    with pytest.raises(ValueError):
        GoldSet(name="n", questions=[ok("1", "a"), ok("1", "b")])


def test_ndcg_never_exceeds_one_with_many_chunks_from_the_same_gold_doc():
    q = Q(GoldRef(doc="a"))
    hits = [H("a", 1), H("a", 2), H("a", 3), H("b", 1)]
    s = retrieval_scores(hits, q, ks=(1, 4))
    assert s.ndcg[4] == pytest.approx(1.0) and s.ndcg[1] == pytest.approx(1.0)
    q2 = Q(GoldRef(doc="a"), GoldRef(doc="b"))
    s2 = retrieval_scores([H("a", 1), H("a", 2), H("b", 1)], q2, ks=(3,))
    assert 0 < s2.ndcg[3] < 1.0  # duplicate 'a' chunk at rank 2 pushed 'b' down
