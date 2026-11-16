"""The semantic cache is a security boundary as much as a speed-up."""
from orca_rag.cache.semantic import SemanticCache, fact_signature
from orca_rag.store.qdrant_store import Principal

CONST = lambda q: [1.0, 0.0, 0.0]  # every question embeds identically => only the guards can tell them apart
EV = [{"type": "answer", "text": "x"}]


def test_users_sharing_tenant_and_groups_do_not_share_entries():
    c = SemanticCache(CONST)
    bob, alice = Principal("bob", "acme", ("group:eng",)), Principal("alice", "acme", ("group:eng",))
    c.put(bob, "col", "what is the budget?", [{"type": "answer", "text": "bob's private figure"}])
    assert c.get(alice, "col", "what is the budget?") is None           # user:bob doc may have fed that answer
    assert c.get(bob, "col", "what is the budget?")[0]["text"] == "bob's private figure"


def test_tenants_and_groups_are_isolated():
    c = SemanticCache(CONST)
    c.put(Principal("u", "acme", ("g1",)), "col", "q?", EV)
    assert c.get(Principal("u", "globex", ("g1",)), "col", "q?") is None
    assert c.get(Principal("u", "acme", ("g1", "g2")), "col", "q?") is None


def test_confusable_identifiers_never_collide_even_with_identical_embeddings():
    c = SemanticCache(CONST)  # embedder says MB-104 == MB-140; only the fact guard can separate them
    p = Principal("u", "acme")
    c.put(p, "col", "What is the total depth of well MB-104?", [{"type": "answer", "text": "7,250 ft"}])
    assert c.get(p, "col", "What is the total depth of well MB-140?") is None
    assert c.get(p, "col", "What is the total depth of well MB-104?")[0]["text"] == "7,250 ft"
    c.put(p, "col", "oil production in March 2021?", EV)
    assert c.get(p, "col", "oil production in March 2022?") is None


def test_paraphrase_with_same_facts_still_hits():
    c = SemanticCache(CONST)
    p = Principal("u", "acme")
    c.put(p, "col", "What is the total depth of well MB-104?", EV)
    assert c.get(p, "col", "Total depth of MB-104, please?") is not None


def test_invalidate_and_ttl():
    c = SemanticCache(CONST, ttl_s=0.0)
    p = Principal("u", "acme")
    c.put(p, "col", "q 1?", EV)
    assert c.get(p, "col", "q 1?") is None  # expired
    c2 = SemanticCache(CONST)
    c2.put(p, "col", "q 1?", EV); c2.invalidate()
    assert c2.get(p, "col", "q 1?") is None


def test_same_number_different_prefix_does_not_collide():
    c = SemanticCache(CONST)
    p = Principal("u", "acme")
    c.put(p, "col", "depth of MB-104?", EV)
    assert c.get(p, "col", "depth of XY-104?") is None


def test_fact_signature():
    assert fact_signature("Depth of MB-104 in 2021, in ft?") == {"mb-104", "2021"}
    assert fact_signature("what is porosity") == frozenset()
