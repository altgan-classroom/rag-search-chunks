"""Qdrant behaviour on the embedded local engine: idempotency, ACL isolation, hybrid, deletes."""
from qdrant_client import QdrantClient

from orca_rag.config.settings import Env, Profile, EmbeddingCfg
from orca_rag.store.qdrant_store import Principal, QdrantStore, point_id


def mk_store():
    prof = Profile(embedding=EmbeddingCfg(dense_model="hash:128", sparse_model="hash-sparse"))
    return QdrantStore(Env(), prof, client=QdrantClient(":memory:"), collection="t")


def child(doc, i, text, tenant="a", acl=("public",), **extra):
    return {"id": point_id(doc, "child", i), "text": text, "embed_text": text,
            "payload": {"kind": "child", "doc_id": doc, "doc_name": doc, "tenant": tenant, "acl": list(acl),
                        "source_key": doc, "page_start": i + 1, "page_end": i + 1, **extra}}


def test_upsert_is_idempotent():
    s = mk_store()
    rows = [child("d1", 0, "pressure falloff test"), child("d1", 1, "injection rate")]
    s.upsert_children(rows); s.upsert_children(rows); s.upsert_children(rows)
    assert s.client.count("t").count == 2


def test_hybrid_finds_exact_term():
    s = mk_store()
    s.upsert_children([child("d1", 0, "the Mt. Simon sandstone porosity"), child("d1", 1, "completely unrelated gardening notes"),
                       child("d1", 2, "ZXQ-4411 valve serial number")])
    hits = s.search("ZXQ-4411", Principal("u", "a"), mode="hybrid", top_k=3)
    assert hits[0].payload["page_start"] == 3


def test_tenant_isolation():
    s = mk_store()
    s.upsert_children([child("d1", 0, "secret acme plan", tenant="acme"), child("d2", 0, "secret globex plan", tenant="globex")])
    hits = s.search("secret plan", Principal("u", "acme"), top_k=10)
    assert {h.payload["tenant"] for h in hits} == {"acme"}


def test_acl_groups_and_user_grants():
    s = mk_store()
    s.upsert_children([child("pub", 0, "handbook policy"), child("hr", 0, "salary bands policy", acl=("group:hr",)),
                       child("bob", 0, "bobs private policy", acl=("user:bob",))])
    names = lambda p: {h.payload["doc_id"] for h in s.search("policy", p, top_k=10)}
    assert names(Principal("eve", "a")) == {"pub"}
    assert names(Principal("amy", "a", groups=("group:hr",))) == {"pub", "hr"}
    assert names(Principal("bob", "a")) == {"pub", "bob"}


def test_delete_doc_by_filter_really_deletes():
    s = mk_store()
    s.upsert_children([child("d1", i, f"chunk {i}") for i in range(5)] + [child("d2", 0, "other")])
    s.delete_doc("d1")
    assert s.client.count("t").count == 1


def test_parent_points_not_searchable_but_fetchable():
    s = mk_store()
    s.upsert_children([child("d1", 0, "child text about packers", parent_id="P1")])
    s.upsert_payload_only([{"id": point_id("d1", "parent", 0), "text": "FULL PARENT about packers and more",
                            "payload": {"kind": "parent", "doc_id": "d1", "parent_id": "P1", "tenant": "a", "acl": ["public"]}}])
    p = Principal("u", "a")
    assert all(h.payload["kind"] == "child" for h in s.search("packers", p))
    assert s.fetch_parents(["P1"], p)["P1"].text.startswith("FULL PARENT")
