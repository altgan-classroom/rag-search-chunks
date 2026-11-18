"""One behavioural contract, run against every registered VectorStore backend. A new backend is "done" when this passes."""
import pytest
from qdrant_client import QdrantClient

from orca_rag.config.settings import ChunkingCfg, EmbeddingCfg, Env, Profile
from orca_rag.ingestion.models import Page, ParsedDocument
from orca_rag.ingestion.pipeline import Ingestor
from orca_rag.store.base import Principal, VectorStore, make_store, point_id
from orca_rag.store.memory import MemoryStore
from orca_rag.store.qdrant_store import QdrantStore

PROF = Profile(embedding=EmbeddingCfg(dense_model="hash:128", sparse_model="hash-sparse"),
               chunking=ChunkingCfg(size=60, overlap=10, parent_size=200))


@pytest.fixture(params=["qdrant", "memory"])
def store(request) -> VectorStore:
    if request.param == "qdrant":
        return QdrantStore(Env(), PROF, client=QdrantClient(":memory:"), collection="t")
    return MemoryStore(Env(), PROF, collection="t")


def child(doc, i, text, tenant="a", acl=("public",), **extra):
    return {"id": point_id(doc, "child", i), "text": text, "embed_text": text,
            "payload": {"kind": "child", "doc_id": doc, "doc_name": doc, "tenant": tenant, "acl": list(acl),
                        "source_key": doc, "page_start": i + 1, "page_end": i + 1, **extra}}


def test_implements_the_protocol(store):
    assert isinstance(store, VectorStore)


def test_upsert_is_idempotent(store):
    rows = [child("d1", 0, "pressure falloff test"), child("d1", 1, "injection rate")]
    store.upsert_children(rows)
    store.upsert_children(rows)
    assert store.stats()["points"] == 2


def test_exact_term_found_by_hybrid(store):
    store.upsert_children([child("d1", 0, "the Mt. Simon sandstone porosity"), child("d1", 1, "completely unrelated gardening notes"),
                           child("d1", 2, "ZXQ-4411 valve serial number")])
    assert store.search("ZXQ-4411", Principal("u", "a"), mode="hybrid", top_k=3)[0].payload["page_start"] == 3


def test_tenant_and_acl_isolation(store):
    store.upsert_children([child("pub", 0, "handbook policy"), child("hr", 0, "salary bands policy", acl=("group:hr",)),
                           child("bob", 0, "bobs private policy", acl=("user:bob",)),
                           child("g", 0, "globex policy", tenant="globex")])
    names = lambda p: {h.payload["doc_id"] for h in store.search("policy", p, top_k=10)}
    assert names(Principal("eve", "a")) == {"pub"}
    assert names(Principal("amy", "a", groups=("group:hr",))) == {"pub", "hr"}
    assert names(Principal("bob", "a")) == {"pub", "bob"}
    assert names(Principal("x", "globex")) == {"g"}


def test_manifest_parents_and_deletes(store):
    store.upsert_payload_only([{"id": point_id("d1", "doc", 0), "text": "", "payload": {
        "kind": "doc", "doc_id": "d1", "source_key": "d1.pdf", "tenant": "a", "acl": ["public"], "sha256": "x"}},
        {"id": point_id("d1", "parent", 0), "text": "whole section text", "payload": {
            "kind": "parent", "doc_id": "d1", "source_key": "d1.pdf", "tenant": "a", "acl": ["public"], "parent_id": "d1:p:0"}}])
    store.upsert_children([child("d1", 0, "some child text", source_key="d1.pdf")])
    assert store.get_manifest("d1.pdf", "a")["sha256"] == "x" and store.get_manifest("d1.pdf", "other") is None
    assert store.fetch_parents(["d1:p:0"], Principal("u", "a"))["d1:p:0"].text == "whole section text"
    assert store.fetch_parents(["d1:p:0"], Principal("u", "other")) == {}
    store.delete_source("d1.pdf", "a")
    assert store.stats()["points"] == 0


def test_full_ingest_and_search_through_either_backend(store):
    ing = Ingestor(store, PROF)
    d = ParsedDocument("Report", "Report.pdf", "sha1", [Page(1, "# Wells\n\nThe total depth of well MB-104 is 4,210 ft.\n\n" + "filler text. " * 80)])
    assert ing.ingest_file("Report.pdf", parsed=d, tenant="a").status == "ingested"
    assert ing.ingest_file("Report.pdf", parsed=d, tenant="a").status == "unchanged"
    assert "4,210" in store.search("total depth of well MB-104", Principal("u", "a"), top_k=3)[0].text


def test_backend_selection_by_name():
    assert isinstance(make_store(Env(store_backend="memory"), PROF), MemoryStore)
    with pytest.raises(ValueError, match="unknown store backend"):
        make_store(Env(store_backend="nope"), PROF)
