import pytest
from qdrant_client import QdrantClient

from orca_rag.config.settings import ChunkingCfg, EmbeddingCfg, Env, Profile
from orca_rag.ingestion.models import Page, ParsedDocument
from orca_rag.ingestion.pipeline import Ingestor
from orca_rag.store.qdrant_store import Principal, QdrantStore


def setup(**chunk):
    prof = Profile(embedding=EmbeddingCfg(dense_model="hash:128", sparse_model="hash-sparse"),
                   chunking=ChunkingCfg(size=60, overlap=10, parent_size=200, **chunk))
    store = QdrantStore(Env(), prof, client=QdrantClient(":memory:"), collection="t")
    return store, Ingestor(store, prof)


def doc(text, sha="s1", key="WellSummary_20140708.pdf"):
    return ParsedDocument(key[:-4], key, sha, [Page(1, "# Title\n\n" + text)])


BODY = "\n\n".join(f"Paragraph {i} about the injection well pressure. " * 3 for i in range(12))


def n(store, kind=None):
    from qdrant_client import models as qm
    f = qm.Filter(must=[qm.FieldCondition(key="kind", match=qm.MatchValue(value=kind))]) if kind else None
    return store.client.count("t", count_filter=f).count


def test_reingest_same_file_is_noop_and_force_is_still_idempotent():
    store, ing = setup()
    r1 = ing.ingest_file("x.pdf", parsed=doc(BODY))
    c = n(store)
    r2 = ing.ingest_file("x.pdf", parsed=doc(BODY))
    assert (r1.status, r2.status) == ("ingested", "unchanged") and n(store) == c
    ing.ingest_file("x.pdf", parsed=doc(BODY), force=True)
    assert n(store) == c  # deterministic ids => rewrite, not duplicate


def test_changed_content_replaces_old_version():
    store, ing = setup()
    ing.ingest_file("x.pdf", parsed=doc("old text about packers. " * 30, sha="v1", key="x.pdf"))
    r = ing.ingest_file("x.pdf", parsed=doc("new text about perforations. " * 30, sha="v2", key="x.pdf"))
    assert r.status == "replaced"
    texts = " ".join(h.text for h in store.search("packers perforations", Principal("u", "default"), top_k=50))
    assert "perforations" in texts and "packers" not in texts
    assert store.get_manifest("x.pdf", "default")["version"] == 2


def test_parent_child_writes_parents_and_manifest():
    store, ing = setup(parent_child=True)
    r = ing.ingest_file("y.pdf", parsed=doc(BODY, key="y.pdf"))
    assert r.parents > 0 and r.chunks > r.parents
    assert n(store, "parent") == r.parents and n(store, "doc") == 1 and n(store, "child") == r.chunks


def test_metadata_extraction_from_filename():
    store, ing = setup()
    ing.ingest_file("a.pdf", parsed=doc(BODY, key="1211523438_20140708_GM1_Well_Summary_Sheet.pdf"))
    m = store.get_manifest("1211523438_20140708_GM1_Well_Summary_Sheet.pdf", "default")
    assert m["doc_year"] == 2014 and m["doc_type"] == "well_summary"


def test_empty_doc_fails_cleanly_without_raising():
    store, ing = setup()
    d = ParsedDocument("e", "e.pdf", "se", [Page(1, "", skipped=True)])
    r = ing.ingest_file("e.pdf", parsed=d)
    assert r.status == "failed" and "no indexable text" in r.error


def test_unsupported_extension_rejected_up_front():
    from orca_rag.ingestion.parsers import parse_file
    with pytest.raises(ValueError):
        parse_file("notes.docx")


def test_contextual_skipped_with_warning_without_llm():
    store, ing = setup(contextual=True)
    r = ing.ingest_file("z.pdf", parsed=doc(BODY, key="z.pdf"))
    assert r.status == "ingested" and any("contextual" in w for w in r.warnings)


def test_ingest_manifest_keeps_tenant_and_acl(tmp_path):
    import json
    for name, body in {"a.md": "# A\n\nAcme injection well pressure report. " * 20,
                       "g.md": "# G\n\nGlobex injection well pressure report. " * 20,
                       "h.md": "# H\n\nAcme HR only injection well note. " * 20}.items():
        (tmp_path / name).write_text(body)
    (tmp_path / "manifest.json").write_text(json.dumps({"name": "m", "files": [
        {"file": "a.md", "tenant": "acme", "acl": ["public"]},
        {"file": "g.md", "tenant": "globex", "acl": ["public"]},
        {"file": "h.md", "tenant": "acme", "acl": ["group:hr"]}]}))
    prof = Profile(embedding=EmbeddingCfg(dense_model="hash:128", sparse_model="hash-sparse"),
                   chunking=ChunkingCfg(size=60, overlap=10, parent_size=200))
    store = QdrantStore(Env(), prof, client=QdrantClient(":memory:"), collection="t")
    reps = Ingestor(store, prof, parse_cache=tmp_path / "cache").ingest_manifest(str(tmp_path))  # a dir containing manifest.json
    assert [r.status for r in reps] == ["ingested"] * 3

    def docs(p):
        return {h.doc for h in store.search("injection well pressure", p, top_k=20)}
    assert docs(Principal("u", "acme")) == {"a"}                       # no Globex, no HR-only
    assert docs(Principal("u", "acme", ("group:hr",))) == {"a", "h"}         # group member also sees the HR note
    assert docs(Principal("u", "globex")) == {"g"}
