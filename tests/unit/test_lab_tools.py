"""Tooling the ingestion lab relies on: `verify`, synthetic load, thread cap."""
import argparse

from orca_rag import cli
from orca_rag.config.settings import ChunkingCfg, EmbeddingCfg, Env, Profile
from orca_rag.ingestion.pipeline import Ingestor
from orca_rag.store.embedders import _threads


def test_synthetic_documents_are_distinct_deterministic_and_carry_a_unique_fact(tmp_path):
    cli._write_synthetic(tmp_path, 5)
    first = {p.name: p.read_text() for p in tmp_path.iterdir()}
    cli._write_synthetic(tmp_path, 5)
    assert first == {p.name: p.read_text() for p in tmp_path.iterdir()}  # same N -> same files (idempotent re-runs)
    assert len(set(first.values())) == 5 and "PX-1007" in first["load_0001.md"]


def test_thread_cap_env(monkeypatch):
    monkeypatch.delenv("ORCA_EMBED_THREADS", raising=False)
    assert _threads() is None
    monkeypatch.setenv("ORCA_EMBED_THREADS", "1")
    assert _threads() == 1


def test_verify_reports_kinds_vectors_and_search(tmp_path, monkeypatch, capsys):
    from qdrant_client import QdrantClient
    from orca_rag.store.qdrant_store import QdrantStore
    prof = Profile(embedding=EmbeddingCfg(dense_model="hash:64", sparse_model="hash-sparse"), chunking=ChunkingCfg(size=60, overlap=10, parent_size=200))
    store = QdrantStore(Env(), prof, client=QdrantClient(":memory:"), collection="t")
    f = tmp_path / "Report.md"
    f.write_text("# Wells\n\nThe total depth of well MB-104 is 4,210 ft.\n\n" + "filler text about monitoring. " * 60)
    assert Ingestor(store, prof, parse_cache=tmp_path / "pc").ingest_file(str(f), tenant="lab").status == "ingested"
    monkeypatch.setattr(cli, "_store", lambda name: (Env(), prof, store))
    assert cli.cmd_verify(argparse.Namespace(profile=None, query="total depth of well MB-104", tenant="lab")) == 0
    out = capsys.readouterr().out
    assert "dense: size=64" in out and "child=" in out and "parent=" in out and "doc=1" in out
    assert "Report.md" in out and "sparse    non-zero terms=" in out and "4,210" in out


def test_verify_says_so_when_nothing_is_ingested(monkeypatch, capsys):
    from qdrant_client import QdrantClient
    from orca_rag.store.qdrant_store import QdrantStore
    prof = Profile(embedding=EmbeddingCfg(dense_model="hash:64", sparse_model="hash-sparse"))
    store = QdrantStore(Env(), prof, client=QdrantClient(":memory:"), collection="t")
    monkeypatch.setattr(cli, "_store", lambda name: (Env(), prof, store))
    assert cli.cmd_verify(argparse.Namespace(profile=None, query="", tenant="lab")) == 1
    assert "NOT FOUND" in capsys.readouterr().out
