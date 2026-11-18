"""Which ingestion failures are worth retrying? A dependency being down is; a bad document is not."""
import httpx
import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from orca_rag.config.settings import Env, Profile, EmbeddingCfg
from orca_rag.ingestion.pipeline import Ingestor, is_transient
from orca_rag.store.qdrant_store import QdrantStore


def test_classification():
    assert is_transient(ConnectionRefusedError()) and is_transient(TimeoutError()) and is_transient(httpx.ConnectError("x"))
    wrapped = ResponseHandlingException(httpx.ConnectError("name or service not known"))  # what qdrant-client raises
    assert is_transient(wrapped)
    assert is_transient(UnexpectedResponse(status_code=503, reason_phrase="x", content=b"", headers=httpx.Headers()))
    assert not is_transient(UnexpectedResponse(status_code=400, reason_phrase="x", content=b"", headers=httpx.Headers()))
    for bad in (ValueError("no indexable text"), RuntimeError("Failed to open file as type pdf"), FileNotFoundError("x"), PermissionError("x")):
        assert not is_transient(bad)


def test_real_unreachable_qdrant_is_reported_transient(tmp_path):
    prof = Profile(embedding=EmbeddingCfg(dense_model="hash:64", sparse_model="hash-sparse"))
    store = QdrantStore(Env(), prof, client=QdrantClient(url="http://127.0.0.1:1", timeout=2), collection="t")
    f = tmp_path / "a.txt"
    f.write_text("# Title\n\n" + "The injection well pressure was stable during the test. " * 20)
    rep = Ingestor(store, prof, parse_cache=tmp_path / "pc").ingest_file(str(f))
    assert rep.status == "failed" and rep.transient, rep.error


def test_corrupt_document_is_permanent(tmp_path):
    prof = Profile(embedding=EmbeddingCfg(dense_model="hash:64", sparse_model="hash-sparse"))
    store = QdrantStore(Env(), prof, client=QdrantClient(":memory:"), collection="t")
    f = tmp_path / "broken.pdf"
    f.write_bytes(bytes(range(256)) * 12)
    rep = Ingestor(store, prof, parse_cache=tmp_path / "pc").ingest_file(str(f))
    assert rep.status == "failed" and not rep.transient
