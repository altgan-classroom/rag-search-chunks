import json

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from orca_rag.api.app import create_app
from orca_rag.config.settings import (AuthEnv, ChunkingCfg, EmbeddingCfg, Env, GenerationCfg, LLMEnv, Profile, RerankCfg,
                                      RetrievalCfg)
from orca_rag.ingestion.models import Page, ParsedDocument
from orca_rag.ingestion.pipeline import Ingestor
from orca_rag.llm.gateway import LLMGateway, default_mock_handler
from orca_rag.store.qdrant_store import QdrantStore

KEYS = {"k-acme": "acme|alice|", "k-globex": "globex|gina|", "k-admin": "acme|root||admin", "k-hr": "acme|hank|group:hr"}


def make(tmp_path, handler=None, **env_kw):
    prof = Profile(name="t", embedding=EmbeddingCfg(dense_model="hash:256", sparse_model="hash-sparse"),
                   chunking=ChunkingCfg(size=60, overlap=8, parent_size=160),
                   retrieval=RetrievalCfg(rerank=RerankCfg(model="lexical"), final_k=4),
                   generation=GenerationCfg(abstain_below=0.0))
    env = Env(data_dir=tmp_path, auth=AuthEnv(api_keys=KEYS), llm=LLMEnv(mock=True), **env_kw)
    store = QdrantStore(env, prof, client=QdrantClient(":memory:"), collection="t")
    ing = Ingestor(store, prof)
    ing.ingest_file("a.pdf", tenant="acme", parsed=ParsedDocument("acme-wells", "a.pdf", "sa", [Page(1, "# Wells\n\nACME well AW1 pressure was 1520 psi at 7000 ft.")]))
    ing.ingest_file("g.pdf", tenant="globex", parsed=ParsedDocument("globex-wells", "g.pdf", "sg", [Page(1, "# Wells\n\nGLOBEX well GW9 pressure was 777 psi at 5000 ft.")]))
    ing.ingest_file("h.pdf", tenant="acme", acl=["group:hr"], parsed=ParsedDocument("acme-hr", "h.pdf", "sh", [Page(1, "# Pay\n\nSalary band pressure data is confidential.")]))
    return create_app(env, prof, store=store, llm=LLMGateway(mock=True, mock_handler=handler))


@pytest.fixture
def client(tmp_path):
    with TestClient(make(tmp_path)) as c:
        yield c


def H(k):
    return {"x-api-key": k}


def frames(resp):
    out = []
    for block in resp.text.split("\n\n"):
        if block.startswith("data: "):
            d = block[6:]
            out.append("[DONE]" if d == "[DONE]" else json.loads(d))
    return out


def test_auth_required(client):
    assert client.post("/v1/chat", json={"message": "hi"}).status_code == 401
    assert client.post("/v1/chat", json={"message": "hi"}, headers=H("nope")).status_code == 401
    assert client.get("/health").status_code == 200


@pytest.mark.parametrize("body", [{"message": ""}, {"message": "   "}, {"message": "x" * 4001}, {"message": "ok", "chat_id": "../etc"},
                                  {"message": "ok", "chat_id": "a" * 65}, {}])
def test_validation_rejects_bad_input(client, body):
    assert client.post("/v1/chat", json=body, headers=H("k-acme")).status_code == 422


def test_sse_stream_shape_and_ui_event_contract(client):
    r = client.post("/v1/chat", json={"message": "What was the AW1 pressure?", "chat_id": "c1"}, headers=H("k-acme"))
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    fr = frames(r)
    types = [f["type"] if f != "[DONE]" else f for f in fr]
    assert types[-1] == "[DONE]" and types[-2] == "done"
    assert types.index("context") < types.index("tokens") < types.index("answer")
    ctx = json.loads(next(f for f in fr if f != "[DONE]" and f["type"] == "context")["text"])
    assert ctx["0"]["doc_name"] == "acme-wells"


def test_tenant_isolation(client):
    ans = lambda k, q: next(f for f in frames(client.post("/v1/chat", json={"message": q}, headers=H(k))) if f != "[DONE]" and f["type"] == "answer")["text"]
    assert "1520" in ans("k-acme", "pressure of well") and "777" not in ans("k-acme", "pressure of well GW9")
    assert "777" in ans("k-globex", "pressure of well GW9")


def test_group_acl_visible_only_to_group(client):
    q = {"message": "salary band confidential pressure data"}
    names = lambda k: {v["doc_name"] for f in frames(client.post("/v1/chat", json=q, headers=H(k))) if f != "[DONE]" and f["type"] == "context" for v in json.loads(f["text"]).values()}
    assert "acme-hr" not in names("k-acme") and "acme-hr" in names("k-hr")


def test_cache_hit_second_time_and_not_shared_across_tenants(client):
    q = {"message": "What was the AW1 pressure at depth?"}
    first = frames(client.post("/v1/chat", json=q, headers=H("k-acme")))
    second = frames(client.post("/v1/chat", json={**q, "chat_id": "other"}, headers=H("k-acme")))
    assert not any(f != "[DONE]" and f.get("cached") for f in first) and any(f != "[DONE]" and f.get("cached") for f in second)
    third = frames(client.post("/v1/chat", json=q, headers=H("k-globex")))
    assert not any(f != "[DONE]" and f.get("cached") for f in third)  # different tenant => miss


def test_generation_failure_arrives_as_error_frame_with_done(tmp_path):
    def boom(task, prompt):
        if task == "generate":
            raise RuntimeError("secret provider detail")
        return default_mock_handler(task, prompt)
    with TestClient(make(tmp_path, boom)) as c:
        r = c.post("/v1/chat", json={"message": "AW1 pressure"}, headers=H("k-acme"))
        fr = frames(r)
        assert r.status_code == 200 and fr[-1] == "[DONE]"
        err = next(f for f in fr if f != "[DONE]" and f["type"] == "error")
        assert "secret provider detail" not in json.dumps(err)


def test_rate_limit(client):
    codes = [client.post("/v1/chat", json={"message": f"q{i} AW1"}, headers=H("k-acme")).status_code for i in range(20)]
    assert 429 in codes and codes[0] == 200


def test_admin_ingest_is_admin_only_and_blocks_traversal(client, tmp_path):
    assert client.post("/v1/admin/ingest", json={"path": "x"}, headers=H("k-acme")).status_code == 403
    assert client.post("/v1/admin/ingest", json={"path": "../../etc/passwd"}, headers=H("k-admin")).status_code == 400
    assert client.post("/v1/admin/ingest", json={"path": "missing.pdf"}, headers=H("k-admin")).status_code == 404


def test_history_is_per_user_and_deletable(client):
    client.post("/v1/chat", json={"message": "AW1 pressure", "chat_id": "same"}, headers=H("k-acme"))
    st = client.app.state.history
    from orca_rag.store.qdrant_store import Principal
    assert st.get(Principal("alice", "acme"), "same") and not st.get(Principal("gina", "globex"), "same")
    assert client.delete("/v1/chats/same", headers=H("k-acme")).json()["deleted"] == 2


def test_ready_metrics_and_cors(client):
    assert client.get("/ready").json()["status"] == "ready"
    assert b"orca_requests_total" in client.get("/metrics").content
    r = client.options("/v1/chat", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"})
    assert r.headers.get("access-control-allow-origin") != "*" and "access-control-allow-origin" not in r.headers
