"""FastAPI service. Everything expensive is built ONCE in `lifespan` and shared (baseline built clients, indexes and
engines per request and mutated global state). Streaming is SSE with explicit `error` and `[DONE]` frames, so a
mid-stream failure is visible to the client instead of looking like a truncated 200."""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from ..cache.semantic import SemanticCache
from ..config.settings import Env, Profile, get_env, load_profile
from ..generation.answer import Answerer
from ..ingestion.pipeline import Ingestor
from ..llm.gateway import LLMGateway
from ..retrieval.pipeline import Retriever
from ..store.qdrant_store import Principal, QdrantStore
from . import metrics
from .history import HistoryStore
from .security import RateLimiter, authenticate, require_admin

logger = logging.getLogger("orca.api")
CHAT_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    chat_id: str = Field("default", description="client conversation id")
    metadata: dict = Field(default_factory=dict, description="accepted for UI compatibility; NOT used for filtering")

    @field_validator("chat_id")
    @classmethod
    def _cid(cls, v: str) -> str:
        if not CHAT_ID.match(v):
            raise ValueError("chat_id must match [A-Za-z0-9_-]{1,64}")
        return v

    @field_validator("message")
    @classmethod
    def _msg(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message cannot be empty")
        return v


class IngestRequest(BaseModel):
    path: str  # relative to <data_dir>/raw
    tenant: str | None = None
    acl: list[str] = Field(default_factory=lambda: ["public"])
    force: bool = False


def create_app(env: Env | None = None, profile: Profile | None = None, *, store: QdrantStore | None = None,
               llm: LLMGateway | None = None) -> FastAPI:
    env = env or get_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        prof = profile or load_profile()
        st = store or QdrantStore(env, prof)
        gw = llm or LLMGateway(mock=env.llm.mock, num_retries=env.llm.num_retries, drop_params=env.llm.drop_params)
        retr = Retriever(st, prof, gw)
        app.state.update = dict(env=env, profile=prof, store=st, llm=gw, retriever=retr,
                                answerer=Answerer(retr, prof, gw), ingestor=Ingestor(st, prof, gw),
                                history=HistoryStore(env.data_dir / "history.sqlite"), limiter=RateLimiter(),
                                cache=SemanticCache(st.dense.embed_query) if env.semantic_cache else None)
        for k, v in app.state.update.items():
            setattr(app.state, k, v)
        logger.info("ready profile=%s collection=%s llm_available=%s", prof.name, st.collection, gw.available)
        yield

    app = FastAPI(title="Orca RAG", version="2.0.0", lifespan=lifespan)
    app.state.env = env  # needed by auth before lifespan attrs on some test clients
    app.add_middleware(CORSMiddleware, allow_origins=env.auth.cors_origins, allow_methods=["GET", "POST", "DELETE"],
                       allow_headers=["Authorization", "Content-Type", "X-API-Key"], allow_credentials=False)

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.rid = rid
        resp = await call_next(request)
        resp.headers["x-request-id"] = rid
        return resp

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        logger.exception("unhandled rid=%s path=%s", getattr(request.state, "rid", "-"), request.url.path)
        return JSONResponse({"error": "internal_error", "request_id": getattr(request.state, "rid", "-")}, status_code=500)

    # ------------------------------------------------------------------ health
    @app.get("/health")
    async def health():  # liveness: process is up
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(request: Request):  # readiness: dependencies reachable
        h = request.app.state.store.health()
        return JSONResponse({"status": "ready" if h["ok"] else "degraded", **h}, status_code=200 if h["ok"] else 503)

    @app.get("/metrics")
    async def prom():
        return Response(metrics.render(), media_type="text/plain; version=0.0.4")

    # ------------------------------------------------------------------ chat
    async def _sse(request: Request, p: Principal, body: ChatRequest, endpoint: str):
        s = request.app.state
        if not s.limiter.allow(f"{p.tenant}:{p.id}"):
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "rate limit exceeded", headers={"Retry-After": "10"})
        history = s.history.get(p, body.chat_id)
        cache: SemanticCache | None = s.cache
        coll = s.store.collection
        cacheable = cache is not None and not history

        async def gen():
            t0, first, final, outcome = time.perf_counter(), None, "", "ok"
            events: list[dict] = []
            try:
                if cacheable and (hit := cache.get(p, coll, body.message)):
                    metrics.CACHE.labels("hit").inc()
                    for e in hit:
                        yield f"data: {json.dumps(e)}\n\n"
                    yield 'data: {"type": "done", "cached": true}\n\n'
                    yield "data: [DONE]\n\n"
                    s.history.add(p, body.chat_id, "user", body.message)
                    s.history.add(p, body.chat_id, "assistant", next((e["text"] for e in hit if e["type"] == "answer"), ""))
                    metrics.REQUESTS.labels(endpoint, "cache_hit").inc()
                    return
                if cacheable:
                    metrics.CACHE.labels("miss").inc()
                async for e in s.answerer.run(body.message, p, history):
                    if first is None and e["type"] in ("tokens", "greeting"):
                        first = time.perf_counter() - t0
                        metrics.TTFT.observe(first)
                    if e["type"] == "answer":
                        final = e["text"]
                    if e["type"] == "error":
                        outcome = "error"
                    if e["type"] == "grounding" and not json.loads(e["text"]).get("ok", True):
                        metrics.UNGROUNDED.inc()
                    events.append(e)
                    yield f"data: {json.dumps(e)}\n\n"
                yield 'data: {"type": "done"}\n\n'
                yield "data: [DONE]\n\n"
                if final and outcome == "ok":
                    s.history.add(p, body.chat_id, "user", body.message)
                    s.history.add(p, body.chat_id, "assistant", final)
                    if "could not find" in final.lower():
                        metrics.ABSTAIN.inc()
                    elif cacheable and all(e["type"] != "grounding" or json.loads(e["text"]).get("ok", True) for e in events):
                        cache.put(p, coll, body.message, [e for e in events if e["type"] != "trace"])
            except Exception:  # noqa: BLE001
                outcome = "error"
                logger.exception("stream failed rid=%s", getattr(request.state, "rid", "-"))
                yield 'data: {"type": "error", "text": "Stream interrupted."}\n\n'
                yield "data: [DONE]\n\n"
            finally:
                metrics.REQUESTS.labels(endpoint, outcome).inc()
                metrics.LATENCY.labels(endpoint).observe(time.perf_counter() - t0)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"})

    @app.post("/v1/chat")
    async def chat(body: ChatRequest, request: Request, p: Principal = Depends(authenticate)):
        return await _sse(request, p, body, "chat")

    @app.post("/v1/report")
    async def report(body: ChatRequest, request: Request, p: Principal = Depends(authenticate)):
        body = body.model_copy(update={"message": f"Write a structured report (executive summary, key findings with data, "
                                                  f"risks/uncertainties) on: {body.message}"})
        return await _sse(request, p, body, "report")

    @app.delete("/v1/chats/{chat_id}")
    async def delete_chat(chat_id: str, request: Request, p: Principal = Depends(authenticate)):
        if not CHAT_ID.match(chat_id):
            raise HTTPException(422, "bad chat_id")
        return {"deleted": request.app.state.history.delete(p, chat_id)}

    # ------------------------------------------------------------------ admin
    @app.post("/v1/admin/ingest")
    async def ingest(body: IngestRequest, request: Request, p: Principal = Depends(require_admin)):
        s = request.app.state
        root = (s.env.data_dir / "raw").resolve()
        target = (root / body.path).resolve()
        if not target.is_relative_to(root):  # path traversal guard (baseline X1)
            raise HTTPException(400, "path escapes the data directory")
        if not target.exists():
            raise HTTPException(404, "no such file or directory")
        import asyncio
        files = [target] if target.is_file() else sorted(x for x in target.rglob("*") if x.is_file())
        reps = await asyncio.get_running_loop().run_in_executor(
            None, lambda: [s.ingestor.ingest_file(str(f), tenant=body.tenant or p.tenant, acl=body.acl, force=body.force) for f in files])
        if s.cache:
            s.cache.invalidate()
        return {"results": [r.dict() for r in reps]}

    return app
