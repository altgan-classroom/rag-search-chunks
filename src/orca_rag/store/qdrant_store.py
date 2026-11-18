"""Qdrant access layer: explicit collection config, payload indexes, deterministic ids, ACL-enforced search.

Fixes baseline REVIEW D1-D8, I3-I5. Notes:
  * Point ids are uuid5(doc_id, kind:ordinal) => re-ingesting is an upsert, deletes are payload-filter based.
  * `kind` = "child" (searchable, has vectors) | "parent" (context text only, no vectors) | "doc" (manifest).
  * Every search carries a *mandatory* tenant/ACL filter built server-side from the authenticated principal.
"""
from __future__ import annotations

import logging
from typing import Any

from qdrant_client import QdrantClient, models as qm

from ..config.settings import Env, Profile
from .base import PUBLIC, Hit, Principal, point_id, register_store  # noqa: F401  (re-exported: existing imports keep working)
from .embedders import DenseEmbedder, get_dense, get_late, get_sparse

logger = logging.getLogger(__name__)
DENSE, SPARSE, LATE = "dense", "sparse", "late"


def acl_filter(p: Principal, extra: list[qm.Condition] | None = None, kinds: tuple[str, ...] = ("child",)) -> qm.Filter:
    must: list[qm.Condition] = [
        qm.FieldCondition(key="tenant", match=qm.MatchValue(value=p.tenant)),
        qm.FieldCondition(key="kind", match=qm.MatchAny(any=list(kinds))),
        qm.FieldCondition(key="acl", match=qm.MatchAny(any=sorted(p.visible_acl()))),
    ]
    return qm.Filter(must=must + (extra or []))


class QdrantStore:
    def __init__(self, env: Env, profile: Profile, *, client: QdrantClient | None = None, collection: str | None = None):
        self.profile, self.env = profile, env
        self.collection = collection or profile.collection_name()
        if client is not None:
            self.client = client
        elif env.qdrant.url:
            self.client = QdrantClient(url=env.qdrant.url, prefer_grpc=env.qdrant.prefer_grpc, timeout=30,
                                       api_key=env.qdrant.api_key.get_secret_value() if env.qdrant.api_key else None)
        else:
            env.qdrant.path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(env.qdrant.path))
        e = profile.embedding
        self.dense: DenseEmbedder = get_dense(e.dense_model, e.doc_prefix, e.query_prefix)
        self.sparse = get_sparse(e.sparse_model) if e.sparse_model else None
        self.late = get_late(e.late_model) if e.late_model else None

    # ------------------------------------------------------------------ collection
    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            return
        vectors = {DENSE: qm.VectorParams(size=self.dense.dim, distance=qm.Distance.COSINE,
                                          hnsw_config=qm.HnswConfigDiff(m=16, ef_construct=128))}
        if self.late:
            vectors[LATE] = qm.VectorParams(size=self.late.dim, distance=qm.Distance.COSINE,
                                            multivector_config=qm.MultiVectorConfig(comparator=qm.MultiVectorComparator.MAX_SIM),
                                            hnsw_config=qm.HnswConfigDiff(m=0))  # rerank-only: no graph
        sparse = {SPARSE: qm.SparseVectorParams(modifier=qm.Modifier.IDF)} if self.sparse else None
        self.client.create_collection(self.collection, vectors_config=vectors, sparse_vectors_config=sparse,
                                      quantization_config=None)
        for key, schema in [("tenant", qm.PayloadSchemaType.KEYWORD), ("kind", qm.PayloadSchemaType.KEYWORD),
                            ("acl", qm.PayloadSchemaType.KEYWORD), ("doc_id", qm.PayloadSchemaType.KEYWORD),
                            ("source_key", qm.PayloadSchemaType.KEYWORD), ("doc_type", qm.PayloadSchemaType.KEYWORD),
                            ("page_start", qm.PayloadSchemaType.INTEGER), ("doc_year", qm.PayloadSchemaType.INTEGER),
                            ("is_table", qm.PayloadSchemaType.BOOL)]:
            if self.env.qdrant.url:  # payload indexes exist only on a Qdrant server (no-ops in embedded mode)
                self.client.create_payload_index(self.collection, key, schema)
        logger.info("created collection %s (dense=%d)", self.collection, self.dense.dim)

    def health(self) -> dict[str, Any]:
        try:
            ok = self.client.collection_exists(self.collection)
            n = self.client.count(self.collection, exact=False).count if ok else 0
            return {"ok": True, "collection": self.collection, "exists": ok, "points": n}
        except Exception as e:  # readiness must report, not raise
            return {"ok": False, "error": type(e).__name__}

    # ------------------------------------------------------------------ writes
    def upsert_children(self, rows: list[dict[str, Any]], batch: int = 64) -> None:
        """rows: {id, text, payload}; embeds + upserts in batches."""
        self.ensure_collection()
        for i in range(0, len(rows), batch):
            part = rows[i : i + batch]
            texts = [r["embed_text"] for r in part]
            dense = self.dense.embed_docs(texts)
            sparse = self.sparse.embed_docs(texts) if self.sparse else None
            late = self.late.embed_docs(texts) if self.late else None
            pts = []
            for j, r in enumerate(part):
                vec: dict[str, Any] = {DENSE: dense[j]}
                if sparse:
                    vec[SPARSE] = qm.SparseVector(indices=sparse[j][0], values=sparse[j][1])
                if late:
                    vec[LATE] = late[j]
                pts.append(qm.PointStruct(id=r["id"], vector=vec, payload={**r["payload"], "text": r["text"]}))
            self.client.upsert(self.collection, points=pts, wait=True)

    def upsert_payload_only(self, rows: list[dict[str, Any]]) -> None:
        """Parents and manifests: no vectors (not searchable, fetched by id/filter)."""
        self.ensure_collection()
        pts = [qm.PointStruct(id=r["id"], vector={}, payload={**r["payload"], "text": r.get("text", "")}) for r in rows]
        for i in range(0, len(pts), 128):
            self.client.upsert(self.collection, points=pts[i : i + 128], wait=True)

    def delete_doc(self, doc_id: str) -> None:
        self.client.delete(self.collection, points_selector=qm.FilterSelector(filter=qm.Filter(must=[
            qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))])), wait=True)

    def delete_source(self, source_key: str, tenant: str) -> None:
        self.client.delete(self.collection, points_selector=qm.FilterSelector(filter=qm.Filter(must=[
            qm.FieldCondition(key="source_key", match=qm.MatchValue(value=source_key)),
            qm.FieldCondition(key="tenant", match=qm.MatchValue(value=tenant))])), wait=True)

    def get_manifest(self, source_key: str, tenant: str) -> dict[str, Any] | None:
        if not self.client.collection_exists(self.collection):
            return None
        pts, _ = self.client.scroll(self.collection, limit=1, with_payload=True, scroll_filter=qm.Filter(must=[
            qm.FieldCondition(key="kind", match=qm.MatchValue(value="doc")),
            qm.FieldCondition(key="source_key", match=qm.MatchValue(value=source_key)),
            qm.FieldCondition(key="tenant", match=qm.MatchValue(value=tenant))]))
        return pts[0].payload if pts else None

    # ------------------------------------------------------------------ reads
    def search(self, query: str, principal: Principal, *, mode: str = "hybrid", top_k: int = 40,
               fusion: str = "rrf", extra: list[qm.Condition] | None = None,
               query_dense: list[float] | None = None) -> list[Hit]:
        if not self.client.collection_exists(self.collection):
            return []
        flt = acl_filter(principal, extra)
        qd = query_dense or self.dense.embed_query(query)
        use_sparse = self.sparse is not None and mode in ("sparse", "hybrid")
        qs = self.sparse.embed_query(query) if use_sparse else None
        sv = qm.SparseVector(indices=qs[0], values=qs[1]) if qs else None
        if mode == "dense" or not use_sparse:
            res = self.client.query_points(self.collection, query=qd, using=DENSE, query_filter=flt,
                                           limit=top_k, with_payload=True).points
        elif mode == "sparse":
            res = self.client.query_points(self.collection, query=sv, using=SPARSE, query_filter=flt,
                                           limit=top_k, with_payload=True).points
        else:
            fq = qm.FusionQuery(fusion=qm.Fusion.DBSF if fusion == "dbsf" else qm.Fusion.RRF)
            res = self.client.query_points(
                self.collection, limit=top_k, with_payload=True, query=fq, query_filter=flt,
                prefetch=[qm.Prefetch(query=qd, using=DENSE, limit=top_k, filter=flt),
                          qm.Prefetch(query=sv, using=SPARSE, limit=top_k, filter=flt)]).points
        return [Hit(str(p.id), float(p.score), (p.payload or {}).get("text", ""), p.payload or {}) for p in res]

    def fetch_parents(self, parent_ids: list[str], principal: Principal) -> dict[str, Hit]:
        if not parent_ids:
            return {}
        flt = acl_filter(principal, kinds=("parent",))
        out: dict[str, Hit] = {}
        pts, _ = self.client.scroll(self.collection, limit=len(parent_ids) * 2, with_payload=True, scroll_filter=qm.Filter(
            must=flt.must + [qm.FieldCondition(key="parent_id", match=qm.MatchAny(any=parent_ids))]))
        for p in pts:
            pl = p.payload or {}
            out[pl["parent_id"]] = Hit(str(p.id), 1.0, pl.get("text", ""), pl)
        return out

    def stats(self) -> dict[str, Any]:
        info = self.client.get_collection(self.collection)
        return {"points": info.points_count, "status": str(info.status)}


register_store("qdrant", lambda env, profile, **kw: QdrantStore(env, profile, **kw))
