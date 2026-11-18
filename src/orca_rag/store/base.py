"""The vector-store contract. Ingestion, retrieval, the API and the eval harness depend on THIS, not on Qdrant.

To add a backend (pgvector, OpenSearch, Weaviate, ...): implement `VectorStore`, register it with `register_store`, and run
`tests/unit/test_store_contract.py` against it: the same behavioural tests (idempotent upserts, tenant + ACL isolation,
exact-term hybrid search, parent fetch, deletes, manifests) every backend must pass. `store/memory.py` is the smallest
complete example (about 130 lines, numpy only); `store/qdrant_store.py` is the production one.

Rules every backend must keep (they are security properties, not preferences):
  * `search` applies the tenant + ACL filter INSIDE the query, never after the top-k;
  * point ids are deterministic (`point_id`), so retried ingestion is an upsert, not a duplicate;
  * `kind` is "child" (searchable) | "parent" (context text, fetched by `parent_id`) | "doc" (manifest of one source).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

PUBLIC = "public"
NS = uuid.UUID("6f1c1c52-3a52-4a39-9d0e-0f0c5b7c9a11")


def point_id(doc_id: str, kind: str, ordinal: int | str) -> str:
    return str(uuid.uuid5(NS, f"{doc_id}:{kind}:{ordinal}"))


@dataclass(frozen=True)
class Principal:
    """Who is asking. Built from the API key / JWT, never from the request body."""
    id: str
    tenant: str = "default"
    groups: tuple[str, ...] = ()
    is_admin: bool = False

    def visible_acl(self) -> set[str]:
        return {PUBLIC, *self.groups, f"user:{self.id}"}


@dataclass
class Hit:
    id: str
    score: float
    text: str
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def doc(self) -> str:
        return self.payload.get("doc_name", "")

    @property
    def pages(self) -> tuple[int | None, int | None]:
        return self.payload.get("page_start"), self.payload.get("page_end")


@runtime_checkable
class VectorStore(Protocol):
    collection: str
    dense: Any  # DenseEmbedder: .dim, .embed_docs(), .embed_query() (the retriever reuses it for MMR)

    def ensure_collection(self) -> None: ...
    def health(self) -> dict[str, Any]: ...
    def stats(self) -> dict[str, Any]: ...
    def upsert_children(self, rows: list[dict[str, Any]], batch: int = 64) -> None: ...
    def upsert_payload_only(self, rows: list[dict[str, Any]]) -> None: ...
    def delete_doc(self, doc_id: str) -> None: ...
    def delete_source(self, source_key: str, tenant: str) -> None: ...
    def get_manifest(self, source_key: str, tenant: str) -> dict[str, Any] | None: ...
    def search(self, query: str, principal: Principal, *, mode: str = "hybrid", top_k: int = 40, fusion: str = "rrf",
               extra: list[Any] | None = None, query_dense: list[float] | None = None) -> list[Hit]: ...
    def fetch_parents(self, parent_ids: list[str], principal: Principal) -> dict[str, Hit]: ...


# ----------------------------------------------------------------------------- registry
_BACKENDS: dict[str, Callable[..., VectorStore]] = {}


def register_store(name: str, factory: Callable[..., VectorStore]) -> None:
    _BACKENDS[name] = factory


def make_store(env, profile, **kw) -> VectorStore:
    """`ORCA_STORE_BACKEND=qdrant|memory|<registered name>` (default qdrant)."""
    from . import memory, qdrant_store  # noqa: F401  (import registers the built-in backends)
    name = getattr(env, "store_backend", "qdrant")
    if name not in _BACKENDS:
        raise ValueError(f"unknown store backend {name!r}; registered: {sorted(_BACKENDS)}")
    return _BACKENDS[name](env, profile, **kw)
