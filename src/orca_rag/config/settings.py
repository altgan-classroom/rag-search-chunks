"""Typed, validated configuration.

Two layers:
  * `Env`      - deployment concerns (urls, keys, paths) from ORCA_* env vars / .env. Secrets are SecretStr.
  * `Profile`  - the *RAG recipe* (chunking, embeddings, retrieval, generation) from configs/<name>.yaml.

Keeping the recipe in a profile is what lets us run `baseline` and `advanced` through the same
eval harness and see the difference. Nothing in the codebase hard-codes a model name.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[3]


# --------------------------------------------------------------------------- recipe (profile)
class EmbeddingCfg(BaseModel):
    dense_model: str = "BAAI/bge-small-en-v1.5"
    dense_dim: int | None = None  # auto-detected when None
    sparse_model: str | None = "Qdrant/bm25"  # None disables sparse vectors
    late_model: str | None = None  # e.g. answerdotai/answerai-colbert-small-v1 (multivector rerank)
    query_prefix: str = ""
    doc_prefix: str = ""


class ChunkingCfg(BaseModel):
    strategy: Literal["fixed_words", "structure", "semantic"] = "structure"
    size: int = Field(400, ge=32, description="Target size. Unit = tokens (structure) or characters (fixed_words).")
    overlap: int = Field(60, ge=0)
    parent_child: bool = True  # index small children, return larger parents
    parent_size: int = Field(1600, ge=64)
    keep_tables_whole: bool = True
    contextual: bool = False  # LLM-written situating context per chunk (Anthropic "contextual retrieval")
    add_doc_title: bool = True

    @model_validator(mode="after")
    def _overlap_lt_size(self):
        if self.overlap >= self.size:
            raise ValueError("chunking.overlap must be < chunking.size")
        return self


class RerankCfg(BaseModel):
    enabled: bool = True
    model: str = "Xenova/ms-marco-MiniLM-L-12-v2"
    top_n: int = 8
    min_score: float | None = None  # applied AFTER rerank, on the reranker's own scale
    candidates: int = 24  # first-stage results sent to the cross-encoder (latency is linear in this)
    blend: float = Field(0.0, ge=0.0, le=1.0)  # 0 = reranker order only; >0 fuses with first-stage order (RRF weight)


class RetrievalCfg(BaseModel):
    mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    top_k: int = 40  # candidates per branch before fusion
    fusion: Literal["rrf", "dbsf"] = "rrf"
    rerank: RerankCfg = RerankCfg()
    mmr_lambda: float | None = None  # None disables MMR; 0.5-0.8 typical
    parent_expand: bool = True
    final_k: int = 8
    # LLM-assisted (need a model; every one degrades gracefully to a no-op without one)
    rewrite_with_history: bool = True
    decompose: bool = True
    hyde: bool = False
    router: bool = True


class GenerationCfg(BaseModel):
    model: str = "anthropic/claude-sonnet-5-5"  # LiteLLM id; override via config / ORCA_LLM__GENERATION_MODEL
    fast_model: str = "anthropic/claude-haiku-5-5"  # routing / rewriting / judging cheap steps
    temperature: float = 0.1
    max_tokens: int = 1200
    timeout_s: float = 60.0
    verify_grounding: bool = True
    abstain_below: float | None = 0.0  # abstain if best rerank/relevance score is below this (None = never)
    max_context_tokens: int = 6000
    corrective: bool = True  # agentic retry loop when retrieval looks insufficient
    max_corrective_rounds: int = 2


class Profile(BaseModel):
    name: str = "advanced"
    description: str = ""
    embedding: EmbeddingCfg = EmbeddingCfg()
    chunking: ChunkingCfg = ChunkingCfg()
    retrieval: RetrievalCfg = RetrievalCfg()
    generation: GenerationCfg = GenerationCfg()

    @classmethod
    def load(cls, name_or_path: str) -> "Profile":
        p = Path(name_or_path)
        if not p.suffix:
            p = ROOT / "configs" / f"{name_or_path}.yaml"
        data = yaml.safe_load(p.read_text()) or {}
        data.setdefault("name", p.stem)
        return cls.model_validate(data)

    def collection_name(self) -> str:
        """A collection is defined by what was embedded + how it was chunked, so profiles can coexist."""
        import hashlib

        c, e = self.chunking, self.embedding
        sig = "|".join(map(str, [e.dense_model, e.sparse_model, e.late_model, c.strategy, c.size, c.overlap,
                                 c.parent_child, c.parent_size, c.contextual, c.add_doc_title]))
        return f"orca_{hashlib.sha1(sig.encode()).hexdigest()[:10]}"


# --------------------------------------------------------------------------- deployment env
class QdrantEnv(BaseModel):
    url: str | None = None  # http://qdrant:6333; None => embedded local mode at `path`
    api_key: SecretStr | None = None
    path: Path = ROOT / "data" / "qdrant_local"
    prefer_grpc: bool = False


class LLMEnv(BaseModel):
    """Provider keys are read by LiteLLM from the standard env vars (ANTHROPIC_API_KEY, OPENAI_API_KEY, ...).
    `mock` makes every LLM call deterministic and offline (tests, CI, keyless demos)."""
    mock: bool = False
    drop_params: bool = True
    num_retries: int = 3


class AuthEnv(BaseModel):
    # api key -> tenant/principal. In production put this behind a secret store / JWT; kept simple on purpose.
    api_keys: dict[str, str] = Field(default_factory=dict, description="{api_key: principal_id}")
    allow_anonymous: bool = False  # dev only
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])


class Env(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ORCA_", env_nested_delimiter="__", env_file=".env", extra="ignore")

    profile: str = "advanced"
    data_dir: Path = ROOT / "data"
    model_cache: Path = ROOT / "data" / "models"
    qdrant: QdrantEnv = QdrantEnv()
    llm: LLMEnv = LLMEnv()
    auth: AuthEnv = AuthEnv()
    semantic_cache: bool = True
    log_level: str = "INFO"

    @field_validator("data_dir", "model_cache")
    @classmethod
    def _abs(cls, v: Path) -> Path:
        return v if v.is_absolute() else ROOT / v


@lru_cache
def get_env() -> Env:
    env = Env()
    os.environ.setdefault("FASTEMBED_CACHE_PATH", str(env.model_cache))
    return env


def load_profile(name: str | None = None) -> Profile:
    return Profile.load(name or get_env().profile)
