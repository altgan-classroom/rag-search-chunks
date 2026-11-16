from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Category = Literal["factoid", "numeric", "table", "multi_hop", "comparison", "temporal", "conflicting", "definition",
                   "follow_up", "unanswerable", "acl", "adversarial", "keyword", "paraphrase"]


class GoldRef(BaseModel):
    doc: str  # doc_name (file stem) - exact match, or prefix match when ending with '*'
    pages: list[int] = Field(default_factory=list)  # empty = any page of the doc counts

    def matches_doc(self, name: str) -> bool:
        return name.startswith(self.doc[:-1]) if self.doc.endswith("*") else name == self.doc


class GoldQuestion(BaseModel):
    id: str
    question: str
    category: Category
    answerable: bool = True
    gold: list[GoldRef] = Field(default_factory=list)
    # keyless answer check: AND over groups, OR within a group. Matched on normalised text (case, commas, spaces).
    must_contain: list[list[str]] = Field(default_factory=list)
    gold_answer: str | None = None
    history: list[tuple[str, str]] = Field(default_factory=list)
    tenant: str = "default"
    groups: list[str] = Field(default_factory=list)
    forbidden_docs: list[str] = Field(default_factory=list)  # ACL tests: must never appear in context
    forbidden_answer: list[str] = Field(default_factory=list)  # strings that must NOT appear in the answer (injection)
    source: Literal["synthetic", "model_authored", "human"] = "synthetic"
    status: Literal["verified", "needs_review", "synthetic"] = "synthetic"
    notes: str = ""

    @model_validator(mode="after")
    def _consistent(self):
        if self.answerable and self.category not in ("acl",) and not self.gold:
            raise ValueError(f"{self.id}: answerable question needs gold refs")
        if not self.answerable and self.must_contain:
            raise ValueError(f"{self.id}: unanswerable question cannot have must_contain")
        return self


class GoldSet(BaseModel):
    name: str
    description: str = ""
    questions: list[GoldQuestion]

    @model_validator(mode="after")
    def _unique(self):
        ids = [q.id for q in self.questions]
        dup = {i for i in ids if ids.count(i) > 1}
        if dup:
            raise ValueError(f"duplicate question ids: {sorted(dup)}")
        # same text is legitimate for different principals / histories (ACL, follow-ups); identical contexts are duplicates
        keys = [(q.question.strip().lower(), q.tenant, tuple(sorted(q.groups)), json.dumps(q.history)) for q in self.questions]
        dupt = {k[0] for k in keys if keys.count(k) > 1}
        if dupt:
            raise ValueError(f"duplicate question texts: {sorted(dupt)[:3]}")
        return self

    @classmethod
    def load(cls, path: str | Path) -> "GoldSet":
        return cls.model_validate(json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.model_dump(), indent=1, ensure_ascii=False))

    def by_category(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for q in self.questions:
            out[q.category] = out.get(q.category, 0) + 1
        return dict(sorted(out.items()))


class CorpusFile(BaseModel):
    file: str  # path relative to the manifest
    tenant: str = "default"
    acl: list[str] = Field(default_factory=lambda: ["public"])


class CorpusManifest(BaseModel):
    name: str
    files: list[CorpusFile]
    root: str = "."

    @classmethod
    def load(cls, path: str | Path) -> "CorpusManifest":
        m = cls.model_validate(json.loads(Path(path).read_text()))
        m.root = str(Path(path).parent)
        return m
