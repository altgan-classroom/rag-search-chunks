"""Named ablation ladders. Each rung is a *cumulative* override on the baseline recipe, so every technique's marginal
effect is visible; `leave_one_out` removes one technique from the final recipe to expose interactions."""
from __future__ import annotations

import copy

from ..config.settings import Profile
from .ablate import deep_update, with_overrides

RUNGS: list[tuple[str, dict]] = [
    ("1+structure-chunks", {"chunking": {"strategy": "structure", "size": 350, "overlap": 50, "parent_child": False, "keep_tables_whole": True}}),
    ("2+parent-child", {"chunking": {"parent_child": True, "parent_size": 1500}, "retrieval": {"parent_expand": True}}),
    ("3+bm25-sparse", {"embedding": {"sparse_model": "Qdrant/bm25"}}),
    ("4+cross-encoder", {"retrieval": {"rerank": {"enabled": True, "candidates": 24, "top_n": 8}, "final_k": 6}}),
    ("5+rerank-blend", {"retrieval": {"rerank": {"blend": 0.3}}}),
    ("6+mmr", {"retrieval": {"mmr_lambda": 0.7}}),
    ("7+query-understanding", {"retrieval": {"rewrite_with_history": True, "router": True, "decompose": True}}),
    ("8+agentic-gen", {"generation": {"verify_grounding": True, "corrective": True, "abstain_below": None}}),
]


def ladder(base: Profile) -> list[tuple[str, Profile]]:
    out = [("0-baseline", base)]
    acc: dict = {}
    for label, ov in RUNGS:
        deep_update(acc, copy.deepcopy(ov))
        out.append((label, with_overrides(base, label, copy.deepcopy(acc))))
    return out


def leave_one_out(base: Profile) -> list[tuple[str, Profile]]:
    full = ladder(base)[-1][1]
    drops = {
        "-structure/parent": {"chunking": {"strategy": "fixed_words", "size": 2048, "overlap": 64, "parent_child": False, "keep_tables_whole": False},
                              "retrieval": {"parent_expand": False}},
        "-parent-child": {"chunking": {"parent_child": False}, "retrieval": {"parent_expand": False}},
        "-bm25": {"embedding": {"sparse_model": "Qdrant/bm42-all-minilm-l6-v2-attentions"}},
        "-cross-encoder": {"retrieval": {"rerank": {"enabled": False}, "mmr_lambda": None}},
        "-blend": {"retrieval": {"rerank": {"blend": 0.0}}},
        "-mmr": {"retrieval": {"mmr_lambda": None}},
        "-query-understanding": {"retrieval": {"rewrite_with_history": False, "router": False, "decompose": False}},
    }
    return [("full", full)] + [(k, with_overrides(full, k, v)) for k, v in drops.items()]
