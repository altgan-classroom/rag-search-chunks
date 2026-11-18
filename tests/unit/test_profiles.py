from pathlib import Path

import pytest

from orca_rag.config.settings import ROOT, Profile
from orca_rag.store.embedders import resolve_prefixes

PROFILES = sorted(p.stem for p in (ROOT / "configs").glob("*.yaml"))


@pytest.mark.parametrize("name", PROFILES)
def test_every_shipped_profile_loads(name):
    p = Profile.load(name)
    assert p.name == name and p.chunking.overlap < p.chunking.size


def test_profiles_that_embed_differently_get_different_collections():
    assert Profile.load("advanced").collection_name() != Profile.load("quality").collection_name()
    assert Profile.load("advanced").collection_name() == Profile.load("full").collection_name()  # query-time-only differences share an index


def test_documented_query_prefixes_are_applied():
    assert resolve_prefixes("snowflake/snowflake-arctic-embed-s")[1].startswith("Represent this sentence")
    assert resolve_prefixes("mixedbread-ai/mxbai-embed-large-v1") == ("", "Represent this sentence for searching relevant passages: ")
    assert resolve_prefixes("BAAI/bge-small-en-v1.5") == ("", "")  # bge-v1.5: instruction optional, not auto-applied
