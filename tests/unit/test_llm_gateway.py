import pytest
from orca_rag.llm.gateway import LLMGateway, default_mock_handler, parse_json_lenient


@pytest.mark.parametrize("raw,expected", [
    ('{"a": 1}', {"a": 1}),
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('Sure! Here you go: ["x", "y"] hope it helps', ["x", "y"]),
    ("not json", None),
])
def test_parse_json_lenient(raw, expected):
    assert parse_json_lenient(raw) == expected


async def test_mock_generate_is_extractive_with_citations():
    g = LLMGateway(mock=True)
    prompt = "<sources>\n[1] Pressure was 1500 psi. More text.\n[2] Depth was 7000 ft.\n</sources>"
    out = await g.complete(prompt, model="x", task="generate")
    assert "[1]" in out and "[2]" in out
    assert g.usage.calls == 1 and g.usage.by_task["generate"] == 1


async def test_json_returns_default_on_garbage():
    g = LLMGateway(mock=True, mock_handler=lambda t, p: "¯\\_(ツ)_/¯")
    assert await g.json("p", model="x", task="route", default={"route": "retrieve"}) == {"route": "retrieve"}


async def test_json_survives_provider_failure():
    def boom(t, p):
        raise RuntimeError("provider down")
    g = LLMGateway(mock=True, mock_handler=boom)
    assert await g.json("p", model="x", default=[]) == []


async def test_stream_mock_yields_pieces():
    g = LLMGateway(mock=True, mock_handler=lambda t, p: "alpha beta gamma")
    pieces = [x async for x in g.stream("p", model="x")]
    assert "".join(pieces) == "alpha beta gamma" and len(pieces) == 3


def test_default_handler_no_sources_abstains():
    assert "could not find" in default_mock_handler("generate", "<sources>\n</sources>").lower()
