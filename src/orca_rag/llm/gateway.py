"""Provider-agnostic LLM access (LiteLLM) with an offline mock.

* All model ids come from the Profile, never from code.
* `LLMGateway.complete / stream / json` are the only call sites in the codebase.
* With `mock=True` (or no handler needed) calls are deterministic and offline: tests, CI and keyless demos.
* Usage (tokens, calls, latency) is accumulated so the eval harness can report cost per technique.
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

MockHandler = Callable[[str, str], str]  # (task, prompt) -> completion


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    by_task: dict[str, int] = field(default_factory=dict)

    def add(self, task: str, pt: int, ct: int, dt: float) -> None:
        self.calls += 1
        self.prompt_tokens += pt
        self.completion_tokens += ct
        self.seconds += dt
        self.by_task[task] = self.by_task.get(task, 0) + 1


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)


def parse_json_lenient(text: str):
    """Parse model JSON even when wrapped in fences or prose. Returns None on failure (callers must fall back)."""
    t = _FENCE.sub("", text.strip())
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        i, j = t.find(open_c), t.rfind(close_c)
        if i != -1 and j > i:
            try:
                return json.loads(t[i : j + 1])
            except json.JSONDecodeError:
                continue
    return None


def _approx_tokens(s: str) -> int:
    return max(1, len(s) // 4)


class LLMGateway:
    def __init__(self, *, mock: bool = False, mock_handler: MockHandler | None = None,
                 num_retries: int = 3, drop_params: bool = True):
        self.mock = mock
        self._handler = mock_handler
        self.num_retries = num_retries
        self.drop_params = drop_params
        self.usage = Usage()

    @property
    def available(self) -> bool:
        """True when calls can succeed (mock, or a provider key is configured)."""
        if self.mock:
            return True
        import os
        return any(os.environ.get(k) for k in (
            "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY", "AZURE_API_KEY",
            "COHERE_API_KEY", "MISTRAL_API_KEY", "GROQ_API_KEY", "OLLAMA_API_BASE"))

    # ------------------------------------------------------------------ core
    async def complete(self, prompt: str, *, model: str, task: str = "generic", system: str | None = None,
                       temperature: float = 0.0, max_tokens: int = 512, timeout_s: float = 60.0) -> str:
        t0 = time.perf_counter()
        if self.mock:
            out = (self._handler or default_mock_handler)(task, prompt)
            self.usage.add(task, _approx_tokens(prompt), _approx_tokens(out), time.perf_counter() - t0)
            return out
        import litellm

        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        resp = await litellm.acompletion(model=model, messages=messages, temperature=temperature,
                                         max_tokens=max_tokens, timeout=timeout_s,
                                         num_retries=self.num_retries, drop_params=self.drop_params)
        out = resp.choices[0].message.content or ""
        u = getattr(resp, "usage", None)
        self.usage.add(task, getattr(u, "prompt_tokens", 0) or _approx_tokens(prompt),
                       getattr(u, "completion_tokens", 0) or _approx_tokens(out), time.perf_counter() - t0)
        return out

    async def stream(self, prompt: str, *, model: str, task: str = "generate", system: str | None = None,
                     temperature: float = 0.1, max_tokens: int = 1200, timeout_s: float = 60.0) -> AsyncIterator[str]:
        t0 = time.perf_counter()
        if self.mock:
            out = (self._handler or default_mock_handler)(task, prompt)
            # emit word-sized pieces so the UI path is exercised realistically
            for piece in re.findall(r"\S+\s*|\s+", out):
                yield piece
            self.usage.add(task, _approx_tokens(prompt), _approx_tokens(out), time.perf_counter() - t0)
            return
        import litellm

        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        resp = await litellm.acompletion(model=model, messages=messages, temperature=temperature,
                                         max_tokens=max_tokens, timeout=timeout_s, stream=True,
                                         num_retries=self.num_retries, drop_params=self.drop_params)
        n = 0
        async for part in resp:
            delta = part.choices[0].delta.content if part.choices else None
            if delta:
                n += 1
                yield delta
        self.usage.add(task, _approx_tokens(prompt), n, time.perf_counter() - t0)

    async def json(self, prompt: str, *, model: str, task: str = "json", default=None, **kw):
        """Structured call that never raises on bad model output: returns `default` instead."""
        try:
            text = await self.complete(prompt, model=model, task=task, **kw)
        except Exception as e:  # network/provider failure must not take the request down
            logger.warning("LLM json call failed (%s): %s", task, type(e).__name__)
            return default
        parsed = parse_json_lenient(text)
        return default if parsed is None else parsed


# ---------------------------------------------------------------------- offline mock behaviour
_CTX = re.compile(r"\[(\d+)\]\s*(.+?)(?=\n\[\d+\]|\Z)", re.S)


def default_mock_handler(task: str, prompt: str) -> str:
    """Deterministic stand-in. For `generate` it writes an *extractive* answer with citations so the full
    pipeline, API and UI can be exercised without a key. Other tasks return conservative no-op answers."""
    if task == "generate":
        ctx = prompt.split("<sources>")[-1].split("</sources>")[0]
        items = _CTX.findall(ctx)
        if not items:
            return "I could not find this in the provided documents."
        parts = []
        for n, body in items[:3]:
            sent = re.split(r"(?<=[.!?])\s+", body.strip().replace("\n", " "))[0][:300]
            parts.append(f"{sent} [{n}]")
        return " ".join(parts)
    if task == "route":
        return json.dumps({"route": "retrieve"})
    if task == "decompose":
        return json.dumps([])
    if task == "rewrite":
        q = re.search(r"Question:\s*(.+)\s*$", prompt, re.S)
        return q.group(1).strip() if q else ""
    if task == "grade":
        return json.dumps({"sufficient": True, "reason": "mock"})
    if task == "verify":
        return json.dumps({"supported": True, "unsupported_claims": []})
    return ""
