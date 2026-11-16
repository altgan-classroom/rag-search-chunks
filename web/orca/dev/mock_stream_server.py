"""Mock retrieval API for developing the UI without the full RAG stack.

    pip install fastapi uvicorn
    python web/orca/dev/mock_stream_server.py          # http://localhost:8080

Emits the same chunk types as the real API (greeting/tokens/context/related/answer).
Pick the wire framing with ?framing=concat|ndjson|sse and the speed with ?delay=0.04.
"""
import asyncio
import json

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ANSWER = (
    "## Quarterly summary\n\n"
    "Revenue grew **12%** year over year [1], driven mainly by cloud services. "
    "Operating margin held steady at 42% [[2]].\n\n"
    "| Segment | Revenue | Growth |\n|---|---|---|\n| Cloud | $24.1B | 20% |\n| Devices | $5.3B | -3% |\n\n"
    "Key risks:\n\n- Currency headwinds (`FX {USD}`)\n- Supply constraints\n- Regulatory scrutiny\n\n"
    "> Note: braces like { and } inside answers must not break the stream parser.\n"
)
CONTEXT = {
    "0": {"doc_name": "msft-q1-24-press.md", "page_num": 3, "chunk": "Revenue was $61.9B"},
    "1": {"doc_name": "msft-q1-24-press.md", "page_num": 7, "chunk": "Operating income"},
}
RELATED = ["What drove cloud growth?", "How did margins compare to last year?", "What guidance was given?"]


def frame(chunk: dict, framing: str) -> bytes:
    body = json.dumps(chunk)
    if framing == "ndjson":
        return (body + "\n").encode()
    if framing == "sse":
        return f"data: {body}\n\n".encode()
    return body.encode()


async def stream(message: str, framing: str, delay: float, report: bool):
    def c(t, text):
        return {"response_id": "mock", "type": t, "text": text, "chat_id": "mock"}

    await asyncio.sleep(0.8)  # retrieval latency
    if "fail" in message.lower():
        yield frame(c("tokens", "Starting to answer, then"), framing)
        await asyncio.sleep(0.3)
        raise RuntimeError("simulated mid-stream failure")
    yield frame(c("context", json.dumps(CONTEXT)), framing)
    text = ("# Report\n\n" if report else "") + ANSWER + f"\n(You asked: {message})"
    # Split into small pieces, keeping whitespace attached, like real LLM tokens.
    i = 0
    while i < len(text):
        step = 3 + (i % 4)
        yield frame(c("tokens", text[i : i + step]), framing)
        i += step
        await asyncio.sleep(delay)
    yield frame(c("answer", text), framing)  # final authoritative text
    for q in RELATED:
        yield frame(c("related", q), framing)
    if framing == "sse":
        yield b"data: [DONE]\n\n"


def endpoint(report: bool):
    async def handler(request: Request, framing: str = "concat", delay: float = 0.04):
        data = await request.json()
        return StreamingResponse(
            stream(data["message"], framing, delay, report),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return handler


app.post("/v1/chat")(endpoint(False))
app.post("/v1/report")(endpoint(True))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
