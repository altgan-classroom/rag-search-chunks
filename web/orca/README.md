# Orca web UI

React + Vite + Tailwind chat UI for the RAG retrieval API. Answers stream token by token.
Design and behaviour are specified in [UI_SPEC.md](./UI_SPEC.md).

## Run

```bash
npm ci
npm run dev          # http://localhost:5173
npm test             # stream parser unit tests
npm run build        # tsc + vite build
```

Without the full RAG stack, start the mock backend (needs `pip install fastapi uvicorn`):

```bash
python dev/mock_stream_server.py     # http://localhost:8080
# try ?framing=sse|ndjson|concat and ?delay=0.04 by editing the request URL in src/lib/api.ts
# send a message containing the word "fail" to see mid-stream error handling
```

## Configuration (build time)

| Variable | Default |
|---|---|
| `VITE_API_URL` | `http://localhost:8080` |
| `VITE_DOCS_BASE_URL` | `https://rag-prod-ingested-docs.s3.ap-south-1.amazonaws.com` |

## Layout

- `src/lib/streamParser.ts` incremental, string-aware parser (concatenated JSON, NDJSON, SSE)
- `src/lib/api.ts` fetch + read loop that emits chunks as they arrive
- `src/hooks/` chat history, streaming state (rAF-throttled), theme
- `src/ui/` presentational components
