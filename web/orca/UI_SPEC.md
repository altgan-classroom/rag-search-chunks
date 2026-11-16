# Orca Web UI: Revamp & Streaming Spec

Status: implemented on branch `feat/ui-revamp-streaming`
Scope: `web/orca` only. The backend contract, endpoints and Docker topology are unchanged.

## 1. Problems with the current UI

| # | Problem | Root cause |
|---|---------|-----------|
| P1 | The answer appears all at once | `App.tsx` reads the whole response into `buffer` and parses JSON only **after** `done`. `setCurrentAnswer` is never called inside the read loop, so nothing renders until the stream ends. |
| P2 | Fragile chunk parsing | The brace counter ignores string literals, so a `{` or `}` inside answer text or the `context` payload corrupts parsing. It only understands back-to-back JSON objects. |
| P3 | Stop discards work | Aborting drops the partial answer instead of saving it. |
| P4 | Flat, dated look | An 839-line `App.tsx` with inline SVGs, a one-colour palette, a global `* { margin: unset }` reset, and Tailwind `md:`/`sm:` redefined as *max-width* breakpoints. |
| P5 | Config is hard-coded | API URL is `http://localhost:8080`. docker-compose passes `REACT_APP_API_URL`, which Vite ignores. |

## 2. Goals

1. **True token streaming.** Text renders as tokens arrive (first token visible in under 100 ms after it reaches the browser) with a live caret.
2. **A rich, modern look.** A defined colour system with gradients, glass surfaces, elevation, motion, and light and dark themes.
3. **Componentised code.** Small single-purpose components, a typed stream client, and hooks instead of one giant file.
4. **Zero backend changes.** Students implement `generate.py` themselves, so the UI must tolerate their framing.

Non-goals: auth, new backend endpoints, a component library, changes to the PDF viewer route (`/file`).

## 3. Contract that must not change

* `POST {API}/v1/chat` (answer mode) and `POST {API}/v1/report` (report mode)
* Request body: `{ "chat_id": string, "message": string, "metadata": {} }`
* Chunk shape: `{ response_id, type, text, chat_id?, metadata? }`

| `type` | UI behaviour |
|--------|-------------|
| `greeting`, `tokens`, `error` | Append `text` to the answer (errors are styled as an error). |
| `answer` | Replace the accumulated answer with `text`. Do this without a flash: swap the content in place and keep scroll position. |
| `context` | `text` is a JSON string `{ "<k>": { doc_name, page_num, chunk? } }`. Ignore `"{}"`. Rendered as source cards. |
| `related` | One suggested follow-up question per chunk. |
| anything else | Ignored (forward compatible). |

Existing features to keep: citation links to `<DOCS_BASE_URL>/<doc>.pdf#page=N`, inline `[n]` citation markers, related questions, chat history in `localStorage` (key `chats`, same shape), rename and delete chat, answer/report mode, stop, regenerate, copy.

## 4. Streaming design

### 4.1 Incremental parser (`src/lib/streamParser.ts`)
A pure, unit-testable class: `push(text) -> StreamChunk[]`.

* Decodes bytes with `TextDecoder({ stream: true })` so multi-byte characters split across reads stay intact.
* **String-aware** JSON object scanner. It tracks `inString` and `\` escapes, so braces inside strings never count.
* Accepts all of the following framings (students' backends vary):
  1. Concatenated objects: `{...}{...}` (the current template behaviour)
  2. NDJSON: `{...}\n{...}\n`
  3. SSE: `data: {...}\n\n`. Strips the `data:` prefix and ignores `[DONE]`, comments and `event:` lines.
* Keeps incomplete trailing data for the next `push`, and never throws on bad JSON (it logs and skips).

### 4.2 Stream client (`src/lib/api.ts`)
`streamChat({ mode, chatId, message, signal, onChunk })` reads the body with `getReader()` and calls `onChunk` **inside the read loop**. It throws a typed `HttpError` on non-2xx and treats `AbortError` as a normal stop.

### 4.3 State (`src/hooks/useChatStream.ts`)
* Accumulates text in a ref and publishes it to React state at most once per animation frame (`requestAnimationFrame`), so long answers do not cause a re-render per token.
* Exposes `status: idle | connecting | streaming | error`, `answer`, `sources`, `related`, `send`, `stop`.
* **Stop** keeps the partial answer, marks it `stopped`, and saves it to history.
* Starting a new request aborts any in-flight one.
* Failures after partial output keep the partial text and show an inline error with a Retry action.

### 4.4 Rendering
* Markdown via the existing `react-markdown` + `remark-gfm`, with styled headings, lists, tables, code blocks and blockquotes.
* A blinking caret is shown while `status === "streaming"`.
* A "Thinking…" shimmer shows while `connecting` (before the first token).
* Auto-scroll sticks to the bottom only while the user is already near the bottom. If they scroll up, show a floating **Jump to latest** button.
* `aria-live="polite"` on the answer region. The caret animation respects `prefers-reduced-motion`.

## 5. Visual design

### 5.1 Tokens (CSS variables, swapped by `.dark` on `<html>`)
* **Brand:** indigo to violet to cyan gradient (`#4f46e5` → `#7c3aed` → `#06b6d4`).
* **Surfaces:** `bg`, `surface`, `surface-2`, `border`, `muted`, `fg`.
* **Semantic:** success, warning, danger.
* **Radius:** 12 / 16 / 24 px. **Elevation:** three shadow levels. **Motion:** 150–250 ms ease-out.
* Theme defaults to the OS preference, is toggleable, and persists to `localStorage` (guarded with try/catch).
* Font: system UI stack plus a monospace stack for code.

### 5.2 Layout
```
+-----------+----------------------------------------------+
| Sidebar   |  Header: title · mode switch · theme toggle  |
|  brand    |----------------------------------------------|
|  New chat |                                              |
|  search   |   Messages (max-w 3xl, centred)              |
|  history  |    user bubble  (gradient)                   |
|  (grouped |    assistant card (markdown + caret)         |
|  by date) |      sources row · related chips · actions   |
|           |                                              |
|           |----------------------------------------------|
|           |  Composer (auto-grow, Enter send, ⏎ newline) |
+-----------+----------------------------------------------+
```
* Empty state: gradient hero, short tagline, four clickable starter prompts.
* Sidebar: collapsible. On narrow screens it becomes an overlay drawer.
* User message: right-aligned gradient bubble. Assistant message: bordered surface card with a gradient avatar.
* Sources: horizontal cards with a numbered badge, document name and page chip. The card opens the PDF at that page.
* Related questions: pill chips with hover lift.
* Message actions: copy, regenerate (last answer only), shown with subtle icon buttons.
* Composer: rounded glass container with focus ring. The send button turns into a **Stop** button while streaming.
* Answer/Report mode: a segmented control in the header.

### 5.3 Responsiveness & accessibility
* Standard Tailwind breakpoints (the `max-width` overrides are removed).
* Visible `:focus-visible` rings, 4.5:1 text contrast in both themes, labelled icon buttons, keyboard-navigable sidebar and chips.

## 6. Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `VITE_API_URL` | `http://localhost:8080` | Retrieval API base URL |
| `VITE_DOCS_BASE_URL` | `https://rag-prod-ingested-docs.s3.ap-south-1.amazonaws.com` | Where cited PDFs live |

Vite reads these at **build time**. `docker-compose.yml` replaces `REACT_APP_API_URL` with `VITE_API_URL` and passes it as a build arg. The browser calls the API directly (CORS is already `*`), so nginx never proxies the stream.

## 7. Dev tooling

`web/orca/dev/mock_stream_server.py` is a small FastAPI mock of `/v1/chat` and `/v1/report` on port 8080. It emits `greeting`, `tokens` (about 40 ms apart), `context`, `related` and `answer` chunks in a selectable framing (`?framing=concat|ndjson|sse`). Use it to see streaming without running the whole stack.

## 8. Acceptance criteria

1. With the mock server, tokens appear progressively, and the first token renders before the response completes.
2. The parser unit tests pass for: split chunks, braces and escaped quotes inside strings, multi-byte characters split across reads, all three framings, `[DONE]`, and malformed JSON.
3. Stop mid-answer keeps and saves the partial text.
4. All existing features in section 3 still work, and old `localStorage` chats load.
5. `npm run build` (tsc + vite) passes. No console errors in either theme. Works at 360 px width.
6. No change to the request body, endpoints or chunk types.

## 9. Out of scope / follow-ups

* (Done) Legacy components that nothing imports any more were deleted. `Citation.tsx` and `reactPDF/` stay because the `/file` PDF viewer uses them.
* Code-split the PDF viewer (the bundle is over 900 kB).
* Persist chats server-side.
