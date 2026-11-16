import { API_KEY, API_URL } from "./config";
import { StreamParser } from "./streamParser";
import type { Mode, StreamChunk } from "./types";

export class HttpError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = "HttpError";
  }
}

export interface StreamChatOptions {
  mode: Mode;
  chatId: string;
  message: string;
  signal?: AbortSignal;
  /** Called for every chunk as soon as it is parsed, inside the read loop. */
  onChunk: (chunk: StreamChunk) => void;
}

/** POSTs to the retrieval API and streams chunks to `onChunk`. */
export async function streamChat({ mode, chatId, message, signal, onChunk }: StreamChatOptions): Promise<void> {
  const endpoint = mode === "report" ? "v1/report" : "v1/chat";
  const res = await fetch(`${API_URL}/${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(API_KEY ? { "X-API-Key": API_KEY } : {}) },
    body: JSON.stringify({ chat_id: chatId, message, metadata: {} }),
    signal,
  });

  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new HttpError(res.status, body || res.statusText || `HTTP ${res.status}`);
  }
  if (!res.body) throw new Error("The server returned an empty response body.");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  const parser = new StreamParser();

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      for (const chunk of parser.push(decoder.decode(value, { stream: true }))) onChunk(chunk);
    }
    for (const chunk of parser.push(decoder.decode())) onChunk(chunk);
  } finally {
    reader.releaseLock?.();
  }
  if (parser.end().truncated) console.warn("[stream] ended in the middle of a chunk");
}

export const isAbortError = (e: unknown) =>
  (e instanceof DOMException && e.name === "AbortError") || (e instanceof Error && e.name === "AbortError");
