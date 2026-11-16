import { useCallback, useEffect, useRef, useState } from "react";
import { HttpError, isAbortError, streamChat } from "../lib/api";
import type { ChatEntry, Grounding, Mode, StreamChunk, Trace } from "../lib/types";

export type StreamStatus = "idle" | "connecting" | "streaming";

export interface PendingTurn {
  chatId: string;
  user: string;
  mode: Mode;
}

export interface LiveState {
  status: StreamStatus;
  pending: PendingTurn | null;
  answer: string;
  citations: string;
  related: string[];
  grounding?: Grounding;
  trace?: Trace;
}

function tryJson<T>(s: string): T | undefined {
  try {
    return JSON.parse(s) as T;
  } catch {
    return undefined;
  }
}

const IDLE: LiveState = { status: "idle", pending: null, answer: "", citations: "", related: [] };

function friendlyError(e: unknown): string {
  if (e instanceof HttpError) return `The server responded with ${e.status}. ${e.message}`.trim();
  if (e instanceof TypeError) return "Lost connection to the API (unreachable or interrupted). Is the retrieval service running?";
  return e instanceof Error ? e.message : String(e);
}

/**
 * Drives one streaming request. Text is accumulated in a ref and published to
 * React at most once per animation frame, so long answers don't re-render per token.
 */
export function useChatStream(onFinish: (chatId: string, entry: ChatEntry) => void) {
  const [live, setLive] = useState<LiveState>(IDLE);
  const abortRef = useRef<AbortController | null>(null);
  const runRef = useRef(0);
  const frameRef = useRef<number | null>(null);
  const acc = useRef({ answer: "", citations: "", related: [] as string[], gotFirst: false, grounding: undefined as Grounding | undefined, trace: undefined as Trace | undefined });
  const onFinishRef = useRef(onFinish);
  onFinishRef.current = onFinish;

  const publish = useCallback(() => {
    frameRef.current = null;
    const a = acc.current;
    setLive((p) =>
      p.pending ? { ...p, status: a.gotFirst ? "streaming" : p.status, answer: a.answer, citations: a.citations, related: [...a.related], grounding: a.grounding, trace: a.trace } : p,
    );
  }, []);

  const schedule = useCallback(() => {
    if (frameRef.current == null) frameRef.current = requestAnimationFrame(publish);
  }, [publish]);

  const cancelFrame = () => {
    if (frameRef.current != null) cancelAnimationFrame(frameRef.current);
    frameRef.current = null;
  };

  const apply = useCallback(
    (c: StreamChunk) => {
      const a = acc.current;
      switch (c.type) {
        case "greeting":
        case "tokens":
          a.answer += c.text;
          break;
        case "error":
          a.answer += (a.answer ? "\n\n" : "") + c.text;
          break;
        case "answer":
          a.answer = c.text; // authoritative final text replaces streamed tokens
          break;
        case "context":
          if (c.text && c.text !== "{}") a.citations = c.text;
          break;
        case "related":
          if (c.text) a.related.push(c.text);
          break;
        case "grounding":
          a.grounding = tryJson<Grounding>(c.text);
          break;
        case "trace":
          a.trace = tryJson<Trace>(c.text);
          break;
        default:
          return;
      }
      a.gotFirst = true;
      schedule();
    },
    [schedule],
  );

  const send = useCallback(
    async (chatId: string, message: string, mode: Mode) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      const run = ++runRef.current;
      cancelFrame();
      acc.current = { answer: "", citations: "", related: [], gotFirst: false, grounding: undefined, trace: undefined };
      const pending: PendingTurn = { chatId, user: message, mode };
      setLive({ ...IDLE, status: "connecting", pending });

      let stopped = false;
      let error: string | undefined;
      try {
        await streamChat({ mode, chatId, message, signal: controller.signal, onChunk: apply });
      } catch (e) {
        if (isAbortError(e)) stopped = true;
        else error = friendlyError(e);
      }
      if (run !== runRef.current) return; // superseded by a newer request

      cancelFrame();
      const a = acc.current;
      abortRef.current = null;
      setLive(IDLE);
      onFinishRef.current(chatId, {
        user: message,
        bot: a.answer,
        citations: a.citations,
        related: a.related,
        mode,
        ...(a.grounding ? { grounding: a.grounding } : {}),
        ...(a.trace ? { trace: a.trace } : {}),
        ...(stopped ? { stopped: true } : {}),
        ...(error ? { error } : {}),
      });
    },
    [apply],
  );

  const stop = useCallback(() => abortRef.current?.abort(), []);

  useEffect(
    () => () => {
      abortRef.current?.abort();
      cancelFrame();
    },
    [],
  );

  return { live, send, stop };
}
