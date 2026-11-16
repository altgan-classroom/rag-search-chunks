export type Mode = "answer" | "report";

/** One chunk of the backend stream. Unknown `type`s are ignored by the UI. */
export interface StreamChunk {
  response_id?: string;
  type: string;
  text: string;
  chat_id?: string;
  metadata?: unknown;
}

export interface Grounding {
  score: number;
  ok: boolean;
  unsupported: string[];
  invalid_citations: number[];
  uncited: number;
  llm_unsupported?: string[] | null;
}

export interface TraceStep {
  name: string;
  ms: number;
  [k: string]: unknown;
}

export interface Trace {
  profile: string;
  route: string;
  queries: string[];
  best_score: number | null;
  total_ms: number;
  steps: TraceStep[];
  abstained?: boolean;
  reason?: string;
}

/** Persisted shape. Compatible with chats saved by the previous UI. */
export interface ChatEntry {
  user: string;
  bot: string;
  citations: string;
  related?: string[];
  mode?: Mode;
  /** True when the user pressed Stop before the answer finished. */
  stopped?: boolean;
  /** Set when the stream failed (partial text is kept in `bot`). */
  error?: string;
  /** Grounding verdict and retrieval trace emitted by the v2 backend (absent on older backends). */
  grounding?: Grounding;
  trace?: Trace;
}

export interface Chat {
  title: string;
  chatEntries: ChatEntry[];
}

export type ChatMap = Record<string, Chat>;
