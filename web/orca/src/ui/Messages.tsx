import { useState } from "react";
import { parseSources } from "../lib/citations";
import type { ChatEntry } from "../lib/types";
import { Alert, Arrow, Check, Copy, Logo, Refresh } from "./icons";
import { Markdown } from "./Markdown";
import { Sources } from "./Sources";

export function UserMessage({ text }: { text: string }) {
  return (
    <div className="flex animate-rise justify-end">
      <div className="max-w-[85%] whitespace-pre-wrap break-words rounded-2xl rounded-br-md bg-brand-gradient px-4 py-2.5 text-[15px] leading-7 text-white shadow-glow">
        {text}
      </div>
    </div>
  );
}

function Thinking() {
  return (
    <div className="flex items-center gap-3 py-1" role="status" aria-label="Searching documents">
      <span className="flex gap-1">
        {[0, 150, 300].map((d) => (
          <span key={d} className="h-2 w-2 animate-bounce rounded-full bg-brand-gradient" style={{ animationDelay: `${d}ms` }} />
        ))}
      </span>
      <span className="animate-shimmer bg-[linear-gradient(90deg,rgb(var(--muted)),rgb(var(--fg)),rgb(var(--muted)))] bg-[length:200%_100%] bg-clip-text text-sm text-transparent">
        Searching your documents…
      </span>
    </div>
  );
}

interface AssistantProps {
  entry: Pick<ChatEntry, "bot" | "citations" | "related" | "stopped" | "error" | "mode">;
  state: "connecting" | "streaming" | "done";
  isLast?: boolean;
  onAsk?: (q: string) => void;
  onRegenerate?: () => void;
}

export function AssistantMessage({ entry, state, isLast, onAsk, onRegenerate }: AssistantProps) {
  const [copied, setCopied] = useState(false);
  const sources = parseSources(entry.citations);
  const live = state !== "done";

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(entry.bot);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable */
    }
  };

  return (
    <div className="flex animate-rise gap-3">
      <Logo size={32} />
      <div className="min-w-0 flex-1">
        <div className="rounded-2xl rounded-tl-md border border-line bg-surface p-4 shadow-card">
          {entry.mode === "report" && !live && (
            <span className="mb-2 inline-block rounded-full bg-brand-soft px-2.5 py-0.5 text-xs font-semibold text-brand">Report</span>
          )}
          {state === "connecting" ? (
            <Thinking />
          ) : (
            <div aria-live="polite" aria-busy={live}>
              <Markdown text={entry.bot} sources={sources} streaming={state === "streaming"} />
            </div>
          )}
          {entry.stopped && <p className="mt-3 text-xs italic text-muted">Generation stopped.</p>}
          {entry.error && (
            <div role="alert" className="mt-3 flex items-start gap-2 rounded-xl border border-danger/30 bg-danger/10 p-3 text-sm text-danger">
              <Alert className="mt-0.5 shrink-0" />
              <span className="min-w-0 flex-1 break-words">{entry.error}</span>
              {onRegenerate && (
                <button onClick={onRegenerate} className="shrink-0 font-semibold underline underline-offset-2">Retry</button>
              )}
            </div>
          )}
          {!live && <Sources sources={sources} />}
        </div>

        {!live && entry.bot && (
          <div className="mt-1.5 flex items-center gap-0.5 pl-1">
            <button className="btn-icon" onClick={copy} aria-label="Copy answer" title="Copy">
              {copied ? <Check className="text-success" /> : <Copy />}
            </button>
            {isLast && onRegenerate && (
              <button className="btn-icon" onClick={onRegenerate} aria-label="Regenerate answer" title="Regenerate">
                <Refresh />
              </button>
            )}
          </div>
        )}

        {!live && isLast && !!entry.related?.length && onAsk && (
          <div className="mt-3">
            <h4 className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted">Related questions</h4>
            <div className="flex flex-wrap gap-2">
              {entry.related.map((q) => (
                <button
                  key={q}
                  onClick={() => onAsk(q)}
                  className="group inline-flex max-w-full items-center gap-1.5 rounded-full border border-line bg-surface px-3.5 py-1.5 text-left text-sm transition hover:-translate-y-0.5 hover:border-brand/50 hover:bg-brand-soft hover:shadow-card"
                >
                  <span className="min-w-0 break-words">{q}</span>
                  <Arrow width={14} height={14} className="shrink-0 text-muted transition group-hover:translate-x-0.5 group-hover:text-brand" />
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
