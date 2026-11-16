import type { Grounding, Trace } from "../lib/types";

/** Grounding verdict + "how was this answered" (retrieval steps and timings) from the v2 backend. */
export function Insights({ grounding, trace }: { grounding?: Grounding; trace?: Trace }) {
  if (!grounding && !trace) return null;
  const bad = grounding && (!grounding.ok || grounding.invalid_citations.length > 0);
  return (
    <div className="mt-4 flex flex-col gap-2 border-t border-line pt-3 text-xs">
      <div className="flex flex-wrap items-center gap-2">
        {trace?.abstained && (
          <span className="rounded-full bg-surface2 px-2.5 py-1 font-medium text-muted">No supporting passages found</span>
        )}
        {grounding && (
          <span
            title={bad ? `Possibly unsupported: ${grounding.unsupported.join(" | ") || "invalid citation"}` : "Every cited statement is supported by the retrieved sources"}
            className={`rounded-full px-2.5 py-1 font-semibold ${bad ? "bg-amber-500/15 text-amber-600 dark:text-amber-400" : "bg-success/15 text-success"}`}
          >
            {bad ? "Check sources" : "Grounded"} · {Math.round(grounding.score * 100)}%
          </span>
        )}
        {trace && <span className="text-muted">{trace.profile} · {(trace.total_ms / 1000).toFixed(1)}s</span>}
      </div>
      {trace && trace.steps.length > 0 && (
        <details className="group">
          <summary className="cursor-pointer select-none text-muted hover:text-fg">How this was answered</summary>
          <ol className="mt-2 space-y-1 text-muted">
            {trace.steps.map((s, i) => (
              <li key={i} className="flex gap-2">
                <span className="w-24 shrink-0 font-mono text-fg">{s.name}</span>
                <span className="w-14 shrink-0 text-right font-mono">{Math.round(s.ms)} ms</span>
                <span className="min-w-0 break-words">
                  {Object.entries(s).filter(([k]) => k !== "name" && k !== "ms").map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v) : String(v)}`).join("  ")}
                </span>
              </li>
            ))}
          </ol>
        </details>
      )}
    </div>
  );
}
