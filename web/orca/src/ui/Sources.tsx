import type { Source } from "../lib/citations";
import { Doc } from "./icons";

export function Sources({ sources }: { sources: Source[] }) {
  if (!sources.length) return null;
  return (
    <section aria-label="Sources" className="mt-5">
      <h4 className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted">Sources</h4>
      <ul className="grid gap-2 sm:grid-cols-2">
        {sources.map((s) => (
          <li key={s.index}>
            <a
              href={s.url}
              target="_blank"
              rel="noopener noreferrer"
              className="group flex items-center gap-3 rounded-xl border border-line bg-surface2/60 p-2.5 transition hover:-translate-y-0.5 hover:border-brand/50 hover:bg-surface hover:shadow-card"
            >
              <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-brand-gradient text-xs font-bold text-white">{s.index}</span>
              <span className="min-w-0 flex-1">
                <span className="block truncate text-sm font-medium">{s.docName}.pdf</span>
                {s.page && <span className="text-xs text-muted">Page {s.page}</span>}
              </span>
              <Doc className="shrink-0 text-muted transition group-hover:text-brand" />
            </a>
          </li>
        ))}
      </ul>
    </section>
  );
}
