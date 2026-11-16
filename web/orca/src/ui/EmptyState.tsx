import { Arrow, Logo } from "./icons";

const STARTERS = [
  { t: "Summarize the key points", d: "A quick overview of the documents" },
  { t: "What are the main risks mentioned?", d: "Surface risks and caveats" },
  { t: "Compare figures across quarters", d: "Tables and trends with sources" },
  { t: "Who are the key people or entities?", d: "Extract names and roles" },
];

export function EmptyState({ onPick }: { onPick: (q: string) => void }) {
  return (
    <div className="flex min-h-full">
    <div className="relative m-auto flex w-full max-w-3xl flex-col items-center px-4 py-12 text-center">
      <div aria-hidden className="pointer-events-none absolute inset-0 -z-10 overflow-hidden">
        <div className="absolute left-[10%] top-[8%] h-64 w-64 animate-blob rounded-full bg-indigo-500/20 blur-3xl" />
        <div className="absolute bottom-[10%] right-[8%] h-72 w-72 animate-blob rounded-full bg-cyan-400/20 blur-3xl [animation-delay:-6s]" />
      </div>
      <Logo size={56} />
      <h1 className="mt-6 text-3xl font-bold tracking-tight sm:text-4xl">
        Ask anything about <span className="gradient-text">your documents</span>
      </h1>
      <p className="mt-3 max-w-md text-muted">Answers stream in live, grounded in your ingested files, with sources you can open at the exact page.</p>
      <div className="mt-8 grid w-full gap-3 sm:grid-cols-2">
        {STARTERS.map((s) => (
          <button
            key={s.t}
            onClick={() => onPick(s.t)}
            className="group flex items-center justify-between gap-3 rounded-2xl border border-line bg-surface/80 p-4 text-left shadow-card backdrop-blur transition hover:-translate-y-0.5 hover:border-brand/50 hover:shadow-pop"
          >
            <span>
              <span className="block text-sm font-semibold">{s.t}</span>
              <span className="text-xs text-muted">{s.d}</span>
            </span>
            <Arrow className="shrink-0 text-muted transition group-hover:translate-x-1 group-hover:text-brand" />
          </button>
        ))}
      </div>
    </div>
    </div>
  );
}
