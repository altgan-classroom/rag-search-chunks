import { useMemo, useState } from "react";
import type { ChatMap } from "../lib/types";
import { Close, Logo, Pencil, Plus, Search, Trash } from "./icons";

interface Props {
  chats: ChatMap;
  activeId: string;
  open: boolean;
  onClose: () => void;
  onSelect: (id: string) => void;
  onNew: () => void;
  onRename: (id: string, title: string) => void;
  onDelete: (id: string) => void;
}

const DAY = 86_400_000;
function group(id: string): string {
  const ts = Number(id.replace(/\D/g, ""));
  if (!ts) return "Earlier";
  const age = Date.now() - ts;
  return age < DAY ? "Today" : age < 2 * DAY ? "Yesterday" : age < 7 * DAY ? "Previous 7 days" : "Earlier";
}
const ORDER = ["Today", "Yesterday", "Previous 7 days", "Earlier"];

export function Sidebar({ chats, activeId, open, onClose, onSelect, onNew, onRename, onDelete }: Props) {
  const [q, setQ] = useState("");
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState("");

  const groups = useMemo(() => {
    const needle = q.trim().toLowerCase();
    const map = new Map<string, [string, string][]>();
    Object.keys(chats)
      .sort()
      .reverse()
      .forEach((id) => {
        const title = chats[id].title;
        if (needle && !title.toLowerCase().includes(needle)) return;
        const g = group(id);
        map.set(g, [...(map.get(g) ?? []), [id, title]]);
      });
    return ORDER.filter((g) => map.has(g)).map((g) => [g, map.get(g)!] as const);
  }, [chats, q]);

  const commit = () => {
    if (editing) onRename(editing, draft);
    setEditing(null);
  };

  return (
    <>
      {open && <div className="fixed inset-0 z-30 bg-black/40 backdrop-blur-sm md:hidden" onClick={onClose} aria-hidden />}
      <aside
        aria-label="Chat history"
        className={`fixed inset-y-0 left-0 z-40 flex w-72 flex-col border-r border-line bg-surface transition-transform duration-200 md:static md:z-auto md:translate-x-0 ${
          open ? "translate-x-0" : "-translate-x-full md:hidden"
        }`}
      >
        <div className="flex items-center gap-3 px-4 pb-3 pt-4">
          <Logo />
          <div className="flex-1 leading-tight">
            <div className="font-bold tracking-tight">Orca</div>
            <div className="text-xs text-muted">Document Q&amp;A</div>
          </div>
          <button className="btn-icon md:hidden" onClick={onClose} aria-label="Close sidebar"><Close /></button>
        </div>

        <div className="space-y-2 px-3">
          <button
            onClick={() => { onNew(); onClose(); }}
            className="flex w-full items-center justify-center gap-2 rounded-xl bg-brand-gradient py-2.5 text-sm font-semibold text-white shadow-glow transition hover:brightness-110 active:scale-[.98]"
          >
            <Plus width={16} height={16} /> New chat
          </button>
          <label className="flex items-center gap-2 rounded-xl border border-line bg-surface2/60 px-3 py-2 text-muted focus-within:border-brand/50">
            <Search width={16} height={16} />
            <input
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="Search chats"
              aria-label="Search chats"
              className="w-full bg-transparent text-sm text-fg outline-none placeholder:text-muted"
            />
          </label>
        </div>

        <nav className="mt-3 flex-1 overflow-y-auto px-2 pb-4">
          {groups.length === 0 && <p className="px-3 py-6 text-center text-sm text-muted">No chats found</p>}
          {groups.map(([label, items]) => (
            <div key={label} className="mb-3">
              <h3 className="px-3 pb-1 pt-2 text-[11px] font-semibold uppercase tracking-wider text-muted">{label}</h3>
              <ul className="space-y-0.5">
                {items.map(([id, title]) => {
                  const active = id === activeId;
                  return (
                    <li key={id} className="group relative">
                      {editing === id ? (
                        <input
                          autoFocus
                          value={draft}
                          onChange={(e) => setDraft(e.target.value)}
                          onBlur={commit}
                          onKeyDown={(e) => {
                            if (e.key === "Enter") commit();
                            if (e.key === "Escape") setEditing(null);
                          }}
                          aria-label="Rename chat"
                          className="w-full rounded-lg border border-brand/50 bg-surface px-3 py-2 text-sm outline-none"
                        />
                      ) : (
                        <>
                          <button
                            onClick={() => { onSelect(id); onClose(); }}
                            aria-current={active ? "page" : undefined}
                            className={`w-full truncate rounded-lg px-3 py-2 pr-16 text-left text-sm transition ${
                              active ? "bg-brand-soft font-medium text-brand" : "hover:bg-surface2"
                            }`}
                          >
                            {title}
                          </button>
                          <span className="absolute right-1 top-1/2 flex -translate-y-1/2 opacity-0 transition focus-within:opacity-100 group-hover:opacity-100 max-md:opacity-100">
                            <button className="btn-icon h-7 w-7" aria-label={`Rename ${title}`} onClick={() => { setEditing(id); setDraft(title); }}><Pencil width={14} height={14} /></button>
                            <button
                              className="btn-icon h-7 w-7 hover:text-danger"
                              aria-label={`Delete ${title}`}
                              onClick={() => { if (confirm("Delete this chat?")) onDelete(id); }}
                            ><Trash width={14} height={14} /></button>
                          </span>
                        </>
                      )}
                    </li>
                  );
                })}
              </ul>
            </div>
          ))}
        </nav>
      </aside>
    </>
  );
}
