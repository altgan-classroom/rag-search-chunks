import type { Mode } from "../lib/types";
import { Menu, Moon, Sun } from "./icons";

interface Props {
  title: string;
  mode: Mode;
  onMode: (m: Mode) => void;
  disabled: boolean;
  theme: "light" | "dark";
  onTheme: () => void;
  onMenu: () => void;
}

export function Header({ title, mode, onMode, disabled, theme, onTheme, onMenu }: Props) {
  return (
    <header className="glass sticky top-0 z-20 flex h-14 shrink-0 items-center gap-2 border-b border-line px-3 sm:px-4">
      <button className="btn-icon" onClick={onMenu} aria-label="Toggle sidebar"><Menu /></button>
      <h2 className="min-w-0 flex-1 truncate text-sm font-semibold">{title}</h2>
      <div role="radiogroup" aria-label="Response mode" className="flex rounded-xl border border-line bg-surface2/70 p-0.5 text-sm">
        {(["answer", "report"] as const).map((m) => (
          <button
            key={m}
            role="radio"
            aria-checked={mode === m}
            disabled={disabled}
            onClick={() => onMode(m)}
            className={`rounded-[10px] px-3 py-1 font-medium capitalize transition disabled:cursor-not-allowed ${
              mode === m ? "bg-brand-gradient text-white shadow-glow" : "text-muted hover:text-fg"
            }`}
          >
            {m}
          </button>
        ))}
      </div>
      <button className="btn-icon" onClick={onTheme} aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}>
        {theme === "dark" ? <Sun /> : <Moon />}
      </button>
    </header>
  );
}
