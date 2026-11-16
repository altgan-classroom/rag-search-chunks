import { useEffect, useRef, useState } from "react";
import { Send, Stop } from "./icons";

interface Props {
  busy: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
  mode: "answer" | "report";
}

export function Composer({ busy, onSend, onStop, mode }: Props) {
  const [value, setValue] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [value]);

  const submit = () => {
    const t = value.trim();
    if (!t || busy) return;
    onSend(t);
    setValue("");
  };

  return (
    <div className="px-4 pb-4 pt-2">
      <div className="mx-auto max-w-3xl">
        <div className="flex items-end gap-2 rounded-2xl border border-line bg-surface p-2 shadow-pop transition focus-within:border-brand/60 focus-within:ring-4 focus-within:ring-brand/15">
          <textarea
            ref={ref}
            rows={1}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                submit();
              }
            }}
            placeholder={mode === "report" ? "Describe the report you want…" : "Ask a question about your documents…"}
            aria-label="Message"
            className="max-h-[200px] min-h-[40px] flex-1 resize-none bg-transparent px-3 py-2 text-[15px] leading-6 outline-none placeholder:text-muted focus-visible:ring-0 focus-visible:ring-offset-0"
          />
          {busy ? (
            <button
              onClick={onStop}
              aria-label="Stop generating"
              title="Stop"
              className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-fg text-bg transition hover:opacity-85 active:scale-95"
            >
              <Stop />
            </button>
          ) : (
            <button
              onClick={submit}
              disabled={!value.trim()}
              aria-label="Send message"
              title="Send"
              className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-brand-gradient text-white shadow-glow transition hover:brightness-110 active:scale-95 disabled:cursor-not-allowed disabled:opacity-40 disabled:shadow-none"
            >
              <Send />
            </button>
          )}
        </div>
        <p className="mt-2 text-center text-xs text-muted">Enter to send · Shift+Enter for a new line · Answers are generated from your documents</p>
      </div>
    </div>
  );
}
