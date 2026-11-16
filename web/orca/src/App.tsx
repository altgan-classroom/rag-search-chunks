import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { useChats } from "./hooks/useChats";
import { useChatStream } from "./hooks/useChatStream";
import { useTheme } from "./hooks/useTheme";
import type { Mode } from "./lib/types";
import { Composer } from "./ui/Composer";
import { EmptyState } from "./ui/EmptyState";
import { Header } from "./ui/Header";
import { ArrowDown } from "./ui/icons";
import { AssistantMessage, UserMessage } from "./ui/Messages";
import { Sidebar } from "./ui/Sidebar";

const NEAR_BOTTOM = 120;

export default function App() {
  const { chats, activeId, setActiveId, newChat, rename, remove, append, dropLast } = useChats();
  const { theme, toggle } = useTheme();
  const [mode, setMode] = useState<Mode>("answer");
  const [sidebar, setSidebar] = useState(() => window.innerWidth >= 768);
  const { live, send, stop } = useChatStream(append);

  useEffect(() => {
    document.title = "Orca · Document Q&A";
  }, []);

  const chat = chats[activeId];
  const entries = chat?.chatEntries ?? [];
  const busy = live.status !== "idle";
  const livePending = live.pending && live.pending.chatId === activeId ? live.pending : null;

  // --- scrolling: follow the stream only while the user is near the bottom
  const scroller = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  const [showJump, setShowJump] = useState(false);

  const scrollToBottom = useCallback((smooth = false) => {
    const el = scroller.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: smooth ? "smooth" : "auto" });
  }, []);

  const onScroll = () => {
    const el = scroller.current;
    if (!el) return;
    const near = el.scrollHeight - el.scrollTop - el.clientHeight < NEAR_BOTTOM;
    stick.current = near;
    setShowJump(!near);
  };

  useLayoutEffect(() => {
    if (stick.current) scrollToBottom();
  }, [live.answer, live.status, entries.length, scrollToBottom]);

  useEffect(() => {
    stick.current = true;
    scrollToBottom();
  }, [activeId, scrollToBottom]);

  const ask = useCallback(
    (text: string) => {
      if (busy) return;
      stick.current = true;
      send(activeId, text, mode);
    },
    [busy, send, activeId, mode],
  );

  const regenerate = useCallback(() => {
    const last = entries[entries.length - 1];
    if (!last || busy) return;
    dropLast(activeId);
    stick.current = true;
    send(activeId, last.user, last.mode ?? mode);
  }, [entries, busy, dropLast, send, activeId, mode]);

  return (
    <>
      <div className="flex h-full overflow-hidden bg-bg">
        <Sidebar
          chats={chats}
          activeId={activeId}
          open={sidebar}
          onClose={() => window.innerWidth < 768 && setSidebar(false)}
          onSelect={setActiveId}
          onNew={newChat}
          onRename={rename}
          onDelete={remove}
        />
        <main className="relative flex min-w-0 flex-1 flex-col">
          <Header
            title={chat?.title ?? "New chat"}
            mode={mode}
            onMode={setMode}
            disabled={busy}
            theme={theme}
            onTheme={toggle}
            onMenu={() => setSidebar((s) => !s)}
          />

          <div ref={scroller} onScroll={onScroll} className="flex-1 overflow-y-auto">
            {entries.length === 0 && !livePending ? (
              <EmptyState onPick={ask} />
            ) : (
              <div className="mx-auto max-w-3xl space-y-6 px-4 py-6">
                {/* One list for committed + live turns, so a finished answer keeps the same DOM node (no remount/flash). */}
                {[...entries, ...(livePending ? [null] : [])].map((e, i) => (
                  <div key={i} className="space-y-4">
                    {e ? (
                      <>
                        <UserMessage text={e.user} />
                        <AssistantMessage
                          entry={e}
                          state="done"
                          isLast={i === entries.length - 1 && !busy}
                          onAsk={ask}
                          onRegenerate={regenerate}
                        />
                      </>
                    ) : (
                      <>
                        <UserMessage text={livePending!.user} />
                        <AssistantMessage
                          entry={{ bot: live.answer, citations: live.citations, related: live.related, mode: livePending!.mode, grounding: live.grounding, trace: live.trace }}
                          state={live.status === "connecting" ? "connecting" : "streaming"}
                        />
                      </>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>

          {showJump && (
            <button
              onClick={() => { stick.current = true; scrollToBottom(true); }}
              className="absolute bottom-32 left-1/2 z-10 flex -translate-x-1/2 animate-rise items-center gap-1.5 rounded-full border border-line bg-surface px-3.5 py-1.5 text-xs font-medium shadow-pop transition hover:border-brand/50"
            >
              <ArrowDown width={14} height={14} /> Jump to latest
            </button>
          )}

          <Composer busy={busy} onSend={ask} onStop={stop} mode={mode} />
        </main>
      </div>
    </>
  );
}
