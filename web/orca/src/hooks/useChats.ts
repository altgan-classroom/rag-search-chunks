import { useCallback, useEffect, useState } from "react";
import type { Chat, ChatEntry, ChatMap } from "../lib/types";

const KEY = "chats";
const newId = () => `chat-${Date.now()}`;

function load(): ChatMap {
  try {
    const raw = localStorage.getItem(KEY);
    const parsed = raw ? JSON.parse(raw) : null;
    if (parsed && typeof parsed === "object" && Object.keys(parsed).length) return parsed;
  } catch {
    /* storage unavailable or corrupt: start fresh */
  }
  return { [newId()]: { title: "New chat", chatEntries: [] } };
}

export function useChats() {
  const [chats, setChats] = useState<ChatMap>(load);
  const [activeId, setActiveId] = useState<string>(() => Object.keys(chats).sort().pop()!);

  useEffect(() => {
    try {
      localStorage.setItem(KEY, JSON.stringify(chats));
    } catch {
      /* ignore quota/private mode */
    }
  }, [chats]);

  const newChat = useCallback(() => {
    // Reuse an existing empty chat instead of piling up blanks.
    const empty = Object.entries(chats).find(([, c]) => c.chatEntries.length === 0);
    if (empty) return setActiveId(empty[0]);
    const id = newId();
    setChats((p) => ({ ...p, [id]: { title: "New chat", chatEntries: [] } }));
    setActiveId(id);
  }, [chats]);

  const rename = useCallback((id: string, title: string) => {
    const t = title.trim();
    if (!t) return;
    setChats((p) => (p[id] ? { ...p, [id]: { ...p[id], title: t } } : p));
  }, []);

  const remove = useCallback(
    (id: string) => {
      const rest = { ...chats };
      delete rest[id];
      if (Object.keys(rest).length === 0) {
        const nid = newId();
        rest[nid] = { title: "New chat", chatEntries: [] };
      }
      setChats(rest);
      if (id === activeId) setActiveId(Object.keys(rest).sort().pop()!);
    },
    [chats, activeId],
  );

  const append = useCallback((id: string, entry: ChatEntry) => {
    setChats((p) => {
      const chat: Chat = p[id] ?? { title: "New chat", chatEntries: [] };
      const title =
        chat.chatEntries.length === 0 && /^New chat/.test(chat.title)
          ? entry.user.slice(0, 48) + (entry.user.length > 48 ? "…" : "")
          : chat.title;
      return { ...p, [id]: { title, chatEntries: [...chat.chatEntries, entry] } };
    });
  }, []);

  const dropLast = useCallback((id: string) => {
    setChats((p) => (p[id] ? { ...p, [id]: { ...p[id], chatEntries: p[id].chatEntries.slice(0, -1) } } : p));
  }, []);

  return { chats, activeId, setActiveId, newChat, rename, remove, append, dropLast };
}
