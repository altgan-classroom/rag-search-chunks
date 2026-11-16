"""Chat history in SQLite, keyed by (tenant, user, chat_id): stateless handlers, no cross-user bleed (baseline S3)."""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

from ..store.qdrant_store import Principal


class HistoryStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._c = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        self._c.execute("""CREATE TABLE IF NOT EXISTS turns(id INTEGER PRIMARY KEY, tenant TEXT, user TEXT, chat TEXT,
                           role TEXT, content TEXT, ts REAL)""")
        self._c.execute("CREATE INDEX IF NOT EXISTS ix ON turns(tenant,user,chat,id)")
        self._c.commit()

    def add(self, p: Principal, chat: str, role: str, content: str) -> None:
        with self._lock:
            self._c.execute("INSERT INTO turns(tenant,user,chat,role,content,ts) VALUES(?,?,?,?,?,?)",
                            (p.tenant, p.id, chat, role, content, time.time()))
            self._c.commit()

    def get(self, p: Principal, chat: str, limit: int = 10) -> list[tuple[str, str]]:
        with self._lock:
            rows = self._c.execute("SELECT role,content FROM turns WHERE tenant=? AND user=? AND chat=? ORDER BY id DESC LIMIT ?",
                                   (p.tenant, p.id, chat, limit)).fetchall()
        return list(reversed(rows))

    def delete(self, p: Principal, chat: str) -> int:
        with self._lock:
            n = self._c.execute("DELETE FROM turns WHERE tenant=? AND user=? AND chat=?", (p.tenant, p.id, chat)).rowcount
            self._c.commit()
        return n
