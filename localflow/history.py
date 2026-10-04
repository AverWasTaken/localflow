"""SQLite dictation history."""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

from .paths import HISTORY_DB

_SCHEMA = """
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    app TEXT,
    raw TEXT,
    cleaned TEXT,
    duration_s REAL,
    stt_ms INTEGER,
    cleanup_ms INTEGER,
    total_ms INTEGER
);
CREATE INDEX IF NOT EXISTS idx_history_ts ON history(ts DESC);
"""


# columns added after the first release: (name, sqlite type)
_NEW_COLUMNS = [("title", "TEXT"), ("style", "TEXT"), ("segments", "INTEGER")]


class History:
    def __init__(self, path: Path = HISTORY_DB, max_entries: int = 5000):
        self.path = path
        self.max_entries = max_entries
        self._lock = threading.Lock()
        c = self._conn()
        try:
            c.executescript(_SCHEMA)
            self._migrate(c)
            c.commit()
        finally:
            c.close()

    @staticmethod
    def _migrate(c: sqlite3.Connection) -> None:
        """Add columns introduced after v1 (old rows keep NULLs)."""
        have = {r[1] for r in c.execute("PRAGMA table_info(history)")}
        for col, typ in _NEW_COLUMNS:
            if col not in have:
                c.execute(f"ALTER TABLE history ADD COLUMN {col} {typ}")

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(str(self.path), timeout=5)
        c.row_factory = sqlite3.Row
        return c

    def _run(self, fn):
        with self._lock:
            c = self._conn()
            try:
                r = fn(c)
                c.commit()
                return r
            finally:
                c.close()

    def add(self, raw: str, cleaned: str, app: str = "", duration_s: float = 0.0,
            stt_ms: int = 0, cleanup_ms: int = 0, total_ms: int = 0, title: str = "", style: str = "",
            segments: int = 0) -> int:
        def f(c):
            cur = c.execute(
                "INSERT INTO history(ts,app,raw,cleaned,duration_s,stt_ms,cleanup_ms,total_ms,title,style,segments) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (time.time(), app, raw, cleaned, duration_s, stt_ms, cleanup_ms, total_ms, title, style, segments))
            c.execute("DELETE FROM history WHERE id NOT IN (SELECT id FROM history ORDER BY id DESC LIMIT ?)",
                      (self.max_entries,))
            return cur.lastrowid
        return self._run(f)

    def list(self, limit: int = 200, search: str = "") -> list[dict]:
        """Newest first. Row keys: id, ts (epoch s), app, raw, cleaned, duration_s, stt_ms, cleanup_ms, total_ms,
        title, style, segments (the last three are NULL on rows from before they existed)."""
        def f(c):
            if search:
                like = f"%{search}%"
                rows = c.execute("SELECT * FROM history WHERE raw LIKE ? OR cleaned LIKE ? OR app LIKE ? OR title LIKE ? "
                                 "OR style LIKE ? ORDER BY id DESC LIMIT ?", (like, like, like, like, like, limit)).fetchall()
            else:
                rows = c.execute("SELECT * FROM history ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in rows]
        return self._run(f)

    def delete(self, entry_id: int) -> None:
        self._run(lambda c: c.execute("DELETE FROM history WHERE id=?", (entry_id,)))

    def clear(self) -> None:
        self._run(lambda c: c.execute("DELETE FROM history"))
