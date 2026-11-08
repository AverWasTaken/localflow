"""SQLite dictation history + daily usage stats.

``history`` rows follow history.enabled / max_entries / retention_days. ``daily_stats`` is separate: it is updated on
every successful insert whatever the history settings are, and is never pruned (lifetime stats)."""
from __future__ import annotations

import datetime as _dt
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

_STATS_SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_stats (
    date TEXT PRIMARY KEY,          -- local calendar day, YYYY-MM-DD
    words INTEGER NOT NULL DEFAULT 0,
    audio_s REAL NOT NULL DEFAULT 0,
    dictations INTEGER NOT NULL DEFAULT 0,
    commands INTEGER NOT NULL DEFAULT 0
);
"""

TYPING_WPM = 40.0       # "time saved" baseline


def _day(ts: float) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(ts))


# columns added after the first release: (name, sqlite type)
_NEW_COLUMNS = [("title", "TEXT"), ("style", "TEXT"), ("segments", "INTEGER")]


class History:
    def __init__(self, path: Path = HISTORY_DB, max_entries: int = 5000, retention_days: float = 0):
        self.path = path
        self.max_entries = max_entries
        self.retention_days = retention_days     # 0 = keep forever
        self._lock = threading.Lock()
        c = self._conn()
        try:
            c.executescript(_SCHEMA)
            self._migrate(c)
            fresh = not c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='daily_stats'").fetchone()
            c.executescript(_STATS_SCHEMA)
            if fresh:
                self._backfill_stats(c)
            self._prune(c)
            c.commit()
        finally:
            c.close()

    @staticmethod
    def _backfill_stats(c: sqlite3.Connection) -> None:
        """First run with stats: seed daily_stats from the history rows that exist (best effort)."""
        agg: dict[str, list] = {}
        for ts, cleaned, dur, style in c.execute("SELECT ts, cleaned, duration_s, style FROM history"):
            a = agg.setdefault(_day(ts or 0), [0, 0.0, 0, 0])
            if style == "command":
                a[3] += 1
            else:
                a[0] += len((cleaned or "").split())
                a[1] += float(dur or 0)
                a[2] += 1
        c.executemany("INSERT OR REPLACE INTO daily_stats(date,words,audio_s,dictations,commands) VALUES(?,?,?,?,?)",
                      [(d, *v) for d, v in agg.items()])

    def _prune(self, c: sqlite3.Connection) -> None:
        if self.retention_days and float(self.retention_days) > 0:
            c.execute("DELETE FROM history WHERE ts < ?", (time.time() - float(self.retention_days) * 86400,))
        c.execute("DELETE FROM history WHERE id NOT IN (SELECT id FROM history ORDER BY id DESC LIMIT ?)",
                  (int(self.max_entries),))

    def set_limits(self, max_entries: int, retention_days: float) -> None:
        """Apply changed history.max_entries / history.retention_days now (prunes immediately)."""
        self.max_entries, self.retention_days = max_entries, retention_days
        self._run(self._prune)

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
            self._prune(c)
            return cur.lastrowid
        return self._run(f)

    # ------------------------------------------------------------------ stats
    def record_stats(self, words: int = 0, audio_s: float = 0.0, kind: str = "dictation",
                     ts: float | None = None) -> None:
        """Count one successful insert (kind "dictation" or "command") on its local day."""
        d = _day(time.time() if ts is None else ts)
        dic, cmd = (0, 1) if kind == "command" else (1, 0)
        self._run(lambda c: c.execute(
            "INSERT INTO daily_stats(date,words,audio_s,dictations,commands) VALUES(?,?,?,?,?) "
            "ON CONFLICT(date) DO UPDATE SET words=words+excluded.words, audio_s=audio_s+excluded.audio_s, "
            "dictations=dictations+excluded.dictations, commands=commands+excluded.commands",
            (d, int(words), float(audio_s), dic, cmd)))

    def stats(self, today: _dt.date | None = None) -> dict:
        """Lifetime usage numbers for the Home page.

        Keys: total_words, words_7d, audio_s, avg_wpm (dictated words per minute of dictation audio),
        time_saved_s (typing those words at 40 WPM minus the time spent speaking, >= 0), streak_days (consecutive
        days with a dictation or command, ending today - or yesterday if nothing yet today), dictations, commands."""
        today = today or _dt.date.today()
        rows = self._run(lambda c: c.execute(
            "SELECT date, words, audio_s, dictations, commands FROM daily_stats").fetchall())
        week0 = (today - _dt.timedelta(days=6)).isoformat()
        total_w = sum(r["words"] for r in rows)
        audio = sum(r["audio_s"] for r in rows)
        active = {r["date"] for r in rows if (r["dictations"] or 0) + (r["commands"] or 0) > 0}
        streak = 0
        d = today if today.isoformat() in active else today - _dt.timedelta(days=1)
        while d.isoformat() in active:
            streak += 1
            d -= _dt.timedelta(days=1)
        return {
            "total_words": int(total_w),
            "words_7d": int(sum(r["words"] for r in rows if week0 <= r["date"] <= today.isoformat())),
            "audio_s": float(audio),
            "avg_wpm": round(total_w / (audio / 60.0), 1) if audio > 0 else 0.0,
            "time_saved_s": max(0.0, total_w / TYPING_WPM * 60.0 - audio),
            "streak_days": streak,
            "dictations": int(sum(r["dictations"] for r in rows)),
            "commands": int(sum(r["commands"] for r in rows)),
        }

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
