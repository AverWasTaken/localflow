"""Recorded transcriptions: SQLite store for file + meeting transcripts and the AI results made from them.

One connection per thread (WAL, busy timeout), so the job thread, the assistant thread and the UI thread can all
use the same ``TranscriptStore``. Meeting/file audio lives on disk (``audio_dir``); only text lives here.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Optional

log = logging.getLogger(__name__)

KINDS = ("file", "meeting")
STATUSES = ("queued", "transcribing", "cleaning", "done", "failed", "interrupted", "cancelled")
ACTIVE_STATUSES = ("queued", "transcribing", "cleaning")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS transcripts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    created REAL NOT NULL,
    updated REAL NOT NULL DEFAULT 0,
    duration_s REAL NOT NULL DEFAULT 0,
    source_path TEXT NOT NULL DEFAULT '',
    audio_dir TEXT NOT NULL DEFAULT '',
    style TEXT NOT NULL DEFAULT 'raw',
    status TEXT NOT NULL DEFAULT 'queued',
    progress REAL NOT NULL DEFAULT 0,
    error TEXT NOT NULL DEFAULT '',
    raw_text TEXT NOT NULL DEFAULT '',
    text TEXT NOT NULL DEFAULT '',
    segments_json TEXT NOT NULL DEFAULT '',
    include_echo INTEGER NOT NULL DEFAULT 0,
    info_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS ix_transcripts_created ON transcripts(created);
CREATE TABLE IF NOT EXISTS ai_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transcript_id INTEGER NOT NULL,
    preset TEXT NOT NULL,
    prompt TEXT NOT NULL DEFAULT '',
    output TEXT NOT NULL DEFAULT '',
    created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_ai_results_tid ON ai_results(transcript_id);
CREATE TABLE IF NOT EXISTS job_progress (
    transcript_id INTEGER NOT NULL,
    key TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (transcript_id, key)
);
"""

_COLUMNS = ("kind", "title", "created", "updated", "duration_s", "source_path", "audio_dir", "style", "status",
            "progress", "error", "raw_text", "text", "segments_json", "include_echo", "info_json")
_LIST_COLUMNS = ("id", "kind", "title", "created", "updated", "duration_s", "source_path", "audio_dir", "style",
                 "status", "progress", "error", "include_echo", "info_json")


def default_db_path() -> Path:
    from .paths import DATA_DIR
    return DATA_DIR / "transcripts.db"


def default_recordings_root() -> Path:
    from .paths import DATA_DIR
    return DATA_DIR / "recordings"


def fmt_ts(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


class TranscriptStore:
    DELETED_MARKER = ".deleted"     # left in an audio dir whose removal failed (file locked); retried later

    def __init__(self, db_path: Optional[Path] = None, recordings_root: Optional[Path] = None):
        self.path = Path(db_path) if db_path else default_db_path()
        self.recordings_root = Path(recordings_root) if recordings_root else default_recordings_root()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.RLock()
        self._all_conns: list[sqlite3.Connection] = []
        self._conns_lock = threading.Lock()
        c = self._conn()
        c.executescript(_SCHEMA)
        c.commit()

    # ------------------------------------------------------------------ connections
    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(str(self.path), timeout=10.0, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            c.execute("PRAGMA busy_timeout=10000")
            self._local.conn = c
            with self._conns_lock:
                self._all_conns.append(c)
        return c

    def close(self) -> None:
        with self._conns_lock:
            conns, self._all_conns = self._all_conns, []
        for c in conns:
            try:
                c.close()
            except Exception:
                pass
        self._local = threading.local()

    def _write(self, sql: str, args: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._write_lock:
            c = self._conn()
            cur = c.execute(sql, tuple(args))
            c.commit()
            return cur

    # ------------------------------------------------------------------ transcripts
    def create(self, kind: str, title: str = "", *, source_path: str = "", audio_dir: str = "", style: str = "raw",
               duration_s: float = 0.0, status: str = "queued", created: Optional[float] = None) -> int:
        if kind not in KINDS:
            raise ValueError(f"bad kind {kind!r}")
        now = time.time()
        cur = self._write(
            "INSERT INTO transcripts(kind,title,created,updated,duration_s,source_path,audio_dir,style,status) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (kind, title, created or now, now, float(duration_s or 0), str(source_path or ""), str(audio_dir or ""),
             style or "raw", status))
        return int(cur.lastrowid)

    def update(self, tid: int, **fields: Any) -> None:
        if not fields:
            return
        bad = set(fields) - set(_COLUMNS)
        if bad:
            raise ValueError(f"unknown transcript fields {sorted(bad)}")
        if "status" in fields and fields["status"] not in STATUSES:
            raise ValueError(f"bad status {fields['status']!r}")
        if "info_json" in fields and not isinstance(fields["info_json"], str):
            fields["info_json"] = json.dumps(fields["info_json"])
        if "segments_json" in fields and not isinstance(fields["segments_json"], str):
            fields["segments_json"] = json.dumps(fields["segments_json"])
        fields.setdefault("updated", time.time())
        cols = ", ".join(f"{k}=?" for k in fields)
        self._write(f"UPDATE transcripts SET {cols} WHERE id=?", (*fields.values(), int(tid)))

    def update_info(self, tid: int, **kv: Any) -> dict:
        """Merge keys into info_json (misc job facts: language, fallbacks, waiting reason...)."""
        with self._write_lock:
            row = self.get(tid)
            if row is None:
                return {}
            info = row["info"]
            for k, v in kv.items():
                if v is None:
                    info.pop(k, None)
                else:
                    info[k] = v
            self.update(tid, info_json=info)
            return info

    @staticmethod
    def _row(r: sqlite3.Row) -> dict:
        d = dict(r)
        try:
            d["info"] = json.loads(d.get("info_json") or "{}")
        except Exception:
            d["info"] = {}
        return d

    def get(self, tid: int) -> Optional[dict]:
        r = self._conn().execute("SELECT * FROM transcripts WHERE id=?", (int(tid),)).fetchone()
        return self._row(r) if r else None

    def segments(self, tid: int) -> dict:
        r = self._conn().execute("SELECT segments_json FROM transcripts WHERE id=?", (int(tid),)).fetchone()
        if not r or not r[0]:
            return {}
        try:
            return json.loads(r[0])
        except Exception:
            return {}

    def list(self, search: str = "", limit: int = 1000) -> list[dict]:
        """Newest first, without the big text columns (``snippet`` = start of the transcript)."""
        cols = ", ".join(_LIST_COLUMNS)
        sql = f"SELECT {cols}, substr(text, 1, 200) AS snippet FROM transcripts"
        args: list[Any] = []
        if search.strip():
            like = "%" + search.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            sql += (" WHERE title LIKE ? ESCAPE '\\' OR text LIKE ? ESCAPE '\\' OR raw_text LIKE ? ESCAPE '\\'"
                    " OR source_path LIKE ? ESCAPE '\\'")
            args += [like] * 4
        sql += " ORDER BY created DESC, id DESC LIMIT ?"
        args.append(int(limit))
        return [self._row(r) for r in self._conn().execute(sql, args).fetchall()]

    def find_by_audio_dir(self, audio_dir: str) -> Optional[dict]:
        r = self._conn().execute("SELECT * FROM transcripts WHERE audio_dir=? ORDER BY id DESC LIMIT 1",
                                 (str(audio_dir),)).fetchone()
        return self._row(r) if r else None

    def ids_with_status(self, statuses: Iterable[str]) -> list[int]:
        st = list(statuses)
        if not st:
            return []
        q = ",".join("?" * len(st))
        return [int(r[0]) for r in self._conn().execute(
            f"SELECT id FROM transcripts WHERE status IN ({q}) ORDER BY id", st).fetchall()]

    def rename(self, tid: int, title: str) -> None:
        self.update(tid, title=str(title).strip())

    def _safe_audio_dir(self, audio_dir: str) -> Optional[Path]:
        """Only directories inside the recordings root are ever deleted (never a user's imported file)."""
        if not audio_dir:
            return None
        try:
            p = Path(audio_dir).resolve()
            root = self.recordings_root.resolve()
        except Exception:
            return None
        if p == root or root not in p.parents:
            log.warning("not deleting audio dir outside the recordings root: %s", p)
            return None
        return p

    def mark_deleted(self, audio_dir: str) -> None:
        """Durable delete intent: recovery removes marked dirs and never creates rows for them."""
        p = self._safe_audio_dir(audio_dir)
        if p is not None and p.exists():
            try:
                (p / self.DELETED_MARKER).write_text("deleted", encoding="utf-8")
            except OSError:
                log.warning("could not mark %s as deleted", p)

    def delete(self, tid: int, delete_audio: bool = True) -> bool:
        """Delete the row, its AI results and progress. The audio dir is marked deleted BEFORE the row goes (a
        crash in between never brings the recording back); with ``delete_audio`` it is removed right away."""
        row = self.get(tid)
        if row is None:
            return False
        self.mark_deleted(row.get("audio_dir") or "")
        with self._write_lock:
            c = self._conn()
            c.execute("DELETE FROM ai_results WHERE transcript_id=?", (int(tid),))
            c.execute("DELETE FROM job_progress WHERE transcript_id=?", (int(tid),))
            c.execute("DELETE FROM transcripts WHERE id=?", (int(tid),))
            c.commit()
        if delete_audio:
            self.remove_audio_dir(row.get("audio_dir") or "")
        return True

    def remove_audio_dir(self, audio_dir: str) -> bool:
        """Remove a recording dir (only inside the recordings root). Retries briefly (Windows file locks); if it
        still fails, a marker file is left so recovery removes it later and never resurrects it."""
        p = self._safe_audio_dir(audio_dir)
        if p is None or not p.exists():
            return True
        err = None
        for i in range(5):
            try:
                shutil.rmtree(p)
                return True
            except FileNotFoundError:
                return True
            except OSError as e:
                err = e
                time.sleep(0.2 * (i + 1))
        log.warning("could not delete %s (%s) - will retry on next start", p, err)
        try:
            (p / self.DELETED_MARKER).write_text("deleted", encoding="utf-8")
        except OSError:
            pass
        return False

    # ------------------------------------------------------------------ AI results
    def add_ai_result(self, transcript_id: int, preset: str, prompt: str, output: str) -> int:
        """Returns the new id, or 0 when the transcript no longer exists (deleted while the AI was answering)."""
        cur = self._write("INSERT INTO ai_results(transcript_id,preset,prompt,output,created) "
                          "SELECT ?,?,?,?,? WHERE EXISTS (SELECT 1 FROM transcripts WHERE id=?)",
                          (int(transcript_id), preset, prompt or "", output or "", time.time(), int(transcript_id)))
        return int(cur.lastrowid) if cur.rowcount else 0

    def list_ai_results(self, transcript_id: int) -> list[dict]:
        rows = self._conn().execute("SELECT * FROM ai_results WHERE transcript_id=? ORDER BY created DESC, id DESC",
                                    (int(transcript_id),)).fetchall()
        return [dict(r) for r in rows]

    def delete_ai_result(self, result_id: int) -> None:
        self._write("DELETE FROM ai_results WHERE id=?", (int(result_id),))

    # ------------------------------------------------------------------ job progress (crash resume)
    def save_progress(self, tid: int, key: str, data: Any) -> None:
        self._write("INSERT OR REPLACE INTO job_progress(transcript_id,key,data) VALUES(?,?,?)",
                    (int(tid), key, json.dumps(data)))

    def load_progress(self, tid: int) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in self._conn().execute("SELECT key, data FROM job_progress WHERE transcript_id=?",
                                         (int(tid),)).fetchall():
            try:
                out[k] = json.loads(v)
            except Exception:
                pass
        return out

    def clear_progress(self, tid: int) -> None:
        self._write("DELETE FROM job_progress WHERE transcript_id=?", (int(tid),))

    # ------------------------------------------------------------------ export
    def export_text(self, tid: int, fmt: str = "txt") -> str:
        row = self.get(tid)
        if row is None:
            raise KeyError(tid)
        text = row["text"] or row["raw_text"] or ""
        if fmt == "txt":
            return text.rstrip() + "\n"
        if fmt != "md":
            raise ValueError(f"unknown export format {fmt!r}")
        created = time.strftime("%Y-%m-%d %H:%M", time.localtime(row["created"]))
        lines = [f"# {row['title'] or 'Transcript'}", "",
                 f"- Date: {created}",
                 f"- Type: {'Meeting recording' if row['kind'] == 'meeting' else 'Audio file'}",
                 f"- Length: {fmt_ts(row['duration_s'])}"]
        if row["kind"] == "file" and row["source_path"]:
            lines.append(f"- Source: {os.path.basename(row['source_path'])}")
        if row["style"] and row["style"] != "raw":
            lines.append(f"- Style: {row['style']}")
        lines += ["", "## Transcript", ""]
        if row["kind"] == "meeting":
            # "[00:01:02] You: text" -> "**[00:01:02] You:** text"
            for para in text.split("\n\n"):
                p = para.strip()
                if p.startswith("[") and "]" in p and ":" in p[p.index("]"):]:
                    close = p.index("]")
                    colon = p.index(":", close)
                    p = f"**{p[:colon + 1]}**{p[colon + 1:]}"
                if p:
                    lines += [p, ""]
        else:
            lines += [text.strip(), ""]
        results = self.list_ai_results(tid)
        if results:
            lines += ["## AI notes", ""]
            for r in reversed(results):
                label = r["preset"].replace("_", " ").title() if r["preset"] != "custom" else (r["prompt"] or "Custom")
                lines += [f"### {label}", "", r["output"].strip(), ""]
        return "\n".join(lines).rstrip() + "\n"
