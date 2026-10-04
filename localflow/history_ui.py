"""History window: searchable list of past dictations shown as cards."""
from __future__ import annotations

import datetime as _dt
import tempfile
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QPushButton, QScrollArea, QVBoxLayout, QWidget)

from .settings_ui import DARK_QSS, enable_dark_titlebar, style_window  # noqa: F401  (DARK_QSS re-exported)

MAX_ENTRIES = 200


def _fmt_ts(ts: float) -> str:
    try:
        d = _dt.datetime.fromtimestamp(float(ts))
    except Exception:
        return ""
    now = _dt.datetime.now()
    clock = d.strftime("%I:%M %p").lstrip("0")
    if d.date() == now.date():
        return f"Today, {clock}"
    if d.date() == (now - _dt.timedelta(days=1)).date():
        return f"Yesterday, {clock}"
    if d.year == now.year:
        return f"{d.strftime('%a %b')} {d.day}, {clock}"
    return f"{d.strftime('%b')} {d.day}, {d.year}, {clock}"


def _nice_app(app: str) -> str:
    app = (app or "").strip()
    if app.lower().endswith(".exe"):
        app = app[:-4]
    return app


def _btn(text: str, role: str = "") -> QPushButton:
    b = QPushButton(text)
    b.setCursor(Qt.PointingHandCursor)
    if role:
        b.setProperty("role", role)
    return b


class _EntryCard(QFrame):
    def __init__(self, row: dict, on_delete):
        super().__init__()
        self.setObjectName("card")
        self._row = row
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 12, 14, 14)
        lay.setSpacing(8)

        head = QHBoxLayout()
        head.setSpacing(8)
        ts = QLabel(_fmt_ts(row.get("ts", 0)))
        ts.setProperty("role", "muted")
        head.addWidget(ts)
        app = _nice_app(row.get("app", ""))
        if app:
            chip = QLabel(app)
            chip.setObjectName("chipAccent")
            if row.get("title"):
                chip.setToolTip(str(row["title"]))
            head.addWidget(chip)
        style = (row.get("style") or "").strip()
        if style:
            sc = QLabel(style)
            sc.setObjectName("chip")
            head.addWidget(sc)
        head.addStretch(1)
        x = _btn("✕", "ghost")
        x.setToolTip("Delete this entry")
        x.setFixedSize(28, 26)
        x.clicked.connect(lambda: on_delete(row["id"]))
        head.addWidget(x)
        lay.addLayout(head)

        cleaned = (row.get("cleaned") or "").strip()
        raw = (row.get("raw") or "").strip()
        main = QLabel(cleaned or raw or "(empty)")
        main.setWordWrap(True)
        main.setTextInteractionFlags(Qt.TextSelectableByMouse)
        main.setStyleSheet("font-size: 14px;")
        main.setCursor(Qt.IBeamCursor)
        lay.addWidget(main)

        if raw and raw != cleaned:
            r = QLabel(raw)
            r.setWordWrap(True)
            r.setTextInteractionFlags(Qt.TextSelectableByMouse)
            r.setStyleSheet("color: #8a8a99; font-size: 12px; font-style: italic;")
            lay.addWidget(r)

        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        timing = QLabel(self._timing_text(row))
        timing.setStyleSheet("color: #70707f; font-size: 11px;")
        bottom.addWidget(timing)
        bottom.addStretch(1)
        self.copy_btn = _btn("Copy", "mini")
        self.copy_btn.clicked.connect(lambda: self._copy(cleaned or raw, self.copy_btn, "Copy"))
        bottom.addWidget(self.copy_btn)
        self.raw_btn = _btn("Copy raw", "mini")
        self.raw_btn.clicked.connect(lambda: self._copy(raw or cleaned, self.raw_btn, "Copy raw"))
        bottom.addWidget(self.raw_btn)
        lay.addLayout(bottom)

    @staticmethod
    def _timing_text(row: dict) -> str:
        parts = []
        d = row.get("duration_s")
        if d:
            parts.append(f"{float(d):.1f}s audio")
        if row.get("stt_ms"):
            parts.append(f"STT {int(row['stt_ms'])} ms")
        if row.get("cleanup_ms"):
            parts.append(f"cleanup {int(row['cleanup_ms'])} ms")
        if row.get("total_ms"):
            parts.append(f"total {int(row['total_ms'])} ms")
        return "  ·  ".join(parts)

    @staticmethod
    def _copy(text: str, btn: QPushButton, label: str) -> None:
        QGuiApplication.clipboard().setText(text or "")
        btn.setText("Copied ✓")
        QTimer.singleShot(1100, lambda: btn.setText(label))


class HistoryWindow(QWidget):
    def __init__(self, history):
        super().__init__()
        self.history = history
        self.setWindowTitle("LocalFlow History")
        self.resize(760, 700)
        self.setMinimumSize(560, 420)
        style_window(self)
        self._sig = None
        self._cards: dict = {}
        self._dark_done = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(26, 22, 26, 22)
        outer.setSpacing(14)

        top = QHBoxLayout()
        top.setSpacing(10)
        title = QLabel("History")
        title.setObjectName("pageTitle")
        top.addWidget(title)
        self.count_label = QLabel("")
        self.count_label.setProperty("role", "muted")
        top.addWidget(self.count_label, 0, Qt.AlignBottom)
        top.addStretch(1)
        self.btn_refresh = _btn("Refresh")
        self.btn_clear = _btn("Clear all", "danger")
        top.addWidget(self.btn_refresh)
        top.addWidget(self.btn_clear)
        outer.addLayout(top)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search dictations...")
        self.search.setClearButtonEnabled(True)
        outer.addWidget(self.search)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.inner = QWidget()
        self.inner.setObjectName("pageInner")
        self.vbox = QVBoxLayout(self.inner)
        self.vbox.setContentsMargins(0, 0, 8, 0)
        self.vbox.setSpacing(10)
        self.scroll.setWidget(self.inner)
        outer.addWidget(self.scroll, 1)

        self.empty = QLabel("")
        self.empty.setAlignment(Qt.AlignCenter)
        self.empty.setProperty("role", "muted")
        self.empty.setStyleSheet("font-size: 14px; padding: 60px;")

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(220)
        self._debounce.timeout.connect(lambda: self.refresh(force=True))
        self.search.textChanged.connect(lambda _t: self._debounce.start())
        self.btn_refresh.clicked.connect(lambda: self.refresh(force=True))
        self.btn_clear.clicked.connect(self._clear_all)

        self._auto = QTimer(self)
        self._auto.setInterval(3000)
        self._auto.timeout.connect(self.refresh)

    # ------------------------------------------------------------ data
    def refresh(self, force: bool = False) -> None:
        q = self.search.text().strip()
        try:
            rows = self.history.list(MAX_ENTRIES, q)
        except Exception:
            rows = []
        sig = (q, tuple((r["id"], r.get("cleaned"), r.get("raw")) for r in rows))
        if sig == self._sig and not force:
            return
        self._sig = sig
        sb = self.scroll.verticalScrollBar()
        pos = sb.value()
        self.inner.setUpdatesEnabled(False)
        try:
            # detach everything from the layout (cards are cached and reused by key)
            while self.vbox.count():
                it = self.vbox.takeAt(0)
                del it
            keys = []
            for r in rows:
                key = (r["id"], r.get("cleaned"), r.get("raw"))
                keys.append(key)
                if key not in self._cards:
                    self._cards[key] = _EntryCard(r, self._delete)
            for key in [k for k in self._cards if k not in set(keys)]:
                w = self._cards.pop(key)
                w.setParent(None)
                w.deleteLater()
            if not rows:
                self.empty.setText("No matches." if q else "No dictations yet. Hold your hotkey and talk.")
                self.vbox.addWidget(self.empty)
                self.empty.show()
            else:
                self.empty.hide()
                for key in keys:
                    self.vbox.addWidget(self._cards[key])
            self.vbox.addStretch(1)
        finally:
            self.inner.setUpdatesEnabled(True)
        n = len(rows)
        plus = "+" if n >= MAX_ENTRIES else ""
        if q:
            self.count_label.setText(f"{n}{plus} {'match' if n == 1 else 'matches'}")
        else:
            self.count_label.setText(f"{n}{plus} {'entry' if n == 1 else 'entries'}")
        QTimer.singleShot(0, lambda: sb.setValue(pos))

    def _delete(self, entry_id: int) -> None:
        try:
            self.history.delete(entry_id)
        except Exception:
            pass
        self.refresh(force=True)

    def _clear_all(self) -> None:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Clear history")
        box.setText("Delete all dictation history?")
        box.setInformativeText("This can't be undone.")
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        box.setDefaultButton(QMessageBox.Cancel)
        box.button(QMessageBox.Yes).setText("Delete all")
        if box.exec() == QMessageBox.Yes:
            try:
                self.history.clear()
            except Exception:
                pass
            self.refresh(force=True)

    # ------------------------------------------------------------ window behaviour
    def showEvent(self, e):
        if not self._dark_done:
            enable_dark_titlebar(self)
            self._dark_done = True
        self.refresh(force=not e.spontaneous())
        self._auto.start()
        super().showEvent(e)

    def hideEvent(self, e):
        self._auto.stop()
        super().hideEvent(e)

    def closeEvent(self, e):
        e.ignore()
        self.hide()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(e)


# ---------------------------------------------------------------- demo
if __name__ == "__main__":
    import sys

    from .history import History
    from .settings_ui import apply_theme

    db = Path(tempfile.gettempdir()) / "localflow_demo_history.db"
    if db.exists():
        db.unlink()
    h = History(path=db)
    h.add("um so i was thinking we should uh refactor the the auth module", "I was thinking we should refactor the auth module.",
          "Code.exe", 3.4, 180, 420, 640)
    h.add("send a message to john saying i'll be late", "I'll be late.", "Discord.exe", 2.1, 140, 310, 480)
    h.add("hello world", "hello world", "notepad.exe", 1.0, 90, 0, 120)
    h.add("this is a much longer dictation " * 6, "This is a much longer dictation. " * 6, "chrome.exe", 14.8, 520, 880, 1500)
    app = QApplication(sys.argv)
    apply_theme(app)
    w = HistoryWindow(h)
    w.show()
    sys.exit(app.exec())
