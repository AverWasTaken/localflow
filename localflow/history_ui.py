"""Dictation history: ``HistoryList`` (searchable list widget, embedded on the Home page) and ``HistoryWindow``
(the same list in its own window, kept as a fallback)."""
from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QScrollArea, QSizePolicy, QVBoxLayout, QWidget)

from .icons import check_icon, muted_icon
from .theme import DARK_QSS, btn as _btn, enable_dark_titlebar, set_btn_icon, style_window  # noqa: F401  (DARK_QSS re-exported)
from .theme import ICON_GAP, confirm as _theme_confirm
from .uifmt import fmt_when as _fmt_ts  # noqa: F401  (kept for older imports)

MAX_ENTRIES = 200


def _nice_app(app: str) -> str:
    app = (app or "").strip()
    if app.lower().endswith(".exe"):
        app = app[:-4]
    return app


def confirm(parent, title: str, text: str, ok_label: str, info: str = "") -> bool:
    """Themed confirm dialog (theme.ConfirmDialog): Cancel is the default, Esc cancels. True = confirmed."""
    return _theme_confirm(parent, title, text, ok_label, info, danger=True)


def flash_copied(btn: QPushButton, label: str) -> None:
    """Show that a copy worked: the icon turns into a green check (and the text, if any, reads "Copied")."""
    name = btn.property("iconName") or ""
    if btn.property("baseTip") is None:
        btn.setProperty("baseTip", btn.toolTip())
    if name:
        btn.setIcon(check_icon(16) if not btn.text() else check_icon(15, pad=ICON_GAP))
    if btn.text():
        btn.setText("Copied")
    btn.setToolTip("Copied")

    def restore():
        if name:
            set_btn_icon(btn, name)
        if btn.text():
            btn.setText(label)
        btn.setToolTip(str(btn.property("baseTip") or ""))
    QTimer.singleShot(1100, btn, restore)   # cancelled if the button is deleted


class _Entry(QFrame):
    """One dictation: app + style + when + actions, the inserted text, the raw transcript when it differs, timing."""

    def __init__(self, row: dict, on_delete):
        super().__init__()
        self.setObjectName("entry")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._row = row
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 12, 12, 14)
        lay.setSpacing(6)

        cleaned = (row.get("cleaned") or "").strip()
        raw = (row.get("raw") or "").strip()

        head = QHBoxLayout()
        head.setSpacing(8)
        app = _nice_app(row.get("app", ""))
        if app:
            a = QLabel(app)
            a.setObjectName("entryApp")
            if row.get("title"):
                a.setToolTip(str(row["title"]))
            head.addWidget(a)
        style = (row.get("style") or "").strip()
        if style:
            sc = QLabel(style)
            sc.setObjectName("chip")
            head.addWidget(sc, 0, Qt.AlignVCenter)
        head.addStretch(1)
        ts = QLabel(_fmt_ts(row.get("ts", 0)))
        ts.setProperty("role", "faint")
        head.addWidget(ts)
        head.addSpacing(6)
        self.copy_btn = _btn("", icon="copy", tip="Copy the inserted text")
        self.copy_btn.clicked.connect(lambda: self._copy(cleaned or raw, self.copy_btn, ""))
        head.addWidget(self.copy_btn)
        self.raw_btn = _btn("", icon="mic", tip="Copy what the speech model heard (before cleanup)")
        self.raw_btn.clicked.connect(lambda: self._copy(raw or cleaned, self.raw_btn, ""))
        head.addWidget(self.raw_btn)
        self.del_btn = _btn("", "iconDanger", icon="trash", tip="Delete from history")
        self.del_btn.clicked.connect(lambda: on_delete(row["id"]))
        head.addWidget(self.del_btn)
        lay.addLayout(head)

        main = QLabel(cleaned or raw or "(empty)")
        main.setObjectName("entryText")
        main.setWordWrap(True)
        main.setTextInteractionFlags(Qt.TextSelectableByMouse)
        main.setCursor(Qt.IBeamCursor)
        main.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)   # wrap instead of widening the page
        main.setContentsMargins(0, 0, 6, 0)
        lay.addWidget(main)

        if raw and raw != cleaned:
            r = QLabel(raw)
            r.setObjectName("entryRaw")
            r.setWordWrap(True)
            r.setTextInteractionFlags(Qt.TextSelectableByMouse)
            r.setToolTip("What the speech model heard, before cleanup")
            r.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            r.setContentsMargins(0, 0, 6, 0)
            lay.addWidget(r)

        timing = QLabel(self._timing_text(row))
        timing.setProperty("role", "faint")
        lay.addSpacing(2)
        lay.addWidget(timing)

    @staticmethod
    def _timing_text(row: dict) -> str:
        parts = []
        d = row.get("duration_s")
        if d:
            parts.append(f"{float(d):.1f} s audio")
        if row.get("stt_ms"):
            parts.append(f"speech {int(row['stt_ms'])} ms")
        if row.get("cleanup_ms"):
            parts.append(f"cleanup {int(row['cleanup_ms'])} ms")
        if row.get("total_ms"):
            parts.append(f"total {int(row['total_ms'])} ms")
        return ", ".join(parts)

    @staticmethod
    def _copy(text: str, btn: QPushButton, label: str) -> None:
        QGuiApplication.clipboard().setText(text or "")
        flash_copied(btn, label)


_EntryCard = _Entry     # older name


class HistoryList(QWidget):
    """Search box + entry list + "Clear all". Refreshes every 3 s only while it is visible.

    ``changed`` fires when the shown rows changed (the Home page uses it to refresh its numbers)."""

    changed = Signal()

    def __init__(self, history, title: str = "History", show_title: bool = True):
        super().__init__()
        self.history = history
        self._sig = None
        self._cards: dict = {}
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(14)

        top = QHBoxLayout()
        top.setSpacing(10)
        if show_title:
            t = QLabel(title)
            t.setObjectName("sectionTitle")
            top.addWidget(t)
        self.count_label = QLabel("")
        self.count_label.setProperty("role", "faint")
        top.addWidget(self.count_label, 0, Qt.AlignVCenter)
        top.addStretch(1)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search history")
        self.search.setClearButtonEnabled(True)
        self.search.setFixedWidth(260)
        self.search.addAction(muted_icon("search", 15), QLineEdit.LeadingPosition)
        top.addWidget(self.search)
        top.addSpacing(4)
        self.btn_clear = _btn("Clear all", "ghostDanger", icon="trash", tip="Delete every entry in the history")
        top.addWidget(self.btn_clear)
        outer.addLayout(top)

        self.scroll = QScrollArea()
        self.scroll.setObjectName("listPanel")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.inner = QWidget()
        self.inner.setObjectName("pageInner")
        self.vbox = QVBoxLayout(self.inner)
        self.vbox.setContentsMargins(0, 2, 2, 2)
        self.vbox.setSpacing(0)
        self.scroll.setWidget(self.inner)
        outer.addWidget(self.scroll, 1)

        self.empty = QLabel("")
        self.empty.setObjectName("emptyState")
        self.empty.setAlignment(Qt.AlignCenter)
        self.empty.setWordWrap(True)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(220)
        self._debounce.timeout.connect(lambda: self.refresh(force=True))
        self.search.textChanged.connect(lambda _t: self._debounce.start())
        self.btn_clear.clicked.connect(self._clear_all)

        self._auto = QTimer(self)
        self._auto.setInterval(3000)
        self._auto.timeout.connect(self.refresh)

    # ------------------------------------------------------------ data
    def refresh(self, force: bool = False) -> None:
        q = self.search.text().strip()
        try:
            rows = self.history.list(MAX_ENTRIES, q) if self.history is not None else []
        except Exception:
            rows = []
        sig = (q, tuple((r["id"], r.get("cleaned"), r.get("raw")) for r in rows))
        if sig == self._sig and not force:
            return
        changed = sig != self._sig
        self._sig = sig
        sb = self.scroll.verticalScrollBar()
        pos = sb.value()
        self.inner.setUpdatesEnabled(False)
        try:
            while self.vbox.count():            # detach everything (entries are cached and reused by key)
                it = self.vbox.takeAt(0)
                del it
            keys = []
            for r in rows:
                key = (r["id"], r.get("cleaned"), r.get("raw"))
                keys.append(key)
                if key not in self._cards:
                    self._cards[key] = _Entry(r, self._delete)
            live = set(keys)
            for key in [k for k in self._cards if k not in live]:
                w = self._cards.pop(key)
                w.setParent(None)
                w.deleteLater()
            if not rows:
                if self.history is None:
                    self.empty.setText("History isn't available.")
                else:
                    self.empty.setText("No matches." if q else
                                       "Nothing here yet. Hold your dictation keys, talk, and let go.")
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
        self.btn_clear.setEnabled(self.history is not None and bool(rows or q))
        QTimer.singleShot(0, lambda: sb.setValue(pos))
        if changed:
            self.changed.emit()

    def _delete(self, entry_id: int) -> None:
        if not confirm(self, "Delete entry", "Delete this dictation from history?", "Delete"):
            return
        try:
            self.history.delete(entry_id)
        except Exception:
            pass
        self.refresh(force=True)

    def _clear_all(self) -> None:
        if confirm(self, "Clear history", "Delete all dictation history?", "Delete all",
                   "This can't be undone. Your usage numbers on Home are kept."):
            try:
                self.history.clear()
            except Exception:
                pass
            self.refresh(force=True)

    # ------------------------------------------------------------ visibility: poll only while shown
    def showEvent(self, e):
        self.refresh(force=not e.spontaneous())
        self._auto.start()
        super().showEvent(e)

    def hideEvent(self, e):
        self._auto.stop()
        super().hideEvent(e)


class HistoryWindow(QWidget):
    """Stand-alone history window (the tray now opens the Home page instead; kept as a fallback)."""

    def __init__(self, history):
        super().__init__()
        self.history = history
        self.setWindowTitle("LocalFlow History")
        self.resize(760, 700)
        self.setMinimumSize(560, 420)
        style_window(self)
        self._dark_done = False
        outer = QVBoxLayout(self)
        outer.setContentsMargins(26, 22, 26, 22)
        self.list = HistoryList(history)
        outer.addWidget(self.list)
        # older attribute names
        self.search, self.count_label, self.btn_clear = self.list.search, self.list.count_label, self.list.btn_clear
        self.refresh = self.list.refresh

    def showEvent(self, e):
        if not self._dark_done:
            enable_dark_titlebar(self)
            self._dark_done = True
        super().showEvent(e)

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
    from .theme import apply_theme

    db = Path(tempfile.gettempdir()) / "localflow_demo_history.db"
    if db.exists():
        db.unlink()
    h = History(path=db)
    h.add("um so i was thinking we should uh refactor the the auth module", "I was thinking we should refactor the auth module.",
          "Code.exe", 3.4, 180, 420, 640)
    h.add("hello world", "hello world", "notepad.exe", 1.0, 90, 0, 120)
    app = QApplication(sys.argv)
    apply_theme(app)
    w = HistoryWindow(h)
    w.show()
    sys.exit(app.exec())
