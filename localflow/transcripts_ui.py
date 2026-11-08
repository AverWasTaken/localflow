"""Transcriptions page: record a meeting, transcribe an audio file, browse transcripts, ask the AI about one.

Talks to the app object: ``app.transcripts`` (TranscriptService or None), ``app.toggle_meeting()``,
``app.import_audio_file()``, ``app.transcript_styles()``, ``app.ask_ai()`` and the ``app.bridge`` signals
(tr_progress, tr_done, meeting_state, meeting_levels, ask_partial, ask_status, ask_done)."""
from __future__ import annotations

import html
import os
import re
import threading
import time
from pathlib import Path

from PySide6.QtCore import QObject, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QGuiApplication, QPainter
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem, QMenu, QPlainTextEdit, QProgressBar,
                               QScrollArea, QSizePolicy, QSplitter, QStackedWidget, QTextBrowser, QVBoxLayout,
                               QWidget)

from .history_ui import confirm, flash_copied
from .palette import OTHERS, REC, SURFACE2, TEXT, TEXT2, TEXT3, YOU
from .theme import btn, hsep, label, repolish
from .uifmt import (ACTIVE, PRESET_LABELS, RESUMABLE, fmt_duration, fmt_elapsed, fmt_when, fmt_when_short,
                    level_unit, parse_turns, preset_label, status_text)

AUDIO_EXTS = "wav mp3 m4a aac ogg opus flac wma webm mp4 mkv mov"   # same list as app.AUDIO_EXTS


def style_label(name: str) -> str:
    return "Raw (no cleanup)" if name == "raw" else name


def _safe_filename(title: str) -> str:
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', " ", title or "").strip().rstrip(".")
    return (s or "transcript")[:80]


def transcript_html(row: dict) -> str:
    """Reader HTML: meetings as speaker turns ("You" / "Others" with timestamps), files as paragraphs."""
    text = (row.get("text") or row.get("raw_text") or "").strip()
    if not text:
        return ""
    out = []
    if row.get("kind") == "meeting":
        for ts, who, body in parse_turns(text):
            b = html.escape(body).replace("\n", "<br>")
            if not who:
                out.append(f'<p style="margin:0 0 16px 0;">{b}</p>')
                continue
            col = YOU if who.lower() == "you" else OTHERS
            out.append(f'<p style="margin:0 0 3px 0;"><span style="color:{col}; font-weight:600;">'
                       f'{html.escape(who)}</span>&nbsp;&nbsp;<span style="color:{TEXT3}; font-size:12px;">'
                       f'{html.escape(ts)}</span></p>'
                       f'<p style="margin:0 0 16px 0;">{b}</p>')
    else:
        for para in text.split("\n\n"):
            p = para.strip()
            if p:
                out.append(f'<p style="margin:0 0 14px 0;">{html.escape(p).replace(chr(10), "<br>")}</p>')
    return (f'<div style="color:{TEXT}; font-size:14px; line-height:150%;">' + "".join(out) + "</div>")


class _Relay(QObject):
    """Background thread -> UI thread."""
    imported = Signal(object)        # new transcript id or None
    deleted = Signal(int)


class LevelMeter(QWidget):
    """Small labelled level bar ("You" / "Others")."""

    def __init__(self, text: str, color: str):
        super().__init__()
        self._text, self._color, self._v, self._shown = text, QColor(color), 0.0, 0.0
        self.setFixedSize(118, 22)
        self._decay = QTimer(self)
        self._decay.setInterval(50)
        self._decay.timeout.connect(self._tick)

    def set_level(self, unit: float) -> None:
        self._v = max(0.0, min(1.0, unit))
        if self._v > self._shown:
            self._shown = self._v          # fast attack
        if not self._decay.isActive():
            self._decay.start()
        self.update()

    def reset(self) -> None:
        self._v = self._shown = 0.0
        self._decay.stop()
        self.update()

    def _tick(self) -> None:
        self._shown = max(self._v, self._shown * 0.82)
        if self._shown < 0.01 and self._v <= 0:
            self._decay.stop()
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(QColor(TEXT2))
        f = p.font()
        f.setPixelSize(12)
        p.setFont(f)
        p.drawText(QRectF(0, 0, 46, self.height()), int(Qt.AlignVCenter | Qt.AlignLeft), self._text)
        bar = QRectF(48, self.height() / 2 - 3, self.width() - 50, 6)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(SURFACE2))
        p.drawRoundedRect(bar, 3, 3)
        if self._shown > 0.01:
            fill = QRectF(bar.left(), bar.top(), max(6.0, bar.width() * self._shown), bar.height())
            p.setBrush(self._color)
            p.drawRoundedRect(fill, 3, 3)


class _RecDot(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(10, 10)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(REC))
        p.drawEllipse(QRectF(1, 1, 8, 8))


class _Row(QWidget):
    """List entry: title, kind + date + length, progress or error line."""

    def __init__(self, row: dict):
        super().__init__()
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 9, 12, 10)
        lay.setSpacing(4)
        self.title = QLabel()
        self.title.setStyleSheet("font-weight: 600;")
        self.title.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        lay.addWidget(self.title)
        meta = QHBoxLayout()
        meta.setSpacing(8)
        self.kind = QLabel()
        meta.addWidget(self.kind)
        self.meta = label(role="faint")
        meta.addWidget(self.meta)
        meta.addStretch(1)
        lay.addLayout(meta)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        lay.addWidget(self.bar)
        self.status = label(role="faint")
        self.status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        lay.addWidget(self.status)
        self.update_row(row)

    def update_row(self, row: dict, msg: str = "") -> None:
        self._full_title = row.get("title") or "Untitled"
        self._elide()
        meeting = row.get("kind") == "meeting"
        self.kind.setText("Meeting" if meeting else "File")
        self.kind.setObjectName("chipRec" if meeting else "chip")
        repolish(self.kind)
        self.meta.setText(f"{fmt_when_short(row.get('created'))}    {fmt_duration(row.get('duration_s'))}")
        self.meta.setToolTip(fmt_when(row.get("created")))
        self.set_status(row.get("status") or "", row.get("progress"), msg, row.get("error") or "")

    def set_status(self, status: str, progress, msg: str = "", error: str = "") -> None:
        active = status in ACTIVE
        self.bar.setVisible(active)
        if active:
            try:
                self.bar.setValue(int(float(progress or 0) * 1000))
            except Exception:
                self.bar.setValue(0)
        if active:
            self.status.setText(status_text(status, progress, msg))
            self.status.setStyleSheet("")
            self.status.show()
        elif status in RESUMABLE:
            self.status.setText(status_text(status))
            self.status.setToolTip(error)
            self.status.setStyleSheet("color: #f06b73;" if status == "failed" else "color: #e2a53b;")
            self.status.show()
        else:
            self.status.hide()

    def _elide(self) -> None:
        w = max(80, self.title.width() or 220)
        self.title.setText(self.title.fontMetrics().elidedText(self._full_title, Qt.ElideRight, w))
        self.title.setToolTip(self._full_title)

    def resizeEvent(self, e):
        self._elide()
        super().resizeEvent(e)


class _NoteCard(QFrame):
    """A saved AI answer."""

    def __init__(self, r: dict, on_delete):
        super().__init__()
        self.setObjectName("panel")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 10, 10, 14)
        lay.setSpacing(6)
        head = QHBoxLayout()
        head.setSpacing(6)
        t = QLabel(preset_label(r.get("preset", ""), r.get("prompt", "")))
        t.setStyleSheet("font-weight: 600;")
        if r.get("preset") == "custom":
            t.setToolTip(r.get("prompt", ""))
        head.addWidget(t)
        head.addWidget(label(fmt_when(r.get("created")), role="faint"))
        head.addStretch(1)
        out = (r.get("output") or "").strip()
        cp = btn("Copy", "ghost")
        cp.clicked.connect(lambda: (QGuiApplication.clipboard().setText(out), flash_copied(cp, "Copy")))
        head.addWidget(cp)
        dl = btn("Delete", "ghostDanger")
        dl.clicked.connect(lambda: on_delete(r))
        head.addWidget(dl)
        lay.addLayout(head)
        body = QLabel()
        body.setTextFormat(Qt.MarkdownText)
        body.setText(out)
        body.setWordWrap(True)
        body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        body.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        body.setStyleSheet("font-size: 13px;")
        lay.addWidget(body)


class TranscriptsPage(QWidget):
    def __init__(self, app=None, config=None):
        super().__init__()
        self.app = app
        self.cfg = config if config is not None else getattr(app, "cfg", None)
        self.svc = getattr(app, "transcripts", None) if app is not None else None
        self._rows: dict[int, dict] = {}
        self._items: dict[int, QListWidgetItem] = {}
        self._cur: int | None = None
        self._cur_row: dict | None = None
        self._dirty = True
        self._rec_on = False
        self._rec_t0 = 0.0
        self._ask_handle = None
        self._ask_tid: int | None = None
        self._ask_text = ""
        self._last_dir = str(Path.home())
        self._relay = _Relay()
        self._relay.imported.connect(self._after_import)
        self._relay.deleted.connect(lambda _tid: self.refresh_list())

        root = QVBoxLayout(self)
        root.setContentsMargins(32, 26, 32, 20)
        root.setSpacing(14)
        root.addWidget(label("Transcriptions", "pageTitle"))

        if self.svc is None:
            ban = QFrame()
            ban.setObjectName("banner")
            bl = QVBoxLayout(ban)
            bl.setContentsMargins(14, 10, 14, 10)
            bl.addWidget(label("Transcriptions aren't available. The transcription service didn't start; "
                               "the log file (About page) says why. Dictation still works.", wrap=True))
            root.addWidget(ban)

        root.addWidget(self._build_actions())
        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)
        split.setHandleWidth(1)
        split.addWidget(self._build_list())
        split.addWidget(self._build_detail())
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([290, 700])
        root.addWidget(split, 1)

        self._elapsed = QTimer(self)
        self._elapsed.setInterval(500)
        self._elapsed.timeout.connect(self._tick_elapsed)
        self._rec_guard = QTimer(self)          # re-sync the record button if no meeting_state arrives
        self._rec_guard.setSingleShot(True)
        self._rec_guard.setInterval(8000)
        self._rec_guard.timeout.connect(self._sync_meeting)

        b = getattr(app, "bridge", None)
        if b is not None:
            b.tr_progress.connect(self._on_progress)
            b.tr_done.connect(self._on_done)
            b.meeting_state.connect(self._on_meeting_state)
            b.meeting_levels.connect(self._on_levels)
            b.ask_partial.connect(self._on_ask_partial)
            b.ask_status.connect(self._on_ask_status)
            b.ask_done.connect(self._on_ask_done)
        if self.svc is None:
            for w in (self.rec_btn, self.file_btn, self.style_combo, self.search):
                w.setEnabled(False)
        self._sync_meeting()
        self._show_detail(None)

    # ------------------------------------------------------------------ building
    def _build_actions(self) -> QWidget:
        w = QFrame()
        w.setObjectName("panel")
        lay = QHBoxLayout(w)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(12)
        self.rec_btn = btn("Record meeting", "record")
        self.rec_btn.setToolTip("Records your microphone as You and this PC's sound as Others. "
                                "Transcribed when you stop.")
        self.rec_btn.setIcon(self._dot_icon())
        self.rec_btn.setIconSize(QSize(10, 10))
        self.rec_btn.setFixedHeight(32)
        self.rec_btn.clicked.connect(self._toggle_record)
        lay.addWidget(self.rec_btn)
        self.rec_time = label("", "statValue")
        self.rec_time.setStyleSheet("font-size: 17px;")
        lay.addWidget(self.rec_time)
        self.m_you = LevelMeter("You", YOU)
        self.m_oth = LevelMeter("Others", OTHERS)
        lay.addWidget(self.m_you)
        lay.addWidget(self.m_oth)
        self.rec_hint = label("Your mic is saved as You, this PC's sound as Others.", role="faint")
        lay.addWidget(self.rec_hint)
        lay.addStretch(1)
        lay.addWidget(hsep(vertical=True))
        lay.addWidget(label("Style", role="muted"))
        self.style_combo = QComboBox()
        self.style_combo.setFixedWidth(170)
        self.style_combo.setToolTip("Raw keeps the words as heard. A style runs AI cleanup on the text.")
        lay.addWidget(self.style_combo)
        self.file_btn = btn("Transcribe file...", "primary")
        self.file_btn.clicked.connect(self._pick_file)
        lay.addWidget(self.file_btn)
        self._fill_styles()
        return w

    @staticmethod
    def _dot_icon():
        from PySide6.QtGui import QIcon, QPixmap
        pm = QPixmap(20, 20)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(REC))
        p.drawEllipse(QRectF(2, 2, 16, 16))
        p.end()
        return QIcon(pm)

    @staticmethod
    def _stop_icon():
        from PySide6.QtGui import QIcon, QPixmap
        pm = QPixmap(20, 20)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("white"))
        p.drawRoundedRect(QRectF(3, 3, 14, 14), 3, 3)
        p.end()
        return QIcon(pm)

    def _build_list(self) -> QWidget:
        w = QWidget()
        w.setMinimumWidth(240)
        w.setMaximumWidth(420)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 14, 0)
        lay.setSpacing(10)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search transcripts")
        self.search.setClearButtonEnabled(True)
        self._search_t = QTimer(self)
        self._search_t.setSingleShot(True)
        self._search_t.setInterval(220)
        self._search_t.timeout.connect(self.refresh_list)
        self.search.textChanged.connect(lambda _t: self._search_t.start())
        lay.addWidget(self.search)
        self.list = QListWidget()
        self.list.setObjectName("trList")
        self.list.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list.currentItemChanged.connect(self._on_select)
        lay.addWidget(self.list, 1)
        self.list_empty = label("", role="muted", wrap=True)
        self.list_empty.setAlignment(Qt.AlignCenter)
        self.list_empty.setStyleSheet("padding: 30px 10px;")
        lay.addWidget(self.list_empty)
        return w

    def _build_detail(self) -> QWidget:
        self.detail_stack = QStackedWidget()
        # 0: nothing selected
        e = QWidget()
        el = QVBoxLayout(e)
        el.addStretch(1)
        self.detail_empty = label("", role="muted", wrap=True)
        self.detail_empty.setAlignment(Qt.AlignCenter)
        el.addWidget(self.detail_empty)
        el.addStretch(2)
        self.detail_stack.addWidget(e)

        # 1: a transcript
        d = QWidget()
        dl = QVBoxLayout(d)
        dl.setContentsMargins(18, 0, 0, 0)
        dl.setSpacing(10)
        self.title_edit = QLineEdit()
        self.title_edit.setObjectName("titleEdit")
        self.title_edit.setToolTip("Click to rename")
        self.title_edit.editingFinished.connect(self._rename)
        mrow = QHBoxLayout()
        mrow.setSpacing(6)
        mrow.addWidget(self.title_edit, 1)
        self.note = label(role="faint")
        mrow.addWidget(self.note)
        self.copy_btn = btn("Copy", "mini")
        self.copy_btn.clicked.connect(self._copy)
        mrow.addWidget(self.copy_btn)
        self.export_btn = btn("Export", "mini")
        self.export_btn.setProperty("menu", "1")
        m = QMenu(self.export_btn)
        m.addAction("Text file (.txt)", lambda: self._export("txt"))
        m.addAction("Markdown (.md)", lambda: self._export("md"))
        self.export_btn.setMenu(m)
        mrow.addWidget(self.export_btn)
        self.del_btn = btn("Delete", "mini")
        self.del_btn.setProperty("role", "danger")
        self.del_btn.setStyleSheet("padding: 4px 10px; font-size: 12px;")
        self.del_btn.clicked.connect(self._delete)
        mrow.addWidget(self.del_btn)
        dl.addLayout(mrow)
        self.meta = label(role="muted")
        self.meta.setContentsMargins(7, 0, 0, 0)
        self.meta.setWordWrap(True)
        dl.addWidget(self.meta)

        # job band: progress + Cancel, or the error + Resume
        self.band = QFrame()
        self.band.setObjectName("panel")
        bl = QHBoxLayout(self.band)
        bl.setContentsMargins(14, 10, 10, 10)
        bl.setSpacing(12)
        bcol = QVBoxLayout()
        bcol.setSpacing(6)
        self.band_text = label(wrap=True)
        bcol.addWidget(self.band_text)
        self.band_bar = QProgressBar()
        self.band_bar.setRange(0, 1000)
        self.band_bar.setTextVisible(False)
        bcol.addWidget(self.band_bar)
        bl.addLayout(bcol, 1)
        self.cancel_btn = btn("Cancel", "mini")
        self.cancel_btn.clicked.connect(self._cancel_job)
        bl.addWidget(self.cancel_btn)
        self.resume_btn = btn("Resume", "primary")
        self.resume_btn.clicked.connect(self._resume_job)
        bl.addWidget(self.resume_btn)
        dl.addWidget(self.band)

        # tabs
        trow = QHBoxLayout()
        trow.setSpacing(10)
        seg = QFrame()
        seg.setObjectName("segBox")
        sl = QHBoxLayout(seg)
        sl.setContentsMargins(3, 3, 3, 3)
        sl.setSpacing(2)
        self.tab_text = btn("Transcript", "seg")
        self.tab_ai = btn("AI notes", "seg")
        self._tabs = QButtonGroup(self)
        for i, b_ in enumerate((self.tab_text, self.tab_ai)):
            b_.setCheckable(True)
            self._tabs.addButton(b_, i)
            sl.addWidget(b_)
        self.tab_text.setChecked(True)
        trow.addWidget(seg)
        trow.addStretch(1)
        self.echo_cb = QCheckBox("Show filtered echo lines")
        self.echo_cb.setToolTip("Lines where your mic picked up the other side's audio are hidden. "
                                "Turn this on to show them.")
        self.echo_cb.toggled.connect(self._echo_toggled)
        trow.addWidget(self.echo_cb)
        dl.addLayout(trow)

        self.body = QStackedWidget()
        self.reader = QTextBrowser()
        self.reader.setObjectName("reader")
        self.reader.setOpenLinks(False)
        self.reader.document().setDocumentMargin(14)
        self.body.addWidget(self.reader)
        self.body.addWidget(self._build_ai())
        dl.addWidget(self.body, 1)
        self._tabs.idClicked.connect(self.body.setCurrentIndex)
        self.detail_stack.addWidget(d)
        return self.detail_stack

    def _build_ai(self) -> QWidget:
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setFrameShape(QFrame.NoFrame)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        inner = QWidget()
        inner.setObjectName("pageInner")
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(0, 4, 10, 10)
        lay.setSpacing(10)
        lay.addWidget(label("Ask the local AI about this transcript. Answers are saved here.", role="muted",
                            wrap=True))
        chips = QHBoxLayout()
        chips.setSpacing(8)
        self.chip_btns = []
        for preset, text in PRESET_LABELS.items():
            b_ = btn(text, "chipBtn")
            b_.clicked.connect(lambda _=False, p=preset: self._ask(p))
            chips.addWidget(b_)
            self.chip_btns.append(b_)
        chips.addStretch(1)
        lay.addLayout(chips)
        crow = QHBoxLayout()
        crow.setSpacing(8)
        self.prompt = QPlainTextEdit()
        self.prompt.setPlaceholderText("Ask your own question, e.g. What did we decide about the launch date?")
        self.prompt.setFixedHeight(64)
        crow.addWidget(self.prompt, 1)
        self.ask_btn = btn("Ask", "primary", 80)
        self.ask_btn.clicked.connect(lambda: self._ask("custom"))
        crow.addWidget(self.ask_btn, 0, Qt.AlignBottom)
        lay.addLayout(crow)
        self.ai_hint = label("", role="faint", wrap=True)
        lay.addWidget(self.ai_hint)

        # live answer
        self.live = QFrame()
        self.live.setObjectName("panel")
        self.live.setStyleSheet("QFrame#panel { border-color: rgba(122,170,255,0.45); }")
        ll = QVBoxLayout(self.live)
        ll.setContentsMargins(16, 10, 10, 14)
        ll.setSpacing(6)
        lh = QHBoxLayout()
        self.live_title = QLabel("")
        self.live_title.setStyleSheet("font-weight: 600;")
        lh.addWidget(self.live_title)
        self.live_status = label(role="faint")
        lh.addWidget(self.live_status)
        lh.addStretch(1)
        self.live_cancel = btn("Stop", "ghost")
        self.live_cancel.clicked.connect(self._cancel_ask)
        lh.addWidget(self.live_cancel)
        ll.addLayout(lh)
        self.live_text = QLabel("")
        self.live_text.setTextFormat(Qt.MarkdownText)
        self.live_text.setWordWrap(True)
        self.live_text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.live_text.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        ll.addWidget(self.live_text)
        lay.addWidget(self.live)
        self.live.hide()

        self.notes_title = label("Saved answers", "sectionTitle")
        self.notes_title.setStyleSheet("font-size: 14px;")
        lay.addSpacing(4)
        lay.addWidget(self.notes_title)
        self.notes_box = QVBoxLayout()
        self.notes_box.setSpacing(10)
        lay.addLayout(self.notes_box)
        self.notes_empty = label("No saved answers yet. Pick one of the buttons above to start.", role="faint",
                                 wrap=True)
        lay.addWidget(self.notes_empty)
        lay.addStretch(1)
        sa.setWidget(inner)
        return sa

    def _fill_styles(self) -> None:
        names = ["raw"]
        try:
            if self.app is not None:
                names = list(self.app.transcript_styles()) or ["raw"]
        except Exception:
            pass
        default = "raw"
        try:
            default = str(self.cfg.get("transcripts.file_style", "raw") or "raw") if self.cfg is not None else "raw"
        except Exception:
            pass
        cfg_default = default
        if self.style_combo.count() and self.style_combo.currentData() != getattr(self, "_style_default", None):
            default = self.style_combo.currentData()      # keep a pick made on this page
        self._style_default = cfg_default
        self.style_combo.blockSignals(True)
        self.style_combo.clear()
        for n in names:
            self.style_combo.addItem(style_label(n), n)
        i = self.style_combo.findData(default)
        self.style_combo.setCurrentIndex(i if i >= 0 else 0)
        self.style_combo.blockSignals(False)

    # ------------------------------------------------------------------ list
    def refresh_list(self) -> None:
        if not self.isVisible():
            self._dirty = True
            return
        self._dirty = False
        rows = []
        if self.svc is not None:
            try:
                rows = self.svc.list(self.search.text().strip())
            except Exception:
                rows = []
        ids = [int(r["id"]) for r in rows]
        same_order = ids == [int(self.list.item(i).data(Qt.UserRole)) for i in range(self.list.count())]
        self.list.blockSignals(True)
        try:
            if same_order:
                for r in rows:
                    tid = int(r["id"])
                    self._rows[tid] = r
                    wdg = self.list.itemWidget(self._items[tid])
                    if wdg is not None:
                        wdg.update_row(r)
            else:
                self.list.clear()
                self._items.clear()
                self._rows = {}
                for r in rows:
                    tid = int(r["id"])
                    self._rows[tid] = r
                    it = QListWidgetItem()
                    it.setData(Qt.UserRole, tid)
                    wdg = _Row(r)
                    it.setSizeHint(QSize(100, wdg.sizeHint().height()))
                    self.list.addItem(it)
                    self.list.setItemWidget(it, wdg)
                    self._items[tid] = it
                if self._cur in self._items:
                    self.list.setCurrentItem(self._items[self._cur])
        finally:
            self.list.blockSignals(False)
        for tid, it in self._items.items():       # active rows grow a progress line
            wdg = self.list.itemWidget(it)
            if wdg is not None:
                it.setSizeHint(QSize(100, wdg.sizeHint().height()))
        q = self.search.text().strip()
        self.list_empty.setVisible(not rows)
        self.list.setVisible(bool(rows))
        if self.svc is None:
            self.list_empty.setText("Not available.")
        else:
            self.list_empty.setText("No matches." if q else "No transcripts yet.")
        if self._cur is not None and self._cur not in self._items and not q:
            self._show_detail(None)
        elif self._cur in self._rows:
            self._load_detail(self._cur, keep=True)
        elif self._cur is None:
            self._show_detail(None)

    def _on_select(self, cur, _prev) -> None:
        if cur is None:
            return
        tid = int(cur.data(Qt.UserRole))
        if tid != self._cur:
            if not (self._ask_handle is not None and self._ask_tid == tid):
                self.tab_text.setChecked(True)          # a new transcript opens on its text
                self.body.setCurrentIndex(0)
            self._load_detail(tid)

    # ------------------------------------------------------------------ detail
    def _show_detail(self, _none) -> None:
        self._cur = None
        self._cur_row = None
        has_any = bool(self._items)
        if self.svc is None:
            self.detail_empty.setText("")
        elif has_any:
            self.detail_empty.setText("Pick a transcript on the left to read it.")
        else:
            self.detail_empty.setText("Record a meeting or transcribe an audio file.\nTranscripts show up here.")
        self.detail_stack.setCurrentIndex(0)

    def _load_detail(self, tid: int, keep: bool = False) -> None:
        if self.svc is None:
            return
        try:
            row = self.svc.get(tid)
        except Exception:
            row = None
        if row is None:
            self._show_detail(None)
            return
        same = keep and self._cur == tid
        self._cur, self._cur_row = tid, row
        self.detail_stack.setCurrentIndex(1)
        if not (same and self.title_edit.hasFocus()):
            self.title_edit.setText(row.get("title") or "")
            self.title_edit.setCursorPosition(0)
        meeting = row.get("kind") == "meeting"
        parts = ["Meeting recording" if meeting else "Audio file", fmt_when(row.get("created"))]
        if row.get("duration_s"):
            parts.append(fmt_duration(row.get("duration_s")))
        st = row.get("style") or "raw"
        parts.append("raw transcript" if st == "raw" else f"{st} style")
        if not meeting and row.get("source_path"):
            parts.append(os.path.basename(row["source_path"]))
        self.meta.setText(",  ".join(p for p in parts if p))
        self.meta.setToolTip(row.get("source_path") or row.get("audio_dir") or "")
        self._update_band(row.get("status") or "", row.get("progress"), "", row.get("error") or "")

        text_html = transcript_html(row)
        sb = self.reader.verticalScrollBar()
        pos = sb.value() if same else 0
        if not text_html:
            status = row.get("status") or ""
            if status in ACTIVE:
                msg = "The transcript appears here when it's done."
            elif status in RESUMABLE:
                msg = "No text yet. Resume the job to finish it."
            elif (row.get("info") or {}).get("no_speech"):
                msg = "No speech was found in this audio."
            else:
                msg = "This transcript is empty."
            text_html = f'<p style="color:{TEXT2}; font-size:14px;">{html.escape(msg)}</p>'
        if text_html != getattr(self, "_last_html", None) or not same:
            self.reader.setHtml(text_html)
            self._last_html = text_html
            QTimer.singleShot(0, lambda: sb.setValue(pos))

        info = row.get("info") or {}
        show_echo = meeting and (bool(info.get("echo_marked")) or bool(row.get("include_echo")))
        self.echo_cb.blockSignals(True)
        self.echo_cb.setChecked(bool(row.get("include_echo")))
        self.echo_cb.blockSignals(False)
        self.echo_cb.setVisible(show_echo and row.get("status") == "done")
        n = info.get("echo_marked")
        if show_echo and n:
            self.echo_cb.setText(f"Show {n} filtered echo line{'s' if n != 1 else ''}")
        has_text = bool((row.get("text") or row.get("raw_text") or "").strip())
        self.copy_btn.setEnabled(has_text)
        self.export_btn.setEnabled(has_text)
        self._load_notes(tid, done=row.get("status") == "done" and has_text)
        self._update_ask_ui()

    def _update_band(self, status: str, progress, msg: str, error: str) -> None:
        active = status in ACTIVE
        resumable = status in RESUMABLE
        self.band.setVisible(active or resumable)
        self.band_bar.setVisible(active)
        self.cancel_btn.setVisible(active)
        self.resume_btn.setVisible(resumable)
        if active:
            try:
                self.band_bar.setValue(int(float(progress or 0) * 1000))
            except Exception:
                self.band_bar.setValue(0)
            self.band_text.setText(status_text(status, progress, msg))
            self.band_text.setStyleSheet("")
        elif resumable:
            txt = {"failed": "Transcription failed", "interrupted": "Transcription was interrupted",
                   "cancelled": "Transcription was cancelled"}.get(status, status)
            if error:
                txt += f": {error}"
            self.band_text.setText(txt + ". Resume picks up where it stopped.")
            self.band_text.setStyleSheet("color: #f06b73;" if status == "failed" else "color: #e2a53b;")

    def _load_notes(self, tid: int, done: bool) -> None:
        while self.notes_box.count():
            it = self.notes_box.takeAt(0)
            w = it.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        rows = []
        try:
            rows = self.svc.list_ai_results(tid) if self.svc is not None else []
        except Exception:
            rows = []
        for r in rows:
            self.notes_box.addWidget(_NoteCard(r, self._delete_note))
        self.notes_empty.setVisible(not rows)
        self.notes_empty.setText("No saved answers yet. Pick one of the buttons above to start." if done else
                                 "You can ask the AI once the transcript is done.")
        self.tab_ai.setText(f"AI notes ({len(rows)})" if rows else "AI notes")
        self._can_ask = done

    def _update_ask_ui(self) -> None:
        busy = self._ask_handle is not None
        can = bool(getattr(self, "_can_ask", False)) and not busy and self.app is not None
        for b_ in self.chip_btns:
            b_.setEnabled(can)
        self.ask_btn.setEnabled(can)
        self.prompt.setEnabled(bool(getattr(self, "_can_ask", False)))
        here = busy and self._ask_tid == self._cur
        self.live.setVisible(here or (bool(self.live_text.text()) and self._ask_tid == self._cur
                                      and self.live_cancel.isHidden()))
        if busy and not here:
            other = (self._rows.get(self._ask_tid) or {}).get("title") or "another transcript"
            self.ai_hint.setText(f"The AI is answering about \"{other}\". One question at a time.")
        else:
            self.ai_hint.setText("")

    # ------------------------------------------------------------------ actions: detail
    def _rename(self) -> None:
        if self._cur is None or self.svc is None or self._cur_row is None:
            return
        t = self.title_edit.text().strip()
        if not t:
            self.title_edit.setText(self._cur_row.get("title") or "")
            return
        if t == (self._cur_row.get("title") or ""):
            return
        try:
            self.svc.rename(self._cur, t)
        except Exception as e:
            self._note(f"Rename failed: {e}")
            return
        self._cur_row["title"] = t
        self.refresh_list()

    def _copy(self) -> None:
        if not self._cur_row:
            return
        QGuiApplication.clipboard().setText((self._cur_row.get("text") or self._cur_row.get("raw_text") or "").strip())
        flash_copied(self.copy_btn, "Copy")

    def _export(self, fmt: str) -> None:
        if self._cur is None or self.svc is None:
            return
        name = _safe_filename(self.title_edit.text()) + f".{fmt}"
        start = os.path.join(self._last_dir, name)
        flt = "Text file (*.txt)" if fmt == "txt" else "Markdown (*.md)"
        path, _ = QFileDialog.getSaveFileName(self, "Export transcript", start, flt)
        if not path:
            return
        self._last_dir = os.path.dirname(path)
        try:
            self.svc.export(self._cur, fmt, path)
        except Exception as e:
            self._note(f"Export failed: {e}")
            return
        self._note(f"Saved {os.path.basename(path)}")

    def _note(self, text: str) -> None:
        self.note.setText(text)
        QTimer.singleShot(3500, lambda: self.note.setText(""))

    def _delete(self) -> None:
        if self._cur is None or self.svc is None:
            return
        row = self._cur_row or {}
        what = "the transcript, its AI notes and the recorded audio" if row.get("kind") == "meeting" else \
            "the transcript and its AI notes (your audio file is not touched)"
        if not confirm(self, "Delete transcript", f"Delete \"{row.get('title') or 'Untitled'}\"?", "Delete",
                       f"This removes {what}. It can't be undone."):
            return
        tid = self._cur
        self.del_btn.setEnabled(False)
        svc, relay = self.svc, self._relay

        def run():
            try:
                svc.delete(tid)
            except Exception:
                pass
            relay.deleted.emit(tid)
        threading.Thread(target=run, name="transcript-ui-delete", daemon=True).start()
        self._show_detail(None)
        QTimer.singleShot(400, lambda: self.del_btn.setEnabled(True))

    def _cancel_job(self) -> None:
        if self._cur is not None and self.svc is not None:
            try:
                self.svc.cancel(self._cur)
            except Exception:
                pass
            self.cancel_btn.setEnabled(False)
            QTimer.singleShot(1500, lambda: self.cancel_btn.setEnabled(True))

    def _resume_job(self) -> None:
        if self._cur is not None and self.svc is not None:
            try:
                ok = self.svc.resume(self._cur)
            except Exception:
                ok = False
            if not ok:
                self._note("Couldn't resume this job.")
            self.refresh_list()

    def _echo_toggled(self, on: bool) -> None:
        if self._cur is None or self.svc is None:
            return
        try:
            self.svc.set_include_echo(self._cur, on)
        except Exception as e:
            self._note(f"Couldn't update: {e}")
        self._load_detail(self._cur, keep=True)

    # ------------------------------------------------------------------ actions: record / import
    def _toggle_record(self) -> None:
        if self.app is None:
            return
        self.rec_btn.setEnabled(False)
        self.rec_btn.setText("Stopping..." if self._rec_on else "Starting...")
        try:
            self.app.toggle_meeting()
        except Exception:
            pass
        self._rec_guard.start()

    def _sync_meeting(self) -> None:
        st = {}
        if self.svc is not None:
            try:
                st = self.svc.meeting_state()
            except Exception:
                st = {}
        self._apply_meeting(st)

    def _apply_meeting(self, st: dict) -> None:
        on = bool(st.get("recording"))
        if on and not self._rec_on:
            self._rec_t0 = time.monotonic() - float(st.get("elapsed") or 0.0)
        self._rec_on = on
        self.rec_btn.setEnabled(self.svc is not None)
        self.rec_btn.setText("Stop recording" if on else "Record meeting")
        self.rec_btn.setProperty("role", "recording" if on else "record")
        self.rec_btn.setIcon(self._stop_icon() if on else self._dot_icon())
        repolish(self.rec_btn)
        for w in (self.rec_time, self.m_you, self.m_oth):
            w.setVisible(on)
        self.rec_hint.setVisible(not on)
        if on:
            self._tick_elapsed()
            if self.isVisible():
                self._elapsed.start()
        else:
            self._elapsed.stop()
            self.m_you.reset()
            self.m_oth.reset()
        if not on and st.get("stop_reason") and st.get("stop_reason") != "stopped by user":
            self.rec_hint.setText(f"Recording stopped: {st['stop_reason']}")
        elif not on:
            self.rec_hint.setText("Your mic is saved as You, this PC's sound as Others.")

    def _tick_elapsed(self) -> None:
        self.rec_time.setText(fmt_elapsed(time.monotonic() - self._rec_t0))

    def _pick_file(self) -> None:
        if self.app is None or self.svc is None:
            return
        try:
            from .app import AUDIO_EXTS as exts
        except Exception:
            exts = AUDIO_EXTS
        flt = "Audio / video (" + " ".join(f"*.{e}" for e in exts.split()) + ");;All files (*)"
        path, _ = QFileDialog.getOpenFileName(self, "Transcribe audio file", self._last_dir, flt)
        if not path:
            return
        self._last_dir = os.path.dirname(path)
        style = self.style_combo.currentData() or "raw"
        app, relay = self.app, self._relay

        def run():
            tid = None
            try:
                tid = app.import_audio_file(path, style)
            except Exception:
                pass
            relay.imported.emit(tid)
        threading.Thread(target=run, name="transcript-ui-import", daemon=True).start()

    def _after_import(self, tid) -> None:
        self.search.blockSignals(True)
        self.search.clear()
        self.search.blockSignals(False)
        self.refresh_list()
        if tid is not None and int(tid) in self._items:
            self.list.setCurrentItem(self._items[int(tid)])

    # ------------------------------------------------------------------ AI
    def _ask(self, preset: str) -> None:
        if self._cur is None or self.app is None or self._ask_handle is not None:
            return
        prompt = None
        if preset == "custom":
            prompt = self.prompt.toPlainText().strip()
            if not prompt:
                self.prompt.setFocus()
                return
        try:
            h = self.app.ask_ai(self._cur, preset, prompt)
        except Exception as e:
            h = None
            self.ai_hint.setText(f"Couldn't ask: {e}")
        if h is None:
            return
        self._ask_handle, self._ask_tid = h, self._cur
        self.live_title.setText(preset_label(preset, prompt or ""))
        self.live_status.setText("Starting...")
        self.live_text.setText("")
        self.live_cancel.show()
        self.tab_ai.setChecked(True)
        self.body.setCurrentIndex(1)
        self._update_ask_ui()

    def _mine(self, handle) -> bool:
        return self._ask_handle is not None and (handle is None or handle is self._ask_handle)

    def _on_ask_partial(self, handle, text: str) -> None:
        if self._mine(handle):
            self.live_status.setText("Writing...")
            self.live_text.setText(text or "")

    def _on_ask_status(self, handle, msg: str) -> None:
        if self._mine(handle):
            self.live_status.setText(msg or "")

    def _on_ask_done(self, handle, row, error: str) -> None:
        if not self._mine(handle):
            return
        cancelled = self._ask_handle is not None and self._ask_handle.stop.is_set() if \
            hasattr(self._ask_handle, "stop") else False
        tid = self._ask_tid
        self._ask_handle = None
        self.live_cancel.hide()
        if error and not cancelled:
            self.live_status.setText("")
            self.live_text.setText(f"Couldn't get an answer: {error}")
        else:
            self.live_text.setText("")
        if tid == self._cur and self._cur is not None:
            row_ = self._cur_row or {}
            self._load_notes(self._cur, done=row_.get("status") == "done")
        self._update_ask_ui()
        if not error and not cancelled:
            self.prompt.clear()

    def _cancel_ask(self) -> None:
        h = self._ask_handle
        if h is not None:
            try:
                h.cancel()
            except Exception:
                pass
            self.live_status.setText("Stopping...")

    def _delete_note(self, r: dict) -> None:
        if not confirm(self, "Delete answer", f"Delete the saved \"{preset_label(r.get('preset', ''), r.get('prompt', ''))}\"?",
                       "Delete"):
            return
        try:
            self.svc.delete_ai_result(int(r["id"]))
        except Exception:
            pass
        if self._cur is not None:
            self._load_notes(self._cur, done=(self._cur_row or {}).get("status") == "done")
            self._update_ask_ui()

    # ------------------------------------------------------------------ bridge signals
    def _on_progress(self, tid: int, status: str, progress: float, msg: str) -> None:
        tid = int(tid)
        it = self._items.get(tid)
        if it is None:
            self.refresh_list()
            return
        wdg = self.list.itemWidget(it)
        if wdg is not None:
            before = wdg.bar.isVisible()
            wdg.set_status(status, progress, msg)
            if wdg.bar.isVisible() != before:
                it.setSizeHint(QSize(100, wdg.sizeHint().height()))
        if tid == self._cur:
            self._update_band(status, progress, msg, "")

    def _on_done(self, tid: int, _status: str) -> None:
        self.refresh_list()

    def _on_meeting_state(self, st) -> None:
        self._rec_guard.stop()
        self._apply_meeting(dict(st or {}))
        if not (st or {}).get("recording"):
            self.refresh_list()

    def _on_levels(self, mic: float, system: float) -> None:
        if self._rec_on and self.isVisible():
            self.m_you.set_level(level_unit(mic))
            self.m_oth.set_level(level_unit(system))

    # ------------------------------------------------------------------ visibility
    def showEvent(self, e):
        super().showEvent(e)
        self._fill_styles()
        self._sync_meeting()
        self.refresh_list()

    def hideEvent(self, e):
        self._elapsed.stop()
        super().hideEvent(e)
