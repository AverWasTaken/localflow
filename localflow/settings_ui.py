"""The LocalFlow window: Home, Transcriptions and the settings pages (General, Dictionary, Snippets, Styles,
About). The theme lives in theme.py and is re-exported here for older imports."""
from __future__ import annotations

import copy
import os
import re
import tempfile
from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QFrame, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
                               QPlainTextEdit, QSlider, QSpinBox, QStackedWidget, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from . import __version__
from . import styles as _styles
from .cleanup import list_models, model_display_name, same_model
from .paths import CONFIG_PATH, LOG_PATH

# ---------------------------------------------------------------- theme (lives in theme.py; re-exported here)
from .theme import (ACCENT, ACCENT_HOVER, ACCENT_PRESSED, ACCENT_TEXT, BG, BORDER, DANGER, DARK_QSS,  # noqa: F401,E402
                    MUTED, SIDEBAR, SURFACE, SURFACE2, TEXT, TEXT2, TEXT3, Card, Switch, _dark_palette, _url,
                    apply_theme, enable_dark_titlebar, label as _label, style_window)
from .theme import btn as _btn, page as _page  # noqa: E402
from .uifmt import (RETENTION_CHOICES, command_combo_error, fmt_chord, parse_mods)  # noqa: E402


def _combo(width: int = 250) -> QComboBox:
    c = QComboBox()
    c.setFixedWidth(width)
    c.setMaxVisibleItems(14)
    return c


def _set_combo_data(combo: QComboBox, data, fallback_label: str | None = None) -> None:
    i = combo.findData(data)
    if i < 0:
        combo.addItem(fallback_label or str(data), data)
        i = combo.count() - 1
    combo.setCurrentIndex(i)


def _table(headers: list[str]) -> QTableWidget:
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(36)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    t.horizontalHeader().setHighlightSections(False)
    t.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    t.setShowGrid(False)
    t.setSelectionBehavior(QAbstractItemView.SelectRows)
    t.setSelectionMode(QAbstractItemView.SingleSelection)
    t.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
                      | QAbstractItemView.AnyKeyPressed)
    t.setFrameShape(QFrame.NoFrame)
    t.setWordWrap(False)
    t.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
    return t


LANGUAGES = [("en", "English"), ("auto", "Auto-detect"), ("es", "Spanish"), ("fr", "French"),
             ("de", "German"), ("it", "Italian"), ("pt", "Portuguese"), ("nl", "Dutch"), ("pl", "Polish"),
             ("ru", "Russian"), ("uk", "Ukrainian"), ("tr", "Turkish"), ("sv", "Swedish"),
             ("ja", "Japanese"), ("ko", "Korean"), ("zh", "Chinese"), ("hi", "Hindi"), ("ar", "Arabic")]

MODELS = [("large-v3-turbo", "large-v3-turbo  (recommended)"),
          ("large-v3", "large-v3  (most accurate)"),
          ("distil-large-v3", "distil-large-v3  (fast)"),
          ("medium.en", "medium.en  (English)"),
          ("small.en", "small.en  (English, light)"),
          ("base.en", "base.en  (English, tiny)"),
          ("tiny.en", "tiny.en  (English, fastest)")]


class _NoWheelCombo(QComboBox):
    """Combo inside a scrolling table: the mouse wheel scrolls the table instead of changing the value."""

    def wheelEvent(self, e):
        e.ignore()


class _DraftCfg:
    """Minimal cfg stand-in for styles.resolve(): live (unsaved) profiles + auto_style over the real config."""

    def __init__(self, base, overrides: dict):
        self._base = base
        self._over = overrides

    def get(self, key, default=None):
        if key in self._over:
            return self._over[key]
        return self._base.get(key, default)


def _style_flags(sd: dict) -> str:
    out = []
    if sd.get("lowercase"):
        out.append("lowercase")
    if sd.get("capitalize"):
        out.append("capitalized")
    stp = sd.get("strip_trailing_period")
    if stp:
        out.append("no final period" + (" (single sentence)" if stp == "single" else ""))
    if sd.get("abbreviations"):
        out.append("abbreviations")
    if sd.get("drop_apostrophes"):
        out.append("no apostrophes")
    if sd.get("email_layout"):
        out.append("email layout")
    return ", ".join(out) or "-"


# ---------------------------------------------------------------- settings window
PAGES = ("home", "transcriptions", "general", "dictionary", "snippets", "styles", "about")


def transcript_style_names(cfg) -> list[str]:
    """Style choices for transcripts ("raw" = no AI cleanup first), same rule as LocalFlowApp.transcript_styles."""
    try:
        names = list((cfg.get("styles", {}) or {}).keys()) or list(_styles.DEFAULT_STYLES)
    except Exception:
        names = list(_styles.DEFAULT_STYLES)
    return ["raw"] + [n for n in names if n != "raw"]


class SettingsWindow(QWidget):
    def __init__(self, config, get_mics: Callable[[], list], autostart_get: Callable[[], bool],
                 autostart_set: Callable[[bool], None],
                 get_last_ctx: Callable[[], tuple[str, str]] | None = None,
                 get_default_mic: Callable[[], str] | None = None, app=None,
                 get_devices: Callable[[], dict] | None = None):
        """``get_last_ctx`` returns ``(exe, title)`` of the last dictated window (for the Styles page's
        "Add rule for last used app" button); None / empty disables the button. ``app`` (LocalFlowApp) powers the
        Home and Transcriptions pages; without it they show an "unavailable" state. ``get_devices`` returns
        recorder.list_devices() for the meeting device pickers (default: that function when ``app`` is given)."""
        super().__init__()
        self.config = config
        self.app = app
        self._get_last_ctx = get_last_ctx
        self._rules: list[dict] = []          # Styles page model: original rule dicts, edited in place
        self._rules_loading = False
        self._get_mics = get_mics
        self._get_default_mic = get_default_mic
        self._autostart_get = autostart_get
        self._autostart_set = autostart_set
        if get_devices is None and app is not None:
            def get_devices():
                from .recorder import list_devices
                return list_devices()
        self._get_devices = get_devices
        self.setWindowTitle("LocalFlow")
        self.resize(1100, 740)
        self.setMinimumSize(900, 600)
        style_window(self)
        self.setFocusPolicy(Qt.ClickFocus)    # so opening the window doesn't drop the caret into a search box
        self._built_dark_titlebar = False
        self._snip_loading = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        body = QHBoxLayout()
        body.setSpacing(0)
        outer.addLayout(body, 1)

        # sidebar
        sb = QWidget()
        sb.setObjectName("sidebarBox")
        sb.setAttribute(Qt.WA_StyledBackground, True)
        sb.setFixedWidth(200)
        sbl = QVBoxLayout(sb)
        sbl.setContentsMargins(0, 20, 0, 14)
        sbl.setSpacing(10)
        brand = QLabel("LocalFlow")
        brand.setObjectName("brand")
        brand.setContentsMargins(24, 0, 0, 8)
        sbl.addWidget(brand)
        self.nav = QListWidget()
        self.nav.setObjectName("sidebar")
        self.nav.setFocusPolicy(Qt.NoFocus)
        self.nav.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sbl.addWidget(self.nav, 1)
        ver = QLabel(f"v{__version__}")
        ver.setProperty("role", "faint")
        ver.setContentsMargins(24, 0, 0, 0)
        sbl.addWidget(ver)
        body.addWidget(sb)

        self.stack = QStackedWidget()
        body.addWidget(self.stack, 1)
        self._page_index: dict[str, int] = {}
        for key, name, builder in (("home", "Home", self._build_home),
                                   ("transcriptions", "Transcriptions", self._build_transcriptions),
                                   (None, "Settings", None),
                                   ("general", "General", self._build_general),
                                   ("dictionary", "Dictionary", self._build_dictionary),
                                   ("snippets", "Snippets", self._build_snippets),
                                   ("styles", "Styles", self._build_styles),
                                   ("about", "About", self._build_about)):
            it = QListWidgetItem(name)
            if key is None:                       # group label between the app pages and the settings pages
                it.setFlags(Qt.NoItemFlags)
                it.setForeground(QColor(TEXT3))
                it.setData(Qt.UserRole, -1)
                self.nav.addItem(it)
                continue
            idx = self.stack.addWidget(builder())
            self._page_index[key] = idx
            it.setData(Qt.UserRole, idx)
            self.nav.addItem(it)
        self._settings_pages = {self._page_index[k] for k in ("general", "dictionary", "snippets", "styles", "about")}
        self.nav.currentRowChanged.connect(self._nav_changed)

        # bottom bar (settings pages only)
        bar = QWidget()
        bar.setObjectName("bottomBar")
        bar.setAttribute(Qt.WA_StyledBackground, True)
        self.bottom_bar = bar
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(24, 12, 24, 12)
        bl.setSpacing(10)
        self.saved_label = QLabel("")
        self.saved_label.setObjectName("saved")
        bl.addWidget(self.saved_label)
        bl.addStretch(1)
        self.btn_close = _btn("Close", w=96)
        self.btn_save = _btn("Save", "primary", 110)
        self.btn_save.setDefault(True)
        bl.addWidget(self.btn_close)
        bl.addWidget(self.btn_save)
        outer.addWidget(bar)
        self.btn_close.clicked.connect(self.close)
        self.btn_save.clicked.connect(self.save)

        self._saved_timer = QTimer(self)
        self._saved_timer.setSingleShot(True)
        self._saved_timer.timeout.connect(lambda: self.saved_label.setText(""))
        self.show_page("home")
        self.reload()

    # ----- navigation -----
    def _nav_changed(self, row: int) -> None:
        it = self.nav.item(row)
        idx = it.data(Qt.UserRole) if it is not None else -1
        if idx is None or idx < 0:
            return
        self.stack.setCurrentIndex(idx)
        self.bottom_bar.setVisible(idx in self._settings_pages)
        self._refresh_last_ctx()

    def _goto_index(self, idx: int) -> None:
        for r in range(self.nav.count()):
            if self.nav.item(r).data(Qt.UserRole) == idx:
                self.nav.setCurrentRow(r)
                return

    def show_page(self, name: str) -> None:
        """Switch to a page by key ("home", "transcriptions", "general", ...). Never reloads the settings."""
        idx = self._page_index.get(name)
        if idx is not None:
            self._goto_index(idx)

    def current_page(self) -> str:
        idx = self.stack.currentIndex()
        return next((k for k, v in self._page_index.items() if v == idx), "")

    def _build_home(self) -> QWidget:
        from .home_ui import HomePage
        self.home_page = HomePage(getattr(self.app, "history", None))
        return self.home_page

    def _build_transcriptions(self) -> QWidget:
        from .transcripts_ui import TranscriptsPage
        self.transcripts_page = TranscriptsPage(self.app, self.config)
        return self.transcripts_page

    # ----- pages -----
    def _build_general(self) -> QWidget:
        sa, lay = _page("General", "Dictation basics. Changes apply when you press Save.")

        # --- input
        c = Card("Input")
        mods = QWidget()
        ml = QHBoxLayout(mods)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(18)
        self.mod_boxes: dict[str, QCheckBox] = {}
        for key, label in (("ctrl", "Ctrl"), ("alt", "Alt"), ("shift", "Shift"), ("win", "Win")):
            cb = QCheckBox(label)
            cb.setCursor(Qt.PointingHandCursor)
            self.mod_boxes[key] = cb
            ml.addWidget(cb)
        ml.addStretch(1)
        c.add_row("Push-to-talk keys", "Hold these modifier keys together to dictate; release to insert. "
                  "Ctrl+Alt is the default.", mods, stack=True)
        self.handsfree_info = QLineEdit()
        self.handsfree_info.setReadOnly(True)
        self.handsfree_info.setFixedWidth(250)
        c.add_row("Hands-free lock", "While holding the hotkey, tap this key to keep recording without "
                  "holding. Tap the hotkey again to finish. (Edit handsfree_key in config.json.)",
                  self.handsfree_info)
        self.mic_combo = _combo(340)
        c.add_row("Microphone", "Auto follows the Windows default recording device, even when you change it later.", self.mic_combo)
        lay.addWidget(c)

        # --- command mode + paste last
        c = Card("Command mode")
        self.cmd_sw = Switch()
        c.add_row("Voice commands", "Hold the command keys and say what to do, like \"make this more formal\" or "
                  "\"translate to German\". With text selected it rewrites the selection, otherwise it writes the "
                  "answer at the cursor.", self.cmd_sw)
        box = QWidget()
        bl_ = QVBoxLayout(box)
        bl_.setContentsMargins(0, 0, 0, 0)
        bl_.setSpacing(6)
        cm = QHBoxLayout()
        cm.setSpacing(18)
        self.cmd_boxes: dict[str, QCheckBox] = {}
        for key, lbl in (("ctrl", "Ctrl"), ("alt", "Alt"), ("shift", "Shift"), ("win", "Win")):
            cb = QCheckBox(lbl)
            cb.setCursor(Qt.PointingHandCursor)
            self.cmd_boxes[key] = cb
            cm.addWidget(cb)
        cm.addStretch(1)
        bl_.addLayout(cm)
        self.cmd_error = _label("", "fieldError", wrap=True)
        self.cmd_error.hide()
        bl_.addWidget(self.cmd_error)
        c.add_row("Command keys", "Hold these together for a command. They can't overlap the dictation keys. "
                  "Ctrl+Win is the default.", box, stack=True)
        self.paste_sw = Switch()
        self.paste_key = QLineEdit()
        self.paste_key.setReadOnly(True)
        self.paste_key.setFixedWidth(140)
        pk = QWidget()
        pkl = QHBoxLayout(pk)
        pkl.setContentsMargins(0, 0, 0, 0)
        pkl.setSpacing(14)
        pkl.addWidget(self.paste_key)
        pkl.addWidget(self.paste_sw)
        c.add_row("Paste last transcript", "Press this shortcut to paste your last dictation again. "
                  "(Change the keys under paste_last in config.json.)", pk)
        lay.addWidget(c)
        for cb in list(self.mod_boxes.values()) + list(self.cmd_boxes.values()):
            cb.toggled.connect(lambda _v: self._validate_cmd())
        self.cmd_sw.toggled.connect(lambda _v: self._validate_cmd())

        # --- speech recognition
        c = Card("Speech recognition")
        self.lang_combo = _combo(250)
        for code, name in LANGUAGES:
            self.lang_combo.addItem(name, code)
        c.add_row("Language", "Spoken language, or let the model detect it.", self.lang_combo)
        self.model_combo = _combo(300)
        for code, name in MODELS:
            self.model_combo.addItem(name, code)
        c.add_row("Model", "Whisper model. *.en models understand English only. Changing it reloads the model.",
                  self.model_combo)
        lay.addWidget(c)

        # --- output
        c = Card("Output")
        self.insert_combo = _combo(250)
        self.insert_combo.addItem("Paste (clipboard + Ctrl+V)", "paste")
        self.insert_combo.addItem("Type (simulated keystrokes)", "type")
        c.add_row("Insert method", "Paste is fastest and works almost everywhere; typing avoids the clipboard.",
                  self.insert_combo)
        self.restore_sw = Switch()
        c.add_row("Restore clipboard", "Put your previous clipboard contents back after pasting.", self.restore_sw)
        self.space_sw = Switch()
        c.add_row("Add trailing space", "Append a space after each dictation.", self.space_sw)
        lay.addWidget(c)

        # --- AI cleanup
        c = Card("AI cleanup")
        self.cleanup_sw = Switch()
        c.add_row("Clean up with local AI", "Removes filler words and fixes punctuation using a small local model.",
                  self.cleanup_sw)
        self.llm_combo = _NoWheelCombo()      # the wheel would switch (and reload) models while scrolling
        self.llm_combo.setFixedWidth(250)
        self.llm_combo.activated.connect(self._model_picked)
        c.add_row("Cleanup model", "The local model that does the cleanup (.gguf files in the models folder). "
                  "Applies right away: the model reloads in a few seconds, basic cleanup is used meanwhile.",
                  self.llm_combo)
        self.minwords_spin = QSpinBox()
        self.minwords_spin.setRange(0, 50)
        self.minwords_spin.setFixedWidth(90)
        c.add_row("Minimum words", "Utterances shorter than this skip the AI and are inserted as heard.",
                  self.minwords_spin)
        self.stream_sw = Switch()
        c.add_row("Transcribe while talking", "For long dictations (7 s+), finish sentences in the background at "
                  "natural pauses so text appears sooner after you release. Thresholds are in config.json under "
                  "\"streaming\".", self.stream_sw)
        lay.addWidget(c)

        # --- meetings + transcription
        c = Card("Meetings and audio files")
        self.meet_mic_combo = _combo(340)
        c.add_row("Meeting microphone", "Recorded as You. Same as dictation uses the microphone picked above.",
                  self.meet_mic_combo)
        self.meet_out_combo = _combo(340)
        c.add_row("Other people's audio", "The speakers or headset you hear the meeting on. LocalFlow records what "
                  "plays there as Others.", self.meet_out_combo)
        self.meet_style_combo = _combo(250)
        c.add_row("Meeting style", "Raw keeps the words as heard. A style runs AI cleanup on each turn.",
                  self.meet_style_combo)
        self.file_style_combo = _combo(250)
        c.add_row("Audio file style", "Preselected when you transcribe an audio file.", self.file_style_combo)
        lay.addWidget(c)

        # --- history
        c = Card("History")
        self.hist_sw = Switch()
        c.add_row("Save dictation history", "Keep what you dictate on the Home page. Usage numbers count either way.",
                  self.hist_sw)
        self.retention_combo = _combo(180)
        for days, lbl in RETENTION_CHOICES:
            self.retention_combo.addItem(lbl, days)
        c.add_row("Keep history for", "Older entries are deleted automatically.", self.retention_combo)
        self.hist_sw.toggled.connect(self.retention_combo.setEnabled)
        lay.addWidget(c)

        # --- game mode
        c = Card("Game mode")
        self.game_sw = Switch()
        c.add_row("Free the GPU while gaming", "Unloads the speech and cleanup models when a listed game starts and "
                  "reloads them shortly after it closes. Only process names are checked.", self.game_sw)
        self.game_mode_combo = _combo(250)
        self.game_mode_combo.addItem("Dictation off", "off")
        self.game_mode_combo.addItem("Dictation on CPU (basic cleanup)", "cpu")
        c.add_row("While a game runs", "CPU dictation uses a small English model and no AI cleanup.",
                  self.game_mode_combo)
        w = QWidget()
        gl = QVBoxLayout(w)
        gl.setContentsMargins(0, 8, 0, 10)
        gl.setSpacing(10)
        hint = QLabel("Game process names (e.g. VALORANT-Win64-Shipping.exe).")
        hint.setObjectName("rowDesc")
        hint.setWordWrap(True)
        gl.addWidget(hint)
        self.game_list = QListWidget()
        self.game_list.setObjectName("plain")
        self.game_list.setMinimumHeight(90)
        self.game_list.setMaximumHeight(140)
        self.game_list.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.game_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        gl.addWidget(self.game_list)
        r = QHBoxLayout()
        r.setSpacing(8)
        self.game_edit = QLineEdit()
        self.game_edit.setPlaceholderText("Add a process name...")
        self.game_edit.returnPressed.connect(self._add_game)
        add = _btn("Add", "primary", 80)
        add.clicked.connect(self._add_game)
        rm = _btn("Remove selected", "danger")
        rm.clicked.connect(self._remove_games)
        r.addWidget(self.game_edit, 1)
        r.addWidget(add)
        r.addWidget(rm)
        gl.addLayout(r)
        c.add_widget(w)
        lay.addWidget(c)

        # --- interface
        c = Card("Interface")
        self.overlay_sw = Switch()
        c.add_row("Show overlay", "The small waveform pill shown while recording.", self.overlay_sw)
        self.sounds_sw = Switch()
        c.add_row("Sounds", "Short start / stop / error cues.", self.sounds_sw)
        vol = QWidget()
        vl = QHBoxLayout(vol)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(10)
        self.vol_slider = QSlider(Qt.Horizontal)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setFixedWidth(190)
        self.vol_label = QLabel("25%")
        self.vol_label.setFixedWidth(40)
        self.vol_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.vol_slider.valueChanged.connect(lambda v: self.vol_label.setText(f"{v}%"))
        vl.addWidget(self.vol_slider)
        vl.addWidget(self.vol_label)
        c.add_row("Sound volume", "", vol)
        self.autostart_sw = Switch()
        c.add_row("Start at login", "Launch LocalFlow in the tray when you sign in to Windows.", self.autostart_sw)
        lay.addWidget(c)
        lay.addStretch(1)
        return sa

    def _build_dictionary(self) -> QWidget:
        sa, lay = _page("Dictionary", "Teach LocalFlow names, jargon and fixes for words it keeps getting wrong.")

        c = Card("Custom words")
        w = QWidget()
        wl = QVBoxLayout(w)
        wl.setContentsMargins(0, 8, 0, 10)
        wl.setSpacing(10)
        hint = QLabel("Names and terms to recognise and spell exactly (e.g. Kubernetes, Postgres, Alex).")
        hint.setObjectName("rowDesc")
        hint.setWordWrap(True)
        wl.addWidget(hint)
        self.words_list = QListWidget()
        self.words_list.setObjectName("plain")
        self.words_list.setMinimumHeight(150)
        self.words_list.setMaximumHeight(210)
        self.words_list.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.words_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        wl.addWidget(self.words_list)
        r = QHBoxLayout()
        r.setSpacing(8)
        self.word_edit = QLineEdit()
        self.word_edit.setPlaceholderText("Add a word or phrase...")
        self.word_edit.returnPressed.connect(self._add_word)
        add = _btn("Add", "primary", 80)
        add.clicked.connect(self._add_word)
        rm = _btn("Remove selected", "danger")
        rm.clicked.connect(self._remove_words)
        r.addWidget(self.word_edit, 1)
        r.addWidget(add)
        r.addWidget(rm)
        wl.addLayout(r)
        c.add_widget(w)
        lay.addWidget(c)

        c = Card("Replacements")
        w = QWidget()
        rl = QVBoxLayout(w)
        rl.setContentsMargins(0, 8, 0, 10)
        rl.setSpacing(10)
        hint = QLabel("Rewrite what the model heard into what you want. Whole words, case-insensitive. "
                      "Double-click a cell to edit.")
        hint.setObjectName("rowDesc")
        hint.setWordWrap(True)
        rl.addWidget(hint)
        self.repl_table = _table(["Heard", "Replace with"])
        self.repl_table.setMinimumHeight(190)
        rl.addWidget(self.repl_table)
        r = QHBoxLayout()
        r.setSpacing(8)
        add = _btn("Add rule", "mini")
        add.clicked.connect(lambda: self._add_table_row(self.repl_table, "", ""))
        rm = _btn("Remove selected", "mini")
        rm.clicked.connect(lambda: self._remove_table_row(self.repl_table))
        r.addWidget(add)
        r.addWidget(rm)
        r.addStretch(1)
        rl.addLayout(r)
        c.add_widget(w)
        lay.addWidget(c)
        lay.addStretch(1)
        return sa

    def _build_snippets(self) -> QWidget:
        sa, lay = _page("Snippets", "Say a trigger phrase and LocalFlow inserts the full text instead.")
        c = Card("Snippets")
        w = QWidget()
        sl = QVBoxLayout(w)
        sl.setContentsMargins(0, 8, 0, 10)
        sl.setSpacing(10)
        self.snip_table = _table(["Trigger phrase", "Expands to"])
        self.snip_table.setMinimumHeight(190)
        self.snip_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive)
        self.snip_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.snip_table.setColumnWidth(0, 230)
        sl.addWidget(self.snip_table)
        r = QHBoxLayout()
        r.setSpacing(8)
        add = _btn("Add snippet", "mini")
        add.clicked.connect(self._add_snippet)
        rm = _btn("Remove selected", "mini")
        rm.clicked.connect(lambda: self._remove_table_row(self.snip_table))
        r.addWidget(add)
        r.addWidget(rm)
        r.addStretch(1)
        sl.addLayout(r)
        lbl = QLabel("Expansion for the selected snippet (multi-line is fine)")
        lbl.setObjectName("rowDesc")
        sl.addWidget(lbl)
        self.snip_edit = QPlainTextEdit()
        self.snip_edit.setMinimumHeight(110)
        self.snip_edit.setMaximumHeight(160)
        self.snip_edit.setPlaceholderText("Select or add a snippet, then type what it should expand to...")
        self.snip_edit.setEnabled(False)
        sl.addWidget(self.snip_edit)
        c.add_widget(w)
        lay.addWidget(c)
        lay.addStretch(1)
        self.snip_table.currentCellChanged.connect(lambda r_, c_, pr, pc: self._snip_select(r_))
        self.snip_edit.textChanged.connect(self._snip_text_changed)
        self.snip_table.cellDoubleClicked.connect(lambda r_, c_: c_ == 1 and self.snip_edit.setFocus())
        return sa

    def _build_styles(self) -> QWidget:
        sa, lay = _page("Styles", "Choose how dictation is written in each app or browser tab. Profile rules are "
                                  "checked top to bottom and the first match wins. Changes apply when you press Save.")
        self._styles_page = sa

        # --- master switch
        c = Card("Automatic style")
        self.auto_sw = Switch()
        c.add_row("Auto style by app", "Pick the writing style from the app (and browser tab) you dictate into. "
                  "When off, everything uses the neutral style.", self.auto_sw)
        self.auto_sw.toggled.connect(lambda _v: self._update_test())
        lay.addWidget(c)

        # --- profiles
        c = Card("Profiles")
        w = QWidget()
        pl = QVBoxLayout(w)
        pl.setContentsMargins(0, 8, 0, 10)
        pl.setSpacing(10)
        hint = QLabel("Order matters: the first rule that matches wins, so put specific rules above general ones. "
                      "App = process names, comma separated (@browsers = all browsers). Title = regex matched "
                      "against the window title (case-insensitive); with both set, both must match. "
                      "Double-click a cell to edit.")
        hint.setObjectName("rowDesc")
        hint.setWordWrap(True)
        pl.addWidget(hint)
        self.rule_table = _table(["Name", "App", "Title regex", "Style"])
        self.rule_table.setMinimumHeight(320)
        hh = self.rule_table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Interactive)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.Stretch)
        hh.setSectionResizeMode(3, QHeaderView.Interactive)
        self.rule_table.setColumnWidth(0, 150)
        self.rule_table.setColumnWidth(3, 150)
        self.rule_table.itemChanged.connect(self._rule_item_changed)
        pl.addWidget(self.rule_table)

        r = QHBoxLayout()
        r.setSpacing(8)
        self.btn_rule_add = _btn("Add rule", "mini")
        self.btn_rule_add.clicked.connect(self._add_rule)
        self.btn_rule_last = _btn("Add rule for last used app", "mini")
        self.btn_rule_last.clicked.connect(self._add_last_app_rule)
        self.btn_rule_rm = _btn("Remove", "mini")
        self.btn_rule_rm.clicked.connect(self._remove_rule)
        self.btn_rule_up = _btn("Move up", "mini")
        self.btn_rule_up.clicked.connect(lambda: self._move_rule(-1))
        self.btn_rule_down = _btn("Move down", "mini")
        self.btn_rule_down.clicked.connect(lambda: self._move_rule(1))
        for b in (self.btn_rule_add, self.btn_rule_last):
            r.addWidget(b)
        r.addStretch(1)
        pl.addLayout(r)
        r = QHBoxLayout()
        r.setSpacing(8)
        for b in (self.btn_rule_up, self.btn_rule_down, self.btn_rule_rm):
            r.addWidget(b)
        r.addStretch(1)
        pl.addLayout(r)

        t = QLabel("Test a window (uses the rules above, even before saving)")
        t.setObjectName("rowDesc")
        pl.addWidget(t)
        tr = QHBoxLayout()
        tr.setSpacing(8)
        self.test_exe = QLineEdit()
        self.test_exe.setPlaceholderText("Process, e.g. chrome.exe")
        self.test_exe.setFixedWidth(190)
        self.test_title = QLineEdit()
        self.test_title.setPlaceholderText("Window title, e.g. Inbox - Gmail - Google Chrome")
        self.test_result = QLabel("")
        self.test_result.setMinimumWidth(180)
        tr.addWidget(self.test_exe)
        tr.addWidget(self.test_title, 1)
        pl.addLayout(tr)
        pl.addWidget(self.test_result)
        self.test_exe.textChanged.connect(lambda _t: self._update_test())
        self.test_title.textChanged.connect(lambda _t: self._update_test())
        c.add_widget(w)
        lay.addWidget(c)

        # --- style definitions
        c = Card("Styles")
        w = QWidget()
        sl = QVBoxLayout(w)
        sl.setContentsMargins(0, 8, 0, 10)
        sl.setSpacing(10)
        hint = QLabel('Styles are defined in config.json under "styles": each has a prompt, and optional '
                      "lowercase, strip_trailing_period, abbreviations, drop_apostrophes, capitalize and "
                      "examples. Edit the file to add or tweak one.")
        hint.setObjectName("rowDesc")
        hint.setWordWrap(True)
        sl.addWidget(hint)
        self.style_table = _table(["Name", "Description", "Behavior"])
        self.style_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.style_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.style_table.setFocusPolicy(Qt.NoFocus)
        sh = self.style_table.horizontalHeader()
        sh.setSectionResizeMode(0, QHeaderView.Interactive)
        sh.setSectionResizeMode(1, QHeaderView.Stretch)
        sh.setSectionResizeMode(2, QHeaderView.Interactive)
        self.style_table.setColumnWidth(0, 120)
        self.style_table.setColumnWidth(2, 250)
        sl.addWidget(self.style_table)
        r = QHBoxLayout()
        r.setSpacing(8)
        ob = _btn("Open config.json", "mini")
        ob.clicked.connect(self._open_config)
        rb = _btn("Reset profiles to defaults", "danger")
        rb.clicked.connect(self._reset_profiles)
        r.addWidget(ob)
        r.addWidget(rb)
        r.addStretch(1)
        sl.addLayout(r)
        c.add_widget(w)
        lay.addWidget(c)
        lay.addStretch(1)
        return sa

    # ----- styles page logic -----
    @staticmethod
    def _plist(rule: dict) -> list[str]:
        p = rule.get("process")
        if isinstance(p, str):
            p = [p]
        return [str(x).strip() for x in (p or []) if str(x).strip()]

    def _draft(self) -> _DraftCfg:
        return _DraftCfg(self.config, {"profiles": self._rules, "auto_style": self.auto_sw.isChecked()})

    def _load_styles_page(self) -> None:
        """Populate the Styles page from the config (discarding unsaved edits)."""
        cfg = self.config
        defs = cfg.get("styles", None) or _styles.DEFAULT_STYLES
        names = [n for n in defs if n != _styles.NEUTRAL]
        self._style_names = [_styles.NEUTRAL] + names
        self._rules = [copy.deepcopy(r) for r in (cfg.get("profiles", None) or []) if isinstance(r, dict)]
        self._rebuild_rules()
        self.style_table.setRowCount(0)
        for n in self._style_names:
            sd = defs.get(n) or _styles.DEFAULT_STYLES.get(n) or {}
            r = self.style_table.rowCount()
            self.style_table.insertRow(r)
            for col, txt in enumerate((n, str(sd.get("description", "")), _style_flags(sd))):
                it = QTableWidgetItem(txt)
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                it.setToolTip(txt)
                self.style_table.setItem(r, col, it)
        self.style_table.setMinimumHeight(min(280, 44 + 36 * len(self._style_names)))
        self._refresh_last_ctx()

    def _rebuild_rules(self, select: int | None = None) -> None:
        """Re-render the rule table from the model (self._rules)."""
        t = self.rule_table
        self._rules_loading = True
        try:
            t.setRowCount(0)
            for rule in self._rules:
                r = t.rowCount()
                t.insertRow(r)
                t.setItem(r, 0, QTableWidgetItem(str(rule.get("name", ""))))
                procs = ", ".join(self._plist(rule))
                it = QTableWidgetItem(procs)
                it.setToolTip(procs)
                t.setItem(r, 1, it)
                it = QTableWidgetItem(str(rule.get("title", "") or ""))
                t.setItem(r, 2, it)
                self._flag_title(it)
                combo = _NoWheelCombo()
                combo.setEditable(True)
                combo.setInsertPolicy(QComboBox.NoInsert)
                combo.addItems(self._style_names)
                combo.setCurrentText(str(rule.get("style", "") or _styles.NEUTRAL))
                combo.currentTextChanged.connect(lambda txt, rule=rule: self._rule_style_changed(rule, txt))
                t.setCellWidget(r, 3, combo)
            if select is not None and 0 <= select < t.rowCount():
                t.setCurrentCell(select, 0)
            else:
                t.clearSelection()
        finally:
            self._rules_loading = False
        self._update_test()

    @staticmethod
    def _regex_error(pat: str) -> str:
        if not pat:
            return ""
        try:
            re.compile(pat, re.IGNORECASE)
            return ""
        except re.error as e:
            return f"Invalid regex: {e}"

    def _flag_title(self, item: QTableWidgetItem) -> bool:
        """Red text + tooltip on an invalid regex; returns True when the pattern is fine."""
        err = self._regex_error(item.text().strip())
        item.setForeground(QColor(DANGER) if err else QColor(TEXT))
        item.setToolTip(err or "Regex searched in the window title (case-insensitive)")
        return not err

    def _rule_item_changed(self, item: QTableWidgetItem) -> None:
        if self._rules_loading:
            return
        row, col = item.row(), item.column()
        if not (0 <= row < len(self._rules)):
            return
        rule = self._rules[row]
        txt = item.text()
        self._rules_loading = True   # setting tooltip/colour below re-emits itemChanged
        try:
            if col == 0:
                rule["name"] = txt.strip()
            elif col == 1:
                rule["process"] = [p.strip() for p in txt.split(",") if p.strip()]
                item.setToolTip(txt)
            elif col == 2:
                rule["title"] = txt.strip()
                self._flag_title(item)
        finally:
            self._rules_loading = False
        self._update_test()

    def _rule_style_changed(self, rule: dict, text: str) -> None:
        if self._rules_loading:
            return
        rule["style"] = text.strip()
        self._update_test()

    def _add_rule(self) -> None:
        self._rules.append({"name": "New rule", "style": _styles.NEUTRAL})
        self._rebuild_rules(select=len(self._rules) - 1)
        self.rule_table.scrollToBottom()
        self.rule_table.editItem(self.rule_table.item(len(self._rules) - 1, 0))

    def _remove_rule(self) -> None:
        r = self.rule_table.currentRow()
        if 0 <= r < len(self._rules):
            del self._rules[r]
            self._rebuild_rules(select=min(r, len(self._rules) - 1))

    def _move_rule(self, delta: int) -> None:
        r = self.rule_table.currentRow()
        n = r + delta
        if 0 <= r < len(self._rules) and 0 <= n < len(self._rules):
            self._rules[r], self._rules[n] = self._rules[n], self._rules[r]
            self._rebuild_rules(select=n)

    def _last_ctx(self) -> tuple[str, str]:
        if self._get_last_ctx is None:
            return "", ""
        try:
            exe, title = self._get_last_ctx() or ("", "")
            return str(exe or "").strip().lower(), str(title or "")
        except Exception:
            return "", ""

    def _refresh_last_ctx(self) -> None:
        exe, _title = self._last_ctx()
        ok = bool(exe)
        self.btn_rule_last.setEnabled(ok)
        self.btn_rule_last.setToolTip(f"New rule at the top for {exe}" if ok else
                                      "Available after you dictate into an app")

    def _add_last_app_rule(self) -> None:
        exe, title = self._last_ctx()
        self._refresh_last_ctx()
        if not exe:
            return
        # reuse an existing app-only rule for this exe instead of duplicating it
        for i, rule in enumerate(self._rules):
            if [p.lower() for p in self._plist(rule)] == [exe] and not rule.get("title"):
                self._rules.insert(0, self._rules.pop(i))
                break
        else:
            ctx = _styles.resolve(exe, title, self._draft())
            self._rules.insert(0, {"name": _styles.pretty_app(exe) or exe, "process": [exe], "style": ctx.style})
        self._rebuild_rules(select=0)
        self.rule_table.scrollToTop()

    def _update_test(self) -> None:
        exe, title = self.test_exe.text().strip(), self.test_title.text()
        if not exe and not title.strip():
            txt, ok = "Enter a process and/or window title to see which style would be used.", False
        else:
            try:
                ctx = _styles.resolve(exe, title, self._draft())
                if not ctx.auto:
                    txt = f"{ctx.style}  -  auto style is off"
                elif ctx.rule:
                    txt = f"{ctx.style}  -  rule: {ctx.rule}"
                else:
                    txt = f"{ctx.style}  -  no rule matched"
                ok = True
            except Exception as e:  # never let a bad rule break typing
                txt, ok = f"error: {e}", False
        self.test_result.setText(txt)
        self.test_result.setStyleSheet(f"color: {ACCENT_TEXT}; font-weight: 600;" if ok else f"color: {MUTED};")

    def _collect_profiles(self) -> list[dict] | None:
        """Rule list for saving, or None (with a message) when a rule has an invalid title regex."""
        out = []
        for i, rule in enumerate(self._rules):
            err = self._regex_error(str(rule.get("title", "") or "").strip())
            if err:
                self._goto_index(self.stack.indexOf(self._styles_page))
                self.rule_table.setCurrentCell(i, 2)
                self._flash(f"Rule '{rule.get('name') or i + 1}': {err}", ok=False)
                return None
            r = copy.deepcopy(rule)          # keeps extra keys such as "note"
            r["name"] = str(r.get("name", "") or "").strip()
            procs = self._plist(r)
            title = str(r.get("title", "") or "").strip()
            for k, v in (("process", procs), ("title", title)):
                if v:
                    r[k] = v
                else:
                    r.pop(k, None)
            r["style"] = _styles.canon_style(str(r.get("style", "") or ""))
            if not (r["name"] or procs or title):
                continue                      # blank row
            out.append(r)
        return out

    def _confirm(self, title: str, text: str) -> bool:
        return QMessageBox.question(self, title, text, QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.No) == QMessageBox.Yes

    def _reset_profiles(self) -> None:
        if not self._confirm("Reset profiles", "Replace all profile rules with the built-in defaults?\n"
                             "Your own rules are discarded when you press Save."):
            return
        self._rules = copy.deepcopy(_styles.DEFAULT_PROFILES)
        self._rebuild_rules()

    @staticmethod
    def _open_config() -> None:
        try:
            os.startfile(str(CONFIG_PATH))  # noqa: S606
        except Exception:
            pass

    def _build_about(self) -> QWidget:
        sa, lay = _page("About")
        c = Card("LocalFlow")
        v = QLabel(f"Version {__version__}")
        c.add_row("Version", "Fully local push-to-talk dictation. Nothing leaves this PC.", v)
        for title, path in (("Config file", CONFIG_PATH), ("Log file", LOG_PATH)):
            box = QWidget()
            bl = QHBoxLayout(box)
            bl.setContentsMargins(0, 0, 0, 0)
            bl.setSpacing(8)
            le = QLineEdit(str(path))
            le.setReadOnly(True)
            le.setCursorPosition(0)
            ob = _btn("Show in folder", "mini")
            ob.clicked.connect(lambda _=False, p=path: self._reveal(p))
            bl.addWidget(le, 1)
            bl.addWidget(ob)
            c.add_row(title, "", box, stack=True)
        lay.addWidget(c)
        lay.addStretch(1)
        return sa

    # ----- helpers -----
    @staticmethod
    def _reveal(path) -> None:
        try:
            import subprocess
            p = Path(path)
            if p.exists():
                subprocess.Popen(["explorer", f"/select,{p}"])
            else:
                os.startfile(str(p.parent))  # noqa: S606
        except Exception:
            pass

    def _add_word(self) -> None:
        w = self.word_edit.text().strip()
        if not w:
            return
        existing = {self.words_list.item(i).text().lower() for i in range(self.words_list.count())}
        if w.lower() not in existing:
            self._append_word(w)
            self.words_list.scrollToBottom()
        self.word_edit.clear()

    def _append_word(self, w: str) -> None:
        it = QListWidgetItem(w)
        it.setFlags(it.flags() | Qt.ItemIsEditable)
        self.words_list.addItem(it)

    def _add_game(self) -> None:
        g = self.game_edit.text().strip()
        if not g:
            return
        existing = {self.game_list.item(i).text().lower() for i in range(self.game_list.count())}
        if g.lower() not in existing:
            self._append_game(g)
            self.game_list.scrollToBottom()
        self.game_edit.clear()

    def _append_game(self, g: str) -> None:
        it = QListWidgetItem(g)
        it.setFlags(it.flags() | Qt.ItemIsEditable)
        self.game_list.addItem(it)

    def _remove_games(self) -> None:
        for it in self.game_list.selectedItems():
            self.game_list.takeItem(self.game_list.row(it))

    def _remove_words(self) -> None:
        for it in self.words_list.selectedItems():
            self.words_list.takeItem(self.words_list.row(it))

    def _add_table_row(self, table: QTableWidget, a: str, b: str, edit: bool = True) -> int:
        r = table.rowCount()
        table.insertRow(r)
        table.setItem(r, 0, QTableWidgetItem(a))
        table.setItem(r, 1, QTableWidgetItem(b))
        if edit:
            table.setCurrentCell(r, 0)
            table.editItem(table.item(r, 0))
        return r

    def _remove_table_row(self, table: QTableWidget) -> None:
        r = table.currentRow()
        if r >= 0:
            table.removeRow(r)

    # snippets: column 1 shows a one-line preview; the real text lives in the item's UserRole
    def _add_snippet(self) -> None:
        r = self._add_snippet_row("", "", edit=True)
        self.snip_table.setCurrentCell(r, 0)

    def _add_snippet_row(self, trigger: str, text: str, edit: bool = False) -> int:
        r = self.snip_table.rowCount()
        self.snip_table.insertRow(r)
        self.snip_table.setItem(r, 0, QTableWidgetItem(trigger))
        it = QTableWidgetItem(self._preview(text))
        it.setData(Qt.UserRole, text)
        it.setFlags(it.flags() & ~Qt.ItemIsEditable)
        self.snip_table.setItem(r, 1, it)
        if edit:
            self.snip_table.setCurrentCell(r, 0)
            self.snip_table.editItem(self.snip_table.item(r, 0))
        return r

    @staticmethod
    def _preview(text: str) -> str:
        return " ↵ ".join(ln.strip() for ln in text.splitlines() if ln.strip()) if text else ""

    def _snip_select(self, row: int) -> None:
        self._snip_loading = True
        try:
            if row < 0 or row >= self.snip_table.rowCount():
                self.snip_edit.setPlainText("")
                self.snip_edit.setEnabled(False)
            else:
                self.snip_edit.setEnabled(True)
                self.snip_edit.setPlainText(self.snip_table.item(row, 1).data(Qt.UserRole) or "")
        finally:
            self._snip_loading = False

    def _snip_text_changed(self) -> None:
        if self._snip_loading:
            return
        r = self.snip_table.currentRow()
        if r < 0:
            return
        it = self.snip_table.item(r, 1)
        txt = self.snip_edit.toPlainText()
        it.setData(Qt.UserRole, txt)
        it.setText(self._preview(txt))

    # ----- load / save -----
    def reload(self) -> None:
        """Populate every control from the current config."""
        cfg = self.config
        mods = {m.strip().lower() for m in str(cfg.get("hotkey", "ctrl+alt")).split("+")}
        for k, cb in self.mod_boxes.items():
            cb.setChecked(k in mods)
        hf = str(cfg.get("handsfree_key", "space"))
        self.handsfree_info.setText(f"Tap {hf.capitalize()} while holding")

        # microphones
        self.mic_combo.clear()
        try:
            default_mic = (self._get_default_mic() if self._get_default_mic else "") or ""
        except Exception:
            default_mic = ""
        match = cfg.get("mic.match", "") or ""
        auto_label = f"Auto (Windows default: {default_mic or 'none found'})"
        if match:
            auto_label = f"Auto (match '{match}', else Windows default: {default_mic or 'none found'})"
        self.mic_combo.addItem(auto_label, "")
        try:
            mics = [m for m in (self._get_mics() or []) if m]
        except Exception:
            mics = []
        seen = set()
        for m in mics:
            if m not in seen:
                seen.add(m)
                self.mic_combo.addItem(m, m)
        dev = cfg.get("mic.device", "") or ""
        if dev:
            _set_combo_data(self.mic_combo, dev, f"{dev} (not connected)")
        else:
            self.mic_combo.setCurrentIndex(0)

        _set_combo_data(self.lang_combo, cfg.get("stt.language", "en"))
        _set_combo_data(self.model_combo, cfg.get("stt.model", "large-v3-turbo"))
        _set_combo_data(self.insert_combo, cfg.get("insert.mode", "paste"))
        self.restore_sw.setChecked(bool(cfg.get("insert.restore_clipboard", True)))
        self.space_sw.setChecked(bool(cfg.get("insert.trailing_space", False)))
        self.cleanup_sw.setChecked(bool(cfg.get("cleanup.enabled", True)))
        self.stream_sw.setChecked(bool(cfg.get("streaming.enabled", True)))
        self.game_sw.setChecked(bool(cfg.get("game_mode.enabled", True)))
        _set_combo_data(self.game_mode_combo, cfg.get("game_mode.while_active", "off"))
        self.game_list.clear()
        for g in cfg.get("game_mode.processes", []) or []:
            self._append_game(str(g))
        self.minwords_spin.setValue(int(cfg.get("cleanup.min_words", 4)))
        self._fill_models()
        self.overlay_sw.setChecked(bool(cfg.get("ui.overlay", True)))
        self.sounds_sw.setChecked(bool(cfg.get("ui.sounds", True)))
        self.vol_slider.setValue(int(round(float(cfg.get("ui.sound_volume", 0.25)) * 100)))
        try:
            self.autostart_sw.setChecked(bool(self._autostart_get()))
        except Exception:
            self.autostart_sw.setChecked(False)
        self._reload_new_general(cfg)

        # dictionary
        self.words_list.clear()
        for w in cfg.get("dictionary.words", []) or []:
            self._append_word(str(w))
        self.repl_table.setRowCount(0)
        for k, v in (cfg.get("dictionary.replacements", {}) or {}).items():
            self._add_table_row(self.repl_table, str(k), str(v), edit=False)
        self.repl_table.clearSelection()

        # snippets
        self.snip_table.setRowCount(0)
        for k, v in (cfg.get("snippets", {}) or {}).items():
            self._add_snippet_row(str(k), str(v))
        self.snip_table.clearSelection()
        self._snip_select(-1)

        # styles
        self.auto_sw.setChecked(bool(cfg.get("auto_style", True)))
        self._load_styles_page()
        self.saved_label.setText("")

    def _collect(self) -> dict | None:
        mods = [k for k in ("ctrl", "alt", "shift", "win") if self.mod_boxes[k].isChecked()]
        if not mods:
            self._flash("Pick at least one hotkey modifier", ok=False)
            self.show_page("general")
            return None
        words, seen = [], set()
        for i in range(self.words_list.count()):
            w = self.words_list.item(i).text().strip()
            if w and w.lower() not in seen:
                seen.add(w.lower())
                words.append(w)
        games, gseen = [], set()
        for i in range(self.game_list.count()):
            g = self.game_list.item(i).text().strip()
            if g and g.lower() not in gseen:
                gseen.add(g.lower())
                games.append(g)
        repl = {}
        for r in range(self.repl_table.rowCount()):
            a = (self.repl_table.item(r, 0).text() if self.repl_table.item(r, 0) else "").strip()
            b = (self.repl_table.item(r, 1).text() if self.repl_table.item(r, 1) else "").strip()
            if a:
                repl[a] = b
        snips = {}
        for r in range(self.snip_table.rowCount()):
            a = (self.snip_table.item(r, 0).text() if self.snip_table.item(r, 0) else "").strip()
            txt = self.snip_table.item(r, 1).data(Qt.UserRole) or ""
            if a and txt:
                snips[a] = txt
        cmd_on = self.cmd_sw.isChecked()
        cmd_mods = [k for k in ("ctrl", "alt", "shift", "win") if self.cmd_boxes[k].isChecked()]
        err = self._validate_cmd()
        if cmd_on and err:
            self._flash(err, ok=False)
            self.show_page("general")
            return None
        profiles = self._collect_profiles()
        if profiles is None:
            return None
        return {
            "hotkey": "+".join(mods),
            "mic": {"device": self.mic_combo.currentData() or ""},
            "stt": {"model": self.model_combo.currentData(), "language": self.lang_combo.currentData()},
            "insert": {"mode": self.insert_combo.currentData(),
                       "restore_clipboard": self.restore_sw.isChecked(),
                       "trailing_space": self.space_sw.isChecked()},
            "cleanup": {"enabled": self.cleanup_sw.isChecked(), "min_words": self.minwords_spin.value()},
            "streaming": {"enabled": self.stream_sw.isChecked()},
            "game_mode": {"enabled": self.game_sw.isChecked(), "processes": games,
                          "while_active": self.game_mode_combo.currentData() or "off"},
            "ui": {"overlay": self.overlay_sw.isChecked(), "sounds": self.sounds_sw.isChecked(),
                   "sound_volume": round(self.vol_slider.value() / 100.0, 2)},
            "dictionary": {"words": words, "replacements": repl},
            "snippets": snips,
            "auto_style": self.auto_sw.isChecked(),
            "profiles": profiles,
            # an invalid combo with command mode off is not saved (the stored one stays)
            "command_mode": {"enabled": cmd_on, **({"hotkey": "+".join(cmd_mods)} if not err else {})},
            "paste_last": {"enabled": self.paste_sw.isChecked()},
            "meetings": {"mic": self.meet_mic_combo.currentData() or "",
                         "output_device": self.meet_out_combo.currentData() or "",
                         "style": self.meet_style_combo.currentData() or "raw"},
            "transcripts": {"file_style": self.file_style_combo.currentData() or "raw"},
            "history": {"enabled": self.hist_sw.isChecked(),
                        "retention_days": self.retention_combo.currentData() or 0},
        }

    def _validate_cmd(self) -> str:
        """Inline check of the command keys against the dictation keys; returns the error ("" = fine)."""
        dic = [k for k, cb in self.mod_boxes.items() if cb.isChecked()]
        cmd = [k for k, cb in self.cmd_boxes.items() if cb.isChecked()]
        err = command_combo_error(dic, cmd)
        on = self.cmd_sw.isChecked()
        for cb in self.cmd_boxes.values():
            cb.setEnabled(on)
        self.cmd_error.setText(err)
        self.cmd_error.setVisible(bool(err) and on)
        return err

    def _reload_new_general(self, cfg) -> None:
        cmd = parse_mods(cfg.get("command_mode.hotkey", "ctrl+win"))
        for k, cb in self.cmd_boxes.items():
            cb.setChecked(k in cmd)
        self.cmd_sw.setChecked(bool(cfg.get("command_mode.enabled", True)))
        self.paste_sw.setChecked(bool(cfg.get("paste_last.enabled", True)))
        self.paste_key.setText(fmt_chord(cfg.get("paste_last.hotkey", "alt+shift+z")) or "-")
        self._validate_cmd()

        devs = {}
        if self._get_devices is not None:
            try:
                devs = self._get_devices() or {}
            except Exception:
                devs = {}
        self.meet_mic_combo.clear()
        self.meet_mic_combo.addItem("Same as dictation", "")
        for n in devs.get("inputs", []) or []:
            self.meet_mic_combo.addItem(n, n)
        m = cfg.get("meetings.mic", "") or ""
        if m:
            _set_combo_data(self.meet_mic_combo, m, f"{m} (not connected)")
        else:
            self.meet_mic_combo.setCurrentIndex(0)
        self.meet_out_combo.clear()
        d_out = devs.get("default_output") or ""
        self.meet_out_combo.addItem(f"Windows default ({d_out})" if d_out else "Windows default output", "")
        for n in devs.get("outputs", []) or []:
            self.meet_out_combo.addItem(n, n)
        o = cfg.get("meetings.output_device", "") or ""
        if o:
            _set_combo_data(self.meet_out_combo, o, f"{o} (not connected)")
        else:
            self.meet_out_combo.setCurrentIndex(0)
        names = transcript_style_names(cfg)
        for combo, key in ((self.meet_style_combo, "meetings.style"), (self.file_style_combo, "transcripts.file_style")):
            combo.clear()
            for n in names:
                combo.addItem("Raw (no cleanup)" if n == "raw" else n, n)
            _set_combo_data(combo, str(cfg.get(key, "raw") or "raw"))

        self.hist_sw.setChecked(bool(cfg.get("history.enabled", True)))
        try:
            days = int(float(cfg.get("history.retention_days", 0) or 0))
        except Exception:
            days = 0
        _set_combo_data(self.retention_combo, days, f"{days} days")
        self.retention_combo.setEnabled(self.hist_sw.isChecked())

    def _fill_models(self) -> None:
        cur = self.config.get("cleanup.model_path", "")
        combo = self.llm_combo
        combo.blockSignals(True)
        try:
            combo.clear()
            idx = -1
            for f in list_models():
                path = f"models/{f}"
                combo.addItem(model_display_name(f), path)
                if idx < 0 and same_model(cur, path):
                    idx = combo.count() - 1
            if idx < 0 and cur:
                combo.addItem(f"{model_display_name(cur)} (current)", cur)
                idx = combo.count() - 1
            combo.setCurrentIndex(idx)
        finally:
            combo.blockSignals(False)

    def _model_picked(self, i: int) -> None:
        # applied on its own (not through Save/_collect), so a stale dropdown can never undo a tray switch
        path = self.llm_combo.itemData(i)
        if not path or same_model(self.config.get("cleanup.model_path", ""), path):
            return
        try:
            self.config.set("cleanup.model_path", path)
        except Exception as e:
            self._flash(f"Could not switch model: {e}", ok=False)
            return
        self._flash(f"Loading {model_display_name(path)}...", ok=True)

    def save(self) -> bool:
        data = self._collect()
        if data is None:
            return False
        try:
            want = self.autostart_sw.isChecked()
            try:
                cur = bool(self._autostart_get())
            except Exception:
                cur = None
            if cur is None or cur != want:
                self._autostart_set(want)
            self.config.update(data)
        except Exception as e:  # keep the window usable
            self._flash(f"Save failed: {e}", ok=False)
            return False
        self._flash("✓  Saved", ok=True)
        return True

    def _flash(self, text: str, ok: bool = True) -> None:
        self.saved_label.setObjectName("saved" if ok else "savedErr")
        self.saved_label.setStyleSheet("")  # force re-polish for the new objectName
        self.saved_label.style().unpolish(self.saved_label)
        self.saved_label.style().polish(self.saved_label)
        self.saved_label.setText(text)
        self._saved_timer.start(2600 if ok else 4500)

    # ----- window behaviour -----
    def showEvent(self, e):
        if not self._built_dark_titlebar:
            enable_dark_titlebar(self)
            self._built_dark_titlebar = True
        if not e.spontaneous():
            self.reload()
        super().showEvent(e)
        if not e.spontaneous():
            QTimer.singleShot(0, self.setFocus)

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

    from .config import Config

    tmp = Path(tempfile.gettempdir()) / "localflow_demo_config.json"
    cfg = Config(path=tmp)
    cfg.update({"dictionary": {"words": ["Kubernetes", "Postgres", "Alex", "Qwen"],
                               "replacements": {"jason": "JSON", "get hub": "GitHub", "pie torch": "PyTorch"}},
                "snippets": {"my email": "you@example.com",
                             "sign off": "Best regards,\nAlex"}})
    state = {"auto": False}
    app = QApplication(sys.argv)
    apply_theme(app)
    w = SettingsWindow(cfg, lambda: ["Microphone (USB Audio Device)", "Headset Mic (Realtek Audio)",
                                     "Line In (USB Audio)"],
                       lambda: state["auto"], lambda v: state.update(auto=v),
                       lambda: ("discord.exe", "general - Discord"))
    w.show()
    sys.exit(app.exec())
