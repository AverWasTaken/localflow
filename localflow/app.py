"""LocalFlow application: wires hotkey -> audio -> pipeline -> insertion, plus tray + UI windows."""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
from typing import Optional

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QColor, QIcon, QPainter, QPixmap, QBrush, QPen
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from . import __version__, autostart, sounds, styles, winutil
from .audio import AudioInput, default_input_name, input_device_names
from .cleanup import Cleaner
from .config import Config
from .history import History
from .hotkey import HotkeyManager
from .inserter import insert_text
from .overlay import Overlay
from .paths import LOG_PATH
from .pipeline import Pipeline
from .streaming import StreamSession
from .stt import Transcriber

log = logging.getLogger("localflow")


def make_icon(state: str = "ready") -> QIcon:
    pm = QPixmap(64, 64)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    col = {"ready": "#7c6cff", "paused": "#6b6b78", "busy": "#f5a524", "error": "#ef4444"}.get(state, "#7c6cff")
    p.setBrush(QBrush(QColor(col)))
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(2, 2, 60, 60, 16, 16)
    p.setBrush(QBrush(QColor("white")))
    for i, h in enumerate((14, 28, 40, 28, 18, 30, 12)):
        x = 9 + i * 7.6
        p.drawRoundedRect(int(x), int(32 - h / 2), 4, h, 2, 2)
    p.end()
    return QIcon(pm)


class Bridge(QObject):
    """Cross-thread -> Qt main thread signals."""
    status = Signal(str, str)        # tooltip text, icon state
    open_settings = Signal()
    open_history = Signal()


class LocalFlowApp:
    def __init__(self, qapp: QApplication):
        self.qapp = qapp
        self.cfg = Config()
        self.history = History(max_entries=int(self.cfg.get("history.max_entries", 5000)))
        self.bridge = Bridge()
        self.overlay = Overlay()
        self.audio = AudioInput(self.cfg, on_level=self._on_level)
        self.audio.on_device_change = lambda name: self._set_status(f"Ready - mic: {name}", "ready")
        self.stt = Transcriber(self.cfg)
        self.cleaner = Cleaner(self.cfg)
        self.pipeline = Pipeline(self.cfg, self.stt, self.cleaner, self.history, insert_text)
        self.hotkey = HotkeyManager(
            os.environ.get("LOCALFLOW_TEST_HOTKEY") or self.cfg.get("hotkey", "ctrl+alt"),
            self.cfg.get("handsfree_key", "space"),
            on_start=lambda: self._ctl.put(("start",)),
            on_stop=lambda: self._ctl.put(("stop",)),
            on_cancel=lambda why: self._ctl.put(("cancel", why)),
            on_handsfree=lambda: self._ctl.put(("handsfree",)))
        self.hotkey.set_enabled(bool(self.cfg.get("enabled", True)))
        self._ctl: queue.Queue = queue.Queue()
        self._jobs: queue.Queue = queue.Queue()
        self._rec_id = 0
        self._recording = False
        self._locked = False
        self._ctx = ("", "")
        self._style = styles.StyleContext()
        self._session: Optional[StreamSession] = None
        self._shutdown = threading.Event()
        self._t_start = 0.0
        self._ui_shown = -1
        self.settings_win = None
        self.history_win = None
        self.tray: Optional[QSystemTrayIcon] = None
        self._status = "Starting..."
        self._ready = False
        self.last_result: dict = {}

    # ------------------------------------------------------------------ overlay/sound helpers
    def _ov(self, state: str, text: str = "") -> None:
        if self.cfg.get("ui.overlay", True) or state in ("error",):
            self.overlay.set_state(state, text)
        elif state == "hidden":
            self.overlay.set_state("hidden")

    def _snd(self, name: str) -> None:
        sounds.play(name, self.cfg)

    def _on_level(self, lvl: float) -> None:
        if self._recording:
            self.overlay.push_level(lvl)
            if self.audio.max_samples_hit:
                self.audio.max_samples_hit = False
                log.warning("max recording duration reached")
                self.hotkey.force_idle()
                self._ctl.put(("stop",))

    # ------------------------------------------------------------------ controller thread (hotkey events)
    def _controller(self) -> None:
        while not self._shutdown.is_set():
            try:
                ev = self._ctl.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                kind = ev[0]
                if kind == "start":
                    self._do_start()
                elif kind == "handsfree":
                    self._do_handsfree()
                elif kind == "stop":
                    self._do_stop()
                elif kind == "cancel":
                    self._do_cancel(ev[1])
            except Exception:
                log.exception("controller error on %s", ev)
                self._recording = False
                self._ov("hidden")

    def _do_start(self) -> None:
        if self._recording:
            return
        self._rec_id += 1
        rid = self._rec_id
        self._locked = False
        self._ctx = winutil.foreground_info()
        self._t_start = time.perf_counter()
        # style is decided from the window the user is dictating INTO, now, not at release
        self._style = styles.resolve(self._ctx[0], self._ctx[1], self.cfg)
        self.audio.begin()
        self._recording = True
        self._session = self._start_session()
        log.info("recording started: app=%s style=%s%s title=%r", self._ctx[0] or "?", self._style.style,
                 f" (rule: {self._style.rule})" if self._style.rule else "", self._ctx[1][:100])
        threading.Timer(0.12, self._show_recording, args=(rid,)).start()

    def _start_session(self) -> Optional[StreamSession]:
        """Chunked transcription while recording (long dictations). Never raises."""
        try:
            if not self.cfg.get("streaming.enabled", True) or not self.stt.ready \
                    or os.environ.get("LOCALFLOW_TEST_WAV"):
                return None
            sess = StreamSession(self.cfg, self.stt, self.cleaner, self._style, self.audio.sample_rate)
            sess.start()
            sess.attach(self.audio)
            return sess
        except Exception:
            log.exception("could not start streaming session - single pass will be used")
            return None

    def _drop_session(self) -> None:
        sess, self._session = self._session, None
        if sess is not None:
            sess.cancel()

    def _style_label(self) -> str:
        return self._style.label if self.cfg.get("auto_style", True) else ""

    def _show_recording(self, rid: int) -> None:
        # only show UI if the user is still holding (not a Ctrl+Alt+<key> shortcut)
        if self._recording and self._rec_id == rid:
            self._ov("handsfree" if self._locked else "recording", self._style_label())
            self._snd("start")
            self._ui_shown = rid

    def _do_handsfree(self) -> None:
        if not self._recording:
            return
        self._locked = True
        self._ov("handsfree", self._style_label())
        self._snd("start")
        self._ui_shown = self._rec_id

    def _do_stop(self) -> None:
        if not self._recording:
            return
        self._recording = False
        t_release = time.perf_counter()
        held = t_release - self._t_start
        min_d = float(self.cfg.get("audio.min_duration_s", 0.3))
        if held < min_d:
            self._drop_session()
            self.audio.cancel()
            log.info("discarded: too short (%.2fs held)", held)
            self._ov("hidden")
            return
        session, self._session = self._session, None
        audio = self.audio.end()
        dur = len(audio) / 16000.0
        self._ov("processing")
        self._snd("stop")
        self._jobs.put((audio, self._ctx, self._style, session, t_release))

    def _do_cancel(self, why: str) -> None:
        if not self._recording:
            return
        self._recording = False
        self._drop_session()
        self.audio.cancel()
        log.info("recording cancelled (%s)", why)
        if why == "esc":
            self._ov("info", "Cancelled")
            self._snd("cancel")
        else:
            self._ov("hidden")

    # ------------------------------------------------------------------ worker thread (STT -> cleanup -> insert)
    def _worker(self) -> None:
        while not self._shutdown.is_set():
            try:
                audio, ctx, sty, session, t_rel = self._jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                waited = 0
                while not self.stt.ready and waited < 120 and not self._shutdown.is_set():
                    time.sleep(0.25)
                    waited += 0.25
                if not self.stt.ready:
                    self._ov("error", "Speech model not ready")
                    self._snd("error")
                    continue
                r = self.pipeline.process(audio, app_exe=ctx[0], app_title=ctx[1], t_release=t_rel,
                                          style_ctx=sty, session=session)
                self.last_result = r
                if r["status"] == "ok":
                    self._ov("hidden")
                elif r["status"] == "discard":
                    log.info("discarded: %s", r["reason"])
                    self._ov("hidden")
                else:
                    self._ov("error", "Couldn't insert text")
                    self._snd("error")
            except Exception:
                log.exception("dictation failed")
                self._ov("error", "Transcription failed")
                self._snd("error")

    # ------------------------------------------------------------------ boot
    def _boot(self) -> None:
        try:
            self.audio.start()
            self._set_status(f"Loading speech model...", "busy")
            self.stt.load()
            try:   # load the VAD used for chunked transcription now, not on the first long dictation
                from faster_whisper.vad import get_vad_model
                get_vad_model()
            except Exception:
                log.exception("VAD preload failed")
            self.cleaner.start()
            gpu = "GPU" if self.stt.device == "cuda" else "CPU (fallback)"
            mic = self.audio.device_name
            self._ready = True
            self._set_status(f"Ready - {self.stt.model_name} on {gpu} - mic: {mic}",
                             "ready" if self.stt.device == "cuda" else "error")
            if self.stt.device != "cuda":
                self._ov("error", "GPU unavailable - using CPU")
            log.info("READY: stt=%s/%s/%s mic=%s", self.stt.model_name, self.stt.device, self.stt.compute, mic)
        except Exception:
            log.exception("boot failed")
            self._set_status("Startup failed - see log", "error")

    def _set_status(self, text: str, icon: str) -> None:
        self._status = text
        self.bridge.status.emit(text, icon if self.cfg.get("enabled", True) else "paused")

    # ------------------------------------------------------------------ config changes
    def _on_config_change(self, old: dict, new: dict) -> None:
        def ch(*path):
            def g(d):
                for p in path:
                    d = d.get(p, {}) if isinstance(d, dict) else {}
                return d
            return g(old) != g(new)
        threading.Thread(target=self._apply_config, args=(ch,), daemon=True, name="cfg-apply").start()

    def _apply_config(self, ch) -> None:
        try:
            if ch("hotkey"):
                self.hotkey.set_hotkey(self.cfg.get("hotkey"))
            if ch("handsfree_key"):
                self.hotkey.set_handsfree_key(self.cfg.get("handsfree_key"))
            if ch("enabled"):
                en = bool(self.cfg.get("enabled"))
                self.hotkey.set_enabled(en)
                self._set_status("Ready" if en else "Paused", "ready" if en else "paused")
                self._ov("info", "Resumed" if en else "Paused")
            if ch("mic") or ch("audio", "keep_open"):
                self.audio.reopen()
                self._set_status(f"Ready - mic: {self.audio.device_name}", "ready")
            if ch("stt", "model") or ch("stt", "device") or ch("stt", "compute_type"):
                self.stt.ready = False
                self._set_status("Reloading speech model...", "busy")
                self.stt.load()
                self._set_status(f"Ready - {self.stt.model_name} ({self.stt.device})", "ready")
            if ch("cleanup", "enabled") and self.cfg.get("cleanup.enabled"):
                self.cleaner.start()
            if ch("ui", "sound_volume"):
                sounds.ensure_sounds(float(self.cfg.get("ui.sound_volume", 0.25)))
        except Exception:
            log.exception("applying config failed")

    # ------------------------------------------------------------------ tray
    def _build_tray(self) -> None:
        self.tray = QSystemTrayIcon(make_icon("busy"))
        self.tray.setToolTip("LocalFlow - starting")
        menu = QMenu()
        self.menu = menu
        a = menu.addAction("Settings...")
        a.triggered.connect(self.show_settings)
        a = menu.addAction("History...")
        a.triggered.connect(self.show_history)
        menu.addSeparator()
        self.act_pause = menu.addAction("Pause dictation")
        self.act_pause.triggered.connect(self.toggle_pause)
        self.act_cleanup = menu.addAction("AI cleanup")
        self.act_cleanup.setCheckable(True)
        self.act_cleanup.triggered.connect(self.toggle_cleanup)
        self.act_autostyle = menu.addAction("Auto style by app")
        self.act_autostyle.setCheckable(True)
        self.act_autostyle.triggered.connect(self.toggle_auto_style)
        self.mic_menu = menu.addMenu("Microphone")
        self.mic_menu.aboutToShow.connect(self._fill_mic_menu)
        menu.addSeparator()
        a = menu.addAction("Open log")
        a.triggered.connect(lambda: os.startfile(str(LOG_PATH)))
        a = menu.addAction("Quit LocalFlow")
        a.triggered.connect(self.quit)
        menu.aboutToShow.connect(self._refresh_menu)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(lambda r: self.show_settings() if r == QSystemTrayIcon.DoubleClick else None)
        self.tray.show()
        self.bridge.status.connect(self._on_status)
        self.bridge.open_settings.connect(self.show_settings)
        self.bridge.open_history.connect(self.show_history)

    def _on_status(self, text: str, icon: str) -> None:
        if self.tray:
            self.tray.setToolTip(f"LocalFlow - {text}")
            self.tray.setIcon(make_icon(icon))

    def _refresh_menu(self) -> None:
        en = bool(self.cfg.get("enabled", True))
        self.act_pause.setText("Pause dictation" if en else "Resume dictation")
        self.act_cleanup.setChecked(bool(self.cfg.get("cleanup.enabled", True)))
        self.act_autostyle.setChecked(bool(self.cfg.get("auto_style", True)))

    def _fill_mic_menu(self) -> None:
        self.mic_menu.clear()
        grp = QActionGroup(self.mic_menu)
        grp.setExclusive(True)
        cur = self.cfg.get("mic.device", "")
        match = self.cfg.get("mic.match", "")
        label = f"Auto (Windows default: {default_input_name() or 'none'})"
        if match:
            label = f"Auto (match '{match}', else Windows default)"
        auto = self.mic_menu.addAction(label[:70])
        auto.setCheckable(True)
        auto.setChecked(not cur)
        grp.addAction(auto)
        auto.triggered.connect(lambda: self.cfg.set("mic.device", ""))
        self.mic_menu.addSeparator()
        for name in input_device_names():
            act = self.mic_menu.addAction(name[:60])
            act.setCheckable(True)
            act.setChecked(name == cur)
            grp.addAction(act)
            act.triggered.connect(lambda _=False, n=name: self.cfg.set("mic.device", n))
        self.mic_menu.addSeparator()
        info = self.mic_menu.addAction(f"In use: {self.audio.device_name}"[:70])
        info.setEnabled(False)

    def toggle_pause(self) -> None:
        self.cfg.set("enabled", not self.cfg.get("enabled", True))

    def toggle_cleanup(self) -> None:
        on = not self.cfg.get("cleanup.enabled", True)
        self.cfg.set("cleanup.enabled", on)
        self._ov("info", "AI cleanup on" if on else "AI cleanup off")

    def toggle_auto_style(self) -> None:
        on = not self.cfg.get("auto_style", True)
        self.cfg.set("auto_style", on)
        self._ov("info", "Auto style on" if on else "Auto style off (neutral)")

    def show_settings(self) -> None:
        if self.settings_win is None:
            from .settings_ui import SettingsWindow
            self.settings_win = SettingsWindow(self.cfg, input_device_names, autostart.is_enabled, autostart.set_enabled,
                                          get_last_ctx=lambda: self._ctx, get_default_mic=default_input_name)
        self.settings_win.show()
        self.settings_win.raise_()
        self.settings_win.activateWindow()

    def show_history(self) -> None:
        if self.history_win is None:
            from .history_ui import HistoryWindow
            self.history_win = HistoryWindow(self.history)
        self.history_win.show()
        self.history_win.raise_()
        self.history_win.activateWindow()

    def quit(self) -> None:
        log.info("quitting")
        self._shutdown.set()
        try:
            self.hotkey.stop()
            self.audio.stop()
            self.cleaner.shutdown()
        finally:
            if self.tray:
                self.tray.hide()
            self.qapp.quit()

    # ------------------------------------------------------------------ run
    def run(self) -> int:
        self._build_tray()
        self.cfg.on_change(self._on_config_change)
        t = QTimer()
        t.setInterval(2000)
        t.timeout.connect(self.cfg.reload_if_changed)
        t.start()
        self._cfg_timer = t
        threading.Thread(target=self._controller, name="controller", daemon=True).start()
        threading.Thread(target=self._worker, name="worker", daemon=True).start()
        ok = self.hotkey.start()
        if not ok:
            log.error("keyboard hook failed to install")
        threading.Thread(target=self._boot, name="boot", daemon=True).start()
        log.info("LocalFlow %s started (hotkey=%s)", __version__, self.cfg.get("hotkey"))
        return self.qapp.exec()


def run() -> int:
    qapp = QApplication.instance() or QApplication([])
    qapp.setQuitOnLastWindowClosed(False)
    qapp.setApplicationName("LocalFlow")
    qapp.setStyle("Fusion")
    try:
        from .settings_ui import apply_theme
        apply_theme(qapp)
    except Exception:
        log.debug("theme not applied", exc_info=True)
    app = LocalFlowApp(qapp)
    return app.run()
