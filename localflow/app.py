"""LocalFlow application: wires hotkey -> audio -> pipeline -> insertion, plus tray + UI windows."""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
from datetime import datetime
from typing import Optional

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QColor, QIcon, QPainter, QPixmap, QBrush, QPen
from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMenu, QSystemTrayIcon

from . import __version__, ask, autostart, fieldprobe, modeldl, sounds, styles, winutil
from .audio import AudioInput, default_input_name, input_device_names
from .audioutil import normalize, peak_window_rms
from .autocomplete import Autocomplete
from .cleanup import (Cleaner, list_model_choices, model_display_name, reasoning_budget_for, resolve_path,
                      same_model)
from .config import Config
from .gamemode import GameWatcher
from .gate import ForegroundGate
from .history import History
from .hotkey import HotkeyManager
from .inserter import (capture_selection, focus_window, foreground_hwnd, insert_text, is_external_target,
                       set_clipboard_text, wait_modifiers_released)
from .outmute import OutputMuter, WinEndpoints
from .overlay import Overlay
from .paths import LOG_PATH, MODELS_DIR
from .pipeline import Pipeline
from .streaming import StreamSession
from .stt import ModelUnloaded, Transcriber
from .tts import TtsManager

try:    # transcripts / meeting recorder: optional, dictation keeps working if their deps are missing
    from .jobs import JobError, TranscriptService
    from .recorder import RecorderError
    _TR_IMPORT_ERROR = ""
except Exception as _e:     # pragma: no cover - logged when the service would be constructed
    TranscriptService = None
    _TR_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"

    class JobError(RuntimeError):
        pass

    class RecorderError(RuntimeError):
        pass

AUDIO_EXTS = "wav mp3 m4a aac ogg opus flac wma webm mp4 mkv mov"

log = logging.getLogger("localflow")


class _DictCfg:
    """Read-only cfg.get() over a plain config dict (old/new snapshots in _on_config_change)."""

    def __init__(self, data):
        self.data = data

    def get(self, dotted, default=None):
        cur = self.data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur


def _reasoning_budget_in(data) -> int:
    """Reasoning budget the configured cleanup model launches with under config ``data`` (0 = off)."""
    try:
        cfg = _DictCfg(data)
        return reasoning_budget_for(cfg, resolve_path(cfg.get("cleanup.model_path", "") or ""))
    except Exception:
        return -1


def make_icon(state: str = "ready") -> QIcon:
    pm = QPixmap(64, 64)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    from . import palette as pal
    col = {"ready": pal.ACCENT, "paused": pal.PAUSED, "busy": pal.WARN, "error": pal.ERROR,
           "game": pal.GAME, "recording": pal.REC}.get(state, pal.ACCENT)
    p.setBrush(QBrush(QColor(col)))
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(2, 2, 60, 60, 16, 16)
    p.setBrush(QBrush(QColor("white")))
    if state == "recording":            # meeting recording: a "record" dot instead of the waveform
        p.drawEllipse(18, 18, 28, 28)
    else:
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
    toast = Signal(str, str)         # overlay state ("info" | "error"), text - from any thread
    # transcripts / meetings (TranscriptService callbacks, re-emitted on the Qt main thread)
    tr_progress = Signal(int, str, float, str)      # transcript id, status, progress 0..1, message
    tr_done = Signal(int, str)                      # transcript id, final status (done|failed|cancelled|interrupted)
    meeting_state = Signal(object)                  # TranscriptService.meeting_state() dict
    meeting_levels = Signal(float, float)           # mic rms, system rms (~10 Hz while recording)
    # AI notes (LocalFlowApp.ask_ai): handle is the jobs.AskHandle returned by ask_ai
    ask_partial = Signal(object, str)               # handle, answer text so far
    ask_status = Signal(object, str)                # handle, status message ("Waiting for the AI model", ...)
    ask_done = Signal(object, object, str)          # handle, saved ai_results row (dict) or None, error ("" = ok)


class LocalFlowApp:
    def __init__(self, qapp: QApplication, real_output_mute: bool = False, real_tts: bool = False):
        self.qapp = qapp
        self.cfg = Config()
        self.history = History(max_entries=int(self.cfg.get("history.max_entries", 5000)),
                               retention_days=float(self.cfg.get("history.retention_days", 0) or 0))
        # foreground (dictation / command) beats background GPU jobs: begin() at hotkey-down, end() once per begin
        self.gate = ForegroundGate()
        self._fg_rids: set[int] = set()
        self._fg_lock = threading.Lock()
        self.bridge = Bridge()
        self.overlay = Overlay()
        self.audio = AudioInput(self.cfg, on_level=self._on_level)
        self.audio.on_device_change = lambda name: self._set_status(f"Ready - mic: {name}", "ready")
        self.stt = Transcriber(self.cfg)
        self.cleaner = Cleaner(self.cfg)
        self.downloads = modeldl.ModelDownloader()     # the default cleanup model is fetched on first start
        self._dl_pct = -1
        self.pipeline = Pipeline(self.cfg, self.stt, self.cleaner, self.history, self._insert_dictation)
        self.hotkey = HotkeyManager(
            os.environ.get("LOCALFLOW_TEST_HOTKEY") or self.cfg.get("hotkey", "ctrl+alt"),
            self.cfg.get("handsfree_key", "space"),
            on_start=lambda mode="dictate": self._ctl.put(("start", mode)),
            on_stop=lambda: self._ctl.put(("stop",)),
            on_cancel=lambda why: self._ctl.put(("cancel", why)),
            on_handsfree=lambda: self._ctl.put(("handsfree",)),
            command_hotkey=self._command_spec(), paste_hotkey=self._paste_spec(),
            on_paste=lambda: self._jobs.put(("paste", foreground_hwnd())))
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
        self.updates = None            # update_ui.UpdateController, made in run() (needs the tray)
        self.tray: Optional[QSystemTrayIcon] = None
        self._status = "Starting..."
        self._ready = False
        self.last_result: dict = {}
        self.last_text = ""            # last successfully inserted non-empty text (dictation or command) or answer
        self._answer_until = 0.0       # monotonic time the overlay answer card hides; toasts wait until then
        self._mode = "dictate"         # mode of the current / last recording: dictate | command
        self._target = (0, "", "")     # command mode: (hwnd, exe, title) of the window at key-down
        self._cmd_stop_sent = False
        self._last_ext_hwnd = 0        # last foreground window of another app (tray "Paste last transcript")
        # game mode: one lock serializes every GPU model load/unload (boot, cfg reload, game enter/exit)
        self._gpu_lock = threading.RLock()
        self.game_active = False       # GPU models are (being) unloaded; guarded by _gpu_lock for writes
        self._game_ui = False          # tray shows the game state
        self.watcher = GameWatcher(self.cfg, self._game_enter, self._game_exit)
        self._meeting_on = False
        self._status_icon = "busy"      # last status icon / text (the tray shows "recording" over it)
        self._status_text = "starting"
        self._toasts: list[tuple[str, str]] = []
        self._toast_timer = False
        self._last_audio_dir = ""
        self.transcripts = self._make_transcripts()
        self.autocomplete = self._make_autocomplete()
        # speakers muted while dictating (audio.mute_output_while_dictating); inert unless the real entry point
        # asks for it, so tests never touch real audio devices
        self.outmute = OutputMuter(WinEndpoints() if real_output_mute and os.name == "nt" else None,
                                   is_recording=lambda: self._recording,
                                   max_s=lambda: float(self.cfg.get("audio.max_duration_s", 300) or 300) + 30)
        # spoken command-mode answers (separate GPU process); inert unless the real entry point asks for it
        self.tts = TtsManager(self.cfg, blocked=lambda: self.game_active or not self.cfg.get("enabled", True),
                              inert=not real_tts, wav_out=os.environ.get("LOCALFLOW_TTS_WAV_OUT", ""))

    # ------------------------------------------------------------------ Tab autocomplete
    def _make_autocomplete(self) -> Optional[Autocomplete]:
        """Never fatal: dictation keeps working if the ghost window / autocomplete can't be set up."""
        if QApplication.instance() is None:     # no GUI (tests): a QWidget here would abort the process
            return None
        try:
            from .ghost import GhostOverlay
            self.ghost = GhostOverlay()
            ac = Autocomplete(self.cfg, self, ghost=self.ghost)
            self.hotkey.key_listener = ac
            return ac
        except Exception:
            log.exception("Tab autocomplete unavailable")
            return None

    def ac_blocked(self) -> str:
        """Why autocomplete may not guess right now ('' = it may): paused, game mode, dictation / command running."""
        if not self.cfg.get("enabled", True):
            return "paused"
        if self.game_active:
            return "game mode"
        if self._recording:
            return "busy (recording)"
        if self._fg_rids:
            return f"busy ({len(self._fg_rids)} job(s) running)"
        return ""

    def ac_server(self) -> tuple[bool, str]:
        s = self.cleaner.server
        return bool(s.ready and not s.suspended), s.base_url

    def _ac_cancel(self, why: str) -> None:
        ac = getattr(self, "autocomplete", None)
        if ac is not None:
            ac.cancel(why)

    # ------------------------------------------------------------------ overlay/sound helpers
    def _ov(self, state: str, text: str = "") -> None:
        if self.cfg.get("ui.overlay", True) or state in ("error",):
            self.overlay.set_state(state, text)
        elif state == "hidden":
            self.overlay.set_state("hidden")

    def _rid_toast(self, rid, state: str, text: str) -> None:
        """Toast for a finished dictation, shown even with ui.overlay off. If a newer recording started meanwhile,
        its pill wins: the toast is queued (bridge.toast waits until that recording / job is done)."""
        if self._recording or self._rec_id != rid:
            self.bridge.toast.emit(state, text)
        else:
            self.overlay.set_state(state, text)

    def _insert_dictation(self, text: str, mode: str = "paste", restore: bool = True,
                          restore_delay_ms: int = 300) -> Optional[dict]:
        """Pipeline inserter: insert_text, or (no text box focused, see fieldprobe) copy to the clipboard."""
        return fieldprobe.deliver(text, cfg=self.cfg, insert=insert_text, copy=set_clipboard_text, mode=mode,
                                  restore=restore, restore_delay_ms=restore_delay_ms)

    def _snd(self, name: str) -> None:
        sounds.play(name, self.cfg)

    def _command_spec(self) -> Optional[str]:
        if not self.cfg.get("command_mode.enabled", True):
            return None
        return str(self.cfg.get("command_mode.hotkey", "ctrl+win") or "") or None

    def _paste_spec(self) -> Optional[str]:
        if not self.cfg.get("paste_last.enabled", True):
            return None
        return str(self.cfg.get("paste_last.hotkey", "alt+shift+z") or "") or None

    # foreground gate: exactly one end() per begin(), keyed by recording id (extra _fg_end calls are no-ops)
    def _fg_begin(self, rid: int) -> None:
        with self._fg_lock:
            if rid in self._fg_rids:
                return
            self._fg_rids.add(rid)
        self._ac_cancel("dictation")        # stop a running guess (frees llama slot 1) and hide the ghost
        self.gate.begin()

    def _fg_end(self, rid: Optional[int]) -> None:
        with self._fg_lock:
            if rid not in self._fg_rids:
                return
            self._fg_rids.discard(rid)
        self.gate.end()

    def _on_level(self, lvl: float) -> None:
        if self._recording:
            self.overlay.push_level(lvl)
            if self._mode == "command" and not self._cmd_stop_sent and \
                    time.perf_counter() - self._t_start > float(self.cfg.get("command_mode.max_duration_s", 60)):
                self._cmd_stop_sent = True
                log.warning("max command duration reached")
                self.hotkey.force_idle()
                self._ctl.put(("stop",))
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
            rid_before, was_rec = self._rec_id, self._recording
            try:
                kind = ev[0]
                if kind == "start":
                    self._do_start(ev[1] if len(ev) > 1 else "dictate")
                elif kind == "handsfree":
                    self._do_handsfree()
                elif kind == "stop":
                    self._do_stop()
                elif kind == "cancel":
                    self._do_cancel(ev[1])
            except Exception:
                log.exception("controller error on %s", ev)
                self._recording = False
                self.outmute.release()
                if was_rec or self._rec_id != rid_before:
                    self._fg_end(self._rec_id)
                self._ov("hidden")

    def _game_blocks(self) -> bool:
        """True while game mode keeps dictation off (no speech model on purpose)."""
        return self.game_active and (self.cfg.get("game_mode.while_active", "off") != "cpu" or self.stt.model is None)

    def _sync_hotkey(self) -> None:
        """Hotkey hook is switched off while game mode blocks dictation (no beeps / injected keys during a game)."""
        self.hotkey.set_enabled(bool(self.cfg.get("enabled", True)) and not self._game_blocks())

    def _do_start(self, mode: str = "dictate") -> None:
        if self._recording:
            return
        self._rec_id += 1
        rid = self._rec_id
        # silence a spoken answer first (and drop answers of older recordings still on their way); the mic
        # pre-roll would hold the voice if it was audible just now
        skip_preroll = self.tts.recording_started(rid)
        self._fg_begin(rid)             # before anything else: background jobs yield the GPU from now on
        if self._game_blocks():
            # no overlay: it could steal focus / flicker over a fullscreen game
            log.info("hotkey ignored: game mode")
            self.hotkey.force_idle()
            self._snd("error")
            self._fg_end(rid)
            return
        if self.cfg.get("audio.mute_output_while_dictating", True) and not self._meeting_on:
            self.outmute.arm(rid)       # muted in _show_recording, once the hold is confirmed
        self._mode = "command" if mode == "command" else "dictate"
        self._answer_until = 0.0        # the recording pill replaces any answer card
        self._locked = False
        self._t_start = time.perf_counter()
        if self._mode == "command":
            hwnd = foreground_hwnd()
            exe, title = winutil.window_info(hwnd) if hwnd else ("", "")
            self._target = (hwnd, exe, title)
            self._ctx = (exe, title)
            self._cmd_stop_sent = False
            self._begin_audio(skip_preroll)
            self._recording = True
            self._session = None        # commands are short: single pass, no streaming
            log.info("command recording started: app=%s title=%r", exe or "?", title[:100])
        else:
            self._ctx = winutil.foreground_info()
            # style is decided from the window the user is dictating INTO, now, not at release
            self._style = styles.resolve(self._ctx[0], self._ctx[1], self.cfg)
            self._begin_audio(skip_preroll)
            self._recording = True
            self._session = self._start_session()
            if self.cfg.get("insert.copy_when_no_field", True):
                fieldprobe.warm()       # UIA thread + the focused app's provider ready before the insert-time check
            log.info("recording started: app=%s style=%s%s title=%r", self._ctx[0] or "?", self._style.style,
                     f" (rule: {self._style.rule})" if self._style.rule else "", self._ctx[1][:100])
        threading.Timer(0.12, self._show_recording, args=(rid,)).start()

    def _begin_audio(self, skip_preroll: bool = False) -> None:
        if skip_preroll:
            drop = getattr(self.audio, "drop_preroll", None)
            if drop is not None:
                drop()
                log.info("mic pre-roll skipped (a spoken answer was playing)")
        self.audio.begin()

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

    def _rec_label(self) -> str:
        return "Command" if self._mode == "command" else self._style_label()

    def _show_recording(self, rid: int) -> None:
        # only show UI if the user is still holding (not a Ctrl+Alt+<key> shortcut)
        if self._recording and self._rec_id == rid:
            self._ov("handsfree" if self._locked else "recording", self._rec_label())
            self._snd("start")
            self._ui_shown = rid
            if not self._meeting_on:    # mute only after the start cue had time to play
                self.outmute.engage(rid, delay=0.2 if self.cfg.get("ui.sounds", True) else 0.0)

    def _do_handsfree(self) -> None:
        if not self._recording:
            return
        self._locked = True
        self._ov("handsfree", self._rec_label())
        self._snd("start")
        self._ui_shown = self._rec_id

    def _do_stop(self) -> None:
        if not self._recording:
            return
        self._recording = False
        self.outmute.release()          # speakers back on before the stop cue and STT
        rid = self._rec_id
        t_release = time.perf_counter()
        held = t_release - self._t_start
        min_d = float(self.cfg.get("audio.min_duration_s", 0.3))
        if held < min_d:
            self._drop_session()
            self.audio.cancel()
            log.info("discarded: too short (%.2fs held)", held)
            self._ov("hidden")
            self._fg_end(rid)
            return
        if self._mode == "command":
            audio = self.audio.end()
            self._ov("processing", "Working…")
            self._snd("stop")
            self._jobs.put(("command", rid, audio, self._target, t_release))
            return
        session, self._session = self._session, None
        audio = self.audio.end()
        self._ov("processing")
        self._snd("stop")
        self._jobs.put(("dictate", rid, audio, self._ctx, self._style, session, t_release))

    def _do_cancel(self, why: str) -> None:
        if not self._recording:
            return
        self._recording = False
        self.outmute.release()
        self._drop_session()
        self.audio.cancel()
        self._fg_end(self._rec_id)
        log.info("recording cancelled (%s)", why)
        if why == "esc":
            self._ov("info", "Cancelled")
            self._snd("cancel")
        else:
            self._ov("hidden")

    # ------------------------------------------------------------------ worker thread (STT -> cleanup -> insert)
    def _worker(self) -> None:
        """One job at a time: ("dictate", rid, audio, ctx, style, session, t_rel), ("command", rid, audio, target,
        t_rel) or ("paste", hwnd). Insertions are serialized here, so they never interleave."""
        while not self._shutdown.is_set():
            try:
                job = self._jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            kind = job[0] if isinstance(job[0], str) else "dictate-legacy"
            try:
                if kind == "dictate":
                    self._run_dictation(*job[1:])
                elif kind == "dictate-legacy":         # (audio, ctx, style, session, t_rel): no gate id
                    self._run_dictation(None, *job)
                elif kind == "command":
                    self._run_command(*job[1:])
                elif kind == "paste":
                    self._run_paste(*job[1:])
                else:
                    log.error("unknown job %r", kind)
            except Exception:
                log.exception("%s job failed", kind)

    def _wait_stt(self) -> bool:
        """Wait (up to 2 min) for the speech model. False = job dropped (overlay already updated)."""
        waited = 0.0
        while not self.stt.ready and waited < 120 and not self._shutdown.is_set() and not self._game_blocks():
            time.sleep(0.25)
            waited += 0.25
        if not self.stt.ready and self._game_blocks():
            log.info("dictation dropped: game mode")
            self._ov("hidden")
            return False
        if not self.stt.ready:
            self._ov("error", "Speech model not ready")
            self._snd("error")
            return False
        return True

    def _run_dictation(self, rid, audio, ctx, sty, session, t_rel) -> None:
        try:
            if not self._wait_stt():
                return
            r = self.pipeline.process(audio, app_exe=ctx[0], app_title=ctx[1], t_release=t_rel,
                                      style_ctx=sty, session=session)
            self.last_result = r
            if r["status"] == "ok":
                if r.get("text"):
                    self.last_text = r["text"]
                if r.get("copy_failed"):        # no text box focused and the clipboard could not be set
                    self._rid_toast(rid, "error", fieldprobe.COPY_FAILED_TOAST)
                    self._snd("error")
                elif r.get("copied"):           # no text box focused: the text is on the clipboard
                    self._rid_toast(rid, "info", fieldprobe.COPIED_TOAST)
                else:
                    self._ov("hidden")
            elif r["status"] == "discard":
                log.info("discarded: %s", r["reason"])
                self._ov("hidden")
            else:
                self._ov("error", "Couldn't insert text")
                self._snd("error")
        except ModelUnloaded:
            log.info("dictation dropped: speech model unloaded (game mode)")
            self._ov("hidden")
        except Exception:
            log.exception("dictation failed")
            self._ov("error", "Transcription failed")
            self._snd("error")
        finally:
            self._fg_end(rid)

    # ------------------------------------------------------------------ command mode
    def _can_capture(self, exe: str) -> bool:
        """False for apps where Ctrl+C is unsafe/ambiguous (terminals interrupt, editors copy the whole line):
        command mode is generate-only there."""
        apps = self.cfg.get("command_mode.no_capture_apps", []) or []
        return bool(exe) and exe.lower() not in {str(a).lower() for a in apps}

    def _command_failed(self, why: str, instruction: str, exe: str, title: str, dur: float,
                        output: str = "", toast: str = "") -> None:
        log.warning("command failed (%s): app=%s instruction=%r", why, exe or "?", instruction[:100])
        self._ov("error", toast or f"Command failed: {why}")
        self._snd("error")
        if instruction and self.cfg.get("history.enabled", True):
            try:
                self.history.add(instruction, output or f"[command failed: {why}]", exe, dur, title=title,
                                 style="command")
            except Exception:
                log.exception("history write failed")

    def _run_command(self, rid, audio, target, t_rel) -> None:
        """Command mode: STT the instruction, copy the selection (if any, and if safe), ask the LLM, replace the
        selection / insert at the cursor. Inserts NOTHING on any failure; never inserts the spoken instruction."""
        hwnd, exe, title = target
        dur = len(audio) / 16000.0
        instruction = text = ""
        try:
            if not self._wait_stt():
                return
            cfg = self.cfg
            if dur < float(cfg.get("audio.min_duration_s", 0.3)) or \
                    peak_window_rms(audio) < float(cfg.get("audio.silence_rms", 0.006)):
                log.info("command discarded: silence")
                self._ov("info", "No command heard")
                return
            words = [x for x in (cfg.get("dictionary", {}) or {}).get("words", []) if str(x).strip()]
            res = self.stt.transcribe(normalize(audio), hotwords=words)
            instruction = (res.text or "").strip()
            if not instruction:
                log.info("command discarded: %s", res.dropped or "empty transcript")
                self._ov("info", "No command heard")
                return
            log.info("command: %r (app=%s, %.1fs audio, stt %d ms)", instruction[:150], exe or "?", dur, res.ms)
            if cfg.get("command_mode.answer_questions", True) and ask.is_question(instruction):
                return self._answer_question(rid, instruction, target, dur, res.ms, t_rel)
            selection = ""
            if self._can_capture(exe):
                if not wait_modifiers_released(2.0):
                    return self._command_failed("keys still held", instruction, exe, title, dur)
                if foreground_hwnd() != hwnd:
                    return self._command_failed("window changed", instruction, exe, title, dur)
                cst, sel = capture_selection(int(cfg.get("command_mode.capture_timeout_ms", 600)), target_hwnd=hwnd)
                if cst == "window":
                    return self._command_failed("window changed", instruction, exe, title, dur)
                if cst == "busy":
                    return self._command_failed("clipboard busy", instruction, exe, title, dur)
                if cst == "uncertain":
                    return self._command_failed("couldn't read the selection", instruction, exe, title, dur)
                selection = sel if cst == "text" else ""
                log.info("command: selection %s", f"{len(selection)} chars" if selection else "none")
            else:
                log.info("command: %s is generate-only (no selection capture)", exe or "unknown app")
            text, info = self.cleaner.command(instruction, selection, exe)
            if not text:
                why = info.get("reason") or "no answer"
                return self._command_failed(why, instruction, exe, title, dur,
                                            toast="Select some text first" if why == "nothing selected" else "")
            if foreground_hwnd() != hwnd:
                return self._command_failed("window changed", instruction, exe, title, dur, output=text,
                                            toast="Window changed - result saved to history")
            ins = insert_text(text, mode=cfg.get("insert.mode", "paste"),
                              restore=bool(cfg.get("insert.restore_clipboard", True)),
                              restore_delay_ms=int(cfg.get("insert.restore_delay_ms", 300)), require_release=True,
                              target_hwnd=hwnd)
            if not ins.get("ok", True):
                why = {"modifiers held": "keys still held"}.get(ins.get("reason"), ins.get("reason") or "insert failed")
                toast = {"keys still held": "Keys held", "window changed": "Window changed"}.get(why, "Couldn't insert")
                return self._command_failed(why, instruction, exe, title, dur, output=text,
                                            toast=f"{toast} - result saved to history")
            self.last_text = text
            total_ms = int((time.perf_counter() - t_rel) * 1000)
            note = f"  [on selection: {len(selection)} chars]" if selection else ""
            if info.get("python"):
                note += "  [python]"
            try:
                self.history.record_stats(kind="command")
            except Exception:
                log.exception("stats write failed")
            if cfg.get("history.enabled", True):
                try:
                    self.history.add(instruction + note, text, exe, dur, res.ms, info.get("ms", 0), total_ms,
                                     title=title, style="command")
                except Exception:
                    log.exception("history write failed")
            log.info("command done: stt %d ms | llm %d ms | total %d ms | %s -> %r", res.ms, info.get("ms", 0),
                     total_ms, exe or "?", text[:120])
            self._ov("hidden")
        except ModelUnloaded:
            log.info("command dropped: speech model unloaded (game mode)")
            self._ov("hidden")
        except Exception as e:
            log.exception("command failed")
            self._command_failed(f"error: {type(e).__name__}", instruction, exe, title, dur, output=text,
                                 toast="Command failed")
        finally:
            self._fg_end(rid)

    def _question_selection(self, hwnd: int, exe: str) -> str:
        """Selected text for a question about it ("what does this mean"); "" whenever it can't be read safely.
        Unlike a command, a question is still answered without it."""
        if not self._can_capture(exe):
            return ""
        if not wait_modifiers_released(2.0) or foreground_hwnd() != hwnd:
            return ""
        cst, sel = capture_selection(int(self.cfg.get("command_mode.capture_timeout_ms", 600)), target_hwnd=hwnd)
        return sel if cst == "text" else ""

    def _answer_question(self, rid, instruction: str, target, dur: float, stt_ms: int, t_rel: float) -> None:
        """Command mode question: answer it in the overlay card (exactly when it is a time span / arithmetic,
        otherwise via the LLM). Nothing is inserted; the answer becomes last_text so paste-last can insert it."""
        hwnd, exe, title = target
        cfg = self.cfg
        question = ask.question_text(instruction)
        selection = ""
        ans = ask.quick_answer(question, datetime.now())
        info = {"ms": 0, "llm": False}
        if not ans:
            if ask.references_text(question):
                selection = self._question_selection(hwnd, exe)
                log.info("command: question selection %s", f"{len(selection)} chars" if selection else "none")
            ans, info = self.cleaner.answer(question, selection, exe)
            if not ans:
                why = info.get("reason") or "no answer"
                return self._command_failed(why, instruction, exe, title, dur, toast=f"Couldn't answer: {why}")
        self.last_text = ans
        total_ms = int((time.perf_counter() - t_rel) * 1000)
        note = f"  [answer, on selection: {len(selection)} chars]" if selection else "  [answer]"
        if info.get("python"):
            note += "  [python]"
        try:
            self.history.record_stats(kind="command")
        except Exception:
            log.exception("stats write failed")
        if cfg.get("history.enabled", True):
            try:
                self.history.add(instruction + note, ans, exe, dur, stt_ms, info.get("ms", 0), total_ms,
                                 title=title, style="command")
            except Exception:
                log.exception("history write failed")
        log.info("command answer (%s): llm %d ms | total %d ms | %r", "llm" if info.get("llm") else "local",
                 info.get("ms", 0), total_ms, ans[:120])
        if self._recording or self._rec_id != rid:      # a newer recording started meanwhile: its UI wins
            log.info("answer not shown: a new recording is in progress")
            return
        duration_ms = min(15000, 4000 + 50 * len(ans))
        spoken_s = self._speak_answer(rid, ans)
        if spoken_s > 0:        # keep the card up while the answer is read out
            duration_ms = max(duration_ms, int(spoken_s * 1000) + 2000)
        self._answer_until = time.monotonic() + duration_ms / 1000.0     # before the gate is released (finally)
        self.overlay.show_answer(question, ans, duration_ms)    # shown even with ui.overlay off, like errors

    def _speak_answer(self, rid, ans: str) -> float:
        """Read a question's answer aloud (command_mode.speak_answers). Seconds of speech expected, 0 = not spoken."""
        if self._meeting_on:            # never talk over (or into) a meeting recording
            return 0.0
        try:
            return float(self.tts.speak(ans, rid) or 0.0)
        except Exception:
            log.exception("tts: speak failed")
            return 0.0

    # ------------------------------------------------------------------ paste last
    def _run_paste(self, hwnd: int = 0) -> None:
        """Re-insert ``last_text`` into ``hwnd``: the window focused when the hotkey was pressed, or (tray) the
        last app window before the menu opened, focused again first. Nothing is pasted if focus moves away
        before the keystroke goes out. 0 = whatever has focus."""
        text = self.last_text
        if not text:
            self._ov("info", "Nothing to paste yet")
            return
        if hwnd and not focus_window(hwnd):
            self._ov("error", "Couldn't switch back to the window")
            self._snd("error")
            return
        cfg = self.cfg
        ins = insert_text(text, mode=cfg.get("insert.mode", "paste"),
                          restore=bool(cfg.get("insert.restore_clipboard", True)),
                          restore_delay_ms=int(cfg.get("insert.restore_delay_ms", 300)),
                          require_release=True, release_timeout=5.0, target_hwnd=hwnd)
        if not ins.get("ok", True):
            self._ov("error", {"modifiers held": "Release the keys to paste", "window changed": "Window changed",
                               "clipboard busy": "Clipboard busy"}.get(ins.get("reason"), "Couldn't paste"))
            self._snd("error")
            return
        log.info("pasted last transcript (%d chars)", len(text))

    def paste_last(self) -> None:
        """Tray: switch back to the window that was active before the menu opened and paste there."""
        self._jobs.put(("paste", self._last_ext_hwnd))

    def copy_last(self) -> None:
        if not self.last_text:
            self._ov("info", "Nothing to copy yet")
            return
        QApplication.clipboard().setText(self.last_text)
        self._ov("info", "Copied last transcript")

    def _track_foreground(self) -> None:
        """Main-thread timer: remember the last window of another app (the tray menu steals focus)."""
        try:
            h = foreground_hwnd()
            if h and h != self._last_ext_hwnd and is_external_target(h):
                self._last_ext_hwnd = h
        except Exception:
            pass

    # ------------------------------------------------------------------ transcripts / meetings
    def _transcripts_paused(self) -> bool:
        """Background transcript jobs hold while game mode is on, while the speech model is not loaded, and while
        it runs on the CPU fallback (a speech model the user configured for the CPU is fine)."""
        if self.game_active or not self.stt.ready:
            return True
        return self.stt.device != str(self.cfg.get("stt.device", "cuda") or "cuda")

    def _make_transcripts(self):
        """The transcripts service (file transcription, meeting recorder, AI notes), or None if it can't start:
        dictation keeps working either way."""
        if TranscriptService is None:
            log.error("transcripts unavailable: %s", _TR_IMPORT_ERROR)
            return None
        b = self.bridge
        try:
            return TranscriptService(
                self.cfg, self.stt, self.cleaner, self.gate, is_paused=self._transcripts_paused,
                request_extra_fn=self.cleaner._request_overrides,
                on_progress=b.tr_progress.emit, on_done=b.tr_done.emit,
                on_meeting=b.meeting_state.emit, on_levels=b.meeting_levels.emit)
        except Exception:
            log.exception("transcripts unavailable: service failed to start")
            return None

    def _recover_transcripts(self) -> None:
        try:
            ids = self.transcripts.recover()
            resumed = bool(self.cfg.get("transcripts.auto_resume", True))
            log.info("transcripts: recovery done (%d interrupted job(s)%s)", len(ids),
                     ", resumed" if ids and resumed else "")
        except Exception:
            log.exception("transcripts: recovery failed")

    @property
    def meeting_recording(self) -> bool:
        return bool(self.transcripts and self.transcripts.recorder.recording)

    def transcript_styles(self) -> list[str]:
        """Style choices for transcripts: "raw" (no AI cleanup) first, then the configured style names."""
        names = list((self.cfg.get("styles", {}) or {}).keys()) or list(styles.DEFAULT_STYLES)
        return ["raw"] + [n for n in names if n != "raw"]

    def start_meeting(self) -> bool:
        """Start the meeting recorder (any thread; blocks while the audio streams open). Errors -> toast."""
        if self.transcripts is None:
            self.bridge.toast.emit("error", "Meeting recording unavailable - see log")
            return False
        try:
            st = self.transcripts.start_meeting()
            log.info("meeting recording started: %s", st.get("dir"))
            return True
        except RecorderError as e:
            log.warning("meeting recording failed to start: %s", e)
            self.bridge.toast.emit("error", str(e) or "Couldn't start recording")
        except Exception as e:
            log.exception("meeting recording failed to start")
            self.bridge.toast.emit("error", f"Couldn't start recording ({type(e).__name__})")
        return False

    def stop_meeting(self) -> Optional[int]:
        """Stop the meeting recorder and queue its transcription (any thread). Returns the transcript id."""
        if self.transcripts is None:
            return None
        try:
            tid = self.transcripts.stop_meeting()
        except Exception:
            log.exception("stopping the meeting recording failed")
            self.bridge.toast.emit("error", "Couldn't stop the recording - see log")
            return None
        log.info("meeting recording stopped (transcript %s)", tid)
        if tid is not None:
            self.bridge.toast.emit("info", "Meeting saved - transcribing")
        return tid

    def toggle_meeting(self) -> None:
        """Tray: start / stop the meeting recording off the UI thread."""
        fn = self.stop_meeting if self.meeting_recording else self.start_meeting
        threading.Thread(target=fn, name="meeting-ctl", daemon=True).start()

    def import_audio_file(self, path: str, style: Optional[str] = None) -> Optional[int]:
        """Queue an audio/video file for transcription (any thread). Returns the transcript id; errors -> toast."""
        if self.transcripts is None:
            self.bridge.toast.emit("error", "Transcription unavailable - see log")
            return None
        name = os.path.basename(path)
        try:
            tid = self.transcripts.import_file(path, style)
        except FileNotFoundError:
            self.bridge.toast.emit("error", f"File not found: {name}")
            return None
        except JobError as e:
            self.bridge.toast.emit("error", str(e))
            return None
        except Exception as e:
            log.exception("import_file failed for %s", path)
            self.bridge.toast.emit("error", f"Couldn't open {name} ({type(e).__name__})")
            return None
        log.info("transcript %s queued: %s (style=%s)", tid, path, style)
        self.bridge.toast.emit("info", f"Queued for transcription: {name}")
        return tid

    def transcribe_file_dialog(self, parent=None) -> None:
        """Main thread: pick a file + style, then queue it (the import runs off the UI thread)."""
        if self.transcripts is None:
            self._notify("error", "Transcription unavailable - see log")
            return
        flt = "Audio / video (" + " ".join(f"*.{e}" for e in AUDIO_EXTS.split()) + ");;All files (*)"
        path, _ = QFileDialog.getOpenFileName(parent, "Transcribe audio file", self._last_audio_dir, flt)
        if not path:
            return
        self._last_audio_dir = os.path.dirname(path)
        choices = self.transcript_styles()
        default = str(self.cfg.get("transcripts.file_style", "raw") or "raw")
        style, ok = QInputDialog.getItem(parent, "Transcribe audio file", "Style (raw = no AI cleanup):", choices,
                                         choices.index(default) if default in choices else 0, False)
        if not ok or not style:
            return
        threading.Thread(target=self.import_audio_file, args=(path, style), name="transcript-import",
                         daemon=True).start()

    def ask_ai(self, tid: int, preset: str, prompt: Optional[str] = None):
        """AI note on a transcript (summary | bullets | key_points | action_items | custom + prompt). Returns the
        jobs.AskHandle (``cancel()``); results arrive on bridge.ask_partial / ask_status / ask_done with it."""
        if self.transcripts is None:
            return None
        b = self.bridge
        holder: dict = {}
        known = threading.Event()

        def h():
            known.wait(2.0)             # callbacks can fire before ask() has returned the handle
            return holder.get("h")
        handle = self.transcripts.ask(tid, preset, prompt,
                                      on_partial=lambda t: b.ask_partial.emit(h(), t),
                                      on_status=lambda m: b.ask_status.emit(h(), m),
                                      on_done=lambda row, err: b.ask_done.emit(h(), row, err))
        holder["h"] = handle
        known.set()
        return handle

    # main-thread handlers for the service signals
    def _on_tr_done(self, tid: int, status: str) -> None:
        if status not in ("done", "failed") or self.transcripts is None:
            return
        try:
            row = self.transcripts.get(tid) or {}
        except Exception:
            row = {}
        title = row.get("title") or f"transcript {tid}"
        if status == "done":
            self._notify("info", f"Transcript ready: {title}" + (" (basic cleanup)" if row.get("error") else ""))
        else:
            self._notify("error", f"Transcript failed: {title}")

    def _on_meeting_state(self, st: dict) -> None:
        was = self._meeting_on
        self._meeting_on = bool(st.get("recording"))
        if self._meeting_on and not was:    # a meeting recording needs the call audio: unmute right away
            self.outmute.release(wait=0)
            self.tts.stop("meeting recording")
        self._apply_tray()
        reason = st.get("stop_reason") or ""
        if was and not self._meeting_on and reason:     # stopped by itself (max length, low disk, write error)
            log.warning("meeting recording auto-stopped: %s", reason)
            self._notify("info" if "maximum" in reason else "error", f"Recording stopped: {reason}")

    def _notify(self, state: str, text: str) -> None:
        """Main thread: overlay toast for background events. Waits while a dictation / command is in progress
        (it would replace the recording pill); skipped in game mode (no overlay over a game)."""
        log.info("notify: %s", text)
        self._toasts.append((state, text))
        self._flush_toasts()

    def _flush_toasts(self) -> None:
        if not self._toasts:
            return
        if self._game_ui:
            self._toasts.clear()
            return
        left = self._answer_until - time.monotonic()        # an answer card is showing: let it finish first
        if self._recording or self.gate.busy() or left > 0:
            if not self._toast_timer:
                self._toast_timer = True

                def again():
                    self._toast_timer = False
                    self._flush_toasts()
                busy = self._recording or self.gate.busy()
                QTimer.singleShot(1000 if busy else int(left * 1000) + 250, again)
            return
        # one pill at a time: the latest error wins over any info (an error must not be hidden by a later success)
        errors = [t for t in self._toasts if t[0] == "error"]
        state, text = errors[-1] if errors else self._toasts[-1]
        forced = text == fieldprobe.COPIED_TOAST        # dictation copied (no text box): shown even with ui.overlay off
        if len(self._toasts) > 1:
            text = f"{text} (+{len(self._toasts) - 1} more)"
        self._toasts.clear()
        if forced:
            self.overlay.set_state(state, text)
        else:
            self._ov(state, text)

    # ------------------------------------------------------------------ boot
    def _boot(self) -> None:
        try:
            self.audio.start()
            with self._gpu_lock:
                if self.game_active:        # game already running at startup: keep the GPU free
                    self.cleaner.server.suspended = True
                    self._game_ui = True
                    self._load_game_cpu_model()
                    self._sync_hotkey()
                    self._ready = True
                    self._set_status(self._game_text(), "game")
                    log.info("READY (game mode): stt=%s/%s", self.stt.model_name, self.stt.device)
                    return
                self._set_status(f"Loading speech model...", "busy")
                self.stt.load()
                try:   # load the VAD used for chunked transcription now, not on the first long dictation
                    from faster_whisper.vad import get_vad_model
                    get_vad_model()
                except Exception:
                    log.exception("VAD preload failed")
                if not self._ensure_cleanup_model():    # still downloading: its completion starts the server
                    self.cleaner.start()
                dev, mname, comp = self.stt.device, self.stt.model_name, self.stt.compute
            gpu = "GPU" if dev == "cuda" else "CPU (fallback)"
            mic = self.audio.device_name
            self._ready = True
            if not self.game_active:        # a game may have started while the models were loading
                self._set_status(f"Ready - {mname} on {gpu} - mic: {mic}", "ready" if dev == "cuda" else "error")
                if dev != "cuda":
                    self._ov("error", "GPU unavailable - using CPU")
            log.info("READY: stt=%s/%s/%s mic=%s", mname, dev, comp, mic)
            if not self.game_active:        # spoken answers: after the dictation models, never under a game
                self.tts.start()
        except Exception:
            log.exception("boot failed")
            self._set_status("Startup failed - see log", "error")

    # ------------------------------------------------------------------ game mode
    def _game_text(self) -> str:
        if self.cfg.get("game_mode.while_active", "off") == "cpu":
            return "Game mode: GPU models unloaded (dictation on CPU)"
        return "Game mode: GPU models unloaded"

    def _load_game_cpu_model(self) -> None:
        if self.cfg.get("game_mode.while_active", "off") != "cpu":
            return
        try:
            self.stt.load(strict=True, device="cpu", compute="int8",
                          name=self.cfg.get("stt.cpu_fallback_model", "small.en"), cpu_threads=4)
        except Exception:
            log.exception("game mode: CPU speech model failed to load - dictation stays off")

    def _game_enter(self) -> None:
        with self._gpu_lock:
            t0 = time.perf_counter()
            self.game_active = True
            self._game_ui = True
            self.outmute.release(wait=0)
            self.tts.shutdown("game mode")      # frees its GPU memory (kills the server process)
            if self._recording:
                self.hotkey.force_idle()
                self._ctl.put(("cancel", "game mode"))
            self.cleaner.suspend()
            self.stt.unload()
            self._sync_hotkey()         # hook off right away in "off" mode (CPU load below can take seconds)
            self._load_game_cpu_model()
            self._sync_hotkey()
            self._set_status(self._game_text(), "game")
            log.info("game mode ON: GPU models unloaded in %.1fs", time.perf_counter() - t0)

    def _game_back(self) -> bool:
        """The game came back while we were reloading: unload again instead of loading under it."""
        if not self.watcher.game_running() or not self.cfg.get("game_mode.enabled", True):
            return False
        log.info("game mode: game relaunched during reload - staying in game mode")
        self.watcher.active = True
        self._game_enter()
        return True

    def _game_exit(self) -> None:
        with self._gpu_lock:
            if not self.game_active:
                return
            t0 = time.perf_counter()
            self._game_ui = False
            self._set_status("Reloading models after game...", "busy")
            loaded = False
            delays = (5, 10, 20, 40)
            for i in range(len(delays) + 1):
                if self._game_back():
                    return
                try:
                    self.stt.load(strict=True)
                    loaded = True
                    break
                except Exception as e:
                    log.warning("game mode: GPU speech load failed (%s: %s) attempt %d/%d", type(e).__name__, e,
                                i + 1, len(delays) + 1)
                    if i < len(delays) and self._shutdown.wait(delays[i]):
                        return
            if self._game_back():
                return
            try:
                if not loaded:
                    log.error("game mode: GPU speech model never loaded - falling back to CPU")
                    self.stt.load()
                try:
                    from faster_whisper.vad import get_vad_model
                    get_vad_model()
                except Exception:
                    log.exception("VAD preload failed")
                if self._ensure_cleanup_model():
                    self.cleaner.server.suspended = False   # model still downloading: its completion starts it
                else:
                    self.cleaner.resume()
            except Exception:
                log.exception("game mode: reload after game failed")
            finally:
                self.game_active = False        # never leave the hotkey dead, whatever failed above
                self._sync_hotkey()
            self.tts.start()                    # spoken answers come back last (after speech + cleanup models)
            if not self.stt.ready:
                self._set_status("Speech model failed to load - see log", "error")
                return
            gpu = "GPU" if self.stt.device == "cuda" else "CPU (fallback)"
            self._set_status(f"Ready - {self.stt.model_name} on {gpu} - mic: {self.audio.device_name}",
                             "ready" if self.stt.device == "cuda" else "error")
            if not self.watcher.game_running():     # disabled by hand while the game still runs: no toast over it
                self._ov("info", "Game closed - models reloaded")
            log.info("game mode OFF: models reloaded in %.1fs (stt=%s)", time.perf_counter() - t0, self.stt.device)

    def _set_status(self, text: str, icon: str) -> None:
        if self._game_ui and icon != "paused":
            text, icon = self._game_text(), "game"
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
        # model_reasoning restarts llama-server only when the budget of the configured model changes (editing the
        # budget while reasoning is off, or another model's entry, needs no reload)
        ch.reasoning_launch_changed = _reasoning_budget_in(old) != _reasoning_budget_in(new)
        threading.Thread(target=self._apply_config, args=(ch,), daemon=True, name="cfg-apply").start()

    def _apply_config(self, ch) -> None:
        try:
            if ch("hotkey"):
                self.hotkey.set_hotkey(self.cfg.get("hotkey"))
            if ch("command_mode", "enabled") or ch("command_mode", "hotkey"):
                self.hotkey.set_command_hotkey(self._command_spec())
            if ch("autocomplete") and getattr(self, "autocomplete", None) is not None:
                self.autocomplete.apply_config()
            if ch("paste_last"):
                self.hotkey.set_paste_hotkey(self._paste_spec())
            if ch("history", "max_entries") or ch("history", "retention_days"):
                self.history.set_limits(int(self.cfg.get("history.max_entries", 5000)),
                                        float(self.cfg.get("history.retention_days", 0) or 0))
            if ch("handsfree_key"):
                self.hotkey.set_handsfree_key(self.cfg.get("handsfree_key"))
            if ch("enabled"):
                en = bool(self.cfg.get("enabled"))
                self._sync_hotkey()
                self._set_status("Ready" if en else "Paused", "ready" if en else "paused")
                self._ov("info", "Resumed" if en else "Paused")
                if not en:
                    self.tts.shutdown("paused")
            if ch("enabled") or any(ch("command_mode", k) for k in ("speak_answers", "enabled", "answer_questions")):
                if self.tts.enabled() and self._ready and not self.game_active and self.cfg.get("enabled", True):
                    self.tts.start()
                elif not self.tts.enabled():
                    self.tts.shutdown("spoken answers off")
            if ch("mic") or ch("audio", "keep_open"):
                self.audio.reopen()
                self._set_status(f"Ready - mic: {self.audio.device_name}", "ready")
            if ch("stt", "model") or ch("stt", "device") or ch("stt", "compute_type"):
                with self._gpu_lock:
                    if self.game_active:
                        # deferred: _game_exit loads from the then-current config
                        log.info("stt config changed during game mode - applied when the game closes")
                    else:
                        self.stt.ready = False
                        self._set_status("Reloading speech model...", "busy")
                        self.stt.load()
                        self._set_status(f"Ready - {self.stt.model_name} ({self.stt.device})", "ready")
            if ch("game_mode", "while_active"):
                with self._gpu_lock:
                    if self.game_active:
                        if self.cfg.get("game_mode.while_active", "off") == "cpu":
                            self._load_game_cpu_model()
                        else:
                            self.stt.unload()
                            self._sync_hotkey()
                        self._sync_hotkey()
                        self._set_status(self._game_text(), "game")
            if any(ch("cleanup", k) for k in ("model_path", "model_args", "server_exe", "gpu_layers", "ctx")) or \
                    (ch("cleanup", "model_reasoning") and getattr(ch, "reasoning_launch_changed", True)):
                self._apply_cleanup_model()
            if ch("cleanup", "enabled") and self.cfg.get("cleanup.enabled"):
                if not self._ensure_cleanup_model():
                    self.cleaner.start()
            if ch("ui", "sound_volume"):
                sounds.ensure_sounds(float(self.cfg.get("ui.sound_volume", 0.25)))
        except Exception:
            log.exception("applying config failed")

    def _apply_cleanup_model(self, background: bool = False) -> None:
        """Cleanup model (or its launch settings) changed: restart llama-server with it. Dictations meanwhile
        fall back to the basic regex cleanup. A failed load stays failed (no silent switch back).
        A downloadable model (modeldl.REGISTRY) that is not in models/ yet is downloaded first; a server running
        the previous model keeps serving until the download finishes and this runs again.
        ``background``: not started by the user (a download finished) - toasts wait for a running dictation."""
        say = self.bridge.toast.emit if background else self._ov
        name = model_display_name(self.cfg.get("cleanup.model_path", ""))
        with self._gpu_lock:
            if self._shutdown.is_set():     # quitting: never (re)start llama-server after cleaner.shutdown()
                return
            if self.game_active:
                log.info("cleanup model changed during game mode - %s loads when the game closes", name)
                return
            if not self.cfg.get("cleanup.enabled", True):
                self.cleaner.restart()      # just stops a server running the old model
                self._set_status(f"Cleanup model set to {name} (AI cleanup is off)", "ready")
                return
            if self._ensure_cleanup_model():
                return
            log.info("switching cleanup model to %s", name)
            self._set_status(f"Loading cleanup model {name}...", "busy")
            say("info", f"Loading {name}...")
            t0 = time.perf_counter()
            ok = self.cleaner.restart()
        if ok:
            log.info("cleanup model %s ready in %.1fs", name, time.perf_counter() - t0)
            self._set_status(f"Ready - cleanup model: {name}", "ready")
            say("info", f"Cleanup model: {name}")
        else:
            why = self.cleaner.server.failed or "timed out"
            log.error("cleanup model %s failed to load: %s", name, why)
            self._set_status(f"Cleanup model {name} failed to load - see log", "error")
            say("error", f"{name} failed to load")

    # ------------------------------------------------------------------ cleanup model download
    def _missing_download(self, path: str) -> Optional[modeldl.ModelInfo]:
        """Registry entry if ``path`` names a downloadable model in models/ whose file is not there yet."""
        info = modeldl.lookup(os.path.basename(str(path or "")))
        if info is None or not same_model(path, f"models/{info.name}"):
            return None
        try:
            if resolve_path(path).is_file():
                return None
        except OSError:
            pass
        return info

    def _ensure_cleanup_model(self) -> bool:
        """Never blocks. If AI cleanup is on and the configured model is a downloadable one that is missing, start
        downloading it (unless that is already running) and return True: the caller must not start llama-server
        now, the download's completion does. A failed download is retried only when this runs again (next app
        start, picking the model again, turning AI cleanup back on): no retry loop."""
        if not self.cfg.get("cleanup.enabled", True):
            return False
        info = self._missing_download(self.cfg.get("cleanup.model_path", ""))
        if info is None:
            return False
        if not self.downloads.active(info.name):
            self._dl_pct = -1
        if self.downloads.start(info.name, self._on_download_progress, self._on_download_done):
            meanwhile = "the current model is used meanwhile" if self.cleaner.server.ready else \
                "dictation works meanwhile with basic cleanup"
            log.info("cleanup model %s is missing - downloading it", info.name)
            self.bridge.toast.emit("info", f"Downloading the cleanup model ({modeldl.gb(info.size)} GB) - {meanwhile}")
            self._set_status(self._download_text(info.name, 0, info.size), "ready")
        return True

    @staticmethod
    def _download_text(name: str, done: int, total: int) -> str:
        pct = int(done * 100 / total) if total else 0
        return (f"Downloading cleanup model {model_display_name(name)} ... {pct}% "
                f"({modeldl.gb(done)} / {modeldl.gb(total)} GB)")

    def _configured(self, name: str) -> bool:
        return same_model(self.cfg.get("cleanup.model_path", ""), f"models/{name}")

    def _on_download_progress(self, name: str, done: int, total: int) -> None:
        """Download thread: tray tooltip / status (bridge.status is a queued signal), at most once per percent."""
        pct = int(done * 100 / total) if total else 0
        if pct == self._dl_pct or not self._configured(name):
            return
        self._dl_pct = pct
        self._set_status(self._download_text(name, done, total), "ready")

    def _on_download_done(self, name: str, ok: bool, err: str) -> None:
        """Download thread: load the model if it is still the configured one (same path as a model switch)."""
        if self._shutdown.is_set():
            return
        disp = model_display_name(name)
        if not ok:
            using = "keeping the current model" if self.cleaner.server.ready else "using basic cleanup"
            self.bridge.toast.emit("error", f"Cleanup model download failed - {using} (see log)")
            if self._configured(name):
                self._set_status(f"Cleanup model download failed ({err[:40]}) - retries on next start", "error")
            return
        if not self._configured(name):
            self.bridge.toast.emit("info", f"{disp} downloaded - pick it under Cleanup model")
            return
        self.bridge.toast.emit("info", f"Cleanup model downloaded - loading {disp}")
        self._apply_cleanup_model(background=True)

    # ------------------------------------------------------------------ tray
    def _build_tray(self) -> None:
        self.tray = QSystemTrayIcon(make_icon("busy"))
        self.tray.setToolTip("LocalFlow - starting")
        menu = QMenu()
        self.menu = menu
        a = menu.addAction("Open LocalFlow")
        a.triggered.connect(lambda: self.show_settings("home"))
        menu.setDefaultAction(a)
        a = menu.addAction("Settings...")
        a.triggered.connect(lambda: self.show_settings("general"))
        menu.addSeparator()
        self.act_meeting = menu.addAction("Start meeting recording")
        self.act_meeting.triggered.connect(self.toggle_meeting)
        self.act_transcribe = menu.addAction("Transcribe audio file...")
        self.act_transcribe.triggered.connect(lambda: self.transcribe_file_dialog())
        menu.addSeparator()
        self.act_paste_last = menu.addAction("Paste last transcript")
        self.act_paste_last.triggered.connect(self.paste_last)
        self.act_copy_last = menu.addAction("Copy last transcript")
        self.act_copy_last.triggered.connect(self.copy_last)
        menu.addSeparator()
        self.act_pause = menu.addAction("Pause dictation")
        self.act_pause.triggered.connect(self.toggle_pause)
        self.act_cleanup = menu.addAction("AI cleanup")
        self.act_cleanup.setCheckable(True)
        self.act_cleanup.triggered.connect(self.toggle_cleanup)
        self.act_game = menu.addAction("Game mode (auto)")
        self.act_game.setCheckable(True)
        self.act_game.triggered.connect(self.toggle_game_mode)
        self.act_autostyle = menu.addAction("Auto style by app")
        self.act_autostyle.setCheckable(True)
        self.act_autostyle.triggered.connect(self.toggle_auto_style)
        self.act_autocomplete = menu.addAction("Tab autocomplete")
        self.act_autocomplete.setCheckable(True)
        self.act_autocomplete.triggered.connect(self.toggle_autocomplete)
        self.mic_menu = menu.addMenu("Microphone")
        self.mic_menu.aboutToShow.connect(self._fill_mic_menu)
        self.model_menu = menu.addMenu("Cleanup model")
        self.model_menu.aboutToShow.connect(self._fill_model_menu)
        menu.addSeparator()
        a = menu.addAction("Check for updates...")
        a.triggered.connect(self.check_for_updates)
        a = menu.addAction("Open log")
        a.triggered.connect(lambda: os.startfile(str(LOG_PATH)))
        a = menu.addAction("Quit LocalFlow")
        a.triggered.connect(self.quit)
        menu.aboutToShow.connect(self._refresh_menu)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(lambda r: self.show_settings("home") if r == QSystemTrayIcon.DoubleClick else None)
        self.tray.show()
        self.bridge.status.connect(self._on_status)
        self.bridge.open_settings.connect(lambda: self.show_settings("general"))
        self.bridge.open_history.connect(self.show_history)
        self.bridge.toast.connect(self._notify)
        self.bridge.tr_done.connect(self._on_tr_done)
        self.bridge.meeting_state.connect(self._on_meeting_state)

    def _on_status(self, text: str, icon: str) -> None:
        self._status_text, self._status_icon = text, icon
        self._apply_tray()

    def _apply_tray(self) -> None:
        if self.tray:
            rec = "Recording meeting - " if self._meeting_on else ""
            self.tray.setToolTip(f"LocalFlow - {rec}{self._status_text}"[:127])
            self.tray.setIcon(make_icon("recording" if self._meeting_on else self._status_icon))

    def _refresh_menu(self) -> None:
        en = bool(self.cfg.get("enabled", True))
        self.act_pause.setText("Pause dictation" if en else "Resume dictation")
        self.act_cleanup.setChecked(bool(self.cfg.get("cleanup.enabled", True)))
        self.act_autostyle.setChecked(bool(self.cfg.get("auto_style", True)))
        self.act_autocomplete.setChecked(bool(self.cfg.get("autocomplete.enabled", True)))
        self.act_autocomplete.setEnabled(getattr(self, "autocomplete", None) is not None)
        self.act_game.setChecked(bool(self.cfg.get("game_mode.enabled", True)))
        has_last = bool(self.last_text)
        self.act_paste_last.setEnabled(has_last)
        self.act_copy_last.setEnabled(has_last)
        have = self.transcripts is not None
        self.act_meeting.setText("Stop meeting recording" if self.meeting_recording else "Start meeting recording")
        self.act_meeting.setEnabled(have)
        self.act_transcribe.setEnabled(have)

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

    def _fill_model_menu(self) -> None:
        """Rebuilt each time it opens, so the check mark follows the config and new .gguf files show up."""
        self.model_menu.clear()
        grp = QActionGroup(self.model_menu)
        grp.setExclusive(True)
        cur = self.cfg.get("cleanup.model_path", "")
        files = list_model_choices()        # files in models/ + downloadable models not fetched yet
        found = False
        for f, label in files:
            prog = self.downloads.progress(f)
            if prog:
                label = f"{model_display_name(f)} (downloading {int(prog[0] * 100 / max(1, prog[1]))}%)"
            act = self.model_menu.addAction(label)
            act.setCheckable(True)
            on = same_model(cur, f"models/{f}")
            found = found or on
            act.setChecked(on)
            grp.addAction(act)
            act.triggered.connect(lambda _=False, n=f: self.select_cleanup_model(n))
        if not files:
            self.model_menu.addAction("No .gguf models in the models folder").setEnabled(False)
        if cur and not found:
            self.model_menu.addAction(f"In use: {model_display_name(cur)}"[:70]).setEnabled(False)
        self.model_menu.addSeparator()
        a = self.model_menu.addAction("Open models folder")
        a.triggered.connect(lambda: os.startfile(str(MODELS_DIR)))

    def select_cleanup_model(self, filename: str) -> None:
        path = f"models/{filename}"
        if same_model(self.cfg.get("cleanup.model_path", ""), path):
            self._ensure_cleanup_model()    # picked again after a failed download: try once more
            return
        self.cfg.set("cleanup.model_path", path)    # saved + _on_config_change restarts the server
        if self.settings_win is not None:
            try:
                self.settings_win.refresh_cleanup_model()
            except Exception:
                log.exception("settings window refresh failed")

    def toggle_pause(self) -> None:
        self.cfg.set("enabled", not self.cfg.get("enabled", True))

    def toggle_cleanup(self) -> None:
        on = not self.cfg.get("cleanup.enabled", True)
        self.cfg.set("cleanup.enabled", on)
        self._ov("info", "AI cleanup on" if on else "AI cleanup off")

    def toggle_game_mode(self) -> None:
        self.cfg.set("game_mode.enabled", not self.cfg.get("game_mode.enabled", True))

    def toggle_autocomplete(self) -> None:
        on = not self.cfg.get("autocomplete.enabled", True)
        self.cfg.set("autocomplete.enabled", on)        # _on_config_change starts / stops it
        self._ov("info", "Tab autocomplete on" if on else "Tab autocomplete off")

    def toggle_auto_style(self) -> None:
        on = not self.cfg.get("auto_style", True)
        self.cfg.set("auto_style", on)
        self._ov("info", "Auto style on" if on else "Auto style off (neutral)")

    def show_settings(self, page: str = "general") -> None:
        """Open the LocalFlow window on a page ("home", "transcriptions", "general", ...)."""
        if self.settings_win is None:
            from .settings_ui import SettingsWindow
            self.settings_win = SettingsWindow(self.cfg, input_device_names, autostart.is_enabled, autostart.set_enabled,
                                               get_last_ctx=lambda: self._ctx, get_default_mic=default_input_name,
                                               app=self)
        if page:
            self.settings_win.show_page(page)     # switching pages never reloads (unsaved edits survive)
        if self.settings_win.isMinimized():
            self.settings_win.showNormal()
        self.settings_win.show()
        self.settings_win.raise_()
        self.settings_win.activateWindow()

    def check_for_updates(self) -> None:
        """Tray / About page: look for a new release now and always report the result."""
        if self.updates is not None:
            self.updates.check(manual=True)

    def update_blocker(self) -> str:
        """What an update install must wait for ("" = nothing): quitting would cut it off."""
        if self.meeting_recording:
            return "the meeting recording"
        if self._recording or self.gate.busy():
            return "the current dictation"
        return ""

    def show_history(self) -> None:
        """History now lives on the Home page of the LocalFlow window."""
        self.show_settings("home")

    def quit(self) -> None:
        log.info("quitting")
        self._shutdown.set()
        self.downloads.cancel_all()     # a model download keeps its .part and resumes on the next start
        self.outmute.shutdown()         # speakers back on even if we quit mid-dictation
        try:
            self.tts.shutdown("quit")
        except Exception:
            log.exception("tts shutdown failed")
        try:
            self.watcher.stop()
            if getattr(self, "autocomplete", None) is not None:
                self.autocomplete.stop()
            self.hotkey.stop()
            self.audio.stop()
            if self.transcripts is not None:
                try:        # saves a running meeting recording, interrupts the job (resumes next start)
                    self.transcripts.shutdown()
                except Exception:
                    log.exception("transcripts shutdown failed")
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
        ft = QTimer()
        ft.setInterval(400)
        ft.timeout.connect(self._track_foreground)
        ft.start()
        self._fg_timer = ft
        self.outmute.recover()          # unmute speakers a crashed / killed run left muted
        threading.Thread(target=self._controller, name="controller", daemon=True).start()
        threading.Thread(target=self._worker, name="worker", daemon=True).start()
        ok = self.hotkey.start()
        if not ok:
            log.error("keyboard hook failed to install")
        if self.autocomplete is not None and self.autocomplete.enabled():
            self.autocomplete.start()
        try:    # first poll BEFORE boot: a game that is already running keeps the GPU models from loading at all
            self.game_active = self.watcher.detect_initial()
        except Exception:
            log.exception("game mode initial poll failed")
        threading.Thread(target=self._boot, name="boot", daemon=True).start()
        if self.transcripts is not None:    # off the UI thread: may convert leftover PCM recordings to FLAC
            threading.Thread(target=self._recover_transcripts, name="transcripts-recover", daemon=True).start()
        self.watcher.start()
        try:
            from .update_ui import UpdateController
            self.updates = UpdateController(self)
            self.updates.start()
        except Exception:
            log.exception("update checks unavailable")
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
    # the real entry point is the only place that may mute real speakers (never under LOCALFLOW_TEST_WAV)
    # spoken answers: off under the test-WAV hook unless the test asks for WAV output (LOCALFLOW_TTS_WAV_OUT)
    app = LocalFlowApp(qapp, real_output_mute=not os.environ.get("LOCALFLOW_TEST_WAV"),
                       real_tts=not os.environ.get("LOCALFLOW_TEST_WAV") or bool(os.environ.get("LOCALFLOW_TTS_WAV_OUT")))
    return app.run()
