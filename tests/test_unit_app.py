"""Offline wiring test for LocalFlowApp: the style/window is captured when recording STARTS, shown on the overlay,
and handed to the pipeline job. Everything hardware-ish is faked (no microphone, no hook, no real window, no GPU).
"""
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from localflow import app as app_mod  # noqa: E402
from localflow.config import Config  # noqa: E402
from localflow.history import History  # noqa: E402

SR = 16000


class FakeOverlay:
    def __init__(self):
        self.calls = []

    def set_state(self, state, text=""):
        self.calls.append((state, text))

    def push_level(self, lvl):
        pass


class FakeAudio:
    sample_rate = SR
    device_name = "fake"
    max_samples_hit = False

    def __init__(self, cfg, on_level=None):
        self.rec = None

    def begin(self):
        self.rec = [np.zeros(1600, np.float32)]

    def pull_chunks(self, since):
        return (None, since) if self.rec is None else (self.rec[since:], len(self.rec))

    def end(self):
        self.rec = None
        return (np.random.default_rng(1).standard_normal(SR * 2) * 0.05).astype(np.float32)

    def cancel(self):
        self.rec = None

    def start(self):
        return True

    def stop(self):
        pass


class FakeSTT:
    ready = True
    device = "cuda"

    def __init__(self, cfg):
        pass


class FakeCleaner:
    def __init__(self, cfg):
        pass

    def _request_overrides(self):
        return {}


class FakeHotkey:
    def __init__(self, *a, **k):
        pass

    def set_enabled(self, *a):
        pass


class FakeService:
    """Stands in for jobs.TranscriptService (no database, no recorder, no job thread)."""
    instances = []

    def __init__(self, cfg, stt, cleaner, gate, is_paused=None, **kw):
        self.cfg, self.stt, self.gate, self.is_paused, self.kw = cfg, stt, gate, is_paused, kw
        self.recorder = SimpleNamespace(recording=False)
        self.rows = {}
        self.shut = False
        FakeService.instances.append(self)

    def get(self, tid):
        return self.rows.get(tid)

    def shutdown(self):
        self.shut = True


class AppWiring(unittest.TestCase):
    def setUp(self):
        d = Path(tempfile.mkdtemp())
        self._saved = {k: getattr(app_mod, k) for k in ("Config", "History", "Overlay", "AudioInput", "Transcriber",
                                                         "Cleaner", "HotkeyManager", "sounds", "TranscriptService")}
        app_mod.TranscriptService = FakeService
        app_mod.sounds = SimpleNamespace(play=lambda *a, **k: None, ensure_sounds=lambda *a, **k: None)   # silence!
        app_mod.Config = lambda: Config(path=d / "cfg.json")
        app_mod.History = lambda max_entries=5000, **kw: History(path=d / "h.db", max_entries=max_entries, **kw)
        app_mod.Overlay = FakeOverlay
        app_mod.AudioInput = FakeAudio
        app_mod.Transcriber = FakeSTT
        app_mod.Cleaner = FakeCleaner
        app_mod.HotkeyManager = FakeHotkey
        self._fg = app_mod.winutil.foreground_info
        self.app = app_mod.LocalFlowApp(None)

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(app_mod, k, v)
        app_mod.winutil.foreground_info = self._fg
        if self.app._session is not None:
            self.app._session.cancel()

    def test_context_captured_at_start_not_release(self):
        fg = {"w": ("discord.exe", "#general - Discord")}
        app_mod.winutil.foreground_info = lambda: fg["w"]
        self.app._do_start()
        self.assertEqual(self.app._style.style, "casual")
        self.assertIsNotNone(self.app._session)                       # streaming session started
        time.sleep(0.3)
        self.assertIn(("recording", "casual · Discord"), self.app.overlay.calls)
        fg["w"] = ("outlook.exe", "Inbox - Outlook")                  # user alt-tabs while talking
        time.sleep(0.2)
        self.app._do_stop()
        kind, rid, audio, ctx, sty, session, t_rel = self.app._jobs.get_nowait()
        self.assertEqual(kind, "dictate")
        self.assertEqual(ctx, ("discord.exe", "#general - Discord"))   # still the window dictated INTO
        self.assertEqual(sty.style, "casual")
        self.assertIsNotNone(session)
        session.cancel()
        self.assertEqual(self.app._session, None)

    def test_auto_style_off_hides_label_and_is_neutral(self):
        self.app.cfg.set("auto_style", False, save=False)
        app_mod.winutil.foreground_info = lambda: ("discord.exe", "x")
        self.app._do_start()
        self.assertEqual(self.app._style.style, "neutral")
        time.sleep(0.3)
        self.assertIn(("recording", ""), self.app.overlay.calls)
        self.app._do_cancel("esc")

    def test_cancel_and_short_press_drop_the_session(self):
        app_mod.winutil.foreground_info = lambda: ("code.exe", "x.py - Visual Studio Code")
        self.app._do_start()
        sess = self.app._session
        self.assertEqual(self.app._style.style, "code")
        self.app._do_cancel("shortcut")
        self.assertIsNone(self.app._session)
        self.assertTrue(sess._closed.is_set())
        self.app._do_start()                                           # released after < 0.3 s
        sess = self.app._session
        self.app._do_stop()
        self.assertIsNone(self.app._session)
        self.assertTrue(sess._closed.is_set())
        self.assertTrue(self.app._jobs.empty())

    def test_streaming_disabled_in_config(self):
        self.app.cfg.set("streaming.enabled", False, save=False)
        app_mod.winutil.foreground_info = lambda: ("notepad.exe", "x")
        self.app._do_start()
        self.assertIsNone(self.app._session)
        self.app._do_cancel("esc")

    def test_last_ctx_for_settings_window(self):
        app_mod.winutil.foreground_info = lambda: ("cursor.exe", "main.py")
        self.app._do_start()
        self.assertEqual(self.app._ctx, ("cursor.exe", "main.py"))
        self.app._do_cancel("esc")


class TranscriptsWiring(unittest.TestCase):
    setUp = AppWiring.setUp
    tearDown = AppWiring.tearDown

    def test_service_built_with_gate_and_signals(self):
        svc = self.app.transcripts
        self.assertIsInstance(svc, FakeService)
        self.assertIs(svc.gate, self.app.gate)
        for k in ("on_progress", "on_done", "on_meeting", "on_levels", "request_extra_fn"):
            self.assertIn(k, svc.kw)

    def test_is_paused_game_not_ready_cpu_fallback(self):
        paused = self.app.transcripts.is_paused
        stt = self.app.stt
        stt.ready, stt.device = True, "cuda"
        self.assertFalse(paused())
        self.app.game_active = True
        self.assertTrue(paused())
        self.app.game_active = False
        stt.ready = False
        self.assertTrue(paused())
        stt.ready, stt.device = True, "cpu"                               # CPU fallback of a cuda config
        self.assertTrue(paused())
        stt.device = "unloaded"
        self.assertTrue(paused())
        self.app.cfg.set("stt.device", "cpu", save=False)                 # CPU chosen on purpose: allowed
        stt.device = "cpu"
        self.assertFalse(paused())

    def test_done_and_failed_toasts(self):
        self.app.transcripts.rows[7] = {"title": "Standup", "error": ""}
        self.app._on_tr_done(7, "done")
        self.assertEqual(self.app.overlay.calls[-1], ("info", "Transcript ready: Standup"))
        self.app.transcripts.rows[8] = {"title": "Talk", "error": "AI cleanup unavailable for part of this transcript"}
        self.app._on_tr_done(8, "done")
        self.assertEqual(self.app.overlay.calls[-1], ("info", "Transcript ready: Talk (basic cleanup)"))
        self.app._on_tr_done(7, "failed")
        self.assertEqual(self.app.overlay.calls[-1], ("error", "Transcript failed: Standup"))
        n = len(self.app.overlay.calls)
        self.app._on_tr_done(7, "cancelled")                              # no toast for cancel / interrupt
        self.assertEqual(len(self.app.overlay.calls), n)

    def test_toast_waits_for_dictation_and_skips_game(self):
        timers = []
        saved = app_mod.QTimer
        app_mod.QTimer = SimpleNamespace(singleShot=lambda ms, fn: timers.append(fn))
        try:
            self.app.gate.begin()
            self.app._notify("info", "Transcript ready: A")
            self.app._notify("info", "Transcript ready: B")
            self.assertNotIn("Transcript ready", " ".join(t for _, t in self.app.overlay.calls))
            self.assertEqual(len(timers), 1)                              # one retry timer, not one per toast
            self.app.gate.end()
            timers.pop()()
            self.assertEqual(self.app.overlay.calls[-1], ("info", "Transcript ready: B (+1 more)"))
            self.app.gate.begin()                                         # an error is not hidden by a later success
            self.app._notify("error", "Recording stopped: low disk space")
            self.app._notify("info", "Transcript ready: C")
            self.app.gate.end()
            timers.pop()()
            self.assertEqual(self.app.overlay.calls[-1], ("error", "Recording stopped: low disk space (+1 more)"))
            self.app._game_ui = True
            self.app._notify("error", "Recording stopped: low disk space")
            self.assertNotEqual(self.app.overlay.calls[-1][1], "Recording stopped: low disk space")
        finally:
            app_mod.QTimer = saved

    def test_meeting_auto_stop_toast_and_tray_state(self):
        self.app._on_meeting_state({"recording": True, "stop_reason": ""})
        self.assertTrue(self.app._meeting_on)
        self.app._on_meeting_state({"recording": False, "stop_reason": "maximum recording length reached"})
        self.assertFalse(self.app._meeting_on)
        self.assertEqual(self.app.overlay.calls[-1], ("info", "Recording stopped: maximum recording length reached"))
        n = len(self.app.overlay.calls)
        self.app._on_meeting_state({"recording": True, "stop_reason": ""})
        self.app._on_meeting_state({"recording": False, "stop_reason": ""})   # stopped by the user: no toast
        self.assertEqual(len(self.app.overlay.calls), n)

    def test_styles_list_raw_first(self):
        names = self.app.transcript_styles()
        self.assertEqual(names[0], "raw")
        self.assertIn("neutral", names)


if __name__ == "__main__":
    unittest.main()
