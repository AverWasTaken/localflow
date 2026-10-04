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


class FakeHotkey:
    def __init__(self, *a, **k):
        pass

    def set_enabled(self, *a):
        pass


class AppWiring(unittest.TestCase):
    def setUp(self):
        d = Path(tempfile.mkdtemp())
        self._saved = {k: getattr(app_mod, k) for k in ("Config", "History", "Overlay", "AudioInput", "Transcriber",
                                                         "Cleaner", "HotkeyManager", "sounds")}
        app_mod.sounds = SimpleNamespace(play=lambda *a, **k: None, ensure_sounds=lambda *a, **k: None)   # silence!
        app_mod.Config = lambda: Config(path=d / "cfg.json")
        app_mod.History = lambda max_entries=5000: History(path=d / "h.db", max_entries=max_entries)
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
        audio, ctx, sty, session, t_rel = self.app._jobs.get_nowait()
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


if __name__ == "__main__":
    unittest.main()
