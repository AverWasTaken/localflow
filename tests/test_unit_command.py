"""Offline tests for command mode, paste-last, the two-combo hotkey state machine, foreground-gate pairing and
history stats/retention. No hook, microphone, GPU, LLM server or real clipboard is used.

   .venv\\Scripts\\python.exe tests\\test_unit_command.py
"""
import datetime as dt
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from localflow import app as app_mod  # noqa: E402
from localflow import cleanup, hotkey as hk  # noqa: E402
from localflow import winutil as w  # noqa: E402
from localflow.config import Config  # noqa: E402
from localflow.gate import ForegroundGate  # noqa: E402
from localflow.history import History  # noqa: E402
from tests import test_unit_app as ta  # noqa: E402

LCTRL, LALT, LSHIFT, LWIN = w.VK_LCONTROL, w.VK_LMENU, w.VK_LSHIFT, w.VK_LWIN
Z, D, SPACE, ESC = 0x5A, 0x44, w.VK_SPACE, w.VK_ESCAPE


# ============================================================================ hotkey state machine
class HK:
    def __init__(self, dictate="ctrl+alt", command="ctrl+win", paste="alt+shift+z"):
        self.ev = []
        self.masks = 0
        self.m = hk.HotkeyManager(dictate, "space", on_start=lambda mode: self.ev.append(("start", mode)),
                                  on_stop=lambda: self.ev.append(("stop",)),
                                  on_cancel=lambda why: self.ev.append(("cancel", why)),
                                  on_handsfree=lambda: self.ev.append(("handsfree",)),
                                  command_hotkey=command, paste_hotkey=paste,
                                  on_paste=lambda: self.ev.append(("paste",)))
        self.m._inject_mask = self._mask

    def _mask(self):
        self.masks += 1

    def down(self, vk):
        return self.m._handle(vk, True)

    def up(self, vk):
        return self.m._handle(vk, False)


class HotkeyModes(unittest.TestCase):
    def test_dictate_and_command_selected_by_exact_combo(self):
        h = HK()
        h.down(LCTRL)
        h.down(LALT)
        self.assertEqual(h.ev, [("start", "dictate")])
        self.assertEqual(h.m.active_mode, "dictate")
        h.up(LALT)
        h.up(LCTRL)
        self.assertEqual(h.ev[-1], ("stop",))
        self.assertEqual(h.m.state, "idle")
        h.ev.clear()
        h.down(LWIN)
        h.down(LCTRL)
        self.assertEqual(h.ev, [("start", "command")])
        self.assertEqual(h.m.active_mode, "command")
        masks = h.masks
        h.up(LWIN)                                  # Win released first: menu mask keeps the Start menu shut
        self.assertEqual(h.ev[-1], ("stop",))
        self.assertEqual(h.masks, masks + 1)
        h.up(LCTRL)
        self.assertEqual(h.m.state, "idle")

    def test_superset_does_not_start_a_second_mode(self):
        h = HK()
        h.down(LCTRL)
        h.down(LALT)
        h.down(LWIN)                # ctrl+alt+win: still the dictation that started first
        self.assertEqual(h.ev, [("start", "dictate")])
        for vk in (LWIN, LALT, LCTRL):
            h.up(vk)
        self.assertEqual(h.ev.count(("stop",)), 1)

    def test_subset_or_equal_command_combo_rejected(self):
        for cmd in ("ctrl", "ctrl+alt", "alt+ctrl+shift", "ctrl+x", "nonsense"):
            h = HK(command=cmd)
            self.assertIsNone(h.m.command, cmd)
            self.assertEqual(list(h.m.combos()), ["dictate"])
            h.down(LCTRL)
            h.up(LCTRL)
            self.assertEqual(h.ev, [], cmd)
        self.assertIsNotNone(HK(command="shift+win").m.command)

    def test_dictation_hotkey_change_revalidates_command(self):
        h = HK()
        h.m.set_hotkey("ctrl+win")                  # now equal to the command combo -> command off
        self.assertIsNone(h.m.command)
        h.m.set_hotkey("ctrl+alt")
        self.assertEqual(h.m.command, frozenset({"ctrl", "win"}))
        h.m.set_command_hotkey(None)
        self.assertIsNone(h.m.command)
        h.m.set_command_hotkey("shift+win")
        self.assertEqual(h.m.command, frozenset({"shift", "win"}))

    def test_esc_handsfree_and_shortcut_in_command_mode(self):
        h = HK()
        h.down(LCTRL)
        h.down(LWIN)
        self.assertTrue(h.down(ESC))                # swallowed
        self.assertTrue(h.up(ESC))
        self.assertEqual(h.ev[-1], ("cancel", "esc"))
        h.up(LWIN)
        h.up(LCTRL)
        self.assertEqual(h.m.state, "idle")
        h.ev.clear()
        # hands-free lock in command mode, combo again stops
        h.down(LCTRL)
        h.down(LWIN)
        self.assertTrue(h.down(SPACE))
        h.up(SPACE)
        h.up(LWIN)
        h.up(LCTRL)
        self.assertEqual(h.m.state, "locked")
        h.down(LCTRL)
        h.down(LWIN)
        self.assertEqual(h.ev, [("start", "command"), ("handsfree",), ("stop",)])
        h.up(LWIN)
        h.up(LCTRL)
        self.assertEqual(h.m.state, "idle")
        h.ev.clear()
        # Ctrl+Win+D (new virtual desktop) passes through and cancels silently
        h.down(LCTRL)
        h.down(LWIN)
        self.assertFalse(h.down(D))
        self.assertEqual(h.ev[-1], ("cancel", "shortcut"))
        h.up(D)
        h.up(LWIN)
        h.up(LCTRL)
        self.assertEqual(h.m.state, "idle")

    def test_paste_chord(self):
        h = HK()
        h.down(LALT)
        h.down(LSHIFT)
        self.assertTrue(h.down(Z))                  # swallowed
        self.assertTrue(h.down(Z))                  # auto-repeat swallowed, no second paste
        self.assertTrue(h.up(Z))
        self.assertEqual(h.ev, [("paste",)])
        self.assertEqual(h.masks, 1)                # mask right away: no menu bar, no layout switch
        h.up(LSHIFT)
        h.up(LALT)
        self.assertEqual(h.m.state, "idle")
        self.assertFalse(h.down(Z))                 # plain z is untouched
        h.up(Z)

    def test_paste_chord_needs_exact_modifiers_and_idle(self):
        h = HK()
        for vk in (LCTRL, LALT, LSHIFT):
            h.down(vk)
        self.assertFalse(h.down(Z))                 # ctrl+alt+shift+z: dictation hold -> real shortcut
        self.assertNotIn(("paste",), h.ev)
        for vk in (Z, LSHIFT, LALT, LCTRL):
            h.up(vk)
        h2 = HK(paste="")
        h2.down(LALT)
        h2.down(LSHIFT)
        self.assertFalse(h2.down(Z))
        self.assertEqual(h2.ev, [])

    def test_paste_chord_collisions_rejected(self):
        self.assertIsNone(HK(dictate="alt").m.paste)              # alt combo would start while building the chord
        self.assertIsNone(HK(command="alt+shift").m.paste)
        self.assertIsNone(HK(paste="alt+shift").m.paste)          # no key
        self.assertIsNone(HK(paste="z").m.paste)                  # no modifier
        self.assertEqual(HK(paste="ctrl+shift+f9").m.paste, (frozenset({"ctrl", "shift"}), 0x78))


# ============================================================================ app: gate pairing + command flow
class CountingGate(ForegroundGate):
    def __init__(self):
        super().__init__()
        self.begins = self.ends = 0

    def begin(self):
        self.begins += 1
        super().begin()

    def end(self):
        self.ends += 1
        super().end()


class AppBase(unittest.TestCase):
    def setUp(self):
        ta.AppWiring.setUp(self)
        a = self.app
        a.gate = CountingGate()
        a.hotkey = mock.Mock()
        a.stt = SimpleNamespace(ready=True, device="cuda", model=object(),
                                transcribe=mock.Mock(return_value=SimpleNamespace(text="make it formal", ms=5,
                                                                                  dropped="")))
        a.cleaner = mock.Mock()
        a.cleaner.command.return_value = ("Dear team, hello.", {"llm": True, "ms": 7, "reason": ""})
        self.inserted = []
        self.patches = [
            mock.patch.object(app_mod, "insert_text", side_effect=self._insert),
            mock.patch.object(app_mod, "capture_selection", return_value=("text", "hey team hi")),
            mock.patch.object(app_mod, "wait_modifiers_released", return_value=True),
            mock.patch.object(app_mod, "foreground_hwnd", return_value=111),
            mock.patch.object(app_mod.winutil, "window_info", return_value=("notepad.exe", "doc - Notepad")),
        ]
        self.mocks = [p.start() for p in self.patches]
        self.capture = self.mocks[1]
        self.fg = self.mocks[3]

    def tearDown(self):
        for p in self.patches:
            p.stop()
        ta.AppWiring.tearDown(self)

    def _insert(self, text, **kw):
        self.inserted.append((text, kw))
        return {"ok": True, "reason": ""}

    def run_jobs(self):
        while not self.app._jobs.empty():
            job = self.app._jobs.get_nowait()
            getattr(self.app, {"dictate": "_run_dictation", "command": "_run_command",
                               "paste": "_run_paste"}[job[0]])(*job[1:])

    def speak(self, mode="command", hold=0.4):
        self.app._do_start(mode)
        time.sleep(hold)
        self.app._do_stop()
        self.run_jobs()

    def assertGateIdle(self):
        g = self.app.gate
        self.assertEqual(g.begins, g.ends)
        self.assertFalse(g.busy())


class GatePairing(AppBase):
    def test_dictation_ok_discard_error_and_exception(self):
        a = self.app
        results = [{"status": "ok", "text": "hello there", "reason": ""},
                   {"status": "discard", "text": "", "reason": "silence"},
                   {"status": "insert-error", "text": "x", "reason": ""}]
        for r in results:
            a.pipeline = SimpleNamespace(process=lambda *x, _r=r, **k: dict(_r))
            self.speak("dictate")
            self.assertGateIdle()
        a.pipeline = SimpleNamespace(process=mock.Mock(side_effect=RuntimeError("boom")))
        self.speak("dictate")
        self.assertGateIdle()
        self.assertEqual(a.gate.begins, 4)
        self.assertEqual(a.last_text, "hello there")         # discards / errors don't overwrite last_text
        self.assertEqual(a.last_result["status"], "insert-error")

    def test_cancel_short_press_game_mode_and_stt_not_ready(self):
        a = self.app
        a._do_start("command")
        self.assertTrue(a.gate.busy())                         # busy from key-down
        a._do_cancel("esc")
        self.assertGateIdle()
        a._do_start("dictate")
        a._do_stop()                                           # < min duration
        self.assertGateIdle()
        a.game_active = True
        a.stt.model = None
        a._do_start("dictate")
        self.assertFalse(a._recording)
        self.assertGateIdle()
        a.game_active = False
        a.stt.model = object()
        a.stt.ready = False
        a._wait_stt = lambda: False
        self.speak("dictate")
        self.assertGateIdle()
        self.speak("command")
        self.assertGateIdle()

    def test_gate_stays_busy_until_command_inserted(self):
        a = self.app
        seen = []
        a.cleaner.command.side_effect = lambda *x: (seen.append(a.gate.busy()), ("ok text", {"ms": 1}))[1]
        self.speak("command")
        self.assertEqual(seen, [True])
        self.assertGateIdle()

    def test_extra_end_calls_are_noops(self):
        a = self.app
        a._do_start("dictate")
        rid = a._rec_id
        a._fg_end(rid)
        a._fg_end(rid)
        a._do_cancel("esc")
        self.assertEqual((a.gate.begins, a.gate.ends), (1, 1))


class CommandFlow(AppBase):
    def test_selection_rewritten_and_inserted(self):
        a = self.app
        self.speak("command")
        a.cleaner.command.assert_called_once_with("make it formal", "hey team hi", "notepad.exe")
        self.assertEqual(len(self.inserted), 1)
        text, kw = self.inserted[0]
        self.assertEqual(text, "Dear team, hello.")
        self.assertTrue(kw["require_release"])
        self.assertEqual(kw["target_hwnd"], 111)                 # re-checked right before the keystroke
        self.assertEqual(self.capture.call_args.kwargs["target_hwnd"], 111)
        self.assertEqual(a.last_text, "Dear team, hello.")
        h = a.history.list()[0]
        self.assertEqual(h["style"], "command")
        self.assertIn("make it formal", h["raw"])
        self.assertIn("on selection", h["raw"])
        self.assertEqual(h["cleaned"], "Dear team, hello.")
        self.assertEqual(a.history.stats()["commands"], 1)
        self.assertIn(("processing", "Working…"), a.overlay.calls)
        self.assertGateIdle()

    def test_capture_reports_window_change(self):
        self.capture.return_value = ("window", "")
        self.speak("command")
        self.app.cleaner.command.assert_not_called()
        self.assertEqual(self.inserted, [])
        self.assertIn("window changed", self.app.history.list()[0]["cleaned"])

    def test_insert_refused_window_changed_keeps_result(self):
        self.mocks[0].side_effect = lambda text, **kw: {"ok": False, "reason": "window changed"}
        self.speak("command")
        self.assertEqual(self.app.last_text, "")
        self.assertEqual(self.app.history.list()[0]["cleaned"], "Dear team, hello.")
        self.assertIn(("error", "Window changed - result saved to history"), self.app.overlay.calls)
        self.assertGateIdle()

    def test_unexpected_error_still_recorded(self):
        self.mocks[0].side_effect = RuntimeError("SendInput blew up")
        self.speak("command")
        self.assertEqual(self.app.history.list()[0]["cleaned"], "Dear team, hello.")
        self.assertIn(("error", "Command failed"), self.app.overlay.calls)
        self.assertGateIdle()

    def test_overlay_label_is_command(self):
        self.app._do_start("command")
        time.sleep(0.3)
        self.assertIn(("recording", "Command"), self.app.overlay.calls)
        self.app._do_cancel("esc")

    def test_llm_failure_inserts_nothing(self):
        a = self.app
        a.cleaner.command.return_value = ("", {"llm": False, "ms": 0, "reason": "AI model not ready"})
        self.speak("command")
        self.assertEqual(self.inserted, [])
        self.assertEqual(a.last_text, "")
        self.assertTrue(any(s == "error" and "not ready" in t for s, t in a.overlay.calls))
        h = a.history.list()[0]
        self.assertEqual(h["style"], "command")
        self.assertIn("command failed", h["cleaned"])
        self.assertEqual(a.history.stats()["commands"], 0)
        self.assertGateIdle()

    def test_window_changed_before_insert(self):
        a = self.app
        hw = iter([111, 111, 222])          # key-down, before capture, before insert
        self.fg.side_effect = lambda: next(hw)
        self.speak("command")
        self.assertEqual(self.inserted, [])
        self.assertEqual(a.history.list()[0]["cleaned"], "Dear team, hello.")    # result kept in history
        self.assertGateIdle()

    def test_window_changed_before_capture(self):
        hw = iter([111, 222])
        self.fg.side_effect = lambda: next(hw)
        self.speak("command")
        self.capture.assert_not_called()
        self.app.cleaner.command.assert_not_called()
        self.assertEqual(self.inserted, [])

    def test_modifiers_never_released_aborts(self):
        self.mocks[2].return_value = False
        self.speak("command")
        self.capture.assert_not_called()
        self.app.cleaner.command.assert_not_called()
        self.assertEqual(self.inserted, [])

    def test_uncertain_or_busy_capture_aborts(self):
        for st in ("uncertain", "busy"):
            self.capture.return_value = (st, "")
            self.speak("command")
        self.app.cleaner.command.assert_not_called()
        self.assertEqual(self.inserted, [])
        self.assertGateIdle()

    def test_no_selection_generates_at_cursor(self):
        self.capture.return_value = ("none", "")
        self.speak("command")
        self.app.cleaner.command.assert_called_once_with("make it formal", "", "notepad.exe")
        self.assertEqual(len(self.inserted), 1)

    def test_terminal_and_editor_are_generate_only(self):
        for exe in ("WindowsTerminal.exe", "Code.exe", "pycharm64.exe"):
            self.mocks[4].return_value = (exe.lower(), "x")
            self.speak("command")
        self.capture.assert_not_called()
        self.assertEqual(len(self.inserted), 3)

    def test_silence_or_empty_instruction_does_nothing(self):
        a = self.app
        a.audio.end = lambda: np.zeros(16000, np.float32)
        self.speak("command")
        a.stt.transcribe.assert_not_called()
        a.audio.end = ta.FakeAudio.end.__get__(a.audio)
        a.stt.transcribe.return_value = SimpleNamespace(text="", ms=3, dropped="no speech (VAD)")
        self.speak("command")
        a.cleaner.command.assert_not_called()
        self.assertEqual(self.inserted, [])
        self.assertIn(("info", "No command heard"), a.overlay.calls)
        self.assertGateIdle()

    def test_insert_refused_when_keys_held(self):
        self._insert = None
        self.mocks[0].side_effect = lambda text, **kw: {"ok": False, "reason": "modifiers held"}
        self.speak("command")
        self.assertEqual(self.app.last_text, "")
        self.assertGateIdle()


class PasteLast(AppBase):
    def test_paste_last_uses_last_inserted_text(self):
        a = self.app
        a._run_paste(0)
        self.assertEqual(self.inserted, [])
        self.assertIn(("info", "Nothing to paste yet"), a.overlay.calls)
        a.last_text = "previous words"
        a._run_paste(0)
        self.assertEqual(self.inserted[-1][0], "previous words")
        self.assertTrue(self.inserted[-1][1]["require_release"])

    def test_tray_paste_refocuses_window_first(self):
        a = self.app
        a.last_text = "abc"
        with mock.patch.object(app_mod, "focus_window", return_value=False) as fw:
            a._run_paste(4242)
            fw.assert_called_once_with(4242)
        self.assertEqual(self.inserted, [])
        with mock.patch.object(app_mod, "focus_window", return_value=True):
            a._run_paste(4242)
        self.assertEqual(self.inserted[-1][0], "abc")

    def test_hotkey_callback_enqueues_paste(self):
        a = self.app
        a._last_ext_hwnd = 99
        a.paste_last()
        self.assertEqual(a._jobs.get_nowait(), ("paste", 99))


# ============================================================================ history stats + retention
class Stats(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp()) / "h.db"

    def test_stats_numbers_and_streak(self):
        h = History(path=self.path)
        today = dt.date(2026, 11, 8)
        ts = lambda d: time.mktime((today - dt.timedelta(days=d)).timetuple()) + 3600 * 12
        h.record_stats(words=120, audio_s=60, ts=ts(0))
        h.record_stats(words=30, audio_s=30, ts=ts(1))
        h.record_stats(kind="command", ts=ts(2))
        h.record_stats(words=1000, audio_s=300, ts=ts(10))       # outside the week, breaks the streak
        s = h.stats(today=today)
        self.assertEqual(s["total_words"], 1150)
        self.assertEqual(s["words_7d"], 150)
        self.assertEqual(s["dictations"], 3)
        self.assertEqual(s["commands"], 1)
        self.assertEqual(s["streak_days"], 3)
        self.assertAlmostEqual(s["avg_wpm"], round(1150 / (390 / 60), 1))
        self.assertAlmostEqual(s["time_saved_s"], 1150 / 40 * 60 - 390)
        # nothing yet today: the streak still counts up to yesterday
        self.assertEqual(h.stats(today=today + dt.timedelta(days=1))["streak_days"], 3)
        self.assertEqual(h.stats(today=today + dt.timedelta(days=2))["streak_days"], 0)

    def test_empty_stats(self):
        s = History(path=self.path).stats()
        self.assertEqual((s["total_words"], s["avg_wpm"], s["time_saved_s"], s["streak_days"]), (0, 0.0, 0.0, 0))

    def test_retention_and_max_entries(self):
        h = History(path=self.path, max_entries=3)
        for i in range(5):
            h.add(f"r{i}", f"c{i}")
        self.assertEqual([r["raw"] for r in h.list()], ["r4", "r3", "r2"])
        h._run(lambda c: c.execute("UPDATE history SET ts = ts - 10*86400 WHERE raw='r2'"))
        h.set_limits(3, 7)
        self.assertEqual([r["raw"] for r in h.list()], ["r4", "r3"])
        h2 = History(path=self.path, max_entries=100, retention_days=0)     # 0 = forever
        self.assertEqual(len(h2.list()), 2)
        # startup prune
        h2._run(lambda c: c.execute("UPDATE history SET ts = ts - 40*86400"))
        self.assertEqual(History(path=self.path, retention_days=30).list(), [])

    def test_stats_survive_history_off_and_pruning(self):
        h = History(path=self.path, max_entries=1)
        h.record_stats(words=5, audio_s=2)
        h.add("a", "b")
        h.clear()
        self.assertEqual(h.stats()["total_words"], 5)

    def test_backfill_from_existing_history(self):
        import sqlite3
        c = sqlite3.connect(str(self.path))
        c.executescript("CREATE TABLE history (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, app TEXT, "
                        "raw TEXT, cleaned TEXT, duration_s REAL, stt_ms INTEGER, cleanup_ms INTEGER, total_ms INTEGER);")
        c.execute("INSERT INTO history(ts,raw,cleaned,duration_s) VALUES(?,?,?,?)", (time.time(), "x", "one two three", 2.0))
        c.commit()
        c.close()
        s = History(path=self.path).stats()
        self.assertEqual((s["total_words"], s["dictations"]), (3, 1))
        self.assertEqual(History(path=self.path).stats()["total_words"], 3)     # backfill runs only once

    def test_pipeline_records_stats_even_with_history_disabled(self):
        from localflow.pipeline import Pipeline
        d = Path(tempfile.mkdtemp())
        cfg = Config(path=d / "c.json")
        cfg.set("history.enabled", False, save=False)
        h = History(path=self.path)
        stt = SimpleNamespace(transcribe=lambda *a, **k: SimpleNamespace(text="hello world again now", ms=1,
                                                                        dropped=""))
        cl = SimpleNamespace(clean=lambda raw, *a, **k: ("Hello world again now.", {"llm": True, "ms": 1, "reason": ""}))
        p = Pipeline(cfg, stt, cl, h, lambda *a, **k: {"ok": True})
        audio = (np.random.default_rng(1).standard_normal(16000 * 2) * 0.05).astype(np.float32)
        r = p.process(audio)
        self.assertEqual(r["status"], "ok")
        self.assertEqual(h.list(), [])
        s = h.stats()
        self.assertEqual((s["total_words"], s["dictations"]), (4, 1))
        p.process(audio, insert=False)          # not inserted: not counted
        self.assertEqual(h.stats()["dictations"], 1)


# ============================================================================ Cleaner.command (mock HTTP)
class FakeResp:
    def __init__(self, content, finish="stop"):
        self._j = {"choices": [{"message": {"content": content}, "finish_reason": finish}]}

    def raise_for_status(self):
        pass

    def json(self):
        return self._j


class CleanerCommand(unittest.TestCase):
    def make(self, model="models/LFM3-1.7B-Q8_0.gguf"):
        cfg = Config(path=Path(tempfile.mkdtemp()) / "c.json")
        cfg.set("cleanup.model_path", model, save=False)
        c = cleanup.Cleaner(cfg)
        c.server.ready = True
        c.server.loaded_model  # noqa: B018 (property exists)
        c.server.proc = None
        self.bodies = []
        return c

    def post(self, content, finish="stop"):
        def f(url, json=None, timeout=None):
            self.bodies.append((json, timeout))
            return FakeResp(content, finish)
        return f

    def test_word_stops_dropped_tag_stops_kept(self):
        c = self.make()
        with mock.patch.object(type(c.server), "loaded_model", new_callable=mock.PropertyMock,
                               return_value=None):
            c.http = SimpleNamespace(post=self.post("Note: done.\n\nActually fine."))
            out, info = c.command("rewrite", "some text here", "notepad.exe")
        body, timeout = self.bodies[0]
        self.assertEqual(out, "Note: done.\n\nActually fine.")
        self.assertTrue(info["llm"])
        self.assertTrue(body["stop"])
        self.assertTrue(all(s.startswith("<") for s in body["stop"]))
        self.assertEqual(timeout, 30.0)
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertIn("<selection>\nsome text here\n</selection>", body["messages"][1]["content"])

    def test_max_tokens_and_generate_only(self):
        c = self.make("models/Qwen3.5-4B-Q8_0.gguf")
        c.http = SimpleNamespace(post=self.post("ok"))
        c.command("shorten", "x" * 3500, "")
        self.assertEqual(self.bodies[-1][0]["max_tokens"], 2500)
        c.command("shorten", "short", "")
        self.assertEqual(self.bodies[-1][0]["max_tokens"], 256)
        c.command("write a haiku", "", "")
        self.assertEqual(self.bodies[-1][0]["max_tokens"], 768)
        self.assertNotIn("stop", self.bodies[-1][0])
        self.assertIn("<instruction>write a haiku</instruction>\n<selection></selection>",
                      self.bodies[-1][0]["messages"][1]["content"])
        out, info = c.command("shorten", "x" * 40000, "")          # does not fit the 8k context
        self.assertEqual(out, "")
        self.assertIn("too long", info["reason"])

    def test_failures_return_empty(self):
        c = self.make()
        c.http = SimpleNamespace(post=self.post("half an ans", finish="length"))
        self.assertEqual(c.command("x", "y", "")[0], "")
        c.http = SimpleNamespace(post=self.post("<think>only thinking"))
        self.assertEqual(c.command("x", "y", "")[0], "")

        def boom(*a, **k):
            raise cleanup.requests.Timeout("slow")
        c.http = SimpleNamespace(post=boom)
        out, info = c.command("x", "y", "")
        self.assertEqual((out, info["reason"]), ("", "AI request timed out"))
        c.server.suspended = True
        self.assertIn("game mode", c.command("x", "y", "")[1]["reason"])
        c.server.suspended = False
        c.cfg.set("cleanup.enabled", False, save=False)
        self.assertIn("disabled", c.command("x", "y", "")[1]["reason"])

    def test_no_selection_sentinel_and_prompt_leak(self):
        c = self.make()
        c.http = SimpleNamespace(post=self.post("NO_SELECTION"))
        self.assertEqual(c.command("make this formal", "", ""), ("", mock.ANY))
        self.assertEqual(c.command("make this formal", "", "")[1]["reason"], "nothing selected")
        c.http = SimpleNamespace(post=self.post("[APP: NOTEPAD.EXE]\n[INSTRUCTION: MAKE THIS ALL UPPERCASE]"))
        self.assertEqual(c.command("make this all uppercase", "", "")[0], "")
        c.http = SimpleNamespace(post=self.post("<instruction>x</instruction> HELLO"))
        self.assertEqual(c.command("x", "hello", "")[0], "")            # echoed instruction tag: rejected
        c.http = SimpleNamespace(post=self.post("<selection>\nHELLO\n</selection>"))
        self.assertEqual(c.command("x", "hello", "")[0], "HELLO")       # a wrapped selection is just unwrapped

    def test_not_ready_waits_then_gives_up(self):
        c = self.make()
        c.server.ready = False
        c.server.failed = None
        with mock.patch.object(cleanup.time, "monotonic", side_effect=[0.0, 0.0, 99.0, 99.0, 99.0]):
            out, info = c.command("x", "y", "")
        self.assertEqual((out, info["reason"]), ("", "AI model not ready"))

    def test_tidy_output(self):
        t = cleanup.tidy_command_output
        self.assertEqual(t("```\nhello\n```", "hi\n"), "hello\n")
        self.assertEqual(t("```\nhello\n```", "```\nhi\n```"), "```\nhello\n```")
        self.assertEqual(t('"Hello there."', "  hi there"), "  Hello there.")
        self.assertEqual(t('"Hello"', '"hi"'), '"Hello"')
        self.assertEqual(t("<selection>Done</selection>", "x"), "Done")
        self.assertEqual(t("<think>hm</think>A list:\n- a\n- b", ""), "A list:\n- a\n- b")

    def test_tidy_keeps_deliberate_changes(self):
        t = cleanup.tidy_command_output
        # asked to remove indentation / trailing blank lines: not undone (one line break kept for a whole line)
        self.assertEqual(t("x = 1\ny = 2", "    x = 1\n    y = 2\n\n\n", "remove the indentation"), "x = 1\ny = 2\n")
        # asked to quote: the quotes stay
        self.assertEqual(t('"hello world"', "hello world", "put this in quotes"), '"hello world"')
        # model kept its own indentation: not doubled
        self.assertEqual(t("    y = 1", "    x = 1", "rename x to y"), "    y = 1")
        self.assertEqual(t("y = 1", "    x = 1", "rename x to y"), "    y = 1")
        self.assertEqual(t("use New;", "    use Old;", "rename the namespace to new"), "    use New;")   # not "space"
        self.assertEqual(t("x", "    y", "convert tabs to spaces"), "x")

    def test_overrides_cannot_raise_max_tokens(self):
        c = self.make("models/Qwen3.5-4B-Q8_0.gguf")
        c.cfg.set("cleanup.model_request", {"Qwen3.5-4B-Q8_0.gguf": {"max_tokens": 20000, "top_k": 5}}, save=False)
        with mock.patch.object(type(c.server), "loaded_model", new_callable=mock.PropertyMock, return_value=None):
            c.http = SimpleNamespace(post=self.post("ok"))
            c.command("write an essay", "", "")
        body = self.bodies[-1][0]
        self.assertEqual(body["max_tokens"], 768)
        self.assertEqual(body["top_k"], 5)                      # other per-model extras still apply


# ============================================================================ inserter strict paths (mocked Win32)
class InserterStrict(unittest.TestCase):
    def setUp(self):
        from localflow import inserter
        self.ins = inserter
        self.sent = []
        self.restored = []
        self.fg = [111]
        p = [mock.patch.object(inserter, "wait_modifiers_released", return_value=True),
             mock.patch.object(inserter, "foreground_hwnd", side_effect=lambda: self.fg[0]),
             mock.patch.object(inserter, "save_clipboard_checked", return_value=[(13, b"o\x00\x00\x00")]),
             mock.patch.object(inserter, "set_clipboard_text", return_value=True),
             mock.patch.object(inserter, "_paste_keystroke", side_effect=lambda: self.sent.append("paste")),
             mock.patch.object(inserter, "_copy_keystroke", side_effect=lambda: self.sent.append("copy")),
             mock.patch.object(inserter, "type_text", side_effect=lambda t: self.sent.append("type")),
             mock.patch.object(inserter, "restore_clipboard",
                               side_effect=lambda s: self.restored.append(s) or True),
             mock.patch.object(inserter, "_open_clipboard", return_value=True),
             mock.patch.object(inserter.u32, "EmptyClipboard", create=True),
             mock.patch.object(inserter.u32, "CloseClipboard", create=True)]
        self.m = [x.start() for x in p]
        self.addCleanup(lambda: [x.stop() for x in p])

    def test_insert_refuses_when_target_lost_focus(self):
        self.fg[0] = 222
        r = self.ins.insert_text("hi", restore_delay_ms=0, require_release=True, target_hwnd=111)
        self.assertEqual((r["ok"], r["reason"]), (False, "window changed"))
        self.assertEqual(self.sent, [])
        r = self.ins.insert_text("hi", mode="type", require_release=True, target_hwnd=111)
        self.assertEqual(self.sent, [])
        self.fg[0] = 111
        r = self.ins.insert_text("hi", restore_delay_ms=0, require_release=True, target_hwnd=111)
        self.assertTrue(r["ok"])
        self.assertEqual(self.sent, ["paste"])

    def test_focus_lost_after_clipboard_set_restores_original(self):
        seq = iter([111, 222])            # ok before set_clipboard_text, changed right before the keystroke
        self.m[1].side_effect = lambda: next(seq)
        r = self.ins.insert_text("hi", restore_delay_ms=0, require_release=True, target_hwnd=111)
        self.assertFalse(r["ok"])
        self.assertEqual(self.sent, [])
        self.assertEqual(self.restored, [[(13, b"o\x00\x00\x00")]])

    def test_clipboard_snapshot_failure(self):
        self.m[2].return_value = None
        r = self.ins.insert_text("hi", restore_delay_ms=0, require_release=True)
        self.assertEqual((r["ok"], r["reason"]), (False, "clipboard busy"))
        self.assertEqual(self.sent, [])
        r = self.ins.insert_text("hi", restore_delay_ms=0)       # dictation: pastes, but never empties the clipboard
        self.assertTrue(r["ok"])
        self.assertEqual(self.sent, ["paste"])
        self.assertEqual(self.restored, [])

    def test_capture_window_changed_restores_and_skips_copy(self):
        seq = iter([111, 222])            # ok before marking, changed before Ctrl+C
        self.m[1].side_effect = lambda: next(seq)
        self.assertEqual(self.ins.capture_selection(50, target_hwnd=111), ("window", ""))
        self.assertEqual(self.sent, [])
        self.assertEqual(len(self.restored), 1)

    def test_capture_restore_failure_aborts(self):
        self.m[7].side_effect = lambda s: False
        with mock.patch.object(self.ins.u32, "GetClipboardSequenceNumber", side_effect=[1, 2, 2]), \
                mock.patch.object(self.ins, "read_clipboard_text", return_value="selected"):
            self.assertEqual(self.ins.capture_selection(200, target_hwnd=111), ("busy", ""))
        self.assertEqual(self.sent, ["copy"])

    def test_post_paste_restore_retries_while_busy(self):
        results = iter([False, False, True])
        self.m[7].side_effect = lambda s: self.restored.append(s) or next(results)
        with mock.patch.object(self.ins.u32, "GetClipboardSequenceNumber", return_value=5):
            r = self.ins.insert_text("hi", restore_delay_ms=0, require_release=True, target_hwnd=111)
        self.assertTrue(r["ok"])
        self.assertEqual(len(self.restored), 3)

    def test_post_paste_restore_skipped_when_user_copied(self):
        with mock.patch.object(self.ins.u32, "GetClipboardSequenceNumber", side_effect=[5, 9]):
            self.ins.insert_text("hi", restore_delay_ms=0, require_release=True, target_hwnd=111)
        self.assertEqual(self.restored, [])

    def test_capture_text(self):
        with mock.patch.object(self.ins.u32, "GetClipboardSequenceNumber", side_effect=[1, 2, 2]), \
                mock.patch.object(self.ins, "read_clipboard_text", return_value="selected"):
            self.assertEqual(self.ins.capture_selection(200, target_hwnd=111), ("text", "selected"))
        self.assertEqual(len(self.restored), 1)


if __name__ == "__main__":
    unittest.main()
