"""Game mode: watcher debounce, llama suspend semantics, deferred STT reload. Offline; no GPU, no real processes."""
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import requests  # noqa: E402

from localflow import cleanup  # noqa: E402
from localflow.config import Config  # noqa: E402
from localflow.gamemode import GameWatcher  # noqa: E402
from localflow.stt import ModelUnloaded, Transcriber  # noqa: E402


def make_cfg():
    return Config(path=Path(tempfile.mkdtemp()) / "cfg.json")


class WatcherTests(unittest.TestCase):
    def setUp(self):
        self.cfg = make_cfg()
        self.cfg.set("game_mode.processes", ["Game.exe"], save=False)
        self.cfg.set("game_mode.exit_grace_s", 10, save=False)
        self.procs = set()
        self.t = [0.0]
        self.events = []
        self.w = GameWatcher(self.cfg, lambda: self.events.append("enter"), lambda: self.events.append("exit"),
                             lister=lambda: self.procs, clock=lambda: self.t[0])

    def poll(self, dt=3.0):
        self.t[0] += dt
        self.w.poll_once()

    def test_enter_immediately_case_insensitive(self):
        self.poll()
        self.assertEqual(self.events, [])
        self.procs = {"game.exe", "explorer.exe"}
        self.poll()
        self.assertEqual(self.events, ["enter"])
        self.poll()
        self.assertEqual(self.events, ["enter"])       # no re-fire

    def test_exit_only_after_grace(self):
        self.procs = {"game.exe"}
        self.poll()
        self.procs = set()
        for _ in range(4):                              # absent 0, 3, 6, 9 s -> still in game mode
            self.poll()
        self.assertEqual(self.events, ["enter"])
        self.poll()                                     # 12 s absent
        self.assertEqual(self.events, ["enter", "exit"])

    def test_brief_absence_does_not_exit(self):
        self.procs = {"game.exe"}
        self.poll()
        self.procs = set()
        self.poll()
        self.poll()
        self.procs = {"game.exe"}
        self.poll()
        self.procs = set()
        self.poll()                                     # grace timer restarted
        self.poll()
        self.assertEqual(self.events, ["enter"])

    def test_disable_while_active_exits_and_reenable_enters(self):
        self.procs = {"game.exe"}
        self.poll()
        self.cfg.set("game_mode.enabled", False, save=False)
        self.poll()
        self.assertEqual(self.events, ["enter", "exit"])
        self.cfg.set("game_mode.enabled", True, save=False)
        self.poll()
        self.assertEqual(self.events, ["enter", "exit", "enter"])

    def test_disabled_never_enters_and_process_list_is_live(self):
        self.cfg.set("game_mode.enabled", False, save=False)
        self.procs = {"game.exe"}
        self.poll()
        self.assertEqual(self.events, [])
        self.cfg.set("game_mode.enabled", True, save=False)
        self.cfg.set("game_mode.processes", ["other.exe"], save=False)
        self.poll()
        self.assertEqual(self.events, [])
        self.cfg.set("game_mode.processes", ["GAME.EXE"], save=False)
        self.poll()
        self.assertEqual(self.events, ["enter"])

    def test_lister_failure_keeps_state_and_callback_failure_is_contained(self):
        self.procs = {"game.exe"}
        self.poll()
        self.w.lister = mock.Mock(side_effect=OSError("boom"))
        self.poll(60)
        self.assertEqual(self.events, ["enter"])
        self.w.on_enter = mock.Mock(side_effect=RuntimeError("x"))
        self.w.active = False
        self.w.lister = lambda: {"game.exe"}
        self.poll()                                     # must not raise
        self.assertTrue(self.w.active)

    def test_detect_initial_sets_state_without_callbacks(self):
        self.procs = {"game.exe"}
        self.assertTrue(self.w.detect_initial())
        self.assertEqual(self.events, [])
        self.poll()
        self.assertEqual(self.events, [])               # already active: no enter


class LlamaSuspendTests(unittest.TestCase):
    def setUp(self):
        self.cfg = make_cfg()
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "llama-server.exe").write_bytes(b"")
        (self.tmp / "m.gguf").write_bytes(b"")
        self.cfg.set("cleanup.server_exe", str(self.tmp / "llama-server.exe"), save=False)
        self.cfg.set("cleanup.model_path", str(self.tmp / "m.gguf"), save=False)
        self.srv = cleanup.LlamaServer(self.cfg)

    def test_start_is_noop_when_suspended(self):
        self.srv.suspended = True
        with mock.patch.object(cleanup.threading, "Thread") as th:
            self.srv.start()
        th.assert_not_called()
        self.assertFalse(self.srv._starting)

    def test_clean_restart_paths_respect_suspended(self):
        c = cleanup.Cleaner(self.cfg)
        c.server.suspended = True
        c.server.failed = "old failure"
        c.server.start = mock.Mock()
        text, info = c.clean("this is a long enough sentence to reach the model", "neutral")
        self.assertEqual(info["reason"], "game mode")
        self.assertFalse(info["llm"])
        c.server.start.assert_not_called()
        # ConnectionError path: suspended flips during the request
        c.server.suspended = False
        c.server.ready = True

        def boom(*a, **k):
            c.server.suspended = True
            raise requests.ConnectionError("down")
        c._llm = boom
        text, info = c.clean("this is a long enough sentence to reach the model", "neutral")
        c.server.start.assert_not_called()
        self.assertTrue(text)

    def test_cleaner_start_noop_when_suspended(self):
        c = cleanup.Cleaner(self.cfg)
        c.server.suspended = True
        c.server.start = mock.Mock()
        c.start()
        c.server.start.assert_not_called()

    def test_resume_clears_failure_and_lockout(self):
        c = cleanup.Cleaner(self.cfg)
        c.server.suspended = True
        c.server.failed = "x"
        c.server._last_start = 10 ** 9
        with mock.patch.object(cleanup.threading, "Thread"), mock.patch.object(c.server, "start") as st:
            c.resume()
        self.assertFalse(c.server.suspended)
        self.assertIsNone(c.server.failed)
        self.assertEqual(c.server._last_start, 0.0)
        st.assert_called_once()

    def test_suspended_after_popen_kills_process_without_failure(self):
        proc = mock.Mock()
        proc.poll.return_value = None
        proc.pid = 1234

        def popen(*a, **k):
            self.srv.suspended = True       # game mode starts while the process is being spawned
            return proc
        with mock.patch.object(cleanup.subprocess, "Popen", side_effect=popen), \
                mock.patch.object(cleanup, "DATA_DIR", self.tmp), \
                mock.patch.object(self.srv, "_kill_stale"):
            self.srv._starting = True
            self.srv._start_impl()
        proc.terminate.assert_called()
        self.assertIsNone(self.srv.failed)
        self.assertIsNone(self.srv.proc)
        self.assertFalse(self.srv.ready)
        self.assertFalse(self.srv._starting)

    def test_suspended_during_health_loop_kills_process_without_failure(self):
        proc = mock.Mock()
        proc.poll.return_value = None
        proc.pid = 1234

        def get(*a, **k):
            self.srv.suspended = True
            raise requests.ConnectionError()
        with mock.patch.object(cleanup.subprocess, "Popen", return_value=proc), \
                mock.patch.object(cleanup, "DATA_DIR", self.tmp), \
                mock.patch.object(cleanup.requests, "get", side_effect=get), \
                mock.patch.object(cleanup.time, "sleep"), \
                mock.patch.object(self.srv, "_make_job", return_value=None), \
                mock.patch.object(self.srv, "_kill_stale"):
            self.srv._start_impl()
        proc.terminate.assert_called()
        self.assertIsNone(self.srv.failed)
        self.assertFalse(self.srv.ready)

    def test_stop_during_start_is_not_a_failure(self):
        proc = mock.Mock()
        proc.poll.return_value = None
        proc.pid = 1234

        def get(*a, **k):
            self.srv.stop(wait_start_s=0)     # proc swapped out underneath the health loop
            proc.poll.return_value = 1
            raise requests.ConnectionError()
        with mock.patch.object(cleanup.subprocess, "Popen", return_value=proc), \
                mock.patch.object(cleanup, "DATA_DIR", self.tmp), \
                mock.patch.object(cleanup.requests, "get", side_effect=get), \
                mock.patch.object(cleanup.time, "sleep"), \
                mock.patch.object(self.srv, "_make_job", return_value=None), \
                mock.patch.object(self.srv, "_kill_stale"):
            self.srv._start_impl()
        self.assertIsNone(self.srv.failed)


class TranscriberUnloadTests(unittest.TestCase):
    def test_unload_then_transcribe_raises_modelunloaded(self):
        t = Transcriber(make_cfg())
        m = SimpleNamespace(model=mock.Mock())
        t._install(m, "cuda", "float16", "large-v3-turbo")
        t.ready = True
        t.unload()
        m.model.unload_model.assert_called_once()
        self.assertIsNone(t.model)
        self.assertFalse(t.ready)
        import numpy as np
        with self.assertRaises(ModelUnloaded):
            t.transcribe(np.zeros(1600, np.float32))


class AppDeferTests(unittest.TestCase):
    def setUp(self):
        from tests import test_unit_app as ta
        self.ta = ta
        ta.AppWiring.setUp(self)

    def tearDown(self):
        self.ta.AppWiring.tearDown(self)

    def test_stt_reload_deferred_during_game_mode(self):
        a = self.app
        a.stt = mock.Mock()
        a.game_active = True
        a._apply_config(lambda *p: p == ("stt", "model"))
        a.stt.load.assert_not_called()
        a.game_active = False
        a._apply_config(lambda *p: p == ("stt", "model"))
        a.stt.load.assert_called_once()

    def test_hotkey_ignored_in_game_mode_without_overlay(self):
        a = self.app
        a.hotkey = mock.Mock()
        a.game_active = True
        a._do_start()
        self.assertFalse(a._recording)
        self.assertEqual(a.overlay.calls, [])
        a.hotkey.force_idle.assert_called()

    def _exit_app(self, load_effects):
        a = self.app
        a.stt = mock.Mock(device="cuda", model_name="m")
        a.stt.load.side_effect = load_effects
        a.cleaner = mock.Mock()
        a.hotkey = mock.Mock()
        a.watcher.game_running = lambda: False
        a.game_active = True
        waits = []
        a._shutdown = mock.Mock()
        a._shutdown.wait = lambda d: waits.append(d) or False
        return a, waits

    def test_exit_retries_with_backoff_then_succeeds(self):
        a, waits = self._exit_app([RuntimeError("oom"), RuntimeError("oom"), None])
        with mock.patch("faster_whisper.vad.get_vad_model"):
            a._game_exit()
        self.assertEqual(waits, [5, 10])
        self.assertEqual(a.stt.load.call_args_list, [mock.call(strict=True)] * 3)
        a.cleaner.resume.assert_called_once()
        self.assertFalse(a.game_active)

    def test_exit_falls_back_to_cpu_after_all_retries(self):
        a, waits = self._exit_app([RuntimeError("oom")] * 5 + [None])
        with mock.patch("faster_whisper.vad.get_vad_model"):
            a._game_exit()
        self.assertEqual(waits, [5, 10, 20, 40])
        self.assertEqual(a.stt.load.call_args_list[-1], mock.call())      # non-strict = CPU fallback allowed
        a.cleaner.resume.assert_called_once()
        self.assertFalse(a.game_active)

    def test_exit_failure_never_leaves_hotkey_dead(self):
        a, waits = self._exit_app([RuntimeError("oom")] * 5 + [RuntimeError("cpu model missing")])
        a.stt.ready = False
        with mock.patch("faster_whisper.vad.get_vad_model"):
            a._game_exit()
        self.assertFalse(a.game_active)
        a.hotkey.set_enabled.assert_called_with(True)

    def test_enter_disables_hotkey_in_off_mode(self):
        a, _ = self._exit_app([None])
        a.game_active = False
        a._game_enter()
        a.hotkey.set_enabled.assert_called_with(False)
        a.stt.unload.assert_called_once()
        a.cleaner.suspend.assert_called_once()

    def test_list_processes_smoke(self):
        from localflow.gamemode import list_processes, _PROCESSENTRY32W
        import ctypes
        procs = list_processes()
        self.assertTrue(any(pid == 4 for pid, _ in procs))
        if ctypes.sizeof(ctypes.c_void_p) == 8:
            self.assertEqual(ctypes.sizeof(_PROCESSENTRY32W), 568)

    def test_exit_aborts_when_game_relaunches(self):
        a, waits = self._exit_app([None])
        a.watcher.game_running = lambda: True
        a._game_exit()
        a.stt.load.assert_not_called()
        a.cleaner.resume.assert_not_called()
        self.assertTrue(a.game_active)
        a.cleaner.suspend.assert_called()

    def test_worker_drops_job_silently_in_game_mode(self):
        a = self.app
        a.stt = SimpleNamespace(ready=False, model=None)
        a.game_active = True
        a._jobs.put((None, ("x.exe", "t"), None, None, 0.0))
        import threading
        th = threading.Thread(target=a._worker, daemon=True)
        th.start()
        time.sleep(1.2)
        a._shutdown.set()
        th.join(2)
        self.assertNotIn(("error", "Speech model not ready"), a.overlay.calls)
        self.assertIn(("hidden", ""), a.overlay.calls)

    def test_alive_has_no_attribute_race(self):
        srv = cleanup.LlamaServer(make_cfg())
        self.assertFalse(srv.alive())


if __name__ == "__main__":
    unittest.main()
