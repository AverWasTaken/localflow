"""Speakers muted while dictating (localflow.outmute). Everything runs against a fake endpoint layer: these tests
never touch a real audio device."""
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from localflow import app as app_mod  # noqa: E402
from localflow import outmute  # noqa: E402
from localflow.outmute import OutputMuter  # noqa: E402

import test_unit_app as tua  # noqa: E402  (module import only: its TestCases are not collected twice)

SPK = "{0.0.0.00000000}.{speakers}"
HP = "{0.0.0.00000000}.{headphones}"
MM = "{0.0.0.00000000}.{multimedia}"


class FakeEndpoints:
    def __init__(self, marker=None):
        self.devs = {SPK: {"name": "Speakers", "muted": False, "present": True},
                     HP: {"name": "Headphones", "muted": False, "present": True},
                     MM: {"name": "Multimedia", "muted": False, "present": True}}
        self.default = [SPK]
        self.calls = []
        self.threads = set()
        self.marker = marker
        self.marker_seen = []         # marker contents at each set_mute(True)
        self.closed = False

    def init(self):
        self.threads.add(threading.get_ident())

    def close(self):
        self.closed = True

    def defaults(self):
        self.threads.add(threading.get_ident())
        return [(d, self.devs[d]["name"]) for d in self.default if self.devs[d]["present"]]

    def get_mute(self, d):
        self.threads.add(threading.get_ident())
        dv = self.devs.get(d)
        return None if dv is None or not dv["present"] else dv["muted"]

    def set_mute(self, d, m):
        self.threads.add(threading.get_ident())
        self.calls.append((d, m))
        dv = self.devs.get(d)
        if dv is None or not dv["present"]:
            return False
        if m and self.marker is not None:
            self.marker_seen.append(json.loads(Path(self.marker).read_text()) if Path(self.marker).exists() else None)
        dv["muted"] = m
        return True

    def muted(self, d):
        return self.devs[d]["muted"]


def wait_until(fn, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if fn():
            return True
        time.sleep(0.01)
    return fn()


class MuterTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.marker = self.dir / "output_mute.json"
        self.ep = FakeEndpoints(self.marker)
        self.rec = {"on": True}
        self.m = OutputMuter(self.ep, marker_path=self.marker, is_recording=lambda: self.rec["on"], poll_s=0.05)

    def tearDown(self):
        self.m.shutdown()

    def mute(self, rid=1, delay=0.0):
        self.m.arm(rid)
        self.m.engage(rid, delay)
        self.assertTrue(wait_until(lambda: self.ep.muted(SPK)))

    def test_mute_and_restore(self):
        self.mute()
        self.assertTrue(self.marker.exists())
        self.assertTrue(self.m.release(wait=1.0))
        self.assertFalse(self.ep.muted(SPK))
        self.assertFalse(self.marker.exists())
        self.assertFalse(self.ep.muted(HP))
        self.assertNotIn(HP, [c[0] for c in self.ep.calls])      # communications / other devices untouched

    def test_marker_written_before_muting(self):
        self.mute()
        self.assertEqual(len(self.ep.marker_seen), 1)
        self.assertIn(SPK, self.ep.marker_seen[0]["devices"])

    def test_already_muted_device_left_alone(self):
        self.ep.devs[SPK]["muted"] = True
        self.m.arm(1)
        self.m.engage(1)
        time.sleep(0.2)
        self.m.release(wait=1.0)
        self.assertTrue(self.ep.muted(SPK))                       # stays muted: it was muted before
        self.assertEqual(self.ep.calls, [])
        self.assertFalse(self.marker.exists())

    def test_multimedia_default_also_muted_when_different(self):
        self.ep.default = [SPK, MM]
        self.mute()
        self.assertTrue(wait_until(lambda: self.ep.muted(MM)))
        self.m.release(wait=1.0)
        self.assertFalse(self.ep.muted(SPK))
        self.assertFalse(self.ep.muted(MM))

    def test_restore_by_id_after_default_changes(self):
        self.mute()
        self.ep.default = [HP]                                     # user switched output mid-dictation
        self.m.release(wait=1.0)
        self.assertFalse(self.ep.muted(SPK))
        self.assertNotIn((HP, False), self.ep.calls)
        self.assertNotIn((HP, True), self.ep.calls)

    def test_release_before_engage_never_mutes(self):
        self.m.arm(1)
        self.m.release()
        self.m.engage(1)
        time.sleep(0.2)
        self.assertEqual(self.ep.calls, [])

    def test_release_during_cue_delay_never_mutes(self):
        self.m.arm(1)
        self.m.engage(1, delay=0.5)
        time.sleep(0.05)
        t0 = time.monotonic()
        self.assertTrue(self.m.release(wait=1.0))
        self.assertLess(time.monotonic() - t0, 0.3)                # the delay wait is interrupted
        time.sleep(0.6)
        self.assertEqual(self.ep.calls, [])

    def test_stale_recording_id_is_ignored(self):
        self.m.arm(2)
        self.m.engage(1)
        time.sleep(0.2)
        self.assertEqual(self.ep.calls, [])

    def test_delay_respected(self):
        self.m.arm(1)
        self.m.engage(1, delay=0.2)
        time.sleep(0.1)
        self.assertFalse(self.ep.muted(SPK))
        self.assertTrue(wait_until(lambda: self.ep.muted(SPK)))

    def test_release_without_mute_returns_immediately(self):
        t0 = time.monotonic()
        self.assertTrue(self.m.release(wait=1.0))
        self.assertLess(time.monotonic() - t0, 0.05)
        self.assertIsNone(self.m._thread)                          # no worker thread was ever needed

    def test_liveness_poll_restores_when_not_recording(self):
        self.mute()
        self.rec["on"] = False                                     # app stopped without calling release
        self.assertTrue(wait_until(lambda: not self.ep.muted(SPK)))
        self.assertTrue(wait_until(lambda: not self.marker.exists()))

    def test_max_duration_restores(self):
        self.m._max_s = lambda: 0.2
        self.mute()
        self.assertTrue(wait_until(lambda: not self.ep.muted(SPK), timeout=3.0))

    def test_shutdown_restores(self):
        self.mute()
        self.m.shutdown(wait=1.0)
        self.assertFalse(self.ep.muted(SPK))
        self.assertTrue(self.ep.closed)

    def test_device_gone_kept_in_marker_and_retried(self):
        self.mute()
        self.ep.devs[SPK]["present"] = False                       # unplugged while muted
        self.ep.default = [HP]
        self.m.release(wait=1.0)
        self.assertTrue(self.marker.exists())
        self.assertIn(SPK, json.loads(self.marker.read_text())["devices"])
        self.ep.devs[SPK]["present"] = True                        # plugged back in (not the default now)
        self.m.release(wait=1.0)                                   # next stop retries
        self.assertFalse(self.ep.muted(SPK))
        self.assertFalse(self.marker.exists())

    def test_device_gone_retried_on_next_start(self):
        self.mute()
        self.ep.devs[SPK]["present"] = False
        self.ep.default = [HP]
        self.m.release(wait=1.0)
        self.ep.devs[SPK]["present"] = True
        self.m.arm(2)
        self.m.engage(2)
        self.assertTrue(wait_until(lambda: self.ep.muted(HP)))     # new default muted
        self.assertTrue(wait_until(lambda: not self.ep.muted(SPK)))  # old one restored
        self.m.release(wait=1.0)
        self.assertFalse(self.ep.muted(HP))
        self.assertFalse(self.marker.exists())

    def test_marker_recovery_at_startup(self):
        self.ep.devs[SPK]["muted"] = True                          # left muted by a crashed run
        self.marker.write_text(json.dumps({"version": 1, "devices": {SPK: "Speakers"}}))
        m2 = OutputMuter(self.ep, marker_path=self.marker)
        m2.recover()
        self.assertTrue(wait_until(lambda: not self.ep.muted(SPK)))
        self.assertTrue(wait_until(lambda: not self.marker.exists()))
        m2.shutdown()

    def test_corrupt_marker_ignored(self):
        self.marker.write_text("{nope")
        m2 = OutputMuter(self.ep, marker_path=self.marker)
        m2.recover()
        self.assertFalse(self.marker.exists())
        self.assertEqual(self.ep.calls, [])

    def test_all_calls_on_one_worker_thread(self):
        self.mute()
        self.m.release(wait=1.0)
        self.assertEqual(len(self.ep.threads), 1)
        self.assertNotIn(threading.get_ident(), self.ep.threads)

    def test_backend_errors_never_raise(self):
        def boom(*a):
            raise RuntimeError("COM exploded")
        self.ep.defaults = boom
        self.m.arm(1)
        self.m.engage(1)
        time.sleep(0.1)
        self.assertTrue(self.m.release(wait=1.0))

    def test_inert_without_backend(self):
        m = OutputMuter(None, marker_path=self.marker)
        m.arm(1)
        m.engage(1)
        self.assertTrue(m.release())
        m.recover()
        m.shutdown()
        self.assertIsNone(m._thread)
        self.assertFalse(m.active)


class AppOutmuteWiring(unittest.TestCase):
    """LocalFlowApp drives the muter: arm at key-down, engage after the confirmed hold, release on every exit."""

    def setUp(self):
        tua.AppWiring.setUp(self)
        self.events = []
        app_mod.sounds = SimpleNamespace(play=lambda name, cfg=None: self.events.append(("sound", name)),
                                         ensure_sounds=lambda *a, **k: None)
        app_mod.winutil.foreground_info = lambda: ("notepad.exe", "x")
        self.app.cfg.set("streaming.enabled", False, save=False)

    def tearDown(self):
        tua.AppWiring.tearDown(self)

    def kinds(self):
        return [c[0] for c in self.app.outmute.calls]

    def test_default_is_on_and_tests_get_no_real_backend(self):
        self.assertTrue(self.app.cfg.get("audio.mute_output_while_dictating"))
        self.assertIsNone(self.app.outmute.backend)

    def test_start_arms_confirmed_hold_engages_stop_releases(self):
        self.app._do_start()
        rid = self.app._rec_id
        self.assertEqual(self.app.outmute.calls[0], ("arm", rid))
        time.sleep(0.3)
        self.assertIn(("engage", rid, 0.2), self.app.outmute.calls)
        self.app._do_stop()
        self.assertEqual(self.kinds()[-1], "release")

    def test_sounds_off_engages_without_delay(self):
        self.app.cfg.set("ui.sounds", False, save=False)
        self.app._do_start()
        time.sleep(0.3)
        self.assertIn(("engage", self.app._rec_id, 0.0), self.app.outmute.calls)
        self.app._do_cancel("esc")

    def test_setting_off_never_arms(self):
        self.app.cfg.set("audio.mute_output_while_dictating", False, save=False)
        self.app._do_start()
        time.sleep(0.3)
        self.assertNotIn("arm", self.kinds())
        self.app._do_cancel("esc")
        self.assertIn("release", self.kinds())                     # exits restore regardless of the setting

    def test_cancel_and_too_short_release(self):
        self.app._do_start()
        self.app._do_cancel("shortcut")                             # Ctrl+Alt+<key> shortcut
        self.assertEqual(self.kinds(), ["arm", "release"])
        self.app._do_start()
        self.app._do_stop()                                         # too short
        self.assertEqual(self.kinds()[-1], "release")
        time.sleep(0.3)
        self.assertNotIn("engage", self.kinds())                    # never confirmed: never muted

    def test_meeting_active_skips_mute(self):
        self.app._meeting_on = True
        self.app._do_start()
        time.sleep(0.3)
        self.assertNotIn("arm", self.kinds())
        self.assertNotIn("engage", self.kinds())
        self.app._do_cancel("esc")

    def test_meeting_starting_mid_dictation_releases(self):
        self.app._do_start()
        time.sleep(0.3)
        self.app._apply_tray = lambda: None
        self.app._on_meeting_state({"recording": True})
        self.assertEqual(self.kinds()[-1], "release")
        self.app._do_cancel("esc")

    def test_controller_exception_releases(self):
        def boom():
            raise RuntimeError("x")
        self.app.audio.begin = boom
        self.app._ctl.put(("start", "dictate"))
        t = threading.Thread(target=self.app._controller, daemon=True)
        t.start()
        self.assertTrue(wait_until(lambda: "release" in self.kinds()))
        self.app._shutdown.set()
        t.join(2)

    def test_real_muter_restores_before_stop_cue(self):
        ep = FakeEndpoints()
        d = Path(tempfile.mkdtemp())
        log = self.events
        orig_set = ep.set_mute

        def set_mute(dev, m):
            log.append(("mute" if m else "unmute", dev))
            return orig_set(dev, m)
        ep.set_mute = set_mute
        self.app.outmute = OutputMuter(ep, marker_path=d / "m.json", is_recording=lambda: self.app._recording)
        try:
            self.app._do_start()
            self.assertTrue(wait_until(lambda: ep.muted(SPK)))
            self.assertEqual([e[0] for e in log][:2], ["sound", "mute"])   # start cue first, then mute
            time.sleep(0.2)
            self.app._do_stop()
            self.assertFalse(ep.muted(SPK))
            names = [e[0] if e[0] != "sound" else e[1] for e in log]
            self.assertLess(names.index("unmute"), names.index("stop"))      # speakers back before the stop cue
        finally:
            self.app.outmute.shutdown()


if __name__ == "__main__":
    unittest.main()
