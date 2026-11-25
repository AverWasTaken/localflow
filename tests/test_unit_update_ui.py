"""Offline tests for localflow/update_ui.py (the update prompt and its controller). The network, the download and
starting setup are mocked; the app is a stand-in with just what the controller uses."""
import os
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from PySide6.QtWidgets import QApplication
    from localflow import update_ui, updater
except ImportError:         # pragma: no cover - PySide6 missing
    QApplication = None

INFO = None if QApplication is None else updater.UpdateInfo(
    version="9.9.9", tag="v9.9.9", html_url="https://example.invalid/r", asset_name="LocalFlow-Setup-9.9.9.exe",
    asset_url="https://example.invalid/s.exe", size=10, sha256=None, notes="* Something new")


class _Cfg:
    def __init__(self):
        self.d = {"updates.check_on_launch": True, "updates.skipped_version": ""}

    def get(self, k, default=None):
        return self.d.get(k, default)

    def set(self, k, v, save=True):
        self.d[k] = v


@unittest.skipIf(QApplication is None, "PySide6 not installed")
class UpdateControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qapp = QApplication.instance() or QApplication([])

    def setUp(self):
        self.blocker = ""
        self.app = SimpleNamespace(cfg=_Cfg(), tray=None, game_active=False, quit=mock.Mock(),
                                   update_blocker=lambda: self.blocker)
        self.ctl = update_ui.UpdateController(self.app)

    def tearDown(self):
        if self.ctl.dialog is not None:
            self.ctl._state = "idle"
            self.ctl.dialog.close()
            self.ctl.dialog.deleteLater()

    def _pump(self, cond, timeout=5.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.qapp.processEvents()
            if cond():
                return True
            time.sleep(0.01)
        return False

    def _check(self, manual, result=None, exc=None):
        kw = {"side_effect": exc} if exc else {"return_value": result}
        with mock.patch.object(updater, "check_latest", **kw):
            self.ctl.check(manual=manual)
            self.assertTrue(self._pump(lambda: not self.ctl._checking))

    def test_manual_latest_and_error(self):
        self._check(True, None)
        self.assertTrue(self.ctl.dialog.isVisible())
        self.assertIn("latest version", self.ctl.dialog.title.text())
        self._check(True, exc=updater.UpdateError("could not reach GitHub: offline"))
        self.assertIn("offline", self.ctl.dialog.text.text())
        self.assertTrue(self.ctl.dialog.release_btn.isVisibleTo(self.ctl.dialog))

    def test_launch_check_quiet_when_nothing_or_error(self):
        self._check(False, None)
        self.assertIsNone(self.ctl.dialog)
        self._check(False, exc=updater.UpdateError("offline"))
        self.assertIsNone(self.ctl.dialog)

    def test_launch_check_disabled(self):
        self.app.cfg.d["updates.check_on_launch"] = False
        with mock.patch.object(updater, "check_latest") as cl:
            self.ctl.check(manual=False)
        cl.assert_not_called()

    def test_offer_skip_and_manual_ignores_skip(self):
        self._check(False, INFO)
        d = self.ctl.dialog
        self.assertIn("v9.9.9", d.title.text())
        self.assertIn("Something new", d.notes.text())
        d.skip_btn.click()
        self.assertEqual(self.app.cfg.d["updates.skipped_version"], "9.9.9")
        self.assertFalse(d.isVisible())
        self._check(False, INFO)            # launch check: skipped version stays quiet
        self.assertFalse(d.isVisible())
        self._check(True, INFO)             # manual check: offered again
        self.assertTrue(d.isVisible())

    def test_later_and_close_hide(self):
        self._check(True, INFO)
        d = self.ctl.dialog
        d.later_btn.click()
        self.assertFalse(d.isVisible())
        self._check(True, None)
        self.assertTrue(d.isVisible())
        d.close_btn.click()
        self.assertFalse(d.isVisible())

    def test_closing_window_cancels_download(self):
        self._check(True, INFO)
        started = []

        def slow_download(info, progress=None, cancel=None):
            started.append(cancel)
            while not cancel.is_set():
                time.sleep(0.01)
            raise updater.UpdateCancelled("cancelled")
        with mock.patch.object(updater, "can_self_update", return_value=True), \
                mock.patch.object(updater, "download", side_effect=slow_download), \
                mock.patch.object(updater, "launch_installer") as li:
            self.ctl.download_and_install()
            self.assertTrue(self._pump(lambda: bool(started)))
            self.ctl.dialog.close()                 # title bar X / Esc
            self.assertFalse(self.ctl.dialog.isVisible())
            self.assertTrue(self._pump(lambda: self.ctl._state == "idle"))
        self.assertTrue(started[0].is_set())
        li.assert_not_called()
        self.app.quit.assert_not_called()

    def test_cancel_after_download_finished_does_not_install(self):
        self._check(True, INFO)
        with mock.patch.object(updater, "can_self_update", return_value=True), \
                mock.patch.object(updater, "launch_installer") as li:
            self.ctl._state = "downloading"
            self.ctl._cancel = threading.Event()
            self.ctl.cancel()                       # user cancels...
            self.ctl._on_downloaded("C:/t/s.exe", "")   # ...but the success signal was already queued
        li.assert_not_called()
        self.assertEqual(self.ctl._state, "idle")
        self.app.quit.assert_not_called()

    def test_installing_flag_set_before_setup_and_rechecked(self):
        self.app.installing_update = False
        seen = []
        with mock.patch.object(updater, "launch_installer",
                               side_effect=lambda p: seen.append(self.app.installing_update)):
            self._check(True, INFO)
            self.ctl._installer, self.ctl._state = "C:/t/s.exe", "waiting"
            self.ctl._install_when_free()
        self.assertEqual(seen, [True])          # new recordings blocked before setup starts
        # a recording is running when the flag goes up: keep waiting, flag cleared
        self.ctl._state = "waiting"
        self.app.installing_update = False
        self.blocker = "the current dictation"
        with mock.patch.object(updater, "launch_installer") as li:
            self.ctl._install_when_free()
        li.assert_not_called()
        self.assertFalse(self.app.installing_update)
        self.assertEqual(self.ctl._state, "waiting")
        self.ctl._wait_timer.stop()

    def test_recording_starting_after_flag_keeps_waiting(self):
        self._check(True, INFO)
        flags, answers = [], iter(["", "the current dictation"])   # clear, then a dictation got in first

        def blocker():
            flags.append(self.app.installing_update)
            return next(answers)
        self.app.update_blocker = blocker
        self.app.installing_update = False
        self.ctl._installer, self.ctl._state = "C:/t/s.exe", "waiting"
        with mock.patch.object(updater, "launch_installer") as li:
            self.ctl._install_when_free()
        self.assertEqual(flags, [False, True])          # flag only goes up after a clear first check
        li.assert_not_called()
        self.assertFalse(self.app.installing_update)
        self.assertEqual(self.ctl._state, "waiting")
        self.ctl._wait_timer.stop()

    def test_deferred_offer_respects_skip_and_switch(self):
        self.blocker = "the current dictation"
        with mock.patch.object(update_ui.QTimer, "singleShot"):
            self._check(False, INFO)
        self.app.cfg.d["updates.skipped_version"] = "9.9.9"      # skipped from a manual check meanwhile
        self.blocker = ""
        self.ctl._offer_when_quiet()
        self.assertIsNone(self.ctl.dialog)
        self.assertIsNone(self.ctl._pending)
        self.app.cfg.d["updates.skipped_version"] = ""
        self.blocker = "x"
        with mock.patch.object(update_ui.QTimer, "singleShot"):
            self._check(False, INFO)
        self.app.cfg.d["updates.check_on_launch"] = False
        self.blocker = ""
        self.ctl._offer_when_quiet()
        self.assertIsNone(self.ctl.dialog)

    def test_offer_waits_while_busy(self):
        self.blocker = "the current dictation"
        with mock.patch.object(update_ui.QTimer, "singleShot") as ss:
            self._check(False, INFO)
        self.assertIsNone(self.ctl.dialog)
        self.assertEqual(ss.call_args[0][0], update_ui.DEFER_MS)
        self.blocker = ""
        self.ctl._offer_when_quiet()
        self.assertTrue(self.ctl.dialog.isVisible())

    def test_source_copy_opens_release_page(self):
        self._check(True, INFO)
        with mock.patch.object(updater, "can_self_update", return_value=False), \
                mock.patch.object(update_ui.QDesktopServices, "openUrl") as ou, \
                mock.patch.object(updater, "download") as dl:
            self.ctl.download_and_install()
        ou.assert_called_once()
        dl.assert_not_called()

    def test_download_install_and_quit(self):
        self._check(True, INFO)
        with mock.patch.object(updater, "can_self_update", return_value=True), \
                mock.patch.object(updater, "download", return_value=Path("C:/t/LocalFlow-Setup-9.9.9.exe")), \
                mock.patch.object(updater, "launch_installer") as li:
            self.ctl.download_and_install()
            self.ctl.download_and_install()     # second click while busy: ignored
            self.assertTrue(self._pump(lambda: li.called))
        li.assert_called_once()
        self.assertEqual(self.ctl._state, "installing")
        self.assertTrue(self._pump(lambda: self.app.quit.called, 3))

    def test_install_waits_for_recording(self):
        self._check(True, INFO)
        self.blocker = "the meeting recording"
        with mock.patch.object(updater, "can_self_update", return_value=True), \
                mock.patch.object(updater, "download", return_value=Path("C:/t/s.exe")), \
                mock.patch.object(updater, "launch_installer") as li:
            self.ctl.download_and_install()
            self.assertTrue(self._pump(lambda: self.ctl._state == "waiting"))
            self.assertIn("meeting recording", self.ctl.dialog.text.text())
            self._pump(lambda: False, 0.3)
            li.assert_not_called()
            self.blocker = ""
            self.assertTrue(self._pump(lambda: li.called, 3))

    def test_cancel_while_waiting(self):
        self._check(True, INFO)
        self.blocker = "the current dictation"
        with mock.patch.object(updater, "can_self_update", return_value=True), \
                mock.patch.object(updater, "download", return_value=Path("C:/t/s.exe")), \
                mock.patch.object(updater, "launch_installer") as li:
            self.ctl.download_and_install()
            self.assertTrue(self._pump(lambda: self.ctl._state == "waiting"))
            self.ctl.dialog.cancel_btn.click()
            self.blocker = ""
            self._pump(lambda: False, 1.3)
        li.assert_not_called()
        self.assertEqual(self.ctl._state, "idle")
        self.app.quit.assert_not_called()

    def test_download_failure_shows_error(self):
        self._check(True, INFO)
        with mock.patch.object(updater, "can_self_update", return_value=True), \
                mock.patch.object(updater, "download", side_effect=updater.UpdateError("checksum")):
            self.ctl.download_and_install()
            self.assertTrue(self._pump(lambda: self.ctl._state == "idle"))
        self.assertIn("checksum", self.ctl.dialog.text.text())
        self.assertTrue(self.ctl.dialog.release_btn.isVisibleTo(self.ctl.dialog))
        self.app.quit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
