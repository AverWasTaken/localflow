"""UI helpers + LocalFlow window behaviour (offscreen; no window is shown on screen)."""
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from localflow import uifmt  # noqa: E402


class Formatting(unittest.TestCase):
    def test_numbers(self):
        self.assertEqual(uifmt.fmt_int(38410), "38,410")
        self.assertEqual(uifmt.fmt_int(None), "0")
        self.assertEqual(uifmt.fmt_wpm(141.6), "142")
        self.assertEqual(uifmt.fmt_wpm(0), "-")
        self.assertEqual(uifmt.fmt_streak(1), "1 day")
        self.assertEqual(uifmt.fmt_streak(6), "6 days")

    def test_saved(self):
        self.assertEqual(uifmt.fmt_saved(0), "0m")
        self.assertEqual(uifmt.fmt_saved(30), "under 1m")
        self.assertEqual(uifmt.fmt_saved(45 * 60), "45m")
        self.assertEqual(uifmt.fmt_saved(3 * 3600 + 12 * 60 + 5), "3h 12m")

    def test_durations(self):
        self.assertEqual(uifmt.fmt_duration(245), "4:05")
        self.assertEqual(uifmt.fmt_duration(3729), "1:02:09")
        self.assertEqual(uifmt.fmt_duration(0), "-")
        self.assertEqual(uifmt.fmt_elapsed(754), "12:34")
        self.assertEqual(uifmt.fmt_elapsed(3725), "1:02:05")

    def test_chord(self):
        self.assertEqual(uifmt.fmt_chord("alt+shift+z"), "Alt+Shift+Z")
        self.assertEqual(uifmt.fmt_combo({"win", "ctrl"}), "Ctrl+Win")

    def test_status_text(self):
        self.assertEqual(uifmt.status_text("transcribing", 0.42, "Transcribing (You)"), "Transcribing 42% (You)")
        self.assertEqual(uifmt.status_text("transcribing", 0.1, "Loading audio"), "Transcribing 10%. Loading audio")
        self.assertEqual(uifmt.status_text("queued", 0, ""), "Waiting")
        self.assertEqual(uifmt.status_text("failed", 0.5, "x"), "Failed")

    def test_level(self):
        self.assertEqual(uifmt.level_unit(0), 0.0)
        self.assertAlmostEqual(uifmt.level_unit(10 ** (-12 / 20)), 1.0, places=3)
        self.assertTrue(0 < uifmt.level_unit(0.02) < 1)

    def test_parse_turns(self):
        t = "[00:00:04] Others: Hi there.\n\n[01:02:03] You: Yes:\nline two\n\nplain para"
        self.assertEqual(uifmt.parse_turns(t), [("00:04", "Others", "Hi there."), ("01:02:03", "You", "Yes:\nline two"),
                                                ("", "", "plain para")])

    def test_preset_label(self):
        self.assertEqual(uifmt.preset_label("key_points"), "Key points")
        self.assertEqual(uifmt.preset_label("custom", "What did we decide?"), "What did we decide?")
        self.assertTrue(uifmt.preset_label("custom", "x" * 100).endswith("..."))


class CommandCombo(unittest.TestCase):
    def test_rules(self):
        err = uifmt.command_combo_error
        self.assertEqual(err({"ctrl", "alt"}, {"ctrl", "win"}), "")
        self.assertIn("different", err({"ctrl", "alt"}, {"ctrl", "alt"}))
        self.assertIn("part of", err({"ctrl", "alt"}, {"ctrl"}))
        self.assertIn("contains", err({"ctrl", "alt"}, {"ctrl", "alt", "shift"}))
        self.assertIn("at least one", err({"ctrl", "alt"}, set()))

    def test_matches_backend_rule(self):
        """Whatever the UI accepts, the hotkey module must accept too (and the other way round)."""
        from itertools import combinations

        from localflow import hotkey
        mods = ("ctrl", "alt", "shift", "win")
        combos = [frozenset(c) for n in range(1, 5) for c in combinations(mods, n)]
        for d in combos:
            for c in combos:
                ui_ok = uifmt.command_combo_error(d, c) == ""
                be_ok = not (c <= d or d <= c)
                self.assertEqual(ui_ok, be_ok, (d, c))
        self.assertTrue(hasattr(hotkey, "parse_combo"))


class Window(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.qapp = QApplication.instance() or QApplication([])

    def make(self, **kw):
        from localflow.config import Config
        from localflow.settings_ui import SettingsWindow
        d = Path(tempfile.mkdtemp())
        cfg = Config(path=d / "config.json")
        w = SettingsWindow(cfg, lambda: [], lambda: False, lambda v: None, get_last_ctx=lambda: ("", ""),
                           get_default_mic=lambda: "", **kw)
        return cfg, w

    def test_pages_and_bar(self):
        cfg, w = self.make()
        self.assertEqual(w.windowTitle(), "LocalFlow")
        self.assertEqual(w.current_page(), "home")
        self.assertTrue(w.bottom_bar.isHidden())
        w.show_page("general")
        self.assertEqual(w.current_page(), "general")
        self.assertFalse(w.bottom_bar.isHidden())
        w.show_page("transcriptions")
        self.assertTrue(w.bottom_bar.isHidden())

    def test_navigation_keeps_unsaved_edits(self):
        cfg, w = self.make()
        w.show_page("general")
        w.minwords_spin.setValue(17)
        for p in ("home", "transcriptions", "styles", "general"):
            w.show_page(p)
        self.assertEqual(w.minwords_spin.value(), 17)

    def test_command_conflict_blocks_save(self):
        cfg, w = self.make()
        w.reload()
        for k, cb in w.cmd_boxes.items():
            cb.setChecked(k == "ctrl")           # Ctrl alone is part of Ctrl+Alt
        self.assertFalse(w.cmd_error.isHidden())
        self.assertFalse(w.save())
        self.assertEqual(cfg.get("command_mode.hotkey"), "ctrl+win")
        w.cmd_sw.setChecked(False)               # off: saves, keeps the stored combo
        self.assertTrue(w.save())
        self.assertFalse(cfg.get("command_mode.enabled"))
        self.assertEqual(cfg.get("command_mode.hotkey"), "ctrl+win")

    def test_new_settings_save(self):
        devs = {"inputs": ["Mic A"], "outputs": ["Phones B"], "default_input": "Mic A", "default_output": "Phones B"}
        cfg, w = self.make(get_devices=lambda: devs)
        w.reload()
        for k, cb in w.cmd_boxes.items():
            cb.setChecked(k in ("ctrl", "shift"))
        w.meet_mic_combo.setCurrentIndex(w.meet_mic_combo.findData("Mic A"))
        w.meet_out_combo.setCurrentIndex(w.meet_out_combo.findData("Phones B"))
        w.retention_combo.setCurrentIndex(w.retention_combo.findData(30))
        w.paste_sw.setChecked(False)
        self.assertTrue(w.save())
        self.assertEqual(cfg.get("command_mode.hotkey"), "ctrl+shift")
        self.assertEqual(cfg.get("meetings.mic"), "Mic A")
        self.assertEqual(cfg.get("meetings.output_device"), "Phones B")
        self.assertEqual(cfg.get("history.retention_days"), 30)
        self.assertFalse(cfg.get("paste_last.enabled"))
        self.assertEqual(cfg.get("paste_last.hotkey"), "alt+shift+z")      # untouched
        self.assertEqual(cfg.get("meetings.max_hours"), 4)                  # deep-merged, not replaced

    def test_transcripts_unavailable(self):
        cfg, w = self.make()
        tp = w.transcripts_page
        self.assertFalse(tp.rec_btn.isEnabled())
        self.assertFalse(tp.file_btn.isEnabled())


class HomeStats(unittest.TestCase):
    def test_values(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from localflow.home_ui import stat_values
        vals = dict((k, v) for k, _l, v in stat_values({"words_7d": 1240, "total_words": 38410, "avg_wpm": 142.2,
                                                        "time_saved_s": 3 * 3600, "streak_days": 6}))
        self.assertEqual(vals, {"words_7d": "1,240", "total_words": "38,410", "avg_wpm": "142",
                                "time_saved_s": "3h 0m", "streak_days": "6 days"})
        self.assertEqual(stat_values(None)[0][2], "0")

    def test_history_list_widget(self):
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        from localflow.history import History
        from localflow.history_ui import HistoryList
        h = History(path=Path(tempfile.mkdtemp()) / "h.db")
        h.add("raw words", "Clean words.", "code.exe", 1.0, 10, 20, 30)
        lst = HistoryList(h)
        lst.refresh(force=True)
        self.assertEqual(len(lst._cards), 1)
        self.assertFalse(lst._auto.isActive())          # polls only while visible
        time.sleep(0)


if __name__ == "__main__":
    unittest.main()
