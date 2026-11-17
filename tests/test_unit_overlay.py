"""Overlay pill + answer card (command-mode questions), rendered offscreen."""
import os
import threading
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

Q = "How many hours is it from 7:30am to 5pm?"
A = "9 hours 30 minutes (9.5 hours)"
LONG = ("It says the deadline moved to Friday because the vendor missed two deliveries, and asks the team to stop "
        "work on the import screen until the new files arrive. The tone is firm but not rude. Softening the first "
        "sentence and adding a thank-you at the end would make it read as collaborative rather than annoyed. "
        "You could also drop the line about the vendor being careless, since it adds blame without adding facts, "
        "and your manager will likely forward it to the vendor anyway, so keep it short and specific.")


class _Screen:
    def __init__(self, rect):
        self._rect = rect

    def availableGeometry(self):
        return self._rect


class OverlayAnswerCard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.qapp = QApplication.instance() or QApplication([])

    def setUp(self):
        from localflow import overlay
        self.O = overlay
        self.ov = overlay.Overlay()

    def tearDown(self):
        self.ov._auto.stop()
        self.ov._frame.stop()
        self.ov.hide()
        self.ov.deleteLater()
        self.qapp.processEvents()

    # ------------------------------------------------------------- helpers
    def finish(self):
        """Run every running animation to its end (emits valueChanged + finished)."""
        ov = self.ov
        for _ in range(3):
            for a in (ov._fade, ov._wanim, ov._lanim, ov._xanim):
                if a.state() == a.State.Running:
                    a.setCurrentTime(a.duration())

    def show_answer(self, q=Q, a=A, ms=5000):
        self.ov._apply_answer(q, a, ms)
        self.finish()

    def card_bottom(self):
        rect, _alpha = self.ov._card_geometry()
        return rect.bottom()

    # ------------------------------------------------------------- tests
    def test_answer_is_a_valid_state(self):
        self.assertIn("answer", self.O.VALID_STATES)

    def test_show_answer_from_hidden(self):
        ov, O = self.ov, self.O
        self.assertFalse(ov.isVisible())
        ov._apply_answer(Q, A, 5000)
        self.assertTrue(ov.isVisible())
        self.assertEqual(ov._state, "answer")
        self.assertEqual(ov._text, O.ANSWER_LABEL)
        self.assertTrue(ov._lanim.state() == ov._lanim.State.Running)
        self.assertEqual(ov._lanim.duration(), O.LIFT_IN_MS)
        self.assertTrue(ov._auto.isActive())
        self.assertEqual(ov._auto.interval(), 5000)
        # mid-way the pill has started to rise and the card is not fully in yet
        ov._lanim.setCurrentTime(O.LIFT_IN_MS // 3)
        self.assertLess(ov._pill_y(), O.REST_Y)
        self.assertLess(ov._card_a, 1.0)
        self.finish()
        self.assertAlmostEqual(ov._lift, 1.0)
        self.assertAlmostEqual(ov._card_a, 1.0)
        self.assertAlmostEqual(ov._card_rise, 0.0)
        card = ov._card
        self.assertEqual((card.question, card.answer), (Q, A))
        self.assertAlmostEqual(ov._pill_y(), O.REST_Y - (card.h + O.CARD_GAP))
        # card bottom lands where the resting pill bottom is
        self.assertAlmostEqual(self.card_bottom(), O.CANVAS_H - O.MARGIN)
        self.assertFalse(ov.grab().isNull())

    def test_pill_never_overlaps_visible_card_while_lifting(self):
        ov, O = self.ov, self.O
        ov._apply_answer(Q * 2, LONG * 2, 5000)          # tallest card = longest travel
        ov._card.h = O.CARD_MAX_H                          # worst case regardless of the test font
        for ms in range(0, O.LIFT_IN_MS + 1, 5):
            ov._lanim.setCurrentTime(ms)
            rect, alpha = ov._card_geometry()
            if rect is not None and alpha > 0.02:
                self.assertLessEqual(ov._pill_y() + O.PILL_H, rect.top() + 0.5, f"overlap at {ms} ms")
        self.finish()
        ov._apply_state("recording", "")                 # and on the way back down
        for ms in range(0, O.LIFT_OUT_MS + 1, 5):
            ov._lanim.setCurrentTime(ms)
            rect, alpha = ov._card_geometry()
            if rect is not None and alpha > 0.02:
                self.assertLessEqual(ov._pill_y() + O.PILL_H, rect.top() + 0.5, f"overlap at {ms} ms (out)")

    def test_show_answer_is_thread_safe(self):
        ov = self.ov
        t = threading.Thread(target=ov.show_answer, args=(Q, A, 4200))
        t.start()
        t.join()
        self.qapp.processEvents()
        self.assertEqual(ov._state, "answer")
        self.assertEqual(ov._card.answer, A)
        self.assertEqual(ov._auto.interval(), 4200)

    def test_show_answer_over_card_replaces_smoothly(self):
        ov = self.ov
        self.show_answer()
        old = ov._card
        ov._apply_answer("Who wrote Pride and Prejudice?", LONG, 9000)
        self.assertIs(ov._card_prev, old)
        self.assertEqual(ov._card.answer, LONG)
        self.assertTrue(ov._xanim.state() == ov._xanim.State.Running)
        self.assertAlmostEqual(ov._lift, 1.0)          # the pill does not drop between answers
        self.assertAlmostEqual(ov._card_a, 1.0)
        ov._xanim.setCurrentTime(ov._xanim.duration() // 2)
        w, h = ov._card_size()
        self.assertTrue(old.h < h < ov._card.h)        # card eases to the new size
        self.assertFalse(ov.grab().isNull())
        self.finish()
        self.assertIsNone(ov._card_prev)
        self.assertEqual(ov._auto.interval(), 9000)
        self.assertAlmostEqual(ov._pill_y(), self.O.REST_Y - (ov._card.h + self.O.CARD_GAP))

    def test_show_answer_during_hide_fade_recovers(self):
        ov = self.ov
        self.show_answer()
        ov._begin_hide()
        ov._fade.setCurrentTime(90)
        ov._apply_answer(Q, "8 hours", 5000)
        self.finish()
        self.assertTrue(ov.isVisible())
        self.assertAlmostEqual(ov.windowOpacity(), 1.0)
        self.assertEqual(ov._card.answer, "8 hours")
        self.assertAlmostEqual(ov._lift, 1.0)

    def test_set_state_collapses_card(self):
        O = self.O
        for state, text in (("recording", ""), ("handsfree", ""), ("processing", "Working…"),
                            ("error", "Microphone not found"), ("info", "Paused")):
            with self.subTest(state=state):
                ov = self.ov
                self.show_answer()
                ov._apply_state(state, text)
                self.assertEqual(ov._mode, state)
                self.assertEqual(ov._lift_dir, "out")
                self.assertTrue(ov._lanim.state() == ov._lanim.State.Running)
                # mid-collapse the outgoing card still paints from its own stored state
                ov._lanim.setCurrentTime(O.LIFT_OUT_MS // 4)
                self.assertIsNotNone(ov._card)
                self.assertEqual(ov._card.answer, A)
                self.assertLess(ov._card_a, 1.0)
                self.assertFalse(ov.grab().isNull())
                self.finish()
                self.assertAlmostEqual(ov._lift, 0.0)
                self.assertIsNone(ov._card)
                self.assertAlmostEqual(ov._pill_y(), O.REST_Y)
                self.assertTrue(ov.isVisible())
                ov._apply_state("hidden", "")
                self.finish()

    def test_hidden_fades_card_out_with_window(self):
        ov = self.ov
        self.show_answer()
        ov._apply_state("hidden", "")
        self.assertTrue(ov._hiding)
        self.assertAlmostEqual(ov._lift, 1.0)          # card stays put while the window fades
        self.finish()
        self.assertFalse(ov.isVisible())
        self.assertAlmostEqual(ov._lift, 0.0)          # reset only once the fade has finished
        self.assertIsNone(ov._card)

    def test_auto_hide_after_duration(self):
        ov = self.ov
        self.show_answer(ms=3000)
        self.assertEqual(ov._auto.interval(), 3000)
        ov._auto.timeout.emit()
        self.finish()
        self.assertFalse(ov.isVisible())

    def test_collapse_mid_rise_and_mid_swap(self):
        ov, O = self.ov, self.O
        ov._apply_answer(Q, A, 5000)
        ov._lanim.setCurrentTime(O.LIFT_IN_MS // 2)
        lift = ov._lift
        ov._apply_state("recording", "")
        self.assertAlmostEqual(ov._lift, lift)          # no jump when reversing
        self.finish()
        self.assertAlmostEqual(ov._lift, 0.0)
        self.show_answer()
        ov._apply_answer(Q, LONG, 5000)
        ov._xanim.setCurrentTime(ov._xanim.duration() // 2)
        ov._apply_state("processing", "")
        self.assertIsNone(ov._card_prev)
        self.finish()
        self.assertIsNone(ov._card)

    def test_long_answer_elides_within_max_height(self):
        ov, O = self.ov, self.O
        card = ov._layout_card(Q * 3, LONG * 2)
        self.assertLessEqual(len(card.lines), O.CARD_MAX_LINES)
        self.assertGreaterEqual(len(card.lines), 2)
        self.assertLessEqual(card.h, O.CARD_MAX_H)
        self.assertLessEqual(card.w, O.CARD_MAX_W)
        self.assertTrue(card.lines[-1].endswith("…"))
        self.assertTrue(card.q_line.endswith("…"))
        # the lifted pill + card still fit inside the fixed canvas
        self.assertGreaterEqual(O.REST_Y - (card.h + O.CARD_GAP), O.MARGIN - 0.5)

    def test_six_lines_fit_with_question_in_segoe(self):
        path = r"C:\Windows\Fonts\segoeui.ttf"
        if not os.path.exists(path):
            self.skipTest("Segoe UI not installed")
        from PySide6.QtGui import QFontDatabase, QFontInfo
        QFontDatabase.addApplicationFont(path)
        ov, O = self.ov, self.O
        if "Segoe UI" not in QFontInfo(ov._a_font).family():
            self.skipTest("Segoe UI not picked up offscreen")
        card = ov._layout_card(Q, LONG * 2)
        self.assertEqual(len(card.lines), O.CARD_MAX_LINES)
        self.assertLessEqual(card.h, O.CARD_MAX_H)

    def test_emoji_wrap_stays_inside_card(self):
        from PySide6.QtGui import QFontMetricsF
        ov, O = self.ov, self.O
        text = "Sure \U0001F600 " * 30 + "done"
        card = ov._layout_card(Q, text)
        fm = QFontMetricsF(ov._a_font)
        inner = card.w - 2 * O.CARD_PAD_X
        for ln in card.lines:
            self.assertLessEqual(fm.horizontalAdvance(ln), inner + 1.0, ln)
        self.assertNotIn("\ufffd", "".join(card.lines))

    def test_lone_surrogate_does_not_break_layout(self):
        card = self.ov._layout_card(Q, "bad \ud83d char")
        self.assertTrue(card.lines)

    def test_interrupted_swap_keeps_text_alpha(self):
        ov = self.ov
        self.show_answer()
        ov._apply_answer(Q, LONG, 5000)
        ov._xanim.setCurrentTime(int(ov._xanim.duration() * 0.375))
        old_a, new_a = ov._swap_alphas()
        visible = max(old_a, new_a)
        self.assertLess(visible, 0.2)
        ov._apply_state("recording", "")            # collapse freezes the card as drawn
        self.assertAlmostEqual(ov._card.text_a, visible, places=3)
        self.finish()
        self.show_answer()
        ov._apply_answer(Q, LONG, 5000)
        ov._xanim.setCurrentTime(int(ov._xanim.duration() * 0.375))
        ov._apply_answer(Q, "8 hours", 5000)        # replace mid-swap: outgoing text keeps its alpha
        self.assertAlmostEqual(ov._card_prev.text_a, visible, places=3)
        self.assertEqual(ov._card.text_a, 1.0)

    def test_text_alpha_reaches_the_paint(self):
        ov, O = self.ov, self.O
        self.show_answer()
        rect, _a = ov._card_geometry()
        dpr = ov.devicePixelRatioF()

        def text_energy():
            img = ov.grab().toImage()
            total = 0
            y0 = int((rect.top() + O.CARD_PAD_T) * dpr)
            y1 = int((rect.bottom() - O.CARD_PAD_B) * dpr)
            x0 = int((rect.left() + O.CARD_PAD_X) * dpr)
            x1 = int((rect.right() - O.CARD_PAD_X) * dpr)
            for y in range(y0, y1, 2):
                for x in range(x0, x1, 2):
                    total += img.pixelColor(x, y).lightness()
            return total

        ov._card.text_a = 1.0
        full = text_energy()
        ov._card.text_a = 0.0
        none = text_energy()
        self.assertGreater(full, none * 1.2)

    def test_show_answer_during_collapse_reverses(self):
        ov, O = self.ov, self.O
        self.show_answer()
        ov._apply_state("recording", "")
        ov._lanim.setCurrentTime(O.LIFT_OUT_MS // 2)
        lift, alpha = ov._lift, ov._card_a
        ov._apply_answer(Q, "8 hours", 5000)
        self.assertEqual(ov._lift_dir, "in")
        self.assertAlmostEqual(ov._lift, lift)       # no jump when reversing
        self.assertAlmostEqual(ov._card_a, alpha)
        self.finish()
        self.assertAlmostEqual(ov._lift, 1.0)
        self.assertEqual(ov._card.answer, "8 hours")
        self.assertEqual(ov._mode, "answer")

    def test_error_info_timers_and_recording_stops_answer_timer(self):
        ov, O = self.ov, self.O
        ov._apply_state("error", "Microphone not found")
        self.assertEqual(ov._auto.interval(), int(O.ERROR_SECS * 1000))
        self.assertTrue(ov._auto.isActive())
        ov._apply_state("info", "Paused")
        self.assertEqual(ov._auto.interval(), int(O.INFO_SECS * 1000))
        self.show_answer(ms=7000)
        self.assertTrue(ov._auto.isActive())
        ov._apply_state("recording", "")
        self.assertFalse(ov._auto.isActive())        # recording never auto-hides

    def test_short_answer_card_width_clamps(self):
        ov, O = self.ov, self.O
        card = ov._layout_card("", "42")
        self.assertEqual(card.lines, ["42"])
        self.assertGreaterEqual(card.w, O.CARD_MIN_W)
        self.assertEqual(card.q_line, "")

    def test_bare_answer_state_and_empty_answer(self):
        ov = self.ov
        ov._apply_answer(Q, "   ", 5000)
        self.assertFalse(ov.isVisible())
        ov._apply_state("answer", "Paris")
        self.finish()
        self.assertEqual(ov._state, "answer")
        self.assertEqual(ov._card.question, "")
        self.assertEqual(ov._card.answer, "Paris")

    def test_resting_pill_bottom_48px_above_work_area(self):
        from PySide6.QtCore import QRect
        ov, O = self.ov, self.O
        for av in (QRect(0, 0, 1920, 1032), QRect(-2560, 120, 2560, 1400), QRect(1920, 0, 1366, 728)):
            with self.subTest(av=av):
                ov._target_screen = lambda av=av: _Screen(av)
                ov._do_reposition()
                pill_bottom = ov.y() + O.REST_Y + O.PILL_H
                self.assertEqual(pill_bottom, av.y() + av.height() - O.BOTTOM_OFFSET)
                self.assertEqual(ov.x() + O.CANVAS_W / 2, av.x() + av.width() / 2)
                self.assertLessEqual(ov.y() + O.CANVAS_H, av.y() + av.height())

    def test_existing_states_still_paint(self):
        ov, O = self.ov, self.O
        for state, text in (("recording", ""), ("recording", "casual · Discord"), ("handsfree", ""),
                            ("processing", ""), ("processing", "Working…"), ("error", "Microphone not found"),
                            ("info", "Paused")):
            with self.subTest(state=state, text=text):
                ov._apply_state(state, text)
                self.finish()
                img = ov.grab().toImage()
                self.assertFalse(img.isNull())
                # the pill is painted at its resting place (bottom of the canvas)
                c = img.pixelColor(int(O.CANVAS_W / 2 * img.devicePixelRatio()),
                                   int((O.REST_Y + O.PILL_H / 2) * img.devicePixelRatio()))
                self.assertGreater(c.alpha(), 200)
                top = img.pixelColor(int(O.CANVAS_W / 2 * img.devicePixelRatio()), 4)
                self.assertEqual(top.alpha(), 0)       # canvas above the resting pill stays transparent


if __name__ == "__main__":
    unittest.main()
