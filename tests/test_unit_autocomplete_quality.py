"""Tab autocomplete quality / reliability: guess filter, chat-template prompt, font + placement of the ghost,
placeholder text, ghost watch tolerance, Backspace suppression lift, diagnostics. Offline (fakes, Qt offscreen)."""
import os
import time
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from localflow import autocomplete as A          # noqa: E402
from localflow import ghost as G                 # noqa: E402


# ----------------------------------------------------------------------------- filter
class Filter(unittest.TestCase):
    def pp(self, raw, before, after="", page=""):
        return A.postprocess_ex(raw, before, after, 80, page)

    def test_digit_garbage_from_notepad_report(self):
        out, why = self.pp("3.1.2.4.5.6.7.8.9.10.11.12.", "So basically DNS stands for ")
        self.assertIsNone(out)
        self.assertIn(why, ("few-letters", "digit-run"))
        self.assertIsNone(self.pp(" 3.1.2.1.2.1.", "So basically DNS stands for")[0])

    def test_scaffold_echo(self):
        self.assertEqual(self.pp(" Notepad. The user is typing:", "So basically")[1], "scaffold")
        self.assertEqual(self.pp(" works. Visible on screen:", "explain how DNS")[1], "scaffold")
        self.assertEqual(self.pp(" Text on screen: hi", "ok")[1], "scaffold")

    def test_good_guess_after_trailing_space(self):
        out, why = self.pp(" Domain Name System, which translates names. More", "So basically DNS stands for ")
        self.assertEqual((out, why), ("Domain Name System, which translates names.", ""))

    def test_number_runs(self):
        for raw in (" 1. 2. 3. 4.", " 2 3 4 5 6 7 8 9", " 1) 2) 3) go", " 1.2.3.4 is out"):
            self.assertIsNone(self.pp(raw, "the list:")[0], raw)

    def test_version_number_in_prose_kept(self):
        self.assertEqual(self.pp(" version 2.3.1 too.", "I see it on")[0], " version 2.3.1 too.")

    def test_lone_number_answers_kept(self):
        self.assertEqual(self.pp(" 10? I have a meeting", "Can we push the standup to")[0], " 10?")
        self.assertEqual(self.pp(" 5,880. The rest", "the totals for october are")[0], " 5,880.")
        self.assertEqual(self.pp(" 11:46 AM?", "see you at")[0], " 11:46 AM?")
        self.assertIsNone(self.pp(" 11/11", "Sam will")[0])

    def test_unbalanced_closers_trimmed(self):
        out = self.pp(' installer, then follow its instructions.")])])])', "To install the app, first open the")[0]
        self.assertEqual(out, " installer, then follow its instructions.")
        self.assertEqual(self.pp(' (see below)', "check the notes")[0], " (see below)")
        self.assertEqual(self.pp(' world"', 'he said "hello')[0], ' world"')

    def test_repeats(self):
        self.assertEqual(self.pp(" at a look at a look at", "yeah I can take a look")[1], "repeat")
        self.assertEqual(self.pp(" so so so so so", "nah i still havent")[1], "repeat")
        self.assertEqual(self.pp(" same as above. ok", "ok so basically the")[0], " same as above.")

    def test_normal_sentences_with_repeated_small_words_kept(self):
        for raw, before in ((" to go to the store to get milk.", "I need"),
                            (" the cat on the mat by the door.", "I saw"),
                            (" a hurry to get back to work and didn't have time to finish", "They were in")):
            self.assertIsNotNone(self.pp(raw, before)[0], raw)

    def test_word_join_after_trailing_space(self):
        self.assertEqual(self.pp("'d like help with anything else!", "Let me know if you ")[1], "word-join")
        self.assertEqual(self.pp("autiful, right?", "be ")[1], "word-join")
        self.assertEqual(self.pp(" Domain Name System.", "DNS stands for ")[0], "Domain Name System.")
        self.assertEqual(self.pp("ing the report.", "I am finish")[0], "ing the report.")    # mid-word: fine

    def test_no_break_space(self):
        self.assertEqual(self.pp("'d like", "you ")[1], "word-join")
        self.assertEqual(self.pp(" go for it", "I think we should")[0], " go for it")
        self.assertTrue(A.build_prompt("be ", "").endswith("<think></think>be"))

    def test_indented_new_line_is_not_a_join(self):
        self.assertEqual(self.pp("Second point.", "First point.\n  ")[0], "Second point.")

    def test_meta_answers_rejected(self):
        for raw in (" provide the rest of your text so I can continue it for you.", " finish what you were saying.",
                    " know that I am an LFM, a Liquid Foundation Model."):
            self.assertEqual(self.pp(raw, "Please")[1], "meta", raw)
        self.assertIsNotNone(self.pp(" thanks for your message.", "hey")[0])
        self.assertIsNotNone(self.pp(" they are most likely to be late.", "I think")[0])

    def test_tail_hash_normalised(self):
        self.assertEqual(A.tail_hash("hey aj will you"), A.tail_hash("hey aj will you\n"))
        self.assertEqual(A.tail_hash("hey aj will"), A.tail_hash("hey aj will﻿"))
        self.assertNotEqual(A.tail_hash("hey aj will"), A.tail_hash("hey aj will you"))

    def test_prompt_echo(self):
        self.assertEqual(self.pp(" few more words they are most likely to type.", "a")[1], "echo-prompt")

    def test_quoted_path_from_page_kept(self):
        path = "localflow/autocomplete.py:194_postprocess_filter_rules"
        out, why = self.pp(" " + path + " too", "please open", page="Changes:\n" + path + " filter")
        self.assertEqual(why, "")

    def test_odd_lead(self):
        for raw in (" | --- |", " ) and", " ## Title", " **bold** text"):
            self.assertEqual(self.pp(raw, "hello")[1], "odd-lead", raw)
        self.assertEqual(self.pp(" `filter` parameter works?", "can you make it so the")[0],
                         " `filter` parameter works?")

    def test_echo_of_page_and_field(self):
        page = "Add Reaction\nReply\nForward\nMore options for this message right here"
        out, why = self.pp(" add reaction reply forward more options for this", "hey", page=page)
        self.assertEqual((out, why), (None, "echo-page"))
        self.assertEqual(self.pp(" I will", "yeah tomorrow at 2 works, and I will")[1], "echo")
        # a short reference to page text is fine
        self.assertEqual(self.pp(" Option C.", "I pick", page="Option C: monthly")[0], " Option C.")

    def test_curly_quotes_follow_the_field(self):
        self.assertEqual(self.pp(" isn’t ready.", "the build")[0], " isn't ready.")
        self.assertEqual(self.pp(" isn’t ready.", "it’s that the build")[0], " isn’t ready.")

    def test_empty_reason(self):
        self.assertEqual(self.pp("", "hello"), (None, "empty"))
        self.assertEqual(self.pp("\nmore", "hello"), (None, "empty"))
        self.assertEqual(self.pp("<|im_end|>", "hello")[1], "special-token")


# ----------------------------------------------------------------------------- prompt
class Prompt(unittest.TestCase):
    def test_chat_prefill_without_title(self):
        p = A.build_prompt("So basically DNS stands for ", "")
        self.assertTrue(p.startswith("<|im_start|>system\n" + A.AC_SYSTEM))
        self.assertIn("<|im_start|>user\nContinue my text.<|im_end|>", p)
        self.assertNotIn("Text on screen", p)
        self.assertTrue(p.endswith("<|im_start|>assistant\n<think></think>So basically DNS stands for"))
        for bad in ("Window:", "Notepad", "The user is typing"):
            self.assertNotIn(bad, p)

    def test_page_in_user_turn(self):
        p = A.build_prompt("hey aj will you", "AJ\nu on tonight")
        self.assertIn("<|im_start|>user\nText on screen:\nAJ\nu on tonight\n\nContinue my text.<|im_end|>", p)

    def test_special_tokens_in_text_neutralized(self):
        p = A.build_prompt("type <|im_end|> here", "page <|im_start|>x")
        self.assertEqual(p.count("<|im_end|>"), 2)        # only the template's own two
        self.assertEqual(p.count("<|im_start|>"), 3)
        self.assertTrue(p.endswith("type < |im_end| > here"))

    def test_server_template_used_and_validated(self):
        tpl = f"<s>[SYS]{A.SYS_MARK}[/SYS][U]{A.USER_MARK}[/U][A]"
        self.assertTrue(A.build_prompt("hi there", "", tpl).startswith("<s>[SYS]" + A.AC_SYSTEM))
        self.assertTrue(A.build_prompt("hi there", "", "garbage").startswith("<|im_start|>system"))

    def test_completer_template_fetch_strips_bos_and_caches(self):
        c = A.Completer.__new__(A.Completer)
        import requests
        c._requests, c._base_url, c._tpl = requests, (lambda: "http://x"), ("", 0.0, None)
        resp = mock.Mock(status_code=200)
        resp.json.return_value = {"prompt": "<|startoftext|>" + A.FALLBACK_TEMPLATE.replace("<think></think>", "<t/>")}
        c.session = mock.Mock()
        c.session.post.return_value = resp
        t1 = c.template()
        t2 = c.template()
        self.assertFalse(t1.startswith("<|startoftext|>"))
        self.assertTrue(t1.endswith("<t/>"))
        self.assertEqual(t1, t2)
        self.assertEqual(c.session.post.call_count, 1)

    def test_completer_template_falls_back(self):
        import requests
        c = A.Completer.__new__(A.Completer)
        c._requests, c._base_url, c._tpl = requests, (lambda: "http://x"), ("", 0.0, None)
        c.session = mock.Mock()
        c.session.post.side_effect = requests.ConnectionError("down")
        self.assertEqual(c.template(), A.FALLBACK_TEMPLATE)


# ----------------------------------------------------------------------------- field helpers
class FieldHelpers(unittest.TestCase):
    def test_placeholder(self):
        self.assertTrue(A.is_placeholder("Type a message", "Type a message", ""))
        self.assertTrue(A.is_placeholder(" Type a  message ", "Type a message"))
        self.assertFalse(A.is_placeholder("Type a message hi", "Type a message"))
        self.assertFalse(A.is_placeholder("", "Type a message"))
        self.assertFalse(A.is_placeholder("hello", ""))

    def test_reader_blanks_placeholder_text(self):
        r = A.UiaReader.__new__(A.UiaReader)
        r.own_pid = os.getpid()
        r.U = mock.Mock(UIA_DocumentControlTypeId=50030)
        el = mock.Mock(CurrentProcessId=os.getpid() + 1, CurrentIsPassword=False, CurrentControlType=50004,
                       CurrentName="Type a message", CurrentHelpText="")
        el.GetRuntimeId.return_value = [1, 2]
        r.focused = lambda: el
        r._editable_multiline = lambda _e, _c, _h, fld: setattr(fld, "multiline", True)

        def uia(_el, fld):
            fld.before, fld.method, fld.caret = "Type a message", "uia", (1, 2, 20)
            return True
        r._uia_text = uia
        fld, why = r.read(1)
        self.assertEqual((fld.before, fld.after, fld.placeholder), ("", "", True))
        self.assertEqual(A.skip_reason(fld, 3), "too short")

    def test_parse_ia2_font(self):
        self.assertEqual(A.parse_ia2_font("font-family:Inter;font-size:12pt;font-weight:400"), (12.0, "Inter"))
        self.assertEqual(A.parse_ia2_font("font-size:16px;font-family:\"gg sans\""), (12.0, "gg sans"))
        self.assertEqual(A.parse_ia2_font(""), (0.0, ""))
        self.assertEqual(A.parse_ia2_font("font-size:0pt"), (0.0, ""))

    def test_refine_caret(self):
        # Notepad: 33 px caret rect, 17 px character box on the same line -> character top / height
        self.assertEqual(A.refine_caret((100, 200, 33), (92, 208, 8, 17)), (100, 208, 17))
        # no caret rect: right edge of the previous character
        self.assertEqual(A.refine_caret(None, (92, 208, 8, 17)), (100, 208, 17))
        # bogus 2 px rects (Chromium) are ignored
        self.assertIsNone(A.refine_caret((1045, 900, 2), (1043, 900, 2, 2)))
        self.assertEqual(A.refine_caret((369, 1339, 25), (360, 1339, 9, 25)), (369, 1339, 25))
        # previous character on another line: caret rect kept
        self.assertEqual(A.refine_caret((10, 300, 20), (500, 280, 8, 18)), (10, 300, 20))

    def test_caret_moved(self):
        self.assertFalse(A.caret_moved((100, 200, 18), (103, 202, 18)))
        self.assertFalse(A.caret_moved((100, 200, 18), (106, 200, 18)))
        self.assertTrue(A.caret_moved((100, 200, 18), (108, 200, 18)))
        self.assertTrue(A.caret_moved((100, 200, 18), (100, 220, 18)))


class DocEndCaret(unittest.TestCase):
    """Chromium: caret at the end of the text compares one step past the DocumentRange end (T3 Code, Discord)."""

    def run_read(self, between):
        from test_unit_autocomplete import _El, _Range, _TP, _reader
        page, field = "Page text. ", "hey aj will you"
        text = page + field + between + "next element"
        s0, s1 = len(page), len(page) + len(field)
        c = s1 + len(between)
        rects = {c - 1: [300.0, 200.0, 1.0, 21.0]}
        fld = A.Field()
        ok = _reader()._uia_text(_El(_TP(_Range(text, s0, s1, rects), _Range(text, c, c, rects))), fld)
        return ok, fld

    def test_line_break_gap_reads_from_document_end(self):
        ok, fld = self.run_read("\n")
        self.assertTrue(ok)
        self.assertEqual((fld.before, fld.after), ("hey aj will you", ""))
        self.assertIn("docend", fld.method)
        fld.multiline = True
        self.assertEqual(A.skip_reason(fld, 3), "")

    def test_text_gap_means_caret_elsewhere(self):
        ok, _fld = self.run_read(" other field text ")
        self.assertFalse(ok)


# ----------------------------------------------------------------------------- watch tolerance
class WatchVerdict(unittest.TestCase):
    def ghost(self, src="uia2"):
        return A.Ghost(token=1, hwnd=1, key_count=1, text=" x", supp_key=(), before_hash=A.tail_hash("hello"),
                       el_id=(1,), caret=(100, 200, 18), field_rect=None, shown_at=10.0, caret_src=src)

    def fld(self, caret=(100, 200, 18), method="uia2", before="hello", el=(1,)):
        return A.Field(before=before, caret=caret, method=method, el_id=el)

    def test_one_failed_read_tolerated(self):
        g = self.ghost()
        self.assertEqual(A.watch_verdict(g, None, "no focus", 11.0), "")
        self.assertEqual(A.watch_verdict(g, self.fld(), "", 11.3), "")          # recovered: counter reset
        self.assertEqual(A.watch_verdict(g, None, "timeout", 11.6), "")
        self.assertIn("re-read failed", A.watch_verdict(g, None, "timeout", 11.9))

    def test_missing_rect_counts_as_miss(self):
        g = self.ghost()
        self.assertEqual(A.watch_verdict(g, self.fld(caret=None), "", 11.0), "")
        self.assertIn("no rect", A.watch_verdict(g, self.fld(caret=None), "", 11.3))

    def test_jitter_and_other_source_ignored(self):
        g = self.ghost()
        self.assertEqual(A.watch_verdict(g, self.fld(caret=(104, 201, 18)), "", 11.0), "")
        self.assertEqual(A.watch_verdict(g, self.fld(caret=(140, 230, 18), method="uia+caretfallback"), "", 11.0),
                         "")

    def test_real_move_after_grace_hides(self):
        g = self.ghost()
        self.assertEqual(A.watch_verdict(g, self.fld(caret=(140, 200, 18)), "", 10.1), "")    # grace period
        self.assertIn("moved", A.watch_verdict(g, self.fld(caret=(140, 200, 18)), "", 10.5))

    def test_text_or_element_change_hides(self):
        g = self.ghost()
        self.assertIn("text changed", A.watch_verdict(g, self.fld(before="hello!"), "", 10.1))
        self.assertIn("other element", A.watch_verdict(self.ghost(), self.fld(el=(2,)), "", 10.1))


# ----------------------------------------------------------------------------- suppression lift
class SuppressionLift(unittest.TestCase):
    def test_lifts_after_eight_words(self):
        key = A.sentence_key(1, (1,), "hey aj will you")
        s = [key]
        self.assertTrue(A.sentence_suppressed(s, 1, (1,), "hey aj will you come"))
        self.assertTrue(A.sentence_suppressed(s, 1, (1,), "hey aj will you come on tonight or tomorrow"))
        self.assertFalse(A.sentence_suppressed(
            s, 1, (1,), "hey aj will you come on tonight or tomorrow i think we can play"))

    def test_lifts_at_sentence_end(self):
        s = [A.sentence_key(1, (1,), "hey aj will you")]
        self.assertFalse(A.sentence_suppressed(s, 1, (1,), "hey aj will you come? ok"))

    def test_old_four_part_keys_still_match(self):
        s = [A.sentence_key(1, (1,), "hey aj will you")[:4]]
        self.assertTrue(A.sentence_suppressed(s, 1, (1,), "hey aj will you come on tonight or tomorrow i think we"))

    def test_prune_drops_lifted_key(self):
        ss = A.SuppressionSet()
        ss.add(A.sentence_key(1, (1,), "hey aj will you"))
        ss.prune(1, (1,), "hey aj will you come on tonight or tomorrow i think we can play")
        self.assertEqual(len(ss), 0)


# ----------------------------------------------------------------------------- ghost font + layout
class GhostFont(unittest.TestCase):
    def test_font_px_from_points(self):
        self.assertEqual(G.ghost_font_px(25, 12.0), (16, True))          # Discord: 12 pt in a 25 px line box
        self.assertEqual(G.ghost_font_px(33, 11.0), (15, True))          # Notepad 11 pt Consolas, 33 px rect
        self.assertEqual(G.ghost_font_px(20, 30.0), (G.font_px(20), False))   # bigger than the line: ignored
        self.assertEqual(G.ghost_font_px(60, 10.0), (G.font_px(60), False))   # tiny vs the rect: ignored
        self.assertEqual(G.ghost_font_px(25, 0.0), (G.font_px(25), False))

    def test_pick_family(self):
        self.assertEqual(G.pick_family("Inter", {"Inter", "Segoe UI"}), "Inter")
        self.assertEqual(G.pick_family("Inter, sans-serif", {"inter"}), "inter")
        self.assertEqual(G.pick_family("gg sans", {"Segoe UI"}), "Segoe UI")
        self.assertEqual(G.pick_family("Consolas", set()), "Consolas")
        self.assertEqual(G.pick_family("'Fira Code', monospace", set()), "Consolas")
        self.assertEqual(G.pick_family("", {"Inter"}), "Segoe UI")

    def test_first_word(self):
        self.assertEqual(G.first_word(" hello there"), " hello")
        self.assertEqual(G.first_word("ing more"), "ing")
        self.assertEqual(G.first_word("   "), "")


class GhostLayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.g = G.GhostOverlay(screens=lambda: [(0, 0, 1280, 720, 1.5)])
        self.lst = mock.Mock()
        self.lst.ghost_shown.return_value = True
        self.g.set_listener(self.lst)

    def tearDown(self):
        self.g.hide()
        self.g.deleteLater()

    def pump(self):
        for _ in range(5):
            self.app.processEvents()

    def test_field_font_size_used(self):
        self.g.show_ghost(1, " see you", (300, 150, 30), None, 5.0, (12.0, "Segoe UI"))   # 20 px logical box
        self.pump()
        self.assertEqual(self.g._font.pixelSize(), 16)
        self.g.show_ghost(2, " see you", (300, 150, 30), None, 5.0)                       # unknown: 0.72 x box
        self.pump()
        self.assertEqual(self.g._font.pixelSize(), G.font_px(20))

    def test_tall_line_box_bottom_aligned(self):
        lay = self.g.layout_for(" see you", (300, 150, 66), None, (12.0, ""))    # 44 px logical box, 16 px font
        x, y, w, h, font, shown = lay
        self.assertEqual(font.pixelSize(), 16)
        self.assertLess(h, 30)
        self.assertEqual(y + h, round(100 + 44))
        lay = self.g.layout_for(" see you", (300, 150, 30), None, (12.0, ""))    # normal box: centred, full height
        self.assertEqual((lay[1], lay[3]), (100, 20 if lay[3] == 20 else lay[3]))
        self.assertGreaterEqual(lay[3], 20)

    def test_no_room_in_field_shows_first_word(self):
        # caret 4 logical px left of the field's right edge: the first word goes past it (the app will wrap)
        self.g.show_ghost(3, " hello there friend", (594, 150, 30), (0, 140, 600, 190), 5.0)
        self.pump()
        self.lst.ghost_shown.assert_called_once()
        self.assertEqual(self.lst.ghost_shown.call_args[0][1], " hello")

    def test_no_room_on_monitor_fails_with_reason(self):
        self.g.show_ghost(4, " hello", (1919, 150, 30), None, 5.0)
        self.pump()
        self.lst.ghost_failed.assert_called_once()
        self.assertIn("no room", self.lst.ghost_failed.call_args[0][1])


# ----------------------------------------------------------------------------- diagnostics
class Diagnostics(unittest.TestCase):
    class Cfg:
        def __init__(self, **d):
            self.d = {f"autocomplete.{k}": v for k, v in d.items()}

        def get(self, k, default=None):
            return self.d.get(k, default)

    def ac(self, **cfg):
        return A.Autocomplete(self.Cfg(**cfg), host=mock.Mock(), ghost=None, reader_factory=lambda: None,
                              completer_factory=lambda: None, typer=lambda s: None, tapper=lambda v: None)

    def test_rate_limited_once_a_minute(self):
        ac = self.ac()
        with self.assertLogs("localflow.autocomplete", "DEBUG") as cm:
            ac._diag("t3.exe", "mid-line via uia/aria")
            ac._diag("t3.exe", "mid-line via uia/aria")
            ac._diag("t3.exe", "too short via uia/aria")
        infos = [r for r in cm.records if r.levelname == "INFO"]
        self.assertEqual(len(infos), 2)
        with mock.patch.object(A.time, "monotonic", return_value=time.monotonic() + 61):
            with self.assertLogs("localflow.autocomplete", "INFO"):
                ac._diag("t3.exe", "mid-line via uia/aria")

    def test_debug_log_logs_every_time(self):
        ac = self.ac(debug_log=True)
        with self.assertLogs("localflow.autocomplete", "INFO") as cm:
            for _ in range(3):
                ac._diag("t3.exe", "mid-line via uia/aria")
        self.assertEqual(len(cm.records), 3)

    def test_layout_failure_logged_with_reason(self):
        ac = self.ac()
        ac._pending = A.Ghost(token=5, hwnd=1, key_count=0, text=" x", supp_key=(), before_hash="", el_id=None,
                              caret=(0, 0, 10), field_rect=None, exe="discord.exe")
        with self.assertLogs("localflow.autocomplete", "INFO") as cm:
            ac.ghost_failed(5, "no room (3 px left)")
        self.assertIn("no room (3 px left)", cm.output[0])
        self.assertIn("discord.exe", cm.output[0])
        self.assertIsNone(ac._pending)


if __name__ == "__main__":
    unittest.main()
