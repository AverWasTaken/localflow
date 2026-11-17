"""Tab autocomplete (autocomplete.py, ghost.py, hotkey wiring, config, server slots). Offline: UIA / IA2 / Win32 /
HTTP are faked; Qt runs on the offscreen platform."""
import os
import threading
import time
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from localflow import autocomplete as A          # noqa: E402
from localflow import ghost as G                 # noqa: E402
from localflow import winutil as w               # noqa: E402


# ----------------------------------------------------------------------------- pure helpers
class PostProcess(unittest.TestCase):
    def p(self, raw, before, after="", n=80):
        return A.postprocess(raw, before, after, n)

    def test_cuts_after_first_sentence(self):
        self.assertEqual(self.p(" see you at 2. Then we can", "Sounds good, I will"), " see you at 2.")

    def test_newline_ends_the_guess(self):
        self.assertEqual(self.p(" and then\nmore stuff", "I went"), " and then")

    def test_space_already_typed(self):
        self.assertEqual(self.p("  and then", "I went "), "and then")

    def test_space_added_after_punctuation(self):
        self.assertEqual(self.p("see you", "Thanks,"), " see you")

    def test_mid_word_kept(self):
        self.assertEqual(self.p("ing the report.", "I am finish"), "ing the report.")

    def test_garbage_and_empty_dropped(self):
        for raw in ("", "   ", "<think>", "<|im_end|>", "...", "!!!", " Me: hi", "\t\n"):
            self.assertIsNone(self.p(raw, "hello there"), raw)

    def test_echo_dropped(self):
        self.assertIsNone(self.p(" I will", "yeah tomorrow at 2 works, and I will"))
        self.assertIsNone(self.p(" the meeting", "so about", after=" the meeting tomorrow"))

    def test_repetition_loop_dropped(self):
        self.assertIsNone(self.p(" word word word word word", "ok so"))

    def test_length_cap_on_word_boundary(self):
        out = self.p(" alpha beta gamma delta epsilon zeta eta theta", "ok so", n=20)
        self.assertLessEqual(len(out), 20)
        self.assertFalse(out.endswith(" "))
        self.assertTrue(" alpha beta gamma delta epsilon".startswith(out))

    def test_control_chars_stripped(self):
        out = self.p(" a​b\x07c d‮e", "hey")
        self.assertEqual(out, " abc de")


class SentenceSuppression(unittest.TestCase):
    def test_current_sentence(self):
        self.assertEqual(A.current_sentence("Hi there. How are you? I was wondering if"), "I was wondering if")
        self.assertEqual(A.current_sentence("line one\nsecond line"), "second line")
        self.assertEqual(A.current_sentence("no boundary yet"), "no boundary yet")

    def test_same_sentence_suppressed_until_next_sentence(self):
        s = A.SuppressionSet()
        s.add(A.sentence_key(5, (1, 2), "Hi. I was wondering if"))
        self.assertTrue(A.sentence_suppressed(s, 5, (1, 2), "Hi. I was wondering if we could"))
        self.assertFalse(A.sentence_suppressed(s, 5, (1, 2), "Hi. I was wondering if we could. Next"))
        self.assertFalse(A.sentence_suppressed(s, 6, (1, 2), "Hi. I was wondering if we could"))   # other window
        self.assertFalse(A.sentence_suppressed(s, 5, (9,), "Hi. I was wondering if we could"))     # other field
        self.assertTrue(A.sentence_suppressed(s, 5, (1, 2), "Hi. I was"))       # edited back: same sentence
        self.assertFalse(A.sentence_suppressed(s, 5, (1, 2), "Hi. Something else"))   # rewritten start

    def test_key_holds_no_text(self):
        k = A.sentence_key(5, None, "Done. Secret plans for tomorrow")
        self.assertNotIn("Secret", repr(k))
        self.assertNotIn("Done", repr(k))
        self.assertEqual(len(k[3]), 8)

    def test_edit_end_of_dismissed_sentence_keeps_it_dismissed(self):
        s = A.SuppressionSet()
        s.add(A.sentence_key(5, (1,), "All good. Canada trip is on"))
        for b in ("All good. Canada trip is o", "All good. Canada trip is off", "All good. Can"):
            s.prune(5, (1,), b)
            self.assertTrue(A.sentence_suppressed(s, 5, (1,), b), b)
        self.assertEqual(len(s), 1)

    def test_dismiss_right_after_a_sentence_end_covers_the_next_sentence(self):
        for at in ("All good.", "All good. "):
            s = A.SuppressionSet()
            s.add(A.sentence_key(5, None, at))
            for b in ("All good. ", "All good. Can", "All good. Can we meet"):
                s.prune(5, None, b)
                self.assertTrue(A.sentence_suppressed(s, 5, None, b), (at, b))
            s.prune(5, None, "All good. Can we meet? Sure")       # that sentence ended: released
            self.assertEqual(len(s), 0)
            self.assertFalse(A.sentence_suppressed(s, 5, None, "All good. Can we meet? Sure"))

    def test_prune_drops_entry_once_the_sentence_moves_on(self):
        s = A.SuppressionSet()
        s.add(A.sentence_key(5, (1,), "All good. Can"))
        s.add(A.sentence_key(6, (1,), "Other window"))
        s.prune(5, (1,), "All good. Canada trip is on")         # same sentence, kept
        self.assertTrue(A.sentence_suppressed(s, 5, (1,), "All good. Canada trip is on"))
        s.prune(5, (1,), "All good. Canada trip is on. Thanks")  # moved on: dropped
        self.assertFalse(A.sentence_suppressed(s, 5, (1,), "All good. Canada trip. Can we"))
        self.assertEqual(len(s), 1)                              # the other window's entry is untouched

    def test_entries_expire(self):
        s = A.SuppressionSet(max_age_s=60)
        s.add(A.sentence_key(5, None, "Hello"))
        s.prune(9, None, "x", now=time.monotonic() + 61)
        self.assertEqual(len(s), 0)

    def test_set_is_bounded(self):
        s = A.SuppressionSet(cap=3)
        for i in range(10):
            s.add((i, None, 1, "x"))
        self.assertEqual(len(s), 3)


class Gating(unittest.TestCase):
    def fld(self, **kw):
        base = dict(before="Hello there, I", after="", caret=(10, 10, 20), multiline=True)
        base.update(kw)
        return A.Field(**base)

    def test_go(self):
        self.assertEqual(A.skip_reason(self.fld(), 3), "")
        self.assertEqual(A.skip_reason(self.fld(after="   \nnext line"), 3), "")

    def test_reasons(self):
        cases = {"password": dict(password=True), "read-only": dict(editable=False),
                 "single-line": dict(multiline=False), "selection": dict(selection=True),
                 "mid-line": dict(after=" more text"), "too short": dict(before="hi"),
                 "line start": dict(before="Hello\n"), "no caret rect": dict(caret=None),
                 "picker": dict(before="thanks @joh")}
        for why, kw in cases.items():
            self.assertEqual(A.skip_reason(self.fld(**kw), 3), why)
        self.assertEqual(A.skip_reason(self.fld(multiline=None), 3), "single-line")   # unknown = no
        for b in ("hey #gen", "nice :smi", "/sha", "thanks @John Sm"):
            self.assertEqual(A.skip_reason(self.fld(before=b), 3), "picker", b)
        for b in ("mail me at a@b", "see http://x", "at 10:30", "and/or", "thanks @John Smith for"):
            self.assertEqual(A.skip_reason(self.fld(before=b), 3), "", b)

    def test_excludes_and_private(self):
        pats = ["code.exe", "keepass*.exe", "1password.exe"]
        self.assertTrue(A.app_excluded("KeePass.exe", pats))
        self.assertTrue(A.app_excluded("keepassxc.exe", pats))
        self.assertTrue(A.app_excluded("code.exe", pats))
        self.assertFalse(A.app_excluded("ms-teams.exe", pats))
        self.assertFalse(A.app_excluded("", pats))
        for t in ("New Tab - InPrivate", "Google - Incognito", "Zen Browser Private Browsing"):
            self.assertTrue(A.is_private_title(t))
        self.assertFalse(A.is_private_title("Chat | Sam | Microsoft Teams"))

    def test_aria_multiline(self):
        self.assertTrue(A.aria_multiline("expanded=false;multiline=true;required=false"))
        self.assertFalse(A.aria_multiline("multiline=false"))
        self.assertIsNone(A.aria_multiline(""))

    def test_trim_page_drops_field_text_keeps_bottom(self):
        page = "\n".join(f"Sam: message {i}" for i in range(400)) + "\nhey can you send"
        out = A.trim_page(page, "hey can you send", 300)
        self.assertLessEqual(len(out), 300)
        self.assertNotIn("hey can you send", out)
        self.assertIn("message 399", out)

    def test_prompt_ends_with_field_text(self):
        p = A.build_prompt("hey can you", "Sam: meeting?")
        self.assertTrue(p.endswith("<|im_start|>assistant\n<think></think>hey can you"))
        self.assertIn("Sam: meeting?", p)
        self.assertNotIn("The user is typing", p)


class SwallowDecision(unittest.TestCase):
    def ghost(self, **kw):
        g = A.Ghost(token=1, hwnd=100, key_count=7, text=" next words", supp_key=(100, None, 0, "x"),
                    before_hash="h", el_id=None, caret=(1, 1, 20), field_rect=None, shown_at=10.0, expires_at=18.0)
        for k, v in kw.items():
            setattr(g, k, v)
        return g

    def test_all_conditions(self):
        g = self.ghost()
        ok = dict(vk=A.VK_TAB, key_count=7, fg_hwnd=100, now=10.5, min_visible_s=0.25)
        self.assertTrue(A.swallow_decision(g, **ok))
        self.assertTrue(A.swallow_decision(g, **{**ok, "vk": A.VK_BACK}))
        self.assertFalse(A.swallow_decision(None, **ok))
        self.assertFalse(A.swallow_decision(g, **{**ok, "vk": 0x41}))
        self.assertFalse(A.swallow_decision(g, **{**ok, "key_count": 8}))       # typed since
        self.assertFalse(A.swallow_decision(g, **{**ok, "fg_hwnd": 101}))       # other window
        self.assertFalse(A.swallow_decision(g, **{**ok, "now": 10.1}))          # visible < 250 ms
        self.assertFalse(A.swallow_decision(g, **{**ok, "now": 18.5}))          # past TTL
        self.assertFalse(A.swallow_decision(self.ghost(shown_at=0.0), **ok))    # not on screen yet
        self.assertFalse(A.swallow_decision(self.ghost(text=""), **ok))


# ----------------------------------------------------------------------------- reader (faked UIA)
class _U:
    TextPatternRangeEndpoint_Start, TextPatternRangeEndpoint_End = 0, 1
    TextUnit_Character = 0
    UIA_TextPatternId, UIA_TextPattern2Id, UIA_ValuePatternId, UIA_LegacyIAccessiblePatternId = 1, 2, 3, 4
    UIA_EditControlTypeId, UIA_DocumentControlTypeId = 50004, 50030
    IUIAutomationTextPattern = IUIAutomationTextPattern2 = IUIAutomationValuePattern = object


class _Range:
    """Text range over a shared string; endpoints may leave the field (like Gecko)."""

    def __init__(self, text, s, e, rects=None):
        self.t, self.s, self.e, self.rects = text, s, e, rects or {}

    def Clone(self):
        return _Range(self.t, self.s, self.e, self.rects)

    def _get(self, ep):
        return self.s if ep == 0 else self.e

    def _set(self, ep, v):
        v = max(0, min(len(self.t), v))
        if ep == 0:
            self.s = v
            self.e = max(self.e, v)
        else:
            self.e = v
            self.s = min(self.s, v)

    def MoveEndpointByRange(self, ep, other, oep):
        self._set(ep, other._get(oep))

    def MoveEndpointByUnit(self, ep, unit, n):
        old = self._get(ep)
        self._set(ep, old + n)
        return self._get(ep) - old

    def CompareEndpoints(self, ep, other, oep):
        return self._get(ep) - other._get(oep)

    def GetText(self, n):
        return self.t[self.s:self.e]

    def GetBoundingRectangles(self):
        if self.e - self.s == 1:
            return self.rects.get(self.s, [])
        return []


class _Sel:
    def __init__(self, r):
        self.r = r
        self.Length = 1

    def GetElement(self, i):
        return self.r


class _TP:
    def __init__(self, doc, caret):
        self.DocumentRange, self.caret = doc, caret

    def QueryInterface(self, _):
        return self

    def GetSelection(self):
        return _Sel(self.caret)


class _El:
    def __init__(self, tp):
        self.tp = tp

    def GetCurrentPattern(self, pid):
        return self.tp if pid == _U.UIA_TextPatternId else None


def _reader():
    r = A.UiaReader.__new__(A.UiaReader)
    r.U, r.before_chars, r.after_chars, r.own_pid = _U, 1500, 300, os.getpid()
    return r


class ReaderText(unittest.TestCase):
    PAGE = "Lots of page text before the box. "
    FIELD = "Hello Sam, I think we"

    def setup(self, caret_off):
        text = self.PAGE + self.FIELD + " " + "page text after"
        s0, s1 = len(self.PAGE), len(self.PAGE) + len(self.FIELD)
        c = s0 + caret_off
        rects = {c - 1: [100.0, 200.0, 8.0, 18.0]}
        doc = _Range(text, s0, s1, rects)
        caret = _Range(text, c, c, rects)
        return _El(_TP(doc, caret))

    def test_ranges_clamped_to_field(self):
        fld = A.Field()
        self.assertTrue(_reader()._uia_text(self.setup(len(self.FIELD)), fld))
        self.assertEqual(fld.before, self.FIELD)        # no page text before the box
        self.assertEqual(fld.after, "")                 # nor after it
        self.assertEqual(fld.caret, (108, 200, 18))     # right edge of the previous character
        self.assertFalse(fld.selection)

    def test_caret_at_field_start_has_no_rect(self):
        fld = A.Field()
        _reader()._uia_text(self.setup(0), fld)
        self.assertEqual(fld.before, "")
        self.assertIsNone(fld.caret)

    def test_mid_text(self):
        fld = A.Field()
        _reader()._uia_text(self.setup(5), fld)
        self.assertEqual(fld.before, "Hello")
        self.assertEqual(fld.after, self.FIELD[5:])
        self.assertTrue(A.caret_mid_line(fld.after))


class ReaderFallbackOrder(unittest.TestCase):
    def make(self, uia_ok, ia2_ok, caret_from=None, pid=None):
        r = _reader()
        calls = []
        el = mock.Mock()
        el.CurrentProcessId = pid if pid is not None else os.getpid() + 1
        el.CurrentIsPassword = False
        el.CurrentControlType = _U.UIA_EditControlTypeId
        el.GetRuntimeId.return_value = [42, 1]
        r.focused = lambda: el

        def editable(_el, _ct, _h, fld):
            fld.multiline, fld.ml_method = True, "aria"
        r._editable_multiline = editable

        def uia(_el, fld):
            calls.append("uia")
            if uia_ok:
                fld.before, fld.method = "Hello there", "uia"
                fld.caret = (1, 2, 3) if caret_from == "uia" else None
            return uia_ok

        def ia2(_h, fld):
            calls.append("ia2")
            if ia2_ok:
                fld.before, fld.method = "Hello there", "ia2"
            return ia2_ok

        r._uia_text, r._ia2_text = uia, ia2
        r._msaa_caret = lambda h: (calls.append("msaa"), (5, 5, 20) if caret_from == "msaa" else None)[1]
        return r, calls

    def test_uia_first(self):
        r, calls = self.make(True, True, "uia")
        fld, why = r.read(1)
        self.assertEqual((why, fld.method, calls), ("", "uia", ["uia"]))
        self.assertEqual(fld.el_id, (42, 1))

    def test_ia2_when_uia_fails(self):
        r, calls = self.make(False, True, "msaa")
        fld, why = r.read(1)
        self.assertEqual(calls, ["uia", "ia2", "msaa"])
        self.assertEqual(fld.caret, (5, 5, 20))
        self.assertIn("ia2", fld.method)

    def test_gui_thread_caret_last(self):
        r, calls = self.make(True, False, None)
        with mock.patch.object(A, "gui_thread_caret", return_value=(9, 9, 16)) as gti:
            fld, why = r.read(1)
        self.assertEqual(calls, ["uia", "msaa"])
        gti.assert_called_once()
        self.assertEqual(fld.caret, (9, 9, 16))

    def test_no_text_access(self):
        r, calls = self.make(False, False)
        fld, why = r.read(1)
        self.assertEqual(why, "no text access")

    def test_own_process_never_read(self):
        r, calls = self.make(True, True, "uia", pid=os.getpid())
        fld, why = r.read(1)
        self.assertIsNone(fld)
        self.assertEqual(why, "own process")
        self.assertEqual(calls, [])

    def test_password_stops_before_text(self):
        r, calls = self.make(True, True, "uia")
        r.focused().CurrentIsPassword = True
        fld, why = r.read(1)
        self.assertTrue(fld.password)
        self.assertEqual(calls, [])
        self.assertEqual(A.skip_reason(fld, 3), "password")


class PageContextOrder(unittest.TestCase):
    def reader(self, doc_tp):
        r = _reader()
        root = mock.Mock()
        root.GetCurrentPattern.return_value = doc_tp
        r._document_of = lambda el: root if doc_tp is not None else None
        r._names_text = mock.Mock(return_value="Name text A\nName text B")
        return r

    def test_field_is_document(self):
        r = self.reader(None)
        out = r.page_text(1, A.Field(page_root=object(), focus_el=object()), 2000)
        self.assertEqual(out, ("", "field is the document"))
        r._names_text.assert_not_called()

    def test_before_field_first(self):
        text = "Sam: are we on for lunch?\nMe: yes\n" + "Hello Sam"
        start = len(text) - len("Hello Sam")
        tp = mock.Mock()
        tp.QueryInterface.return_value = tp
        tp.RangeFromChild.return_value = _Range(text, start, len(text))
        r = self.reader(tp)
        out, method = r.page_text(1, A.Field(focus_el=object(), before="Hello Sam"), 2000)
        self.assertEqual(method, "before-field")
        self.assertIn("are we on for lunch?", out)
        tp.GetVisibleRanges.assert_not_called()
        r._names_text.assert_not_called()

    def test_visible_ranges_when_before_field_fails(self):
        tp = mock.Mock()
        tp.QueryInterface.return_value = tp
        tp.RangeFromChild.side_effect = OSError("no")
        rng = mock.Mock()
        rng.GetText.return_value = "Visible message"
        tp.GetVisibleRanges.return_value = mock.Mock(Length=1, GetElement=lambda i: rng)
        r = self.reader(tp)
        out, method = r.page_text(1, A.Field(focus_el=object()), 2000)
        self.assertEqual((out, method), ("Visible message", "visible-ranges"))
        r._names_text.assert_not_called()           # never the slow tree scan inside a web document

    def test_names_only_without_document(self):
        r = self.reader(None)
        out, method = r.page_text(1, A.Field(focus_el=object()), 2000)
        self.assertEqual(method, "names")
        self.assertIn("Name text B", out)


# ----------------------------------------------------------------------------- controller with fakes
class _Host:
    def __init__(self):
        self.blocked = ""

    def ac_blocked(self):
        return self.blocked

    def ac_server(self):
        return True, "http://127.0.0.1:1"


class _Cfg:
    def __init__(self, **over):
        self.d = {"autocomplete.debounce_ms": 20, "autocomplete.min_visible_ms": 30, "autocomplete.exclude_apps": []}
        self.d.update({f"autocomplete.{k}": v for k, v in over.items()})

    def get(self, k, default=None):
        return self.d.get(k, default)


class _FakeReader:
    def __init__(self):
        self.field = A.Field(before="Hello Sam, I think we", after="", caret=(100, 200, 18), multiline=True,
                             el_id=(1, 2), field_rect=(0, 190, 800, 230), method="uia", ml_method="aria")
        self.reads = 0

    def read(self, hwnd):
        self.reads += 1
        import copy
        return copy.copy(self.field), ""

    def page_text(self, hwnd, fld, cap):
        return "Sam: lunch tomorrow?", "visible-ranges"


class _FakeCompleter:
    def __init__(self, out=" should meet at noon. Also"):
        self.out, self.prompts = out, []

    def complete(self, prompt, **kw):
        self.prompts.append(prompt)
        return self.out

    def close(self):
        pass


class _FakeGhost:
    def __init__(self):
        self.listener, self.shown, self.hidden = None, [], []

    def set_listener(self, l):
        self.listener = l

    def show_ghost(self, token, text, caret, field_rect, ttl, font=None):
        self.shown.append((token, text))
        self.fonts = getattr(self, "fonts", []) + [font]
        self.listener.ghost_shown(token, text, ttl)

    def hide_ghost(self, token=0):
        self.hidden.append(token)


class Controller(unittest.TestCase):
    HWND = 4242

    def setUp(self):
        patches = [mock.patch.object(A, "_fg_hwnd", side_effect=lambda: self.fg),
                   mock.patch.object(A, "_window_pid", return_value=1),
                   mock.patch.object(A, "_mouse_down", return_value=False),
                   mock.patch.object(A.w, "window_info", return_value=("ms-teams.exe", "Chat | Sam"))]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.fg = self.HWND
        self.reader, self.comp, self.ghost = _FakeReader(), _FakeCompleter(), _FakeGhost()
        self.typed, self.tapped = [], []
        self.host = _Host()
        self.ac = A.Autocomplete(_Cfg(), self.host, ghost=self.ghost, reader_factory=lambda: self.reader,
                                 completer_factory=lambda: self.comp, typer=self.typed.append,
                                 tapper=self.tapped.append)
        self.ac.start()
        self.addCleanup(self.ac.stop)

    def wait_for(self, cond, timeout=2.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if cond():
                return True
            time.sleep(0.01)
        return False

    def type_and_wait_ghost(self):
        n = len(self.ghost.shown)
        self.ac.note_key(0x41)
        self.assertTrue(self.wait_for(lambda: len(self.ghost.shown) > n and self.ac._ghost is not None))
        time.sleep(0.05)        # past min_visible_ms

    def test_guess_shown_after_pause(self):
        self.type_and_wait_ghost()
        self.assertEqual(self.ghost.shown[-1][1], " should meet at noon.")
        self.assertIn("Sam: lunch tomorrow?", self.comp.prompts[-1])
        self.assertTrue(self.comp.prompts[-1].endswith("Hello Sam, I think we"))

    def test_tab_accepts(self):
        self.type_and_wait_ghost()
        self.assertTrue(self.ac.want_swallow(A.VK_TAB))
        self.assertTrue(self.wait_for(lambda: self.typed))
        self.assertEqual(self.typed, [" should meet at noon."])
        self.assertEqual(self.tapped, [])
        n = len(self.ghost.shown)                       # chained: the next guess comes without a key press
        self.assertTrue(self.wait_for(lambda: len(self.ghost.shown) > n))

    def test_tab_replayed_when_text_changed(self):
        self.type_and_wait_ghost()
        self.assertTrue(self.ac.want_swallow(A.VK_TAB))
        self.reader.field.before = "something else entirely"
        self.assertTrue(self.wait_for(lambda: self.tapped))
        self.assertEqual(self.tapped, [A.VK_TAB])
        self.assertEqual(self.typed, [])

    def test_backspace_dismisses_and_suppresses_sentence(self):
        self.type_and_wait_ghost()
        self.assertTrue(self.ac.want_swallow(A.VK_BACK))
        self.assertIsNone(self.ac._ghost)
        self.assertEqual(self.typed, [])
        n = len(self.comp.prompts)
        self.reader.field.before = "Hello Sam, I think we should"       # same sentence, more typing
        self.ac.note_key(0x41)
        time.sleep(0.2)
        self.assertEqual(len(self.comp.prompts), n)                     # no new guess
        self.reader.field.before = "Hello Sam, I think we should. Also"  # next sentence
        self.ac.note_key(0x41)
        self.assertTrue(self.wait_for(lambda: len(self.comp.prompts) > n))

    def test_key_hides_and_blocks_tab(self):
        self.type_and_wait_ghost()
        tok = self.ac._ghost.token
        self.ac.note_key(0x42)
        self.assertIsNone(self.ac._ghost)
        self.assertIn(tok, self.ghost.hidden)
        self.assertFalse(self.ac.want_swallow(A.VK_TAB))

    def test_other_window_no_swallow(self):
        self.type_and_wait_ghost()
        self.fg = 777
        self.assertFalse(self.ac.want_swallow(A.VK_TAB))

    def test_blocked_or_excluded_no_guess(self):
        self.host.blocked = "busy"
        self.ac.note_key(0x41)
        time.sleep(0.15)
        self.assertEqual(self.comp.prompts, [])
        self.host.blocked = ""
        with mock.patch.object(A.w, "window_info", return_value=("KeePassXC.exe", "db")):
            self.ac.cfg.d["autocomplete.exclude_apps"] = ["keepass*.exe"]
            self.ac.note_key(0x41)
            time.sleep(0.15)
        self.assertEqual(self.comp.prompts, [])
        with mock.patch.object(A.w, "window_info", return_value=("zen.exe", "Page — Private Browsing")):
            self.ac.note_key(0x41)
            time.sleep(0.15)
        self.assertEqual(self.comp.prompts, [])
        self.assertEqual(self.reader.reads, 0)

    def test_cancel_hides(self):
        self.type_and_wait_ghost()
        self.ac.cancel("dictation")
        self.assertIsNone(self.ac._ghost)
        self.assertFalse(self.ac.want_swallow(A.VK_TAB))

    def test_typed_text_is_sanitized(self):
        self.comp.out = " meet\tat noon."
        self.type_and_wait_ghost()
        self.assertTrue(self.ac.want_swallow(A.VK_TAB))
        self.assertTrue(self.wait_for(lambda: self.typed))
        self.assertEqual(self.typed, [" meetat noon."])

    def test_stop_and_restart(self):
        self.ac.stop()
        self.assertFalse(self.ac.running)
        self.assertFalse(self.ac.want_swallow(A.VK_TAB))
        self.ac.start()
        self.type_and_wait_ghost()


# ----------------------------------------------------------------------------- hotkey wiring
class HotkeyListener(unittest.TestCase):
    def setUp(self):
        from localflow.hotkey import HotkeyManager
        self.hk = HotkeyManager("ctrl+alt", on_start=lambda m: None, on_stop=lambda: None,
                                on_cancel=lambda why: None, on_handsfree=lambda: None)
        self.kl = mock.Mock()
        self.kl.want_swallow.return_value = True
        self.hk.key_listener = self.kl

    def test_tab_swallowed_with_key_up(self):
        self.assertTrue(self.hk._handle(w.VK_TAB, True))
        self.assertTrue(self.hk._handle(w.VK_TAB, False))
        self.kl.note_key.assert_not_called()

    def test_backspace_swallowed_only_once(self):
        self.assertTrue(self.hk._handle(w.VK_BACK, True))
        self.assertFalse(self.hk._handle(w.VK_BACK, False))      # key-up goes through

    def test_not_swallowed_with_modifier(self):
        self.hk._handle(w.VK_LSHIFT, True)
        self.assertFalse(self.hk._handle(w.VK_TAB, True))
        self.kl.want_swallow.assert_not_called()
        self.kl.note_key.assert_called_with(w.VK_TAB)

    def test_declined_tab_is_counted(self):
        self.kl.want_swallow.return_value = False
        self.assertFalse(self.hk._handle(w.VK_TAB, True))
        self.kl.note_key.assert_called_once_with(w.VK_TAB)

    def test_letters_counted_modifiers_not(self):
        self.hk._handle(0x41, True)
        self.hk._handle(0x41, False)
        self.hk._handle(w.VK_LSHIFT, True)
        self.hk._handle(w.VK_LSHIFT, False)
        self.kl.note_key.assert_called_once_with(0x41)

    def test_not_while_dictating(self):
        self.hk._handle(w.VK_LCONTROL, True)
        self.hk._handle(w.VK_LMENU, True)
        self.assertEqual(self.hk.state, "hold")
        self.hk._handle(w.VK_TAB, True)
        self.kl.want_swallow.assert_not_called()

    def test_paused_ignores(self):
        self.hk.set_enabled(False)
        self.assertFalse(self.hk._handle(w.VK_TAB, True))
        self.kl.want_swallow.assert_not_called()
        self.kl.note_key.assert_not_called()


# ----------------------------------------------------------------------------- ghost window
class GhostMath(unittest.TestCase):
    SCREENS = [(0, 0, 1280, 720, 1.5), (1920, 0, 1920, 1080, 1.0), (-1707, 0, 1707, 960, 1.5)]

    def test_phys_to_logical_per_monitor(self):
        x, y, dpr, s = G.phys_to_logical(300, 150, self.SCREENS)
        self.assertEqual((x, y, dpr), (200, 100, 1.5))
        x, y, dpr, s = G.phys_to_logical(3000, 500, self.SCREENS)
        self.assertEqual((x, y, dpr), (3000, 500, 1.0))
        x, y, dpr, s = G.phys_to_logical(-1707 + 150, 300, self.SCREENS)      # left monitor, negative x
        self.assertAlmostEqual(x, -1707 + 100)
        self.assertAlmostEqual(y, 200)
        self.assertIsNone(G.phys_to_logical(99999, 0, self.SCREENS))

    def test_fit_text(self):
        wd = lambda s: 10.0 * len(s)
        self.assertEqual(G.fit_text(" hello there friend", wd, 1000), " hello there friend")
        self.assertEqual(G.fit_text(" hello there friend", wd, 125), " hello there")
        self.assertEqual(G.fit_text(" hello there", wd, 15), "")
        self.assertEqual(G.fit_text("x", wd, 0), "")

    def test_font_from_caret(self):
        self.assertEqual(G.font_px(20), 14)
        self.assertEqual(G.font_px(2), 9)
        self.assertEqual(G.font_px(500), 48)


class GhostWidget(unittest.TestCase):
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

    def test_flags_never_activate_click_through(self):
        from PySide6.QtCore import Qt
        f = self.g.windowFlags()
        for flag in (Qt.Tool, Qt.FramelessWindowHint, Qt.WindowStaysOnTopHint, Qt.WindowDoesNotAcceptFocus,
                     Qt.WindowTransparentForInput):
            self.assertTrue(f & flag, flag)
        for attr in (Qt.WA_ShowWithoutActivating, Qt.WA_TranslucentBackground, Qt.WA_TransparentForMouseEvents):
            self.assertTrue(self.g.testAttribute(attr), attr)
        self.assertEqual(self.g.focusPolicy(), Qt.NoFocus)

    def test_show_maps_dpi_and_reports_text(self):
        self.g.show_ghost(3, " see you soon", (300, 150, 30), (0, 140, 1200, 190), 5.0)
        self.pump()
        self.assertTrue(self.g.isVisible())
        self.lst.ghost_shown.assert_called_once()
        tok, text, ttl = self.lst.ghost_shown.call_args[0]
        self.assertEqual((tok, text), (3, " see you soon"))
        self.assertEqual(self.g.x(), 201)                   # 300 phys / 1.5 + 1
        self.assertEqual(self.g.y(), 100)
        self.assertEqual(self.g.native_ex_style & G.GHOST_EX_STYLE, G.GHOST_EX_STYLE)

    def test_clipped_to_field(self):
        self.g.show_ghost(4, " " + "verylongword " * 40, (300, 150, 30), (0, 140, 600, 190), 5.0)
        self.pump()
        if self.lst.ghost_shown.called:
            text = self.lst.ghost_shown.call_args[0][1]
            self.assertLess(len(text), 200)
            self.assertLessEqual(self.g.x() + self.g.width(), 600 / 1.5 + 6)
        else:
            self.lst.ghost_failed.assert_called_once_with(4, mock.ANY)

    def test_hide_by_token(self):
        self.g.show_ghost(5, " hi there", (300, 150, 30), None, 5.0)
        self.pump()
        self.g.hide_ghost(99)                                # stale token: ignored
        self.pump()
        self.assertTrue(self.g.isVisible())
        self.g.hide_ghost(5)
        self.pump()
        self.assertFalse(self.g.isVisible())

    def test_stale_listener_hides(self):
        self.lst.ghost_shown.return_value = False
        self.g.show_ghost(6, " hi there", (300, 150, 30), None, 5.0)
        self.pump()
        self.assertFalse(self.g.isVisible())

    def test_off_screen_fails(self):
        self.g.show_ghost(7, " hi there", (99999, 150, 30), None, 5.0)
        self.pump()
        self.lst.ghost_failed.assert_called_once_with(7, "caret off screen")
        self.assertFalse(self.g.isVisible())


# ----------------------------------------------------------------------------- config + server
class ConfigAndServer(unittest.TestCase):
    def test_defaults(self):
        from localflow import config
        d = config.DEFAULTS["autocomplete"]
        self.assertTrue(d["enabled"])
        self.assertIn("autocomplete", config.PERSISTED_KEYS)
        for exe in ("code.exe", "windowsterminal.exe", "excel.exe", "1password.exe", "keepass*.exe",
                    "keepassxc.exe", "bitwarden.exe"):
            self.assertIn(exe, d["exclude_apps"])
        for k in A.AC_DEFAULTS:
            if k != "exclude_apps":
                self.assertIn(k, d)

    def test_persisted_into_existing_config(self):
        import json
        import tempfile
        from pathlib import Path
        from localflow import config
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "config.json"
            p.write_text(json.dumps({"hotkey": "ctrl+alt"}), encoding="utf-8")
            c = config.Config(path=p)
            self.assertTrue(c.get("autocomplete.enabled"))
            saved = json.loads(p.read_text(encoding="utf-8"))
            self.assertTrue(saved["autocomplete"]["enabled"])
            c.update({"autocomplete": {"enabled": False}})
            self.assertFalse(json.loads(p.read_text(encoding="utf-8"))["autocomplete"]["enabled"])
            self.assertEqual(c.get("autocomplete.debounce_ms"), 350)          # deep merge kept the rest

    def test_server_two_slots_full_ctx_each(self):
        from localflow.cleanup import LlamaServer
        cfg = mock.Mock()
        vals = {"cleanup.server_exe": "bin/llama/llama-server.exe", "cleanup.model_path": "models/x.gguf",
                "cleanup.port": 8765, "cleanup.gpu_layers": 99, "cleanup.ctx": 8192}
        cfg.get.side_effect = lambda k, d=None: vals.get(k, d)
        cmd, _sig = LlamaServer(cfg).launch()
        self.assertEqual(cmd[cmd.index("--parallel") + 1], "2")
        self.assertEqual(cmd[cmd.index("-c") + 1], "16384")


class AppWiring(unittest.TestCase):
    def test_blocked_reasons(self):
        from localflow.app import LocalFlowApp
        a = LocalFlowApp.__new__(LocalFlowApp)
        a.cfg = mock.Mock()
        a.cfg.get.side_effect = lambda k, d=None: d
        a.game_active, a._recording, a._fg_rids = False, False, set()
        self.assertEqual(a.ac_blocked(), "")
        a._fg_rids = {3}
        self.assertEqual(a.ac_blocked(), "busy (1 job(s) running)")
        a._fg_rids, a._recording = set(), True
        self.assertEqual(a.ac_blocked(), "busy (recording)")
        a._recording = False
        a._fg_rids, a.game_active = set(), True
        self.assertEqual(a.ac_blocked(), "game mode")
        a.cfg.get.side_effect = lambda k, d=None: False if k == "enabled" else d
        self.assertEqual(a.ac_blocked(), "paused")

    def test_fg_begin_cancels_guess(self):
        from localflow.app import LocalFlowApp
        a = LocalFlowApp.__new__(LocalFlowApp)
        a._fg_lock, a._fg_rids = threading.Lock(), set()
        a.gate = mock.Mock()
        a.autocomplete = mock.Mock()
        a._fg_begin(1)
        a.autocomplete.cancel.assert_called_once()
        a.gate.begin.assert_called_once()


if __name__ == "__main__":
    unittest.main()
