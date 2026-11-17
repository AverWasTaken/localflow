"""No text box focused -> copy to the clipboard + toast (fieldprobe.py, uia.py, inserter clipboard history, pipeline
and app wiring). Offline: UIA / Win32 / clipboard are faked; Qt runs on the offscreen platform."""
import os
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from localflow import fieldprobe as F       # noqa: E402
from localflow import inserter               # noqa: E402
from localflow import uia as U               # noqa: E402
from localflow.pipeline import Pipeline      # noqa: E402

import test_unit_app as ta                   # noqa: E402
from localflow import app as app_mod         # noqa: E402


def facts(**kw):
    base = dict(exe="app.exe", fg_class="SomeWindow")
    base.update(kw)
    return F.Facts(**base)


class DecisionTable(unittest.TestCase):
    def k(self, overrides=None, **kw):
        d = F.decide(facts(**kw), overrides)
        return d.kind, d.shell

    # ---- field
    def test_uia_edit_is_field(self):
        self.assertEqual(self.k(ctype=F.CT_EDIT, framework="Chrome"), ("field", False))

    def test_system_caret_is_field(self):
        self.assertEqual(self.k(caret=True, ctype=F.CT_BUTTON), ("field", False))

    def test_edit_window_classes_are_field(self):
        for cls in ("Edit", "RichEdit20W", "RICHEDIT50W", "Scintilla", "ConsoleWindowClass"):
            self.assertEqual(self.k(focus_class=cls), ("field", False), cls)

    def test_terminals_are_field(self):
        for cls in ("CASCADIA_HOSTING_WINDOW_CLASS", "ConsoleWindowClass"):
            self.assertEqual(self.k(fg_class=cls, ctype=50025), ("field", False), cls)

    def test_editable_value_is_field(self):
        self.assertEqual(self.k(ctype=50026, value_ro=False), ("field", False))     # contenteditable Group

    def test_desktop_rename_box_is_field(self):
        self.assertEqual(self.k(fg_class="Progman", caret=True, focus_class="Edit"), ("field", False))

    def test_edit_beats_taskbar(self):
        self.assertEqual(self.k(fg_class="Shell_TrayWnd", ctype=F.CT_EDIT), ("field", False))

    # ---- none
    def test_desktop_and_taskbar_are_none_without_keystroke(self):
        for cls in ("Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd"):
            self.assertEqual(self.k(fg_class=cls, ctype=F.CT_LISTITEM), ("none", True), cls)

    def test_desktop_is_none_even_when_uia_timed_out(self):
        self.assertEqual(self.k(fg_class="WorkerW", uia="timeout"), ("none", True))

    def test_non_text_controls_are_none(self):
        for ct in (F.CT_BUTTON, F.CT_LISTITEM, F.CT_TREEITEM, F.CT_HYPERLINK, F.CT_MENUITEM, F.CT_TABITEM):
            self.assertEqual(self.k(ctype=ct), ("none", False), ct)

    def test_gecko_read_only_page_is_none(self):
        self.assertEqual(self.k(fg_class="MozillaWindowClass", ctype=F.CT_DOCUMENT, framework="Gecko",
                                legacy_ro=True, children=1), ("none", False))
        self.assertEqual(self.k(ctype=F.CT_DOCUMENT, framework="Gecko", value_ro=True, children=1), ("none", False))

    def test_chromium_read_only_page_is_none(self):
        self.assertEqual(self.k(fg_class="Chrome_WidgetWin_1", ctype=F.CT_DOCUMENT, framework="Chrome",
                                value_ro=True, children=1), ("none", False))

    # ---- unknown (= behave as before)
    def test_chromium_doc_without_children_is_unknown(self):
        self.assertEqual(self.k(ctype=F.CT_DOCUMENT, framework="Chrome", value_ro=True, children=0),
                         ("unknown", False))

    def test_doc_without_read_only_evidence_is_unknown(self):
        self.assertEqual(self.k(ctype=F.CT_DOCUMENT, framework="Gecko", legacy_ro=False, children=1),
                         ("unknown", False))

    def test_non_browser_doc_is_unknown(self):
        self.assertEqual(self.k(ctype=F.CT_DOCUMENT, framework="Win32", value_ro=True, children=1),
                         ("unknown", False))

    def test_pane_window_custom_group_are_unknown(self):
        for ct in (50033, 50032, 50025, 50026, 50020):
            self.assertEqual(self.k(ctype=ct), ("unknown", False), ct)

    def test_uia_failures_are_unknown(self):
        for st in ("timeout", "busy", "error:COMError", "no focus"):
            self.assertEqual(self.k(uia=st, ctype=F.CT_BUTTON), ("unknown", False), st)

    def test_own_process_is_unknown(self):
        self.assertEqual(self.k(own=True, fg_class="Progman"), ("unknown", False))
        self.assertEqual(self.k(uia_own=True), ("unknown", False))
        self.assertEqual(self.k(no_window=True), ("unknown", False))

    # ---- override
    def test_override_map(self):
        self.assertEqual(self.k({"app.exe": "field"}, fg_class="Progman"), ("field", False))
        self.assertEqual(self.k({"app.exe": "none"}, ctype=F.CT_EDIT), ("none", True))
        self.assertEqual(self.k({"other.exe": "none"}, ctype=F.CT_EDIT), ("field", False))

    def test_describe_has_no_text_fields(self):
        s = facts(ctype=F.CT_DOCUMENT, framework="Gecko", legacy_ro=True, children=1).describe()
        self.assertIn("type=Document", s)
        self.assertIn("fw=Gecko", s)


class FakeThread:
    def __init__(self, result=None, exc=None):
        self.result, self.exc, self.calls = result, exc, []

    def call(self, fn, timeout):
        self.calls.append(timeout)
        if self.exc:
            raise self.exc
        return self.result


class Gather(unittest.TestCase):
    def setUp(self):
        self.gti = SimpleNamespace(hwndCaret=0, hwndFocus=0)
        self.p = [mock.patch.object(U, "_fg_hwnd", return_value=100),
                  mock.patch.object(U, "_window_pid", return_value=4242),
                  mock.patch.object(U, "class_name", side_effect=lambda h: {100: "Chrome_WidgetWin_1",
                                                                            7: "Edit"}.get(h, "")),
                  mock.patch.object(U, "gui_thread_info", side_effect=lambda h: self.gti),
                  mock.patch.object(F.winutil, "window_info", return_value=("chrome.exe", "t"))]
        for p in self.p:
            p.start()

    def tearDown(self):
        for p in self.p:
            p.stop()

    def test_caret_skips_uia(self):
        self.gti.hwndCaret = 5
        t = FakeThread({"ctype": F.CT_BUTTON})
        f = F.gather(0.15, thread=t)
        self.assertEqual(t.calls, [])
        self.assertEqual(F.decide(f).kind, "field")

    def test_focus_edit_skips_uia(self):
        self.gti.hwndFocus = 7
        t = FakeThread()
        self.assertEqual(F.decide(F.gather(0.15, thread=t)).kind, "field")
        self.assertEqual(t.calls, [])

    def test_uia_facts_merged(self):
        t = FakeThread({"ctype": F.CT_DOCUMENT, "framework": "Chrome", "value_ro": True, "children": 1})
        f = F.gather(0.15, thread=t)
        self.assertEqual(len(t.calls), 1)
        self.assertLessEqual(t.calls[0], 0.15)
        self.assertEqual((f.exe, f.fg_class), ("chrome.exe", "Chrome_WidgetWin_1"))
        self.assertEqual(F.decide(f).kind, "none")

    def test_timeout_busy_error_are_unknown(self):
        for exc, st in ((TimeoutError("UIA call timed out"), "timeout"), (TimeoutError("UIA thread busy"), "busy"),
                        (RuntimeError("x"), "error:RuntimeError")):
            f = F.gather(0.15, thread=FakeThread(exc=exc))
            self.assertEqual(f.uia, st)
            self.assertEqual(F.decide(f).kind, "unknown")

    def test_own_window_never_touches_uia(self):
        t = FakeThread({"ctype": F.CT_BUTTON})
        with mock.patch.object(U, "_window_pid", return_value=os.getpid()):
            f = F.gather(0.15, thread=t)
        self.assertEqual(t.calls, [])
        self.assertEqual(F.decide(f).reason, "own process")

    def test_probe_never_raises_and_reads_overrides(self):
        cfg = SimpleNamespace(get=lambda k, d=None: {"insert.no_field_override": {"Chrome.EXE": "FIELD"}}.get(k, d))
        d = F.probe(cfg, gather_fn=lambda t: facts(exe="chrome.exe", ctype=F.CT_BUTTON))
        self.assertEqual((d.kind, d.reason), ("field", "override"))
        d = F.probe(None, gather_fn=mock.Mock(side_effect=OSError("boom")))
        self.assertEqual(d.kind, "unknown")


class FakeEl:
    def __init__(self, pid=10, ctype=F.CT_DOCUMENT, fw="Gecko", cls="", value_ro=None, state=None):
        self.CurrentProcessId, self.CurrentControlType = pid, ctype
        self.CurrentFrameworkId, self.CurrentClassName = fw, cls
        self._v, self._s = value_ro, state

    def GetCurrentPattern(self, pid):
        if pid == F.UIA_VALUE_PATTERN and self._v is not None:
            return SimpleNamespace(QueryInterface=lambda i: SimpleNamespace(CurrentIsReadOnly=self._v))
        if pid == F.UIA_LEGACY_PATTERN and self._s is not None:
            return SimpleNamespace(QueryInterface=lambda i: SimpleNamespace(CurrentState=self._s))
        return None


class UiaJob(unittest.TestCase):
    def ctx(self, el, child=True):
        return U.UiaContext(SimpleNamespace(IUIAutomationValuePattern=1, IUIAutomationLegacyIAccessiblePattern=2),
                            SimpleNamespace(GetFocusedElement=lambda: el),
                            SimpleNamespace(GetFirstChildElement=lambda e: object() if child else None), 99)

    def test_gecko_document_facts(self):
        out = F._uia_job(self.ctx(FakeEl(state=0x40 | 0x100000)))
        self.assertEqual(out, {"ctype": F.CT_DOCUMENT, "framework": "Gecko", "el_class": "", "legacy_ro": True,
                               "children": 1})

    def test_value_pattern_and_no_children(self):
        out = F._uia_job(self.ctx(FakeEl(fw="Chrome", value_ro=True), child=False))
        self.assertEqual((out["value_ro"], out["children"]), (True, 0))

    def test_children_only_for_documents(self):
        out = F._uia_job(self.ctx(FakeEl(ctype=F.CT_BUTTON)))
        self.assertNotIn("children", out)

    def test_own_process_and_no_focus(self):
        self.assertEqual(F._uia_job(self.ctx(FakeEl(pid=99))), {"uia_own": True})
        self.assertEqual(F._uia_job(self.ctx(None)), {"uia": "no focus"})


class SharedThread(unittest.TestCase):
    def make(self, factory=None):
        return U.UiaThread("test-uia", context_factory=factory or (lambda: "CTX"))

    def test_call_returns_value_on_the_uia_thread(self):
        t = self.make()
        names = []
        self.assertEqual(t.call(lambda ctx: (names.append(threading.current_thread().name), ctx)[1], 2.0), "CTX")
        self.assertEqual(names, ["test-uia"])

    def test_timeout_drops_late_result_and_busy_fails_fast(self):
        t = self.make()
        release = threading.Event()
        with self.assertRaises(TimeoutError):
            t.call(lambda ctx: release.wait(2) and "late", 0.05)
        t0 = time.perf_counter()
        with self.assertRaisesRegex(TimeoutError, "busy"):
            t.call(lambda ctx: "x", 1.0)                  # still running the hung job: no queueing behind it
        self.assertLess(time.perf_counter() - t0, 0.05)
        release.set()
        for _ in range(100):
            if not t._pending:
                break
            time.sleep(0.01)
        self.assertEqual(t.call(lambda ctx: "fresh", 1.0), "fresh")    # the late "late" was dropped

    def test_warm_skips_own_and_missing_windows(self):
        posted = []
        fake = SimpleNamespace(post=lambda fn: posted.append(fn) or True)
        with mock.patch.object(U, "shared", return_value=fake), mock.patch.object(U, "_fg_hwnd", return_value=100):
            with mock.patch.object(U, "_window_pid", return_value=os.getpid()):
                F.warm()
            self.assertEqual(posted, [])                    # our own window: no UIA call at all
            with mock.patch.object(U, "_window_pid", return_value=4242):
                F.warm()
            self.assertEqual(posted, [F._warm_job])
        with mock.patch.object(U, "shared", return_value=fake), mock.patch.object(U, "_fg_hwnd", return_value=0):
            F.warm()
        self.assertEqual(len(posted), 1)

    def test_job_error_is_wrapped(self):
        t = self.make()
        with self.assertRaisesRegex(U.UiaJobError, "ValueError: bad"):
            t.call(lambda ctx: (_ for _ in ()).throw(ValueError("bad")), 1.0)

    def test_setup_failure(self):
        t = self.make(factory=mock.Mock(side_effect=OSError("no uia")))
        with self.assertRaises(Exception):
            t.call(lambda ctx: 1, 1.0)
        with self.assertRaisesRegex(RuntimeError, "UIA unavailable"):
            t.call(lambda ctx: 1, 1.0)
        self.assertFalse(t.post(lambda ctx: 1))


class Cfg(dict):
    def get(self, k, d=None):
        return super().get(k, d)


class Deliver(unittest.TestCase):
    def run_(self, kind="none", shell=False, mode="paste", enabled=True, ins_ret=None, copy_ok=True):
        self.ins, self.copies = [], []
        probe = mock.Mock(return_value=F.Decision(kind, "r", shell))

        def insert(text, **kw):
            self.ins.append(kw)
            return ins_ret

        def copy(text, **kw):
            self.copies.append(kw)
            return copy_ok
        cfg = Cfg({"insert.copy_when_no_field": enabled})
        out = F.deliver("hi", cfg=cfg, insert=insert, copy=copy, probe_fn=probe, mode=mode, restore=True,
                        restore_delay_ms=300)
        return out, probe

    def test_disabled_is_unchanged_and_never_probes(self):
        out, probe = self.run_(enabled=False, ins_ret={"ok": True})
        probe.assert_not_called()
        self.assertEqual(self.ins, [{"mode": "paste", "restore": True, "restore_delay_ms": 300}])
        self.assertEqual(out, {"ok": True})

    def test_field_and_unknown_are_unchanged(self):
        for kind in ("field", "unknown"):
            out, _ = self.run_(kind=kind, ins_ret=None)
            self.assertEqual(self.ins, [{"mode": "paste", "restore": True, "restore_delay_ms": 300}])
            self.assertIsNone(out)
            self.assertEqual(self.copies, [])

    def test_uia_none_in_paste_mode_still_pastes_but_keeps_clipboard(self):
        out, _ = self.run_(ins_ret={"mode": "paste", "ok": True, "reason": ""})
        self.assertEqual(self.ins, [{"mode": "paste", "restore": False, "restore_delay_ms": 300,
                                     "clipboard_history": True, "fallback_type": False}])
        self.assertTrue(out["copied"])
        self.assertNotIn("copy_failed", out)

    def test_none_paste_with_fake_inserter_returning_nothing(self):
        out, _ = self.run_(ins_ret=None)
        self.assertTrue(out["copied"])

    def test_desktop_none_skips_keystroke(self):
        out, _ = self.run_(shell=True)
        self.assertEqual(self.ins, [])
        self.assertEqual(self.copies, [{"history": True}])
        self.assertTrue(out["copied"])

    def test_type_mode_none_just_copies(self):
        out, _ = self.run_(mode="type")
        self.assertEqual(self.ins, [])
        self.assertTrue(out["copied"])

    def test_copy_failures(self):
        out, _ = self.run_(shell=True, copy_ok=False)
        self.assertTrue(out["copy_failed"])
        self.assertNotIn("copied", out)
        out, _ = self.run_(ins_ret={"mode": "paste", "ok": False, "reason": "clipboard busy"})
        self.assertTrue(out["copy_failed"])


class InserterClipboard(unittest.TestCase):
    def setUp(self):
        self.sent = []
        self.p = [mock.patch.object(inserter, "wait_modifiers_released", return_value=True),
                  mock.patch.object(inserter, "_paste_keystroke", side_effect=lambda: self.sent.append("paste")),
                  mock.patch.object(inserter, "type_text", side_effect=lambda t: self.sent.append("type")),
                  mock.patch.object(inserter, "save_clipboard_checked", return_value=[])]
        for p in self.p:
            p.start()

    def tearDown(self):
        for p in self.p:
            p.stop()

    def test_no_type_fallback_when_disabled(self):
        with mock.patch.object(inserter, "set_clipboard_text", return_value=False) as s:
            r = inserter.insert_text("hi", restore=False, clipboard_history=True, fallback_type=False)
        s.assert_called_once_with("hi", history=True)
        self.assertEqual(self.sent, [])
        self.assertEqual((r["ok"], r["reason"]), (False, "clipboard busy"))

    def test_default_still_falls_back_to_typing(self):
        with mock.patch.object(inserter, "set_clipboard_text", return_value=False) as s:
            inserter.insert_text("hi", restore=False)
        s.assert_called_once_with("hi", history=False)
        self.assertEqual(self.sent, ["type"])

    def _formats(self, history):
        reg, setd = {}, []

        def register(name):
            reg[len(reg) + 100] = name
            return len(reg) + 99

        fake = SimpleNamespace(OpenClipboard=lambda h: True, CloseClipboard=lambda: True, EmptyClipboard=lambda: True,
                               RegisterClipboardFormatW=register,
                               SetClipboardData=lambda f, h: setd.append((f, h)) or 1)
        with mock.patch.object(inserter, "u32", fake), \
                mock.patch.object(inserter, "_hglobal_from_bytes", side_effect=lambda b: b):
            self.assertTrue(inserter.set_clipboard_text("hi", history=history))
        return {reg.get(f, f): h for f, h in setd}

    def test_history_visible_but_never_cloud(self):
        got = self._formats(True)
        self.assertNotIn("ExcludeClipboardContentFromMonitorProcessing", got)
        self.assertEqual(got["CanIncludeInClipboardHistory"], b"\x01\x00\x00\x00")
        self.assertEqual(got["CanUploadToCloudClipboard"], b"\x00\x00\x00\x00")

    def test_default_hidden_from_history(self):
        got = self._formats(False)
        self.assertEqual(got["ExcludeClipboardContentFromMonitorProcessing"], b"\x01\x00\x00\x00")
        self.assertEqual(got["CanIncludeInClipboardHistory"], b"\x00\x00\x00\x00")
        self.assertEqual(got["CanUploadToCloudClipboard"], b"\x00\x00\x00\x00")


class PipelineCarriesFlag(unittest.TestCase):
    def run_(self, ret):
        cfg = Cfg({"cleanup.enabled": False, "insert.mode": "paste", "insert.restore_clipboard": True})
        stt = SimpleNamespace(transcribe=lambda *a, **k: SimpleNamespace(text="hello there", ms=1, dropped=""))
        cl = SimpleNamespace(clean=lambda t, s, w: (t, {"llm": False, "ms": 0, "reason": ""}))
        import numpy as np
        audio = (np.sin(np.arange(16000) / 5) * 0.3).astype("float32")
        return Pipeline(cfg, stt, cl, None, lambda *a, **k: ret).process(audio)

    def test_flags(self):
        r = self.run_({"copied": True})
        self.assertEqual(r["status"], "ok")
        self.assertTrue(r["copied"])
        self.assertTrue(self.run_({"copy_failed": True})["copy_failed"])
        for ret in (None, {}, {"ok": True}):
            r = self.run_(ret)
            self.assertNotIn("copied", r)
            self.assertNotIn("copy_failed", r)


class AppFlow(ta.AppWiring):
    def setUp(self):
        super().setUp()
        self.toasts = []
        self.app.bridge.toast.connect(lambda s, t: self.toasts.append((s, t)))
        self.app.stt.ready = True

    def dictate(self, r, rid=None):
        a = self.app
        a.pipeline = SimpleNamespace(process=lambda *x, **k: dict(r))
        a._run_dictation(a._rec_id if rid is None else rid, None, ("x.exe", "t"), None, None, 0.0)

    def test_copied_toast_even_with_overlay_off(self):
        self.app.cfg.set("ui.overlay", False)
        self.dictate({"status": "ok", "text": "hello", "copied": True})
        self.assertEqual(self.app.overlay.calls[-1], ("info", "Copied to clipboard"))
        self.assertEqual(self.app.last_text, "hello")

    def test_copy_failed_error_toast(self):
        self.dictate({"status": "ok", "text": "hello", "copy_failed": True})
        self.assertEqual(self.app.overlay.calls[-1], ("error", "Couldn't copy to clipboard"))
        self.assertEqual(self.app.last_text, "hello")

    def test_normal_insert_hides(self):
        self.dictate({"status": "ok", "text": "hello"})
        self.assertEqual(self.app.overlay.calls[-1][0], "hidden")

    def test_new_recording_keeps_its_pill(self):
        a = self.app
        a._recording = True
        n = len(a.overlay.calls)
        self.dictate({"status": "ok", "text": "hello", "copied": True})
        self.assertEqual(len(a.overlay.calls), n)              # the recording pill was not replaced
        self.assertEqual(self.toasts, [("info", "Copied to clipboard")])
        a._recording = False
        self.toasts.clear()
        a._rec_id += 1                                          # a newer recording came and went
        self.dictate({"status": "ok", "text": "hello", "copied": True}, rid=a._rec_id - 1)
        self.assertEqual(self.toasts, [("info", "Copied to clipboard")])

    def test_queued_copied_toast_shown_with_overlay_off(self):
        a = self.app
        a.cfg.set("ui.overlay", False)
        a._toasts = [("info", "Copied to clipboard")]
        a._flush_toasts()
        self.assertEqual(a.overlay.calls[-1], ("info", "Copied to clipboard"))
        n = len(a.overlay.calls)
        a._toasts = [("info", "Meeting saved - transcribing")]      # other background toasts still respect it
        a._flush_toasts()
        self.assertEqual(len(a.overlay.calls), n)

    def test_warm_up_only_for_dictation(self):
        a = self.app
        with mock.patch.object(app_mod.winutil, "window_info", return_value=("x.exe", "t")):
            a._do_start("command")
            a._do_cancel("test")
            self.assertEqual(self.warm_calls, [])
            a._do_start("dictate")
            a._do_cancel("test")
        self.assertEqual(self.warm_calls, [1])
        a.cfg.set("insert.copy_when_no_field", False)
        a._do_start("dictate")
        a._do_cancel("test")
        self.assertEqual(self.warm_calls, [1])

    def test_inserter_goes_through_deliver(self):
        calls = []
        with mock.patch.object(app_mod, "insert_text", side_effect=lambda t, **k: calls.append(k) or {"ok": True}), \
                mock.patch.object(app_mod.fieldprobe, "probe", return_value=F.Decision("unknown", "x")):
            self.app._insert_dictation("hi", mode="paste", restore=True, restore_delay_ms=300)
        self.assertEqual(calls, [{"mode": "paste", "restore": True, "restore_delay_ms": 300}])
        self.assertIs(self.app.pipeline.inserter.__func__, app_mod.LocalFlowApp._insert_dictation)


if __name__ == "__main__":
    unittest.main()
