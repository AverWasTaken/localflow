"""Offline tests for switching the cleanup model: model list, think-tag stripping, per-model extras and
Cleaner.restart() (with a fake llama-server; nothing is spawned)."""
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from localflow import cleanup  # noqa: E402
from localflow.cleanup import Cleaner, list_models, model_display_name, same_model, strip_think  # noqa: E402
from localflow.config import Config  # noqa: E402

LFM = "models/LFM3-1.7B-Q8_0.gguf"
QWEN = "models/Qwen3.5-4B-Q8_0.gguf"


def tmp_cfg():
    return Config(path=Path(tempfile.mkdtemp()) / "config.json")


class ModelList(unittest.TestCase):
    def test_lists_gguf_only(self):
        d = Path(tempfile.mkdtemp())
        for n in ("b-model.gguf", "A-model.GGUF", "mmproj-vision.gguf", "c.gguf.part", "notes.txt"):
            (d / n).write_bytes(b"x")
        (d / "dir.gguf").mkdir()
        self.assertEqual(list_models(d), ["A-model.GGUF", "b-model.gguf"])

    def test_missing_dir(self):
        self.assertEqual(list_models(Path(tempfile.mkdtemp()) / "nope"), [])

    def test_display_name(self):
        self.assertEqual(model_display_name(LFM), "LFM3-1.7B-Q8_0")
        self.assertEqual(model_display_name(r"C:\x\Qwen3.5-4B-Q8_0.gguf"), "Qwen3.5-4B-Q8_0")

    def test_same_model(self):
        self.assertTrue(same_model(LFM, str(cleanup.PROJECT_DIR / LFM)))
        self.assertTrue(same_model(LFM, LFM.replace("/", "\\").upper()))
        self.assertFalse(same_model(LFM, QWEN))
        self.assertFalse(same_model("", LFM))


class StripThink(unittest.TestCase):
    def test_plain_text_unchanged(self):
        self.assertEqual(strip_think("Hello there."), "Hello there.")

    def test_reasoning_block_removed(self):
        self.assertEqual(strip_think("<think>let me see</think>Hello.").strip(), "Hello.")

    def test_answer_repeated_after_close_tag(self):
        self.assertEqual(strip_think("Draft.\n\nLet me fix this:\n\nDraft.</think>Final.").strip(), "Final.")

    def test_unterminated_reasoning_is_no_answer(self):
        self.assertEqual(strip_think("<think>We need to clean up the transcript"), "")
        self.assertEqual(strip_think("<think>reasoning</think>"), "")

    def test_stray_tags_dropped(self):
        self.assertEqual(strip_think("Hi <think>there"), "Hi there")

    def test_empty(self):
        self.assertEqual(strip_think(""), "")
        self.assertEqual(strip_think(None), "")


class PerModelExtras(unittest.TestCase):
    def test_qwen_command_line_unchanged(self):
        cfg = tmp_cfg()
        cfg.set("cleanup.model_path", QWEN, save=False)
        cmd, _ = cleanup.LlamaServer(cfg).launch()
        self.assertEqual(cmd[-1], "--no-mmproj")
        self.assertEqual(cmd[1:3], ["-m", str(cleanup.PROJECT_DIR / QWEN)])

    def test_model_args_appended_for_matching_model_only(self):
        cfg = tmp_cfg()
        cfg.set("cleanup.model_args", {"lfm3-1.7b-q8_0.gguf": ["--reasoning-budget", 0]}, save=False)
        s = cleanup.LlamaServer(cfg)
        cfg.set("cleanup.model_path", LFM, save=False)
        cmd, sig_lfm = s.launch()
        self.assertEqual(cmd[-2:], ["--reasoning-budget", "0"])
        cfg.set("cleanup.model_path", QWEN, save=False)
        cmd, sig_qwen = s.launch()
        self.assertEqual(cmd[-1], "--no-mmproj")
        self.assertNotEqual(sig_lfm, sig_qwen)

    def test_bad_model_args_ignored(self):
        cfg = tmp_cfg()
        cfg.set("cleanup.model_path", LFM, save=False)
        cfg.set("cleanup.model_args", {"LFM3-1.7B-Q8_0.gguf": "--oops"}, save=False)
        self.assertEqual(cleanup.LlamaServer(cfg).launch()[0][-1], "--no-mmproj")

    def test_request_overrides(self):
        cfg = tmp_cfg()
        cl = Cleaner(cfg)
        cfg.set("cleanup.model_path", LFM, save=False)
        self.assertIn("</think>", cl._request_overrides()["stop"])     # default LFM3 stop list
        cfg.set("cleanup.model_path", QWEN, save=False)
        self.assertEqual(cl._request_overrides(), {})
        cfg.set("cleanup.model_request", {"Qwen3.5-4B-Q8_0.gguf": {"stop": ["x"], "messages": []}}, save=False)
        self.assertEqual(cl._request_overrides(), {"stop": ["x"]})

    def test_request_overrides_follow_loaded_model(self):
        cfg = tmp_cfg()
        cl = Cleaner(cfg)
        cfg.set("cleanup.model_path", LFM, save=False)
        cfg.set("cleanup.model_path", QWEN, save=False)
        cl.server.launched = ("exe", str(cleanup.PROJECT_DIR / LFM).lower(), ())
        self.assertIn("stop", cl._request_overrides())

    def test_model_args_map_replaced_on_update(self):
        cfg = tmp_cfg()
        cfg.update({"cleanup": {"model_request": {}}})
        self.assertEqual(cfg.get("cleanup.model_request"), {})

    def test_llm_applies_overrides_and_strips(self):
        cfg = tmp_cfg()
        cfg.set("cleanup.model_path", LFM, save=False)
        cl = Cleaner(cfg)
        resp = mock.Mock()
        resp.json.return_value = {"choices": [{"finish_reason": "stop",
                                               "message": {"content": "Hi.</think>Hello there.",
                                                           "reasoning_content": ""}}]}
        cl.http = mock.Mock()
        cl.http.post.return_value = resp
        out = cl._llm("hello there you know", {"name": "neutral", "prompt": ""}, [])
        self.assertEqual(out, "Hello there.")
        body = cl.http.post.call_args.kwargs["json"]
        self.assertIn("\n\nLet me", body["stop"])
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})


class StopFilter(unittest.TestCase):
    STOPS = ["</think>", "<think>", "\n\nLet me", "\n\nActually", "\n\n(", " (Note", " Wait, let me", " Hmm."]

    def test_unspoken_stops_kept(self):
        self.assertEqual(cleanup.unspoken_stops(self.STOPS, "send the budget to dana today"), self.STOPS)

    def test_spoken_words_dropped(self):
        raw = "send the budget to dana today open paren note marcus also needs a copy close paren"
        self.assertNotIn(" (Note", cleanup.unspoken_stops(self.STOPS, raw))
        raw = "hmm let me think about it actually no wait let me check"
        got = cleanup.unspoken_stops(self.STOPS, raw)
        for s in ("\n\nLet me", "\n\nActually", " Wait, let me", " Hmm."):
            self.assertNotIn(s, got)
        self.assertIn("</think>", got)                      # "think" was said, but tag stops always stay

    def test_symbol_stop(self):
        self.assertNotIn("\n\n(", cleanup.unspoken_stops(self.STOPS, "call (555) 123"))
        self.assertNotIn("\n\n(", cleanup.unspoken_stops(self.STOPS, "new paragraph open parenthesis aside"))
        self.assertIn("\n\n(", cleanup.unspoken_stops(self.STOPS, "call me later."))

    def test_word_variants_count_as_spoken(self):
        # a stop matches text, so a longer spoken word or a missing apostrophe must also disable it
        self.assertNotIn(" (Note", cleanup.unspoken_stops(self.STOPS, "notes from marcus are included"))
        self.assertNotIn("\n\nLet's", cleanup.unspoken_stops(["\n\nLet's"], "lets go"))
        self.assertNotIn("\n\nLet me", cleanup.unspoken_stops(self.STOPS, "let meet at noon"))
        self.assertIn("\n\nLet me", cleanup.unspoken_stops(self.STOPS, "outlet me"))

    def test_truncation_never_hits_spoken_text(self):
        """Simulate the server: cut the faithful cleanup at the first remaining stop; nothing may be cut."""
        cases = [("dana today notes from marcus are included", "Dana today (Notes from Marcus are included)."),
                 ("lets go new paragraph lets eat", "Let's go.\n\nLet's eat."),
                 ("new paragraph open paren aside close paren", "\n\n(aside)"),
                 ("new paragraph actually i am fine", "\n\nActually, I am fine."),
                 ("hmm. ok", "Hmm. Ok.")]
        from localflow.config import LFM3_STOPS
        for raw, faithful in cases:
            for s in cleanup.unspoken_stops(LFM3_STOPS, raw):
                self.assertNotIn(s, faithful, f"{s!r} would cut {faithful!r}")

    def test_bad_input(self):
        self.assertEqual(cleanup.unspoken_stops("oops", "x"), [])
        self.assertEqual(cleanup.unspoken_stops(None, "x"), [])

    def _body(self, raw, content="Ok."):
        cfg = tmp_cfg()
        cfg.set("cleanup.model_path", LFM, save=False)
        cl = Cleaner(cfg)
        resp = mock.Mock()
        resp.json.return_value = {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
        cl.http = mock.Mock()
        cl.http.post.return_value = resp
        cl._llm(raw, {"name": "neutral", "prompt": ""}, [])
        return cl.http.post.call_args.kwargs["json"]

    def test_llm_filters_spoken_stops(self):
        body = self._body("please send it to dana today note marcus needs a copy")
        self.assertNotIn(" (Note", body["stop"])
        self.assertIn("\n\nLet me", body["stop"])

    def test_qwen_request_has_no_stop(self):
        cfg = tmp_cfg()
        cfg.set("cleanup.model_path", QWEN, save=False)
        cl = Cleaner(cfg)
        resp = mock.Mock()
        resp.json.return_value = {"choices": [{"finish_reason": "stop", "message": {"content": "Ok."}}]}
        cl.http = mock.Mock()
        cl.http.post.return_value = resp
        cl._llm("hello there you know", {"name": "neutral", "prompt": ""}, [])
        body = cl.http.post.call_args.kwargs["json"]
        self.assertEqual(set(body), {"messages", "temperature", "top_k", "max_tokens", "cache_prompt", "stream",
                                     "chat_template_kwargs"})


class LateConnectionError(unittest.TestCase):
    def make(self):
        cfg = tmp_cfg()
        cl = Cleaner(cfg)
        cl.server.ready = True
        cl.server.proc = object()
        cl.server.start = mock.Mock()
        return cl

    def test_error_from_replaced_process_is_ignored(self):
        cl = self.make()
        new_proc = object()

        def llm(*a, **k):
            cl.server.proc = new_proc            # restart() swapped the process while this request was out
            raise cleanup.requests.ConnectionError("old socket")
        cl._llm = llm
        text, info = cl.clean("so um this is a longer test sentence")
        self.assertTrue(info["reason"].startswith("error"))
        cl.server.start.assert_not_called()
        self.assertTrue(cl.server.ready)

    def test_no_recovery_while_a_restart_runs(self):
        cl = self.make()
        cl._llm = mock.Mock(side_effect=cleanup.requests.ConnectionError("old socket"))
        with cl._restart_lock:                       # restart() in progress
            cl.clean("so um this is a longer test sentence")
        cl.server.start.assert_not_called()

    def test_error_from_current_process_recovers(self):
        cl = self.make()
        cl._llm = mock.Mock(side_effect=cleanup.requests.ConnectionError("dead"))
        cl.clean("so um this is a longer test sentence")
        cl.server.start.assert_called_once()
        self.assertFalse(cl.server.ready)


class SettingsDropdown(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.qapp = QApplication.instance() or QApplication([])

    def make(self):
        from localflow.settings_ui import SettingsWindow
        cfg = tmp_cfg()
        w = SettingsWindow(cfg, lambda: [], lambda: False, lambda v: None, get_last_ctx=lambda: ("", ""),
                           get_default_mic=lambda: "")
        w.reload()
        return cfg, w

    def test_pick_applies_and_save_keeps_speech_model(self):
        cfg, w = self.make()
        names = [f for f in list_models() if not same_model(f"models/{f}", cfg.get("cleanup.model_path"))]
        if not names:
            self.skipTest("needs a second .gguf in models/")
        i = w.llm_combo.findData(f"models/{names[0]}")
        w.llm_combo.setCurrentIndex(i)
        w._model_picked(i)
        self.assertEqual(cfg.get("cleanup.model_path"), f"models/{names[0]}")
        stt = cfg.get("stt.model")
        self.assertTrue(w.save())
        self.assertEqual(cfg.get("stt.model"), stt)        # regression: the two dropdowns once shared a name
        self.assertEqual(cfg.get("cleanup.model_path"), f"models/{names[0]}")

    def test_save_does_not_undo_tray_switch(self):
        cfg, w = self.make()
        cfg.set("cleanup.model_path", "models/elsewhere.gguf", save=False)    # e.g. picked in the tray
        self.assertTrue(w.save())
        self.assertEqual(cfg.get("cleanup.model_path"), "models/elsewhere.gguf")
        w.reload()
        self.assertEqual(w.llm_combo.currentData(), "models/elsewhere.gguf")   # shown as "(current)"


class FakeServer:
    """Stands in for LlamaServer: start() 'loads' whatever the config says after a short delay."""

    def __init__(self, cfg, fail=False, delay=0.05):
        self.cfg = cfg
        self.ready = False
        self.failed = None
        self.suspended = False
        self.launched = None
        self._last_start = 123.0
        self.fail = fail
        self.delay = delay
        self.starts = 0
        self.stops = 0

    def launch(self):
        return [], ("exe", self.cfg.get("cleanup.model_path"))

    @property
    def loaded_model(self):
        return None

    def _abs(self, p):
        return cleanup.resolve_path(p)

    def start(self, wait=False):
        if self.ready:
            return
        self.starts += 1
        sig = self.launch()[1]

        def run():
            time.sleep(self.delay)
            if self.fail:
                self.failed = "boom"
            else:
                self.launched = sig
                self.ready = True
        threading.Thread(target=run, daemon=True).start()

    def stop(self, wait_start_s=10.0):
        self.stops += 1
        self.ready = False


class Restart(unittest.TestCase):
    def make(self, **kw):
        cfg = tmp_cfg()
        cfg.set("cleanup.model_path", QWEN, save=False)
        cl = Cleaner(cfg)
        cl.server = FakeServer(cfg, **kw)
        cl._warm = lambda: None
        return cfg, cl

    def test_switch_restarts_with_new_model(self):
        cfg, cl = self.make()
        cl.server.ready, cl.server.launched = True, ("exe", QWEN)
        old_http = cl.http
        cfg.set("cleanup.model_path", LFM, save=False)
        self.assertTrue(cl.restart(timeout_s=5))
        s = cl.server
        self.assertEqual((s.stops, s.starts), (1, 1))
        self.assertEqual(s.launched, ("exe", LFM))
        self.assertIsNot(cl.http, old_http)            # fresh connection pool for the new process

    def test_backoff_reset(self):
        cfg, cl = self.make()
        cl.server.failed = "earlier failure"
        cl.server._last_start = time.monotonic()
        self.assertTrue(cl.restart(timeout_s=5))
        self.assertEqual(cl.server._last_start, 0.0)
        self.assertIsNone(cl.server.failed)

    def test_noop_when_same_model_running(self):
        cfg, cl = self.make()
        cl.server.ready, cl.server.launched = True, ("exe", QWEN)
        self.assertTrue(cl.restart(timeout_s=5))
        self.assertEqual((cl.server.stops, cl.server.starts), (0, 0))

    def test_noop_when_suspended(self):
        cfg, cl = self.make()
        cl.server.suspended = True
        cfg.set("cleanup.model_path", LFM, save=False)
        self.assertTrue(cl.restart(timeout_s=5))
        self.assertEqual((cl.server.stops, cl.server.starts), (0, 0))

    def test_cleanup_disabled_only_stops_old_model(self):
        cfg, cl = self.make()
        cl.server.ready, cl.server.launched = True, ("exe", QWEN)
        cfg.set("cleanup.enabled", False, save=False)
        cfg.set("cleanup.model_path", LFM, save=False)
        self.assertTrue(cl.restart(timeout_s=5))
        self.assertEqual((cl.server.stops, cl.server.starts), (1, 0))

    def test_failure_reported(self):
        cfg, cl = self.make(fail=True)
        cfg.set("cleanup.model_path", LFM, save=False)
        self.assertFalse(cl.restart(timeout_s=5))
        self.assertEqual(cl.server.failed, "boom")

    def test_timeout_reported(self):
        cfg, cl = self.make(delay=2.0)
        self.assertFalse(cl.restart(timeout_s=0.2))

    def test_config_changed_during_load_restarts_again(self):
        cfg, cl = self.make(delay=0.2)
        cfg.set("cleanup.model_path", LFM, save=False)
        threading.Timer(0.05, lambda: cfg.set("cleanup.model_path", QWEN, save=False)).start()
        self.assertTrue(cl.restart(timeout_s=5))
        self.assertEqual(cl.server.launched, ("exe", QWEN))
        self.assertEqual(cl.server.starts, 2)


if __name__ == "__main__":
    unittest.main()
