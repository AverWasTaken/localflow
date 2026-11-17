"""Offline tests for localflow/modeldl.py (background download of the default cleanup model) and the model pickers
listing a downloadable model that is not in models/ yet. Downloads come from a local http.server; the registry is
patched to a small fake file, so nothing touches the network or the real models/ folder."""
import hashlib
import os
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from localflow import cleanup, modeldl  # noqa: E402
from localflow.config import Config, DEFAULTS  # noqa: E402

DATA = bytes((i * 7 + 3) % 256 for i in range(300_000))      # ~0.3 MB fake model
SHA = hashlib.sha256(DATA).hexdigest()
FAKE = "Fake-LocalFlow-test-Q8_0.gguf"


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        srv = self.server
        srv.requests.append((self.path, self.headers.get("Range")))
        if self.path.startswith("/redirect"):
            self.send_response(302)
            self.send_header("Location", "/file")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path != "/file":
            self.send_error(404)
            return
        data = srv.data
        rng = self.headers.get("Range")
        start = 0
        if rng and srv.honor_range:
            start = int(rng.split("=", 1)[1].split("-", 1)[0])
            if start >= len(data):
                self.send_response(416)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        else:
            self.send_response(200)
        body = data[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if srv.truncate_at is not None:
            self.wfile.write(body[:srv.truncate_at])
            self.wfile.flush()
            self.close_connection = True
            return
        self.wfile.write(body)


class _Server:
    def __init__(self):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.httpd.requests = []
        self.httpd.data = DATA
        self.httpd.honor_range = True
        self.httpd.truncate_at = None
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def url(self, path):
        return f"http://127.0.0.1:{self.httpd.server_address[1]}{path}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class Base(unittest.TestCase):
    def setUp(self):
        p = mock.patch.dict(os.environ, {"NO_PROXY": "*", "no_proxy": "*"})     # never via a system proxy
        p.start()
        self.addCleanup(p.stop)
        self.srv = _Server()
        self.addCleanup(self.srv.close)
        self.dir = Path(tempfile.mkdtemp())

    def info(self, path="/file", sha=SHA, size=len(DATA)):
        return modeldl.ModelInfo(FAKE, self.srv.url(path), size, sha)

    def part(self):
        return self.dir / (FAKE + ".part")


class Registry(unittest.TestCase):
    def test_release_url_exact(self):
        info = modeldl.REGISTRY["LFM3-1.7B-LocalFlow-r1e3-Q8_0.gguf"]
        self.assertEqual(info.url, "https://github.com/AverWasTaken/localflow/releases/download/v1.2.0/"
                                   "LFM3-1.7B-LocalFlow-r1e3-Q8_0.gguf")
        self.assertEqual(info.size, 1846237728)
        self.assertEqual(info.sha256, "d7581422f333d2aef23118966a14c6f9363f2e6c6316d39b79c8535efc070e16")

    def test_lookup_case_insensitive(self):
        self.assertIs(modeldl.lookup("lfm3-1.7b-localflow-r1e3-q8_0.GGUF"), modeldl.REGISTRY[modeldl.DEFAULT_MODEL])
        self.assertIsNone(modeldl.lookup("Qwen3.5-4B-Q8_0.gguf"))

    def test_default_config_uses_registry_model(self):
        self.assertEqual(DEFAULTS["cleanup"]["model_path"], f"models/{modeldl.DEFAULT_MODEL}")

    def test_saved_model_path_is_kept(self):
        d = Path(tempfile.mkdtemp())
        (d / "config.json").write_text('{"cleanup": {"model_path": "models/Qwen3.5-4B-Q8_0.gguf"}}', encoding="utf-8")
        self.assertEqual(Config(path=d / "config.json").get("cleanup.model_path"), "models/Qwen3.5-4B-Q8_0.gguf")

    def test_finetune_gets_compact_prompt_and_stops(self):
        m = Path(modeldl.DEFAULT_MODEL)
        cfg = Config(path=Path(tempfile.mkdtemp()) / "config.json")
        self.assertEqual(cleanup._per_model_family(cfg, "model_prompt", m), "compact")
        self.assertEqual(cleanup._per_model_family(cfg, "model_request", m), {"stop": ["<think>", "</think>"]})

    def test_gb(self):
        self.assertEqual(modeldl.gb(1846237728), "1.7")


class Download(Base):
    def test_success_verifies_and_renames(self):
        seen = []
        out = modeldl.download(self.info(), self.dir, progress=lambda d, t: seen.append((d, t)))
        self.assertEqual(out, self.dir / FAKE)
        self.assertEqual(out.read_bytes(), DATA)
        self.assertFalse(self.part().exists())
        self.assertEqual(seen[-1], (len(DATA), len(DATA)))

    def test_hash_mismatch_deletes_part(self):
        with self.assertRaises(modeldl.DownloadError) as cm:
            modeldl.download(self.info(sha="0" * 64), self.dir)
        self.assertIn("checksum", str(cm.exception))
        self.assertFalse(self.part().exists())
        self.assertFalse((self.dir / FAKE).exists())

    def test_resume_with_range(self):
        self.part().write_bytes(DATA[:123_456])
        modeldl.download(self.info(), self.dir)
        self.assertEqual((self.dir / FAKE).read_bytes(), DATA)
        self.assertEqual(self.srv.httpd.requests, [("/file", "bytes=123456-")])

    def test_range_ignored_restarts(self):
        self.srv.httpd.honor_range = False
        self.part().write_bytes(b"x" * 1000)      # junk: must be overwritten, not appended to
        modeldl.download(self.info(), self.dir)
        self.assertEqual((self.dir / FAKE).read_bytes(), DATA)

    def test_follows_redirect_and_keeps_range(self):
        self.part().write_bytes(DATA[:50_000])
        modeldl.download(self.info("/redirect"), self.dir)
        self.assertEqual((self.dir / FAKE).read_bytes(), DATA)
        self.assertEqual(self.srv.httpd.requests, [("/redirect", "bytes=50000-"), ("/file", "bytes=50000-")])

    def test_interrupted_keeps_part_then_resumes(self):
        self.srv.httpd.truncate_at = 100_000
        with self.assertRaises(modeldl.DownloadError):
            modeldl.download(self.info(), self.dir)
        self.assertEqual(self.part().read_bytes(), DATA[:100_000])
        self.assertFalse((self.dir / FAKE).exists())
        self.srv.httpd.truncate_at = None
        modeldl.download(self.info(), self.dir)
        self.assertEqual((self.dir / FAKE).read_bytes(), DATA)
        self.assertEqual(self.srv.httpd.requests[-1], ("/file", "bytes=100000-"))

    def test_http_error_keeps_nothing_and_raises(self):
        with self.assertRaises(modeldl.DownloadError) as cm:
            modeldl.download(self.info("/nope"), self.dir)
        self.assertIn("404", str(cm.exception))
        self.assertFalse((self.dir / FAKE).exists())

    def test_complete_part_is_only_verified(self):
        self.part().write_bytes(DATA)
        modeldl.download(self.info(), self.dir)
        self.assertEqual((self.dir / FAKE).read_bytes(), DATA)
        self.assertEqual(self.srv.httpd.requests, [])

    def test_416_starts_over_once(self):
        self.srv.httpd.data = DATA[:1000]          # server copy shorter than our .part claims
        info = modeldl.ModelInfo(FAKE, self.srv.url("/file"), 1000, hashlib.sha256(DATA[:1000]).hexdigest())
        self.part().write_bytes(b"y" * 1000 + b"")  # == size: verified locally -> mismatch -> deleted
        with self.assertRaises(modeldl.DownloadError):
            modeldl.download(info, self.dir)
        self.assertFalse(self.part().exists())
        self.part().write_bytes(b"z" * 10)
        self.srv.httpd.data = DATA[:5]             # range start 10 >= 5 -> 416 -> start over without a range
        info = modeldl.ModelInfo(FAKE, self.srv.url("/file"), 1000, "0" * 64)
        with self.assertRaises(modeldl.DownloadError):
            modeldl.download(info, self.dir)
        self.assertEqual([r for _, r in self.srv.httpd.requests], ["bytes=10-", None])

    def test_cancel_keeps_part(self):
        ev = threading.Event()

        def prog(done, total):
            ev.set()
        with mock.patch.object(modeldl, "CHUNK", 4096), mock.patch.object(modeldl, "PROGRESS_EVERY_S", 0.0):
            with self.assertRaises(modeldl.DownloadCancelled):
                modeldl.download(self.info(), self.dir, progress=prog, cancel=ev)
        self.assertTrue(self.part().exists())
        self.assertFalse((self.dir / FAKE).exists())


class Downloader(Base):
    def test_background_download_single_flight(self):
        info = self.info()
        done = threading.Event()
        results = []
        gate = threading.Event()
        orig = modeldl.download

        def slow(*a, **k):
            gate.wait(5)
            return orig(*a, **k)
        with mock.patch.dict(modeldl.REGISTRY, {FAKE: info}), mock.patch.object(modeldl, "download", slow):
            dl = modeldl.ModelDownloader(self.dir)
            self.assertIsNotNone(dl.needs_download(FAKE))
            self.assertTrue(dl.start(FAKE, on_done=lambda n, ok, err: (results.append((n, ok, err)), done.set())))
            self.assertTrue(dl.active(FAKE))
            self.assertFalse(dl.start(FAKE))            # one download per file
            self.assertEqual(dl.progress(FAKE), (0, len(DATA)))
            gate.set()
            self.assertTrue(done.wait(10))
            self.assertEqual(results, [(FAKE, True, "")])
            self.assertFalse(dl.active(FAKE))
            self.assertIsNone(dl.needs_download(FAKE))  # present now
            self.assertFalse(dl.start(FAKE))

    def test_failure_reported_and_retry_allowed(self):
        done = threading.Event()
        results = []
        with mock.patch.dict(modeldl.REGISTRY, {FAKE: self.info("/nope")}):
            dl = modeldl.ModelDownloader(self.dir)
            self.assertTrue(dl.start(FAKE, on_done=lambda n, ok, err: (results.append((ok, err)), done.set())))
            self.assertTrue(done.wait(10))
            self.assertFalse(results[0][0])
            self.assertIn("404", results[0][1])
            self.assertIn(FAKE, dl.last_error)
            done.clear()
            self.assertTrue(dl.start(FAKE, on_done=lambda n, ok, err: done.set()))   # no retry loop; a new start works
            self.assertTrue(done.wait(10))

    def test_no_done_callback_after_cancel(self):
        info = self.info()
        calls = []
        with mock.patch.dict(modeldl.REGISTRY, {FAKE: info}):
            dl = modeldl.ModelDownloader(self.dir)
            orig = modeldl.download

            def finish_then_quit(*a, **k):      # download completes just as the app quits
                out = orig(*a, **k)
                dl.cancel_all()
                return out
            with mock.patch.object(modeldl, "download", finish_then_quit):
                self.assertTrue(dl.start(FAKE, on_done=lambda *x: calls.append(x)))
                for _ in range(200):
                    if not dl.active(FAKE):
                        break
                    time.sleep(0.02)
        self.assertFalse(dl.active(FAKE))
        self.assertEqual(calls, [])
        self.assertTrue((self.dir / FAKE).exists())

    def test_unknown_or_cancelled_never_starts(self):
        dl = modeldl.ModelDownloader(self.dir)
        self.assertFalse(dl.start("Qwen3.5-4B-Q8_0.gguf"))
        with mock.patch.dict(modeldl.REGISTRY, {FAKE: self.info()}):
            dl.cancel_all()
            self.assertFalse(dl.start(FAKE))


class PickerList(unittest.TestCase):
    def test_choices_include_missing_registry_model(self):
        d = Path(tempfile.mkdtemp())
        (d / "Qwen3.5-4B-Q8_0.gguf").write_bytes(b"x")
        info = modeldl.ModelInfo(FAKE, "http://x/y", 1846237728, "0" * 64)
        with mock.patch.dict(modeldl.REGISTRY, {FAKE: info}, clear=True):
            self.assertEqual(cleanup.list_model_choices(d), [
                (FAKE, "Fake-LocalFlow-test-Q8_0 (download, 1.7 GB)"),
                ("Qwen3.5-4B-Q8_0.gguf", "Qwen3.5-4B-Q8_0"),
            ])
            (d / (FAKE + ".part")).write_bytes(b"x")    # a partial download does not count as present
            self.assertEqual(cleanup.list_model_choices(d)[0][1], "Fake-LocalFlow-test-Q8_0 (download, 1.7 GB)")
            (d / FAKE).write_bytes(b"x")                 # downloaded: listed like any other file
            self.assertEqual(cleanup.list_model_choices(d), [(FAKE, "Fake-LocalFlow-test-Q8_0"),
                                                             ("Qwen3.5-4B-Q8_0.gguf", "Qwen3.5-4B-Q8_0")])

    def test_settings_combo_lists_downloadable_model(self):
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        from localflow.settings_ui import SettingsWindow
        info = modeldl.ModelInfo(FAKE, "http://x/y", 1846237728, "0" * 64)
        with mock.patch.dict(modeldl.REGISTRY, {FAKE: info}):
            cfg = Config(path=Path(tempfile.mkdtemp()) / "config.json")
            w = SettingsWindow(cfg, lambda: [], lambda: False, lambda v: None, get_last_ctx=lambda: ("", ""),
                               get_default_mic=lambda: "")
            w.reload()
            i = w.llm_combo.findData(f"models/{FAKE}")
            self.assertGreaterEqual(i, 0)
            self.assertEqual(w.llm_combo.itemText(i), "Fake-LocalFlow-test-Q8_0 (download, 1.7 GB)")
            # picking it again (already configured, still missing) asks the app to retry the download
            picked = []
            w.app = SimpleNamespace(select_cleanup_model=picked.append)
            cfg.set("cleanup.model_path", f"models/{FAKE}", save=False)
            w._model_picked(i)
            self.assertEqual(picked, [FAKE])


class AppEnsure(unittest.TestCase):
    """LocalFlowApp._ensure_cleanup_model on a bare instance (no GUI, no models loaded)."""

    def make(self, model=f"models/{FAKE}", enabled=True):
        from localflow import app as app_mod
        a = app_mod.LocalFlowApp.__new__(app_mod.LocalFlowApp)
        a.cfg = Config(path=Path(tempfile.mkdtemp()) / "config.json")
        a.cfg.set("cleanup.model_path", model, save=False)
        a.cfg.set("cleanup.enabled", enabled, save=False)
        a.downloads = SimpleNamespace(started=[], start=lambda n, p, d: a.downloads.started.append(n) or True,
                                      active=lambda n: False)
        a.toasts, a.statuses = [], []
        a.bridge = SimpleNamespace(toast=SimpleNamespace(emit=lambda s, t: a.toasts.append((s, t))))
        a._set_status = lambda t, i: a.statuses.append((t, i))
        a.cleaner = SimpleNamespace(server=SimpleNamespace(ready=False))
        a._dl_pct = -1
        a._shutdown = threading.Event()
        return a

    def test_missing_registry_model_starts_download(self):
        info = modeldl.ModelInfo(FAKE, "http://x/y", 1846237728, "0" * 64)
        with mock.patch.dict(modeldl.REGISTRY, {FAKE: info}):
            a = self.make()
            self.assertTrue(a._ensure_cleanup_model())
            self.assertEqual(a.downloads.started, [FAKE])
            self.assertIn("basic cleanup", a.toasts[0][1])
            self.assertTrue(a.statuses[0][0].startswith("Downloading cleanup model Fake-LocalFlow-test-Q8_0 ... 0%"))

    def test_other_or_disabled_models_untouched(self):
        info = modeldl.ModelInfo(FAKE, "http://x/y", 1846237728, "0" * 64)
        with mock.patch.dict(modeldl.REGISTRY, {FAKE: info}):
            for a in (self.make("models/Qwen-not-there.gguf"), self.make(enabled=False),
                      self.make(f"C:/elsewhere/{FAKE}")):
                self.assertFalse(a._ensure_cleanup_model())
                self.assertEqual(a.downloads.started, [])

    def test_progress_text_once_per_percent(self):
        info = modeldl.ModelInfo(FAKE, "http://x/y", 1846237728, "0" * 64)
        with mock.patch.dict(modeldl.REGISTRY, {FAKE: info}):
            a = self.make()
            a._on_download_progress(FAKE, 777_000_000, 1846237728)
            a._on_download_progress(FAKE, 777_000_001, 1846237728)
            self.assertEqual(a.statuses, [("Downloading cleanup model Fake-LocalFlow-test-Q8_0 ... 42% (0.7 / 1.7 GB)",
                                           "ready")])

    def test_done_loads_configured_model(self):
        a = self.make()
        calls = []
        a._apply_cleanup_model = lambda background=False: calls.append(background)
        a._on_download_done(FAKE, True, "")
        self.assertEqual(calls, [True])
        a.cfg.set("cleanup.model_path", "models/Other.gguf", save=False)    # user switched away meanwhile
        a._on_download_done(FAKE, True, "")
        self.assertEqual(calls, [True])
        self.assertIn("pick it", a.toasts[-1][1])
        a._on_download_done(FAKE, False, "HTTP 404")
        self.assertEqual(a.toasts[-1][0], "error")


if __name__ == "__main__":
    unittest.main()
