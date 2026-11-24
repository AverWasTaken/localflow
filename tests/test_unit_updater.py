"""Offline tests for localflow/updater.py (in-app updates from GitHub releases). urllib and subprocess are mocked:
nothing touches the network, and no setup program is started."""
import hashlib
import io
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from localflow import updater  # noqa: E402
from localflow.config import DEFAULTS  # noqa: E402

DATA = bytes((i * 13 + 5) % 256 for i in range(200_000))
SHA = hashlib.sha256(DATA).hexdigest()


def _asset(version="1.3.0", size=len(DATA), digest=f"sha256:{SHA}", name=None, **extra):
    a = {"name": name or f"LocalFlow-Setup-{version}.exe", "size": size, "state": "uploaded",
         "browser_download_url": f"https://github.com/AverWasTaken/localflow/releases/download/v{version}/"
                                 f"LocalFlow-Setup-{version}.exe"}
    if digest is not None:
        a["digest"] = digest
    a.update(extra)
    return a


def _release(tag="v1.3.0", assets=None, **extra):
    r = {"tag_name": tag, "html_url": f"https://github.com/AverWasTaken/localflow/releases/tag/{tag}",
         "draft": False, "prerelease": False, "body": "## What's Changed\n* Faster things by @a in https://x/1",
         "assets": [_asset()] if assets is None else assets}
    r.update(extra)
    return r


class _Resp(io.BytesIO):
    """Stands in for the urlopen() response (context manager + read)."""
    status = 200

    def getcode(self):
        return self.status


def _info(**kw):
    base = dict(version="1.3.0", tag="v1.3.0", html_url="https://example.invalid/r", asset_name="LocalFlow-Setup-1.3.0.exe",
                asset_url="https://example.invalid/setup.exe", size=len(DATA), sha256=SHA, notes="")
    base.update(kw)
    return updater.UpdateInfo(**base)


class VersionTests(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(updater.parse_version("v1.2.3"), (1, 2, 3))
        self.assertEqual(updater.parse_version("1.10.0"), (1, 10, 0))
        for bad in ("", None, "v1.2", "v1.2.3-beta", "latest", "v1.2.3.4", "vx.y.z", " v1. 2.3"):
            self.assertIsNone(updater.parse_version(bad), bad)

    def test_compare_is_numeric(self):
        self.assertTrue(updater.is_newer("v1.10.0", "1.9.9"))
        self.assertTrue(updater.is_newer("v2.0.0", "1.99.99"))
        self.assertTrue(updater.is_newer("v1.2.1", "1.2.0"))
        self.assertFalse(updater.is_newer("v1.2.0", "1.2.0"))
        self.assertFalse(updater.is_newer("v1.1.9", "1.2.0"))
        self.assertFalse(updater.is_newer("garbage", "1.2.0"))
        self.assertTrue(updater.is_newer("v1.0.0", "dev"))         # unparseable current = 0.0.0

    def test_current_version_reads_package(self):
        with mock.patch("localflow.__version__", "0.0.1"):
            self.assertEqual(updater.current_version(), "0.0.1")

    def test_digest(self):
        self.assertEqual(updater.parse_digest(f"sha256:{SHA.upper()}"), SHA)
        self.assertIsNone(updater.parse_digest("sha512:" + "a" * 128))
        self.assertIsNone(updater.parse_digest("sha256:xyz"))
        self.assertIsNone(updater.parse_digest(None))


class ParseReleaseTests(unittest.TestCase):
    def test_newer_release_with_asset(self):
        i = updater.parse_release(_release(), "1.2.0")
        self.assertIsNotNone(i)
        self.assertEqual((i.version, i.tag, i.asset_name, i.size, i.sha256),
                         ("1.3.0", "v1.3.0", "LocalFlow-Setup-1.3.0.exe", len(DATA), SHA))
        self.assertTrue(i.asset_url.endswith("/v1.3.0/LocalFlow-Setup-1.3.0.exe"))
        self.assertIn("Faster things", i.notes)

    def test_not_newer(self):
        self.assertIsNone(updater.parse_release(_release(), "1.3.0"))
        self.assertIsNone(updater.parse_release(_release(), "1.4.0"))

    def test_missing_asset_is_no_update(self):
        self.assertIsNone(updater.parse_release(_release(assets=[]), "1.2.0"))

    def test_wrong_version_asset_ignored(self):
        rel = _release(assets=[_asset(version="1.2.0"), _asset(name="LocalFlow-Setup-1.3.0.zip")])
        self.assertIsNone(updater.parse_release(rel, "1.2.0"))
        rel = _release(assets=[_asset(version="1.2.0"), _asset()])
        self.assertEqual(updater.parse_release(rel, "1.2.0").asset_name, "LocalFlow-Setup-1.3.0.exe")

    def test_garbage_tags_and_flags(self):
        for tag in ("1.3.0", "v1.3", "v1.3.0-rc1", "nightly", ""):
            self.assertIsNone(updater.parse_release(_release(tag=tag), "1.2.0"), tag)
        self.assertIsNone(updater.parse_release(_release(draft=True), "1.2.0"))
        self.assertIsNone(updater.parse_release(_release(prerelease=True), "1.2.0"))
        self.assertIsNone(updater.parse_release("not a dict", "1.2.0"))

    def test_no_digest_and_bad_assets(self):
        i = updater.parse_release(_release(assets=[_asset(digest=None)]), "1.2.0")
        self.assertIsNone(i.sha256)
        self.assertIsNone(updater.parse_release(_release(assets=[_asset(size=0)]), "1.2.0"))
        self.assertIsNone(updater.parse_release(_release(assets=[_asset(state="new")]), "1.2.0"))
        bad = _asset()
        bad["browser_download_url"] = "http://insecure/setup.exe"
        self.assertIsNone(updater.parse_release(_release(assets=[bad]), "1.2.0"))


class CheckLatestTests(unittest.TestCase):
    def test_request_and_result(self):
        body = json.dumps(_release()).encode()
        with mock.patch.object(updater.urllib.request, "urlopen", return_value=_Resp(body)) as uo:
            i = updater.check_latest(current="1.2.0")
        self.assertEqual(i.version, "1.3.0")
        req = uo.call_args[0][0]
        self.assertEqual(req.full_url, updater.LATEST_API_URL)
        self.assertEqual(req.get_header("User-agent"), "LocalFlow/1.2.0")
        self.assertEqual(req.get_header("Accept"), "application/vnd.github+json")
        self.assertLessEqual(uo.call_args[1]["timeout"], 15)

    def test_current_is_latest(self):
        body = json.dumps(_release(tag="v1.2.0", assets=[_asset("1.2.0")])).encode()
        with mock.patch.object(updater.urllib.request, "urlopen", return_value=_Resp(body)):
            self.assertIsNone(updater.check_latest(current="1.2.0"))

    def test_errors(self):
        def http(code):
            return urllib.error.HTTPError(updater.LATEST_API_URL, code, "x", {}, io.BytesIO(b""))
        with mock.patch.object(updater.urllib.request, "urlopen", side_effect=http(404)):
            self.assertIsNone(updater.check_latest(current="1.2.0"))        # no release published at all
        for exc in (http(500), http(403), urllib.error.URLError("offline"), TimeoutError("slow")):
            with mock.patch.object(updater.urllib.request, "urlopen", side_effect=exc):
                with self.assertRaises(updater.UpdateError):
                    updater.check_latest(current="1.2.0")
        with mock.patch.object(updater.urllib.request, "urlopen", return_value=_Resp(b"<html>")):
            with self.assertRaises(updater.UpdateError):
                updater.check_latest(current="1.2.0")


class SkipTests(unittest.TestCase):
    def test_should_offer(self):
        i = _info()
        self.assertFalse(updater.should_offer(None))
        self.assertTrue(updater.should_offer(i, ""))
        self.assertFalse(updater.should_offer(i, "1.3.0"))
        self.assertFalse(updater.should_offer(i, "v1.3.0"))
        self.assertTrue(updater.should_offer(i, "1.3.0", manual=True))     # manual checks ignore the skip
        self.assertTrue(updater.should_offer(i, "1.2.9"))                  # skipping an older one doesn't hide this

    def test_config_defaults(self):
        self.assertEqual(DEFAULTS["updates"], {"check_on_launch": True, "skipped_version": ""})


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _dl(self, info, data=DATA, **kw):
        with mock.patch.object(updater.urllib.request, "urlopen", side_effect=lambda *a, **k: _Resp(data)) as uo:
            return updater.download(info, dest_dir=self.d, **kw), uo

    def test_ok_with_progress(self):
        seen = []
        p, uo = self._dl(_info(), progress=lambda d, t: seen.append((d, t)))
        self.assertEqual(p, self.d / "LocalFlow-Setup-1.3.0.exe")
        self.assertEqual(p.read_bytes(), DATA)
        self.assertEqual(seen[-1], (len(DATA), len(DATA)))
        self.assertFalse((self.d / "LocalFlow-Setup-1.3.0.exe.part").exists())

    def test_sha_mismatch_deletes(self):
        with self.assertRaises(updater.UpdateError):
            self._dl(_info(sha256="0" * 64))
        self.assertEqual(list(self.d.iterdir()), [])

    def test_size_mismatch_deletes(self):
        with self.assertRaises(updater.UpdateError):
            self._dl(_info(), data=DATA[:-10])          # short
        self.assertEqual(list(self.d.iterdir()), [])
        with self.assertRaises(updater.UpdateError):
            self._dl(_info(size=len(DATA) - 10, sha256=None))    # server sends more than expected
        self.assertEqual(list(self.d.iterdir()), [])

    def test_no_digest_checks_size_only(self):
        p, _ = self._dl(_info(sha256=None))
        self.assertEqual(p.read_bytes(), DATA)

    def test_cancel_deletes(self):
        ev = threading.Event()
        ev.set()
        with self.assertRaises(updater.UpdateCancelled):
            self._dl(_info(), cancel=ev)
        self.assertEqual(list(self.d.iterdir()), [])

    def test_http_error(self):
        err = urllib.error.HTTPError("u", 404, "Not Found", {}, io.BytesIO(b""))
        with mock.patch.object(updater.urllib.request, "urlopen", side_effect=err):
            with self.assertRaises(updater.UpdateError):
                updater.download(_info(), dest_dir=self.d)
        self.assertEqual(list(self.d.iterdir()), [])

    def test_reuses_verified_file_and_removes_old_ones(self):
        (self.d / "LocalFlow-Setup-1.2.0.exe").write_bytes(b"old")
        (self.d / "LocalFlow-Setup-1.3.0.exe").write_bytes(DATA)
        with mock.patch.object(updater.urllib.request, "urlopen") as uo:
            p = updater.download(_info(), dest_dir=self.d)
        uo.assert_not_called()
        self.assertEqual(p.read_bytes(), DATA)
        self.assertFalse((self.d / "LocalFlow-Setup-1.2.0.exe").exists())

    def test_bad_existing_file_is_downloaded_again(self):
        (self.d / "LocalFlow-Setup-1.3.0.exe").write_bytes(b"x" * len(DATA))
        p, uo = self._dl(_info())
        uo.assert_called_once()
        self.assertEqual(p.read_bytes(), DATA)


class InstallTests(unittest.TestCase):
    def test_can_self_update(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertFalse(updater.can_self_update(Path(t)))
            (Path(t) / "unins000.dat").write_bytes(b"")
            self.assertTrue(updater.can_self_update(Path(t)))

    def test_installer_command(self):
        cmd = updater.installer_command(r"C:\Temp\LocalFlow-update\LocalFlow-Setup-1.3.0.exe",
                                        r"C:\Users\A B\AppData\Local\Programs\LocalFlow" + "\\")
        self.assertEqual(cmd, r'"C:\Temp\LocalFlow-update\LocalFlow-Setup-1.3.0.exe" /SILENT /NORESTART '
                              r'/DIR="C:\Users\A B\AppData\Local\Programs\LocalFlow" /RELAUNCH /NOTASK')
        self.assertNotIn("SUPPRESSMSGBOXES", cmd)
        self.assertIn('/DIR="D:\\"', updater.installer_command("s.exe", "D:"))
        self.assertTrue(updater.installer_command("s.exe", "C:\\x", "C:\\l.log").endswith(' /LOG="C:\\l.log"'))

    def test_launch_installer(self):
        with mock.patch.object(updater.subprocess, "Popen") as po:
            updater.launch_installer(Path("/tmp/LocalFlow-update/LocalFlow-Setup-1.3.0.exe"), root="C:\\LF")
        args, kw = po.call_args
        self.assertIn("/SILENT /NORESTART /DIR=\"C:\\LF\" /RELAUNCH /NOTASK", args[0])
        self.assertIn("LocalFlow-Setup-1.3.0.log", args[0])
        self.assertEqual(kw["creationflags"], updater.DETACHED_PROCESS | updater.CREATE_NEW_PROCESS_GROUP)
        self.assertTrue(kw["close_fds"])
        self.assertEqual(kw["cwd"], str(Path("/tmp/LocalFlow-update")))


class NotesTests(unittest.TestCase):
    def test_trim(self):
        body = ("## What's Changed\n* Add updates by @me in https://github.com/x/pull/2\n* Fix **bug**\n\n"
                "## New Contributors\n* @me made their first contribution in https://github.com/x/pull/1\n\n"
                "**Full Changelog**: https://github.com/x/compare/v1...v2")
        self.assertEqual(updater.trim_notes(body), "- Add updates\n- Fix bug")
        long = "\n".join(f"* item {i}" for i in range(30))
        out = updater.trim_notes(long, max_lines=5)
        self.assertEqual(out.count("\n"), 5)
        self.assertTrue(out.endswith("..."))
        self.assertEqual(updater.trim_notes(""), "")


if __name__ == "__main__":
    unittest.main()
