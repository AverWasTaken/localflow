"""In-app updates: ask GitHub for the latest release, download its setup program, run it and let it replace us.

The release workflow (.github/workflows/release.yml) attaches ``LocalFlow-Setup-X.Y.Z.exe`` to the GitHub release
for tag ``vX.Y.Z``. ``check_latest()`` reads ``/releases/latest`` (which never returns drafts or prereleases) and
reports an update only when that release is newer than this copy *and* already has its setup program attached
(CI uploads it a few minutes after the release appears). ``download()`` fetches it into
``%TEMP%\\LocalFlow-update`` and checks its size and SHA-256 (GitHub publishes a digest per asset).
``launch_installer()`` starts it with ``/SILENT /RELAUNCH``: setup waits for this copy to quit, installs over
it and starts LocalFlow again. Only a copy made by that setup program can update itself (``can_self_update``);
a source checkout gets the release page instead.

Standard library only, no Qt, so it can be unit tested anywhere."""
from __future__ import annotations

import hashlib
import http.client
import json
import logging
import os
import re
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .paths import PROJECT_DIR

log = logging.getLogger(__name__)

REPO = "AverWasTaken/localflow"
LATEST_API_URL = f"https://api.github.com/repos/{REPO}/releases/latest"
RELEASES_URL = f"https://github.com/{REPO}/releases/latest"
CHECK_TIMEOUT_S = 10.0          # the release check: connect + read
DOWNLOAD_TIMEOUT_S = 30.0       # the setup download: connect / per-read
CHUNK = 64 << 10
PROGRESS_EVERY_S = 0.2
MAX_NOTES_CHARS = 4000          # release notes kept from the API (the dialog trims further)

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200

_VERSION_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class UpdateError(RuntimeError):
    pass


class UpdateCancelled(UpdateError):
    pass


@dataclass(frozen=True)
class UpdateInfo:
    version: str                # "1.3.0"
    tag: str                    # "v1.3.0"
    html_url: str               # the release page
    asset_name: str             # "LocalFlow-Setup-1.3.0.exe"
    asset_url: str              # browser_download_url
    size: int                   # bytes
    sha256: Optional[str]       # lowercase hex, or None when GitHub gave no digest
    notes: str = ""             # release notes (markdown), may be empty


# ----------------------------------------------------------------------------- versions
def current_version() -> str:
    from . import __version__       # read at call time (tests patch localflow.__version__)
    return __version__


def parse_version(text) -> Optional[tuple[int, int, int]]:
    """'v1.2.3' / '1.2.3' -> (1, 2, 3); anything else (prerelease suffixes, garbage) -> None."""
    m = _VERSION_RE.match(str(text or "").strip())
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def is_newer(candidate, current) -> bool:
    """True when ``candidate`` is a valid version greater than ``current`` (numeric compare: 1.10.0 > 1.9.9).
    An unparseable current version counts as 0.0.0."""
    c = parse_version(candidate)
    if c is None:
        return False
    return c > (parse_version(current) or (0, 0, 0))


def installer_name(version: str) -> str:
    """The setup program's asset name for a version. The release workflow uploads exactly this name."""
    return f"LocalFlow-Setup-{version}.exe"


def parse_digest(value) -> Optional[str]:
    """GitHub asset digest 'sha256:<hex>' -> lowercase hex; anything else -> None."""
    s = str(value or "").strip()
    if not s.lower().startswith("sha256:"):
        return None
    hexpart = s[7:].strip().lower()
    return hexpart if _SHA256_RE.match(hexpart) else None


def parse_release(data, current: Optional[str] = None) -> Optional[UpdateInfo]:
    """An UpdateInfo when the release JSON ``data`` is newer than ``current`` and has its setup program attached,
    else None (not newer, not a vX.Y.Z tag, draft / prerelease, or setup not uploaded yet)."""
    if not isinstance(data, dict) or data.get("draft") or data.get("prerelease"):
        return None
    tag = str(data.get("tag_name") or "").strip()
    ver = parse_version(tag)
    if ver is None or not tag.startswith("v"):
        return None
    current = current_version() if current is None else current
    if not is_newer(tag, current):
        return None
    version = "%d.%d.%d" % ver
    want = installer_name(version)
    for a in data.get("assets") or []:
        if not isinstance(a, dict) or a.get("name") != want:
            continue
        url = str(a.get("browser_download_url") or "")
        try:
            size = int(a.get("size") or 0)
        except (TypeError, ValueError):
            size = 0
        if not url.startswith("https://") or size <= 0:
            return None
        if a.get("state") not in (None, "uploaded"):      # still uploading
            return None
        notes = str(data.get("body") or "")[:MAX_NOTES_CHARS]
        return UpdateInfo(version=version, tag=tag, html_url=str(data.get("html_url") or RELEASES_URL),
                          asset_name=want, asset_url=url, size=size, sha256=parse_digest(a.get("digest")),
                          notes=notes)
    return None         # release exists but CI has not attached the setup program yet


def check_latest(current: Optional[str] = None, timeout: float = CHECK_TIMEOUT_S) -> Optional[UpdateInfo]:
    """Ask GitHub for the latest release. Returns an UpdateInfo when an update can be offered, None when this copy
    is current (or the new release has no setup program yet). Raises UpdateError when GitHub can't be reached or
    answers with something unexpected."""
    current = current_version() if current is None else current
    req = urllib.request.Request(LATEST_API_URL, headers={
        "User-Agent": f"LocalFlow/{current}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read(4 << 20)
    except urllib.error.HTTPError as e:
        e.close()
        if e.code == 404:       # no published release at all
            return None
        if e.code in (403, 429):
            raise UpdateError("GitHub is limiting update checks right now. Try again later.") from e
        raise UpdateError(f"GitHub answered HTTP {e.code} {e.reason}") from e
    except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
        raise UpdateError(f"could not reach GitHub: {getattr(e, 'reason', e)}") from e
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise UpdateError("GitHub sent an unreadable answer") from e
    return parse_release(data, current)


def should_offer(info: Optional[UpdateInfo], skipped: str = "", manual: bool = False) -> bool:
    """Whether to show the update prompt. Launch checks respect a version the user chose to skip; manual checks
    ("Check for updates") always offer it."""
    if info is None:
        return False
    if manual:
        return True
    return not (skipped and parse_version(skipped) == parse_version(info.version))


# ----------------------------------------------------------------------------- download
def download_dir() -> Path:
    return Path(tempfile.gettempdir()) / "LocalFlow-update"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _unlink(p: Path) -> None:
    try:
        p.unlink()
    except OSError:
        pass


def _open(url: str, timeout: float):
    req = urllib.request.Request(url, headers={"User-Agent": f"LocalFlow/{current_version()}",
                                               "Accept": "application/octet-stream",
                                               "Accept-Encoding": "identity"})
    return urllib.request.urlopen(req, timeout=timeout)     # follows GitHub's redirect to its download host


def download(info: UpdateInfo, progress: Optional[Callable[[int, int], None]] = None,
             cancel: Optional[threading.Event] = None, dest_dir: Optional[Path] = None,
             timeout: float = DOWNLOAD_TIMEOUT_S) -> Path:
    """Download the setup program into ``dest_dir`` (default %TEMP%\\LocalFlow-update) through a ``.part`` file.
    Blocks; returns the final path. Checks the size, and the SHA-256 when GitHub gave one; a file that fails either
    is deleted. Raises UpdateError (UpdateCancelled when ``cancel`` is set). ``progress(done, total)`` is called
    from this thread at most every PROGRESS_EVERY_S and once at the end."""
    d = Path(dest_dir) if dest_dir is not None else download_dir()
    d.mkdir(parents=True, exist_ok=True)
    final = d / info.asset_name
    part = d / (info.asset_name + ".part")
    want = (info.sha256 or "").lower() or None
    for old in d.glob("LocalFlow-Setup-*.exe*"):        # earlier updates' leftovers
        if old.name not in (final.name,):
            _unlink(old)
    if final.is_file():         # downloaded before (e.g. the user cancelled the install): reuse only if verified
        if want and final.stat().st_size == info.size and _sha256_file(final) == want:
            if progress is not None:
                progress(info.size, info.size)
            return final
        _unlink(final)
    _unlink(part)
    h = hashlib.sha256()
    done = 0
    try:
        try:
            resp = _open(info.asset_url, timeout)
        except urllib.error.HTTPError as e:
            e.close()
            raise UpdateError(f"download failed: HTTP {e.code} {e.reason}") from e
        except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
            raise UpdateError(f"could not connect: {getattr(e, 'reason', e)}") from e
        with resp, open(part, "wb") as f:
            last = 0.0
            while True:
                if cancel is not None and cancel.is_set():
                    raise UpdateCancelled("cancelled")
                try:
                    b = resp.read(CHUNK)
                except (OSError, http.client.HTTPException) as e:
                    raise UpdateError(f"download interrupted: {e}") from e
                if not b:
                    break
                done += len(b)
                if done > info.size:
                    raise UpdateError("the server sent more data than expected - download discarded")
                f.write(b)
                h.update(b)
                now = time.monotonic()
                if progress is not None and now - last >= PROGRESS_EVERY_S:
                    last = now
                    progress(done, info.size)
        if done != info.size:
            raise UpdateError(f"download ended early ({done} of {info.size} bytes)")
        got = h.hexdigest()
        if want and got != want:
            raise UpdateError("the downloaded file failed its checksum - download discarded")
        os.replace(part, final)
    except BaseException:
        _unlink(part)
        raise
    if progress is not None:
        progress(info.size, info.size)
    log.info("update: downloaded %s (%d bytes, sha256 %s%s)", final, info.size, got,
             "" if want else ", no digest to compare")
    return final


# ----------------------------------------------------------------------------- install
def app_root() -> Path:
    return PROJECT_DIR


def can_self_update(root: Optional[Path] = None) -> bool:
    """True for a copy installed by the setup program (it leaves unins000.dat in the install folder). A source
    checkout or dev run can't update itself: it gets the release page instead."""
    r = Path(root) if root is not None else app_root()
    return (r / "unins000.dat").is_file()


def installer_command(path, root=None, log_path=None) -> str:
    """The setup command line. /SILENT shows only setup's progress window (and any error); /RELAUNCH makes the
    new setup wait for this copy to quit, leave the startup task alone and start LocalFlow again when done.
    /NOTASK: never re-register the startup task during an update. register-task.ps1 always re-registers and asks
    for admin whenever it isn't already elevated; the task (if any) already points at this folder and its
    definition doesn't change between versions."""
    r = str(Path(root) if root is not None else app_root()).rstrip("\\/")
    if len(r) == 2 and r[1] == ":":     # "C:" -> "C:\" (a bare drive is the drive's current directory)
        r += "\\"
    cmd = f'"{path}" /SILENT /NORESTART /DIR="{r}" /RELAUNCH /NOTASK'
    if log_path:
        cmd += f' /LOG="{log_path}"'
    return cmd


def launch_installer(path, root=None) -> subprocess.Popen:
    """Start the downloaded setup program, detached from this process, so the caller can quit right after.
    Setup inherits this process's rights (elevated when LocalFlow runs from its startup task), so it can stop
    the old copy and finish without an admin prompt."""
    p = Path(path)
    log_path = p.with_suffix(".log")
    cmd = installer_command(p, root, log_path)
    log.info("update: starting %s", cmd)
    return subprocess.Popen(cmd, cwd=str(p.parent), close_fds=True,
                            creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP)


# ----------------------------------------------------------------------------- display helpers
_BY_AUTHOR_RE = re.compile(r"\s+by @[\w.-]+ in https?://\S+\s*$")


def trim_notes(text: str, max_lines: int = 10, max_chars: int = 700) -> str:
    """Release notes (GitHub markdown) -> a few plain lines for the update prompt: headings and bold markers
    dropped, bullets as "- ", the "Full Changelog" link and the "New Contributors" section left out."""
    out: list[str] = []
    skip_section = False
    for raw in str(text or "").replace("\r\n", "\n").split("\n"):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            skip_section = heading.lower().startswith("new contributors")
            if not skip_section and heading.lower() not in ("what's changed", "whats changed"):
                out.append(heading.replace("**", ""))
            continue
        if skip_section or line.lower().startswith("**full changelog**"):
            continue
        if line[:2] in ("* ", "- ", "+ "):
            line = "- " + line[2:].strip()
        line = _BY_AUTHOR_RE.sub("", line).replace("**", "")
        out.append(line)
    more = len(out) > max_lines
    s = "\n".join(out[:max_lines])
    if len(s) > max_chars:
        s, more = s[:max_chars].rsplit(" ", 1)[0], True
    return s + ("\n..." if more and s else "")


def mb(n: float) -> str:
    return f"{n / (1 << 20):.1f}"
