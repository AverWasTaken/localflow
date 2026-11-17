"""Downloads of known cleanup models (the LocalFlow fine-tune) into models/, in the background.

Only files named in REGISTRY can be downloaded; each has a pinned size and sha256. A download streams into
``<name>.part`` next to the final file (hashing while it streams), resumes an existing .part with an HTTP Range
request when the server allows it, checks the size and hash, then renames the .part into place. A failed download
keeps the .part for the next try (deleted only when the hash is wrong). Nothing retries on its own: the app starts
a download again on its next start or when the user picks the model again. Standard library only (urllib)."""
from __future__ import annotations

import hashlib
import http.client
import logging
import os
import re
import shutil
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .paths import MODELS_DIR

log = logging.getLogger(__name__)

MODEL_RELEASE_TAG = "v1.2.0"
RELEASE_BASE_URL = "https://github.com/AverWasTaken/localflow/releases/download"
CHUNK = 1 << 20                 # 1 MiB reads
SOCKET_TIMEOUT_S = 30.0         # connect / per-read timeout
PROGRESS_EVERY_S = 0.5          # progress callback at most ~2x per second
DISK_MARGIN = 200 << 20         # keep this much free beyond the rest of the download


@dataclass(frozen=True)
class ModelInfo:
    name: str       # file name in models/
    url: str
    size: int       # bytes
    sha256: str     # lowercase hex


def _release_asset(name: str, size: int, sha256: str) -> ModelInfo:
    return ModelInfo(name, f"{RELEASE_BASE_URL}/{MODEL_RELEASE_TAG}/{name}", size, sha256)


DEFAULT_MODEL = "LFM3-1.7B-LocalFlow-r1e3-Q8_0.gguf"
REGISTRY: dict[str, ModelInfo] = {
    DEFAULT_MODEL: _release_asset(DEFAULT_MODEL, 1846237728,
                                  "d7581422f333d2aef23118966a14c6f9363f2e6c6316d39b79c8535efc070e16"),
}


class DownloadError(RuntimeError):
    pass


class DownloadCancelled(DownloadError):
    pass


def lookup(filename: str) -> Optional[ModelInfo]:
    """Registry entry for a model file name (case-insensitive), or None."""
    low = str(filename or "").lower()
    for name, info in REGISTRY.items():
        if name.lower() == low:
            return info
    return None


def gb(n: float) -> str:
    """Bytes -> '1.7' (GiB, as Windows Explorer shows sizes)."""
    return f"{n / (1 << 30):.1f}"


def _parse_content_range(value: str) -> Optional[tuple[int, int]]:
    """'bytes 100-199/200' -> (100, 200); total may be '*' -> -1. None if unparseable."""
    m = re.match(r"\s*bytes\s+(\d+)-(\d+)/(\d+|\*)\s*$", value or "")
    if not m:
        return None
    return int(m.group(1)), (-1 if m.group(3) == "*" else int(m.group(3)))


def _hash_file(path: Path, h, cancel: Optional[threading.Event]) -> None:
    with open(path, "rb") as f:
        while True:
            if cancel is not None and cancel.is_set():
                raise DownloadCancelled("cancelled")
            b = f.read(CHUNK)
            if not b:
                return
            h.update(b)


def _open(url: str, start: int, timeout: float):
    headers = {"User-Agent": "LocalFlow-model-download", "Accept-Encoding": "identity"}
    if start > 0:
        headers["Range"] = f"bytes={start}-"
    # urllib follows the 302 to objects.githubusercontent.com and carries the Range header along
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout)


def download(info: ModelInfo, dest_dir: Optional[Path] = None,
             progress: Optional[Callable[[int, int], None]] = None,
             cancel: Optional[threading.Event] = None, timeout: float = SOCKET_TIMEOUT_S) -> Path:
    """Download ``info`` into ``dest_dir`` (default models/). Blocks; returns the final path. Raises DownloadError
    (DownloadCancelled when ``cancel`` was set). ``progress(done, total)`` is called at most every PROGRESS_EVERY_S
    and once at the end."""
    d = Path(dest_dir) if dest_dir is not None else MODELS_DIR
    d.mkdir(parents=True, exist_ok=True)
    final = d / info.name
    part = d / (info.name + ".part")
    if final.is_file() and final.stat().st_size == info.size:
        return final
    want = info.sha256.lower()
    retried_416 = False
    while True:
        have = part.stat().st_size if part.is_file() else 0
        if have > info.size:
            log.warning("model download: %s is larger than expected - starting over", part.name)
            part.unlink()
            have = 0
        h = hashlib.sha256()
        if have == info.size:       # complete from an earlier run: just verify
            _hash_file(part, h, cancel)
            break
        try:
            free = shutil.disk_usage(d).free
        except OSError:
            free = None
        if free is not None and free < info.size - have + DISK_MARGIN:
            raise DownloadError(f"not enough disk space in {d} ({gb(free)} GB free, "
                                f"{gb(info.size - have)} GB needed)")
        if have > 0:        # hash what is already there before connecting (a slow disk must not idle the socket)
            _hash_file(part, h, cancel)
        try:
            resp = _open(info.url, have, timeout)
        except urllib.error.HTTPError as e:
            e.close()
            if e.code == 416 and have > 0 and not retried_416:      # server rejects our range: start over once
                log.warning("model download: server refused to resume %s (416) - starting over", info.name)
                part.unlink()
                retried_416 = True
                continue
            raise DownloadError(f"HTTP {e.code} {e.reason} for {info.url}") from e
        except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
            raise DownloadError(f"could not connect: {getattr(e, 'reason', e)}") from e
        with resp:
            status = getattr(resp, "status", None) or resp.getcode()
            mode = "ab"
            if have > 0 and status == 206:
                cr = _parse_content_range(resp.headers.get("Content-Range", ""))
                if cr is None or cr[0] != have or (cr[1] not in (-1, info.size)):
                    raise DownloadError(f"unexpected Content-Range {resp.headers.get('Content-Range')!r}")
                log.info("model download: resuming %s at %s / %s GB", info.name, gb(have), gb(info.size))
            elif status == 200:
                if have > 0:
                    log.info("model download: server ignored the range request - restarting %s", info.name)
                    h = hashlib.sha256()
                have, mode = 0, "wb"
            else:
                raise DownloadError(f"unexpected HTTP status {status} for {info.url}")
            done = have
            last = 0.0
            try:
                with open(part, mode) as f:
                    while True:
                        if cancel is not None and cancel.is_set():
                            raise DownloadCancelled("cancelled")
                        b = resp.read(CHUNK)
                        if not b:
                            break
                        if done + len(b) > info.size:
                            f.close()
                            part.unlink()
                            raise DownloadError(f"server sent more than {info.size} bytes - download discarded")
                        f.write(b)
                        h.update(b)
                        done += len(b)
                        now = time.monotonic()
                        if progress is not None and now - last >= PROGRESS_EVERY_S:
                            last = now
                            progress(done, info.size)
            except (OSError, urllib.error.URLError, http.client.HTTPException) as e:   # timeout / reset: keep .part
                raise DownloadError(f"download interrupted at {gb(done)} / {gb(info.size)} GB: {e}") from e
        if done != info.size:
            raise DownloadError(f"download ended early at {gb(done)} / {gb(info.size)} GB")
        break
    got = h.hexdigest()
    if got != want:
        try:
            part.unlink()
        except OSError:
            pass
        raise DownloadError(f"checksum mismatch for {info.name} (expected {want}, got {got}) - download discarded")
    os.replace(part, final)
    if progress is not None:
        progress(info.size, info.size)
    log.info("model download: %s complete and verified", info.name)
    return final


class ModelDownloader:
    """Background downloads, at most one per file. Callbacks run on the download thread."""

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
        self._lock = threading.Lock()
        self._active: dict[str, threading.Thread] = {}
        self._progress: dict[str, tuple[int, int]] = {}
        self._cancel = threading.Event()
        self.last_error: dict[str, str] = {}

    def needs_download(self, filename: str) -> Optional[ModelInfo]:
        """Registry entry if ``filename`` is a known model that is not in models/ yet."""
        info = lookup(filename)
        if info is None or (self.models_dir / info.name).is_file():
            return None
        return info

    def active(self, filename: str) -> bool:
        info = lookup(filename)
        with self._lock:
            return info is not None and info.name in self._active

    def progress(self, filename: str) -> Optional[tuple[int, int]]:
        """(bytes done, total) of a running download, else None."""
        info = lookup(filename)
        with self._lock:
            if info is None or info.name not in self._active:
                return None
            return self._progress.get(info.name, (0, info.size))

    def start(self, filename: str, on_progress: Optional[Callable[[str, int, int], None]] = None,
              on_done: Optional[Callable[[str, bool, str], None]] = None) -> bool:
        """Start downloading ``filename`` (a REGISTRY name) unless it is present, already downloading or the
        downloader was cancelled. True if a new download started. ``on_done(name, ok, error)``; a cancelled
        download (app quit) calls nothing."""
        info = self.needs_download(filename)
        if info is None:
            return False
        with self._lock:
            if info.name in self._active or self._cancel.is_set():
                return False
            self._progress[info.name] = (0, info.size)
            t = threading.Thread(target=self._run, args=(info, on_progress, on_done),
                                 name=f"model-dl-{info.name}", daemon=True)
            self._active[info.name] = t
        t.start()
        return True

    def _run(self, info: ModelInfo, on_progress, on_done) -> None:
        ok, err = False, ""

        def prog(done: int, total: int) -> None:
            with self._lock:
                self._progress[info.name] = (done, total)
            if on_progress is not None:
                try:
                    on_progress(info.name, done, total)
                except Exception:
                    log.exception("model download progress callback failed")

        log.info("model download: %s from %s", info.name, info.url)
        try:
            download(info, self.models_dir, progress=prog, cancel=self._cancel)
            ok = True
        except DownloadCancelled:
            log.info("model download: %s cancelled (kept the partial file for next time)", info.name)
            with self._lock:
                self._active.pop(info.name, None)
            return
        except Exception as e:
            err = str(e) if isinstance(e, DownloadError) else f"{type(e).__name__}: {e}"
            log.error("model download: %s failed: %s", info.name, err)
        with self._lock:
            self._active.pop(info.name, None)
            if ok:
                self.last_error.pop(info.name, None)
            else:
                self.last_error[info.name] = err
        if self._cancel.is_set():       # app is quitting: the file is kept, nothing may load it now
            return
        if on_done is not None:
            try:
                on_done(info.name, ok, err)
            except Exception:
                log.exception("model download done callback failed")

    def cancel_all(self) -> None:
        """App quit: stop every download (each keeps its .part) and refuse new ones."""
        self._cancel.set()
