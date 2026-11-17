"""Shared UI Automation plumbing: comtypes import on an MTA thread, small Win32 focus helpers, and ``UiaThread`` -
one lazily started COM (MTA) thread that owns its own CUIAutomation object and runs posted jobs with a deadline.

Used by autocomplete.py (helpers; its worker thread keeps its own UiaReader because that loop also blocks on HTTP)
and fieldprobe.py (the "is a text box focused?" check right before a dictation is inserted).

Rules for ``UiaThread`` jobs: a job is ``fn(ctx)`` and runs on the UIA thread; COM pointers never leave it (return
plain Python values only). A caller that stops waiting abandons the job: its late result is dropped. While a job is
still queued or running, new calls fail fast instead of piling up behind a hung app.
"""
from __future__ import annotations

import ctypes
import logging
import os
import queue
import sys
import threading
from ctypes import wintypes
from typing import Any, Callable, Optional

from . import winutil as w

log = logging.getLogger(__name__)

u32 = w.user32


# ----------------------------------------------------------------------------- Win32 helpers
class _GTI(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("flags", wintypes.DWORD), ("hwndActive", wintypes.HWND),
                ("hwndFocus", wintypes.HWND), ("hwndCapture", wintypes.HWND), ("hwndMenuOwner", wintypes.HWND),
                ("hwndMoveSize", wintypes.HWND), ("hwndCaret", wintypes.HWND), ("rcCaret", wintypes.RECT)]


def _fg_hwnd() -> int:
    try:
        return int(u32.GetForegroundWindow() or 0)
    except Exception:
        return 0


def _window_pid(hwnd: int) -> int:
    pid = wintypes.DWORD(0)
    try:
        u32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    except Exception:
        return 0
    return int(pid.value)


def gui_thread_info(hwnd: int) -> Optional[_GTI]:
    """GetGUIThreadInfo of the thread that owns ``hwnd`` (focus / caret windows of that thread), or None."""
    try:
        g = _GTI()
        g.cbSize = ctypes.sizeof(_GTI)
        tid = u32.GetWindowThreadProcessId(hwnd, None)
        if not tid or not u32.GetGUIThreadInfo(tid, ctypes.byref(g)):
            return None
        return g
    except Exception:
        return None


def gui_thread_caret(hwnd: int) -> Optional[tuple[int, int, int]]:
    """System caret of the window's GUI thread (classic Win32 edits), screen coordinates."""
    try:
        g = gui_thread_info(hwnd)
        if g is None or not g.hwndCaret:
            return None
        h = g.rcCaret.bottom - g.rcCaret.top
        if h <= 0:
            return None
        pt = wintypes.POINT(g.rcCaret.left, g.rcCaret.top)
        u32.ClientToScreen(g.hwndCaret, ctypes.byref(pt))
        return int(pt.x), int(pt.y), int(h)
    except Exception:
        return None


def class_name(hwnd: int) -> str:
    if not hwnd:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(256)
        u32.GetClassNameW(hwnd, buf, 256)
        return buf.value
    except Exception:
        return ""


# ----------------------------------------------------------------------------- comtypes (worker threads only)
_com_lock = threading.Lock()


def _import_comtypes_mta():
    """Import comtypes on the calling (worker) thread so its import-time CoInitializeEx is MTA, and generate the
    UIA / MSAA wrappers. Returns (comtypes, comtypes.client, UIAutomationClient module)."""
    with _com_lock:
        had = hasattr(sys, "coinit_flags")
        old = getattr(sys, "coinit_flags", None)
        sys.coinit_flags = 0        # COINIT_MULTITHREADED
        try:
            import comtypes
            import comtypes.client
        finally:
            if had:
                sys.coinit_flags = old
            else:
                try:
                    del sys.coinit_flags
                except AttributeError:
                    pass
        try:
            comtypes.CoInitializeEx(0)      # no-op if the import above already did it on this thread
        except OSError:
            pass
        comtypes.client.GetModule("UIAutomationCore.dll")
        from comtypes.gen import UIAutomationClient as U
        return comtypes, comtypes.client, U


class UiaJobError(Exception):
    """A job raised on the UIA thread (message: original type name + text)."""


class UiaContext:
    """What a job gets: the UIAutomationClient module ``U``, the ``uia`` object, a control-view ``walker`` and our
    own pid. Only valid on the UIA thread."""

    def __init__(self, U, uia, walker, own_pid: int):
        self.U, self.uia, self.walker, self.own_pid = U, uia, walker, own_pid


class UiaThread:
    """One lazily started MTA thread with its own CUIAutomation. ``call(fn, timeout)`` runs ``fn(ctx)`` there."""

    def __init__(self, name: str = "uia", uia_timeout_ms: int = 250, context_factory: Optional[Callable] = None):
        self.name = name
        self.uia_timeout_ms = uia_timeout_ms
        self._factory = context_factory or self._make_context
        self._lock = threading.Lock()
        self._q: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._pending = 0                  # jobs queued or running (guarded by _lock)
        self.init_error: Optional[str] = None

    # ---- caller side
    def _submit(self, fn: Callable[[UiaContext], Any], box: dict) -> bool:
        with self._lock:
            if self.init_error is not None or self._pending:
                return False
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name=self.name, daemon=True)
                self._thread.start()
            self._pending += 1
        self._q.put((fn, box))
        return True

    def call(self, fn: Callable[[UiaContext], Any], timeout: float) -> Any:
        """Result of ``fn(ctx)``. TimeoutError when it misses ``timeout`` or another job is still running;
        RuntimeError when UIA could not be set up; any exception ``fn`` raised."""
        box: dict = {"done": threading.Event()}
        if not self._submit(fn, box):
            if self.init_error is not None:
                raise RuntimeError(f"UIA unavailable: {self.init_error}")
            raise TimeoutError("UIA thread busy")
        if not box["done"].wait(timeout):
            box["abandoned"] = True
            raise TimeoutError("UIA call timed out")
        if "error" in box:
            raise box["error"]
        return box.get("value")

    def post(self, fn: Callable[[UiaContext], Any]) -> bool:
        """Fire and forget (warm-up). False when it was dropped (busy / unavailable)."""
        return self._submit(fn, {"done": threading.Event(), "abandoned": True})

    # ---- UIA thread
    def _make_context(self) -> UiaContext:
        _comtypes, client, U = _import_comtypes_mta()
        try:
            uia = client.CreateObject(U.CUIAutomation8, interface=U.IUIAutomation2)
            try:
                uia.ConnectionTimeout = self.uia_timeout_ms
                uia.TransactionTimeout = self.uia_timeout_ms
            except Exception:
                pass
        except Exception:
            uia = client.CreateObject(U.CUIAutomation, interface=U.IUIAutomation)
        return UiaContext(U, uia, uia.ControlViewWalker, os.getpid())

    def _run(self) -> None:
        try:
            ctx = self._factory()
        except Exception as e:
            log.warning("UIA thread %s: setup failed (%s)", self.name, type(e).__name__)
            with self._lock:
                self.init_error = type(e).__name__
            ctx = None
        while True:
            fn, box = self._q.get()
            try:
                if ctx is None:
                    box["error"] = RuntimeError(f"UIA unavailable: {self.init_error}")
                else:
                    try:
                        box["value"] = fn(ctx)
                    except Exception as e:     # a fresh exception: no traceback frames holding COM pointers
                        box["error"] = UiaJobError(f"{type(e).__name__}: {str(e)[:200]}")
            finally:
                with self._lock:
                    self._pending -= 1
                box["done"].set()
                box = fn = None             # drop references (late results of abandoned calls die here)


_shared: Optional[UiaThread] = None
_shared_lock = threading.Lock()


def shared() -> UiaThread:
    """The process-wide UIA thread (started on first use, whatever the autocomplete setting is)."""
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = UiaThread("uia-probe")
        return _shared
