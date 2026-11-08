"""Insert text into the focused app: clipboard + Ctrl+V (with full clipboard save/restore) or SendInput typing."""
from __future__ import annotations

import ctypes
import logging
import os
import time
from ctypes import wintypes
from typing import Optional

from . import winutil as w

log = logging.getLogger(__name__)

u32, k32 = w.user32, w.kernel32

CF_TEXT, CF_BITMAP, CF_METAFILEPICT, CF_UNICODETEXT, CF_ENHMETAFILE, CF_PALETTE = 1, 2, 3, 13, 14, 9
CF_DSPBITMAPDELAY = 0x0082
_GDI_FORMATS = {CF_BITMAP, CF_METAFILEPICT, CF_ENHMETAFILE, CF_PALETTE}
GMEM_MOVEABLE = 0x0002

u32.OpenClipboard.argtypes = [wintypes.HWND]
u32.OpenClipboard.restype = wintypes.BOOL
u32.CloseClipboard.restype = wintypes.BOOL
u32.EmptyClipboard.restype = wintypes.BOOL
u32.EnumClipboardFormats.argtypes = [wintypes.UINT]
u32.EnumClipboardFormats.restype = wintypes.UINT
u32.GetClipboardData.argtypes = [wintypes.UINT]
u32.GetClipboardData.restype = wintypes.HANDLE
u32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
u32.SetClipboardData.restype = wintypes.HANDLE
u32.RegisterClipboardFormatW.argtypes = [wintypes.LPCWSTR]
u32.RegisterClipboardFormatW.restype = wintypes.UINT
u32.GetClipboardSequenceNumber.restype = wintypes.DWORD
k32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
k32.GlobalAlloc.restype = wintypes.HANDLE
k32.GlobalLock.argtypes = [wintypes.HANDLE]
k32.GlobalLock.restype = ctypes.c_void_p
k32.GlobalUnlock.argtypes = [wintypes.HANDLE]
k32.GlobalSize.argtypes = [wintypes.HANDLE]
k32.GlobalSize.restype = ctypes.c_size_t
k32.GlobalFree.argtypes = [wintypes.HANDLE]


def _open_clipboard(retries: int = 20, delay: float = 0.025) -> bool:
    for _ in range(retries):
        if u32.OpenClipboard(None):
            return True
        time.sleep(delay)
    return False


def _hglobal_from_bytes(data: bytes):
    h = k32.GlobalAlloc(GMEM_MOVEABLE, len(data) or 1)
    if not h:
        return None
    p = k32.GlobalLock(h)
    if not p:
        k32.GlobalFree(h)
        return None
    ctypes.memmove(p, data, len(data))
    k32.GlobalUnlock(h)
    return h


def save_clipboard() -> list[tuple[int, bytes]]:
    """Snapshot every HGLOBAL-backed clipboard format (text, HTML, RTF, DIB, file lists, ...)."""
    return save_clipboard_checked() or []


def save_clipboard_checked() -> Optional[list[tuple[int, bytes]]]:
    """Like save_clipboard, but None when the clipboard could not be opened (vs [] for an empty clipboard)."""
    saved: list[tuple[int, bytes]] = []
    if not _open_clipboard():
        log.warning("clipboard busy; not saving")
        return None
    try:
        fmt = 0
        while True:
            fmt = u32.EnumClipboardFormats(fmt)
            if not fmt:
                break
            if fmt in _GDI_FORMATS or fmt == CF_DSPBITMAPDELAY or 0x0200 <= fmt <= 0x02FF or 0x0300 <= fmt <= 0x03FF:
                continue   # GDI handles / private ranges: skip (DIB variants are saved instead)
            try:
                h = u32.GetClipboardData(fmt)
                if not h:
                    continue
                size = k32.GlobalSize(h)
                if not size or size > 64 * 1024 * 1024:
                    continue
                p = k32.GlobalLock(h)
                if not p:
                    continue
                try:
                    saved.append((fmt, ctypes.string_at(p, size)))
                finally:
                    k32.GlobalUnlock(h)
            except Exception:
                continue
    finally:
        u32.CloseClipboard()
    return saved


def restore_clipboard(saved: list[tuple[int, bytes]]) -> bool:
    """Put a save_clipboard() snapshot back. False if the clipboard could not be opened."""
    if not _open_clipboard():
        log.warning("clipboard busy; could not restore")
        return False
    ok = True
    try:
        u32.EmptyClipboard()
        for fmt, data in saved:
            h = _hglobal_from_bytes(data)
            if not h:
                ok = False
            elif not u32.SetClipboardData(fmt, h):
                k32.GlobalFree(h)
                ok = False
    finally:
        u32.CloseClipboard()
    if not ok:
        log.warning("clipboard restore incomplete")
    return ok


def restore_clipboard_retry(saved: list[tuple[int, bytes]], attempts: int = 4) -> bool:
    """restore_clipboard, retried for ~2 s while another app holds the clipboard open."""
    for _ in range(attempts):
        if restore_clipboard(saved):
            return True
    log.error("could not restore the clipboard after %d attempts", attempts)
    return False


def set_clipboard_text(text: str) -> bool:
    if not _open_clipboard():
        return False
    try:
        u32.EmptyClipboard()
        h = _hglobal_from_bytes((text + "\0").encode("utf-16-le"))
        if not h or not u32.SetClipboardData(CF_UNICODETEXT, h):
            return False
        # keep dictated text out of Win+V clipboard history / cloud clipboard
        for name in ("ExcludeClipboardContentFromMonitorProcessing",):
            f = u32.RegisterClipboardFormatW(name)
            hh = _hglobal_from_bytes(b"\x01\x00\x00\x00")
            if f and hh and not u32.SetClipboardData(f, hh):
                k32.GlobalFree(hh)
        f = u32.RegisterClipboardFormatW("CanIncludeInClipboardHistory")
        hh = _hglobal_from_bytes(b"\x00\x00\x00\x00")
        if f and hh and not u32.SetClipboardData(f, hh):
            k32.GlobalFree(hh)
        f = u32.RegisterClipboardFormatW("CanUploadToCloudClipboard")
        hh = _hglobal_from_bytes(b"\x00\x00\x00\x00")
        if f and hh and not u32.SetClipboardData(f, hh):
            k32.GlobalFree(hh)
        return True
    finally:
        u32.CloseClipboard()


def wait_modifiers_released(timeout: float = 2.0) -> bool:
    """Block until no Ctrl/Alt/Shift/Win is physically held (a held Alt would turn Ctrl+V into Ctrl+Alt+V)."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if not w.physical_modifiers_down():
            return True
        time.sleep(0.01)
    return False


def _paste_keystroke() -> None:
    I = w._key_input
    w.send_inputs([
        I(vk=w.VK_CONTROL), I(vk=0x56), I(vk=0x56, flags=w.KEYEVENTF_KEYUP), I(vk=w.VK_CONTROL, flags=w.KEYEVENTF_KEYUP),
    ])


def type_text(text: str) -> None:
    """Type via SendInput Unicode events (newline -> Shift+Enter so chat apps do not send)."""
    I = w._key_input
    batch: list = []
    for ch in text.replace("\r\n", "\n"):
        if ch == "\n":
            batch += [I(vk=w.VK_SHIFT), I(vk=w.VK_RETURN), I(vk=w.VK_RETURN, flags=w.KEYEVENTF_KEYUP),
                      I(vk=w.VK_SHIFT, flags=w.KEYEVENTF_KEYUP)]
            continue
        if ch == "\t":
            batch += [I(vk=w.VK_TAB), I(vk=w.VK_TAB, flags=w.KEYEVENTF_KEYUP)]
            continue
        data = ch.encode("utf-16-le")
        for i in range(0, len(data), 2):
            code = int.from_bytes(data[i:i + 2], "little")
            batch += [I(scan=code, flags=w.KEYEVENTF_UNICODE),
                      I(scan=code, flags=w.KEYEVENTF_UNICODE | w.KEYEVENTF_KEYUP)]
        if len(batch) >= 200:
            w.send_inputs(batch)
            batch = []
    w.send_inputs(batch)


def insert_text(text: str, mode: str = "paste", restore: bool = True, restore_delay_ms: int = 300,
                require_release: bool = False, release_timeout: float = 2.0, target_hwnd: int = 0) -> dict:
    """Insert text into the focused window. Returns timing info.

    Strict mode (command mode / paste-last; dictation keeps the old lenient behaviour):
    ``require_release``: if modifier keys are still held after ``release_timeout``, or the clipboard can't be
    snapshotted for restoring, insert NOTHING and return ok=False with info["reason"].
    ``target_hwnd``: re-checked right before the keystroke goes out; if another window has focus by then, insert
    nothing (reason "window changed")."""
    t0 = time.perf_counter()
    info = {"mode": mode, "wait_ms": 0, "ok": True, "reason": ""}
    released = wait_modifiers_released(release_timeout)
    info["wait_ms"] = int((time.perf_counter() - t0) * 1000)
    if not released and require_release:
        log.warning("modifier keys still held after %.1f s; not inserting", release_timeout)
        info.update(ok=False, reason="modifiers held")
        return info
    if not released:
        log.warning("modifier keys still held after 2 s; inserting anyway")

    def wrong_window() -> bool:
        if target_hwnd and foreground_hwnd() != target_hwnd:
            log.warning("foreground window changed before insertion; not inserting")
            info.update(ok=False, reason="window changed")
            return True
        return False

    if mode == "type":
        if not wrong_window():
            type_text(text)
        return info
    saved = save_clipboard_checked() if restore else []
    if saved is None:
        if require_release:
            info.update(ok=False, reason="clipboard busy")
            return info
        restore = False         # can't snapshot: leave our text on the clipboard rather than emptying it
    if wrong_window():
        return info
    if not set_clipboard_text(text):
        log.error("could not set clipboard; falling back to typing")
        info["mode"] = "type-fallback"
        if not wrong_window():
            type_text(text)
        return info
    seq_after_set = u32.GetClipboardSequenceNumber()
    if wrong_window():
        if restore:
            restore_clipboard_retry(saved)
        return info
    _paste_keystroke()
    if restore:
        time.sleep(restore_delay_ms / 1000.0)
        # only restore if nobody else (the user) changed the clipboard in the meantime; retry while it is busy
        for _ in range(4):
            if u32.GetClipboardSequenceNumber() != seq_after_set:
                log.info("clipboard changed during paste; not restoring")
                break
            if restore_clipboard(saved):
                break
            seq_after_set = u32.GetClipboardSequenceNumber()     # a partial restore changed it (our own write)
        else:
            log.error("could not restore the clipboard after paste")
    return info


# ----------------------------------------------------------------------------- selection capture (command mode)
VK_C = 0x43


def read_clipboard_text() -> Optional[str]:
    """CF_UNICODETEXT of the clipboard, or None (no text / clipboard busy)."""
    if not _open_clipboard(retries=6, delay=0.01):
        return None
    try:
        h = u32.GetClipboardData(CF_UNICODETEXT)
        if not h:
            return None
        p = k32.GlobalLock(h)
        if not p:
            return None
        try:
            return ctypes.wstring_at(p)
        finally:
            k32.GlobalUnlock(h)
    finally:
        u32.CloseClipboard()


def _copy_keystroke() -> None:
    I = w._key_input
    w.send_inputs([I(vk=w.VK_CONTROL), I(vk=VK_C), I(vk=VK_C, flags=w.KEYEVENTF_KEYUP),
                   I(vk=w.VK_CONTROL, flags=w.KEYEVENTF_KEYUP)])


def capture_selection(timeout_ms: int = 600, target_hwnd: int = 0) -> tuple[str, str]:
    """Copy the focused app's selection with Ctrl+C without losing the user's clipboard.

    Returns (status, text): "text" (selection copied), "none" (clipboard never changed: nothing selected),
    "uncertain" (the clipboard changed but held no text in time: caller must abort), "busy" (clipboard could not
    be opened, or the original could not be put back: caller must abort), "window" (``target_hwnd`` lost focus
    before the Ctrl+C went out). The original clipboard is restored right after reading, on every path that
    touched it. The caller must make sure no modifier key is held (Ctrl+C would become Ctrl+Alt+C ...)."""
    saved = save_clipboard_checked()
    if saved is None:
        return "busy", ""
    if target_hwnd and foreground_hwnd() != target_hwnd:
        return "window", ""
    if not _open_clipboard():
        return "busy", ""
    try:
        u32.EmptyClipboard()        # mark: a stale copy can't be mistaken for the selection
    finally:
        u32.CloseClipboard()
    seq0 = u32.GetClipboardSequenceNumber()
    changed, text, wrong = False, None, False
    restored = False
    try:
        if target_hwnd and foreground_hwnd() != target_hwnd:    # focus moved while we waited for the clipboard
            wrong = True
        else:
            _copy_keystroke()
            end = time.monotonic() + timeout_ms / 1000.0
            while time.monotonic() < end:
                time.sleep(0.015)
                if u32.GetClipboardSequenceNumber() != seq0:
                    changed = True
                    t = read_clipboard_text()
                    if t:
                        text = t
                        break
    finally:
        restored = restore_clipboard_retry(saved)
    if not restored:
        return "busy", ""
    if wrong:
        return "window", ""
    if text:
        return "text", text
    return ("uncertain" if changed else "none"), ""


# ----------------------------------------------------------------------------- foreground window helpers
u32.IsWindow.argtypes = [wintypes.HWND]
u32.IsWindow.restype = wintypes.BOOL
u32.SetForegroundWindow.argtypes = [wintypes.HWND]
u32.SetForegroundWindow.restype = wintypes.BOOL

# windows that are never a paste target (taskbar, tray overflow, ...): the tray menu briefly focuses them
_SHELL_CLASSES = {"shell_traywnd", "shell_secondarytraywnd", "notifyiconoverflowwindow",
                  "toplevelwindowforoverflowxamlisland", "progman", "workerw", "xaml_windowedpopupclass"}


def foreground_hwnd() -> int:
    return int(u32.GetForegroundWindow() or 0)


def _class_name(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(256)
    u32.GetClassNameW(hwnd, buf, 256)
    return buf.value


def _window_pid(hwnd: int) -> int:
    pid = wintypes.DWORD(0)
    u32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return int(pid.value)


def is_external_target(hwnd: int) -> bool:
    """A window of another process that text could be pasted into (not ours, not the taskbar/desktop)."""
    try:
        return bool(hwnd) and _window_pid(hwnd) != os.getpid() and _class_name(hwnd).lower() not in _SHELL_CLASSES
    except Exception:
        return False


def focus_window(hwnd: int, timeout: float = 0.6) -> bool:
    """Bring ``hwnd`` to the foreground (works right after the user clicked our tray menu). True when it is."""
    if not hwnd or not u32.IsWindow(hwnd):
        return False
    if foreground_hwnd() == hwnd:
        return True
    u32.SetForegroundWindow(hwnd)
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if foreground_hwnd() == hwnd:
            time.sleep(0.05)        # let the window take keyboard focus before we paste
            return True
        time.sleep(0.02)
    return False
