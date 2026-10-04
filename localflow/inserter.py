"""Insert text into the focused app: clipboard + Ctrl+V (with full clipboard save/restore) or SendInput typing."""
from __future__ import annotations

import ctypes
import logging
import time
from ctypes import wintypes

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
    saved: list[tuple[int, bytes]] = []
    if not _open_clipboard():
        log.warning("clipboard busy; not saving")
        return saved
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


def restore_clipboard(saved: list[tuple[int, bytes]]) -> None:
    if not _open_clipboard():
        log.warning("clipboard busy; could not restore")
        return
    try:
        u32.EmptyClipboard()
        for fmt, data in saved:
            h = _hglobal_from_bytes(data)
            if h and not u32.SetClipboardData(fmt, h):
                k32.GlobalFree(h)
    finally:
        u32.CloseClipboard()


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


def insert_text(text: str, mode: str = "paste", restore: bool = True, restore_delay_ms: int = 300) -> dict:
    """Insert text into the focused window. Returns timing info."""
    t0 = time.perf_counter()
    info = {"mode": mode, "wait_ms": 0, "ok": True}
    released = wait_modifiers_released(2.0)
    info["wait_ms"] = int((time.perf_counter() - t0) * 1000)
    if not released:
        log.warning("modifier keys still held after 2 s; inserting anyway")
    if mode == "type":
        type_text(text)
        return info
    saved = save_clipboard() if restore else []
    if not set_clipboard_text(text):
        log.error("could not set clipboard; falling back to typing")
        info["mode"] = "type-fallback"
        type_text(text)
        return info
    seq_after_set = u32.GetClipboardSequenceNumber()
    _paste_keystroke()
    if restore:
        time.sleep(restore_delay_ms / 1000.0)
        # only restore if nobody else (the user) changed the clipboard in the meantime
        if u32.GetClipboardSequenceNumber() == seq_after_set:
            restore_clipboard(saved)
        else:
            log.info("clipboard changed during paste; not restoring")
    return info
