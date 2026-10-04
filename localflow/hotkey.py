"""Global push-to-talk hotkey using a WH_KEYBOARD_LL hook (ctypes).

Behaviour
- Hold the configured modifier combo (default Ctrl+Alt, either side) -> on_start().
- Release it -> on_stop().
- Any non-modifier key while held (a real shortcut like Ctrl+Alt+T) -> on_cancel("shortcut"),
  the key is passed through untouched.
- Tap Space while held -> hands-free lock (on_handsfree); keys can be released. Press the combo
  again to stop, Esc to cancel.  Esc while holding also cancels.
- When Alt/Win is part of the combo, a dummy unassigned keystroke (VK 0xE8) is injected just before
  the Alt/Win key-up so Windows does not open the menu bar / Start menu (AutoHotkey's "menu mask").

The hook callback only flips a tiny state machine and calls the registered callbacks, which must
return immediately (the app just enqueues work). A watchdog resyncs stuck key state and re-installs
the hook if Windows silently dropped it.
"""
from __future__ import annotations

import ctypes
import logging
import os
import threading
import time
from ctypes import wintypes
from typing import Callable, Optional

from . import winutil as w

log = logging.getLogger(__name__)

WH_KEYBOARD_LL = 13
WM_KEYDOWN, WM_KEYUP, WM_SYSKEYDOWN, WM_SYSKEYUP = 0x100, 0x101, 0x104, 0x105
WM_TIMER, WM_QUIT = 0x113, 0x12
LLKHF_INJECTED = 0x10

LRESULT = ctypes.c_ssize_t
WPARAM = ctypes.c_size_t
LPARAM = ctypes.c_ssize_t
HOOKPROC = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int, WPARAM, LPARAM)
TIMERPROC = ctypes.WINFUNCTYPE(None, wintypes.HWND, wintypes.UINT, ctypes.c_size_t, wintypes.DWORD)


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [("vkCode", wintypes.DWORD), ("scanCode", wintypes.DWORD), ("flags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", w.ULONG_PTR)]


u32 = w.user32
u32.SetWindowsHookExW.argtypes = [ctypes.c_int, HOOKPROC, wintypes.HINSTANCE, wintypes.DWORD]
u32.SetWindowsHookExW.restype = ctypes.c_void_p
u32.UnhookWindowsHookEx.argtypes = [ctypes.c_void_p]
u32.UnhookWindowsHookEx.restype = wintypes.BOOL
u32.CallNextHookEx.argtypes = [ctypes.c_void_p, ctypes.c_int, WPARAM, LPARAM]
u32.CallNextHookEx.restype = LRESULT
u32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
u32.SetTimer.argtypes = [wintypes.HWND, ctypes.c_size_t, wintypes.UINT, TIMERPROC]
u32.SetTimer.restype = ctypes.c_size_t
u32.KillTimer.argtypes = [wintypes.HWND, ctypes.c_size_t]
u32.PostThreadMessageW.argtypes = [wintypes.DWORD, wintypes.UINT, WPARAM, LPARAM]
w.kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
w.kernel32.GetModuleHandleW.restype = wintypes.HMODULE
w.kernel32.GetCurrentThreadId.restype = wintypes.DWORD

_VK_TO_GROUP: dict[int, str] = {}
for _g, _vks in w.MODIFIER_VKS.items():
    for _v in _vks:
        _VK_TO_GROUP[_v] = _g
# generic (non-sided) VKs may show up from some injectors
_VK_TO_GROUP.update({w.VK_CONTROL: "ctrl", w.VK_MENU: "alt", w.VK_SHIFT: "shift"})

_KEYNAME_TO_VK = {"space": w.VK_SPACE, "tab": w.VK_TAB, "enter": w.VK_RETURN, "backspace": w.VK_BACK}


def parse_hotkey(spec: str) -> frozenset[str]:
    mods = {p.strip().lower() for p in spec.replace(" ", "").split("+") if p.strip()}
    aliases = {"control": "ctrl", "ctl": "ctrl", "menu": "alt", "windows": "win", "super": "win", "cmd": "win"}
    mods = {aliases.get(m, m) for m in mods}
    mods &= set(w.MODIFIER_VKS)
    return frozenset(mods) if mods else frozenset({"ctrl", "alt"})


class HotkeyManager:
    def __init__(self, hotkey: str = "ctrl+alt", handsfree_key: str = "space", *,
                 on_start: Callable[[], None], on_stop: Callable[[], None],
                 on_cancel: Callable[[str], None], on_handsfree: Callable[[], None]):
        self.required = parse_hotkey(hotkey)
        self.handsfree_vk = _KEYNAME_TO_VK.get(handsfree_key.lower(), w.VK_SPACE)
        self._cb_start, self._cb_stop = on_start, on_stop
        self._cb_cancel, self._cb_handsfree = on_cancel, on_handsfree
        self.enabled = True
        self.state = "idle"          # idle | hold | locked | suppressed
        self._armed_stop = False
        self._needs_mask = False
        self._down: dict[int, float] = {}   # vk -> time pressed (physical keys we know are down)
        self._swallow: set[int] = set()     # vks whose down/up/repeat we swallow
        self._lock = threading.RLock()
        self._hook = None
        self._proc = HOOKPROC(self._hook_proc)       # keep a reference alive
        self._timer_proc = TIMERPROC(self._on_timer)
        self._thread: Optional[threading.Thread] = None
        self._thread_id = 0
        self._ready = threading.Event()
        self._last_event = time.monotonic()
        self._desync_since: Optional[float] = None
        self.installed = False
        self.events_seen = 0

    # ------------------------------------------------------------------ public
    def set_hotkey(self, spec: str) -> None:
        with self._lock:
            self.required = parse_hotkey(spec)
            if self.state in ("hold", "locked"):
                self._fire(self._cb_cancel, "hotkey-changed")
            self.state = "suppressed" if self._held_mods() else "idle"
        log.info("hotkey set to %s", "+".join(sorted(self.required)))

    def set_handsfree_key(self, name: str) -> None:
        self.handsfree_vk = _KEYNAME_TO_VK.get(name.lower(), w.VK_SPACE)

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self.enabled = enabled
            if not enabled and self.state in ("hold", "locked"):
                self._fire(self._cb_cancel, "paused")
                self.state = "suppressed"

    def force_idle(self) -> None:
        """Called by the app when it ends a recording itself (e.g. max duration)."""
        with self._lock:
            if self.state in ("hold", "locked"):
                self.state = "suppressed" if self._held_mods() else "idle"

    def start(self) -> bool:
        self._thread = threading.Thread(target=self._run, name="hotkey-hook", daemon=True)
        self._thread.start()
        ok = self._ready.wait(5)
        return ok and self.installed

    def stop(self) -> None:
        if self._thread_id:
            u32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)

    # ------------------------------------------------------------------ hook thread
    def _install(self) -> bool:
        if self._hook:
            u32.UnhookWindowsHookEx(self._hook)
            self._hook = None
        hmod = w.kernel32.GetModuleHandleW(None)
        self._hook = u32.SetWindowsHookExW(WH_KEYBOARD_LL, self._proc, hmod, 0)
        self.installed = bool(self._hook)
        if not self.installed:
            log.error("SetWindowsHookExW failed: %s", ctypes.get_last_error())
        return self.installed

    def _run(self) -> None:
        self._thread_id = w.kernel32.GetCurrentThreadId()
        self._install()
        log.info("keyboard hook installed=%s", self.installed)
        self._ready.set()
        timer = u32.SetTimer(None, 0, 400, self._timer_proc)
        msg = wintypes.MSG()
        while u32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            u32.TranslateMessage(ctypes.byref(msg))
            u32.DispatchMessageW(ctypes.byref(msg))
        u32.KillTimer(None, timer)
        if self._hook:
            u32.UnhookWindowsHookEx(self._hook)
            self._hook = None
        self.installed = False

    def _hook_proc(self, n_code, w_param, l_param):
        try:
            if n_code >= 0:
                kb = KBDLLHOOKSTRUCT.from_address(l_param)
                if kb.dwExtraInfo != w.INJECT_MARK and not (kb.vkCode == w.VK_LCONTROL and kb.scanCode == 0x21D):
                    self.events_seen += 1
                    self._last_event = time.monotonic()
                    down = w_param in (WM_KEYDOWN, WM_SYSKEYDOWN)
                    if self._handle(kb.vkCode, down):
                        return 1
        except Exception:
            log.exception("hook handler error")
        return u32.CallNextHookEx(None, n_code, w_param, l_param)

    # ------------------------------------------------------------------ state machine
    def _held_mods(self) -> set[str]:
        held = set()
        for vk in self._down:
            g = _VK_TO_GROUP.get(vk)
            if g:
                held.add(g)
        return held

    def _fire(self, cb: Callable, *args) -> None:
        try:
            cb(*args)
        except Exception:
            log.exception("hotkey callback failed")

    def _inject_mask(self) -> None:
        if os.environ.get("LOCALFLOW_NO_MASK"):      # test hook: negative control for the menu-mask test
            return
        w.send_inputs([w._key_input(vk=w.VK_UNASSIGNED), w._key_input(vk=w.VK_UNASSIGNED, flags=w.KEYEVENTF_KEYUP)])

    def _handle(self, vk: int, down: bool) -> bool:
        """Returns True to swallow the event."""
        with self._lock:
            # swallowed control keys (space lock, esc cancel) incl. auto-repeat
            if vk in self._swallow:
                if not down:
                    self._swallow.discard(vk)
                    self._down.pop(vk, None)
                return True

            mod = _VK_TO_GROUP.get(vk)
            if down:
                self._down[vk] = time.monotonic()
            else:
                self._down.pop(vk, None)
            if not self.enabled:
                return False
            held = self._held_mods()
            req = self.required
            st = self.state

            if mod:
                if down:
                    if st == "idle" and held == req:
                        self.state = "hold"
                        self._armed_stop = False
                        self._needs_mask = bool(req & {"alt", "win"})
                        self._fire(self._cb_start)
                    elif st == "locked" and self._armed_stop and req <= held:
                        self.state = "suppressed"
                        self._fire(self._cb_stop)
                    return False
                # ---- modifier released
                if st == "hold" and not req <= held:
                    self.state = "suppressed" if held & req else "idle"
                    self._fire(self._cb_stop)
                elif st == "locked" and not req <= held:
                    self._armed_stop = True
                elif st == "suppressed" and not held & req:
                    self.state = "idle"
                if self.state == "idle" and held & req == set():
                    pass
                if mod in ("alt", "win") and self._needs_mask:
                    other_down = any(v != vk and _VK_TO_GROUP.get(v) == mod for v in self._down)
                    if not other_down:
                        self._inject_mask()
                        self._needs_mask = False
                return False

            # ---- non-modifier key
            if not down:
                return False
            if st == "hold":
                if vk == self.handsfree_vk:
                    self.state = "locked"
                    self._armed_stop = False
                    self._swallow.add(vk)
                    self._fire(self._cb_handsfree)
                    return True
                if vk == w.VK_ESCAPE:
                    self._swallow.add(vk)
                    self.state = "suppressed"
                    self._fire(self._cb_cancel, "esc")
                    return True
                # a genuine shortcut (Ctrl+Alt+T, ...): cancel silently, pass the key through
                self.state = "suppressed"
                self._needs_mask = False
                self._fire(self._cb_cancel, "shortcut")
                return False
            if st == "locked" and vk == w.VK_ESCAPE:
                self._swallow.add(vk)
                self.state = "suppressed" if held & req else "idle"
                self._fire(self._cb_cancel, "esc")
                return True
            return False

    # ------------------------------------------------------------------ watchdog (hook thread timer)
    def _on_timer(self, hwnd, msg, timer_id, tick) -> None:
        try:
            now = time.monotonic()
            with self._lock:
                # 1) drop keys we think are down but the OS says are up (missed key-up)
                for vk in list(self._down):
                    if now - self._down[vk] > 0.4 and not w.is_vk_down(vk):
                        log.warning("resync: missed key-up for vk 0x%x", vk)
                        self._handle(vk, False)
                # 2) hotkey physically down but hook has seen nothing -> hook was probably dropped
                phys = w.physical_modifiers_down()
                if self.enabled and self.state == "idle" and phys == set(self.required) and \
                        not self.required <= self._held_mods():
                    if self._desync_since is None:
                        self._desync_since = now
                    elif now - self._desync_since > 1.0:
                        log.warning("hotkey held but hook silent - reinstalling hook")
                        self._install()
                        self._desync_since = None
                else:
                    self._desync_since = None
        except Exception:
            log.exception("watchdog error")
