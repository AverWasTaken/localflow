"""Global push-to-talk hotkey using a WH_KEYBOARD_LL hook (ctypes).

Behaviour
- Two modifier combos: "dictate" (default Ctrl+Alt) and optionally "command" (default Ctrl+Win). Holding exactly
  one of them (either side) -> on_start(mode). Neither may be a subset of the other (command mode is then off).
- Release it -> on_stop().
- Any non-modifier key while held (a real shortcut like Ctrl+Alt+T) -> on_cancel("shortcut"),
  the key is passed through untouched.
- Tap Space while held -> hands-free lock (on_handsfree); keys can be released. Press the combo
  again to stop, Esc to cancel.  Esc while holding also cancels. Same in both modes.
- Optional "paste last" chord (default Alt+Shift+Z): in idle with exactly its modifiers held, its key is swallowed,
  a menu mask is injected (no menu bar / keyboard-layout switch) and on_paste() fires.
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


_ALIASES = {"control": "ctrl", "ctl": "ctrl", "menu": "alt", "windows": "win", "super": "win", "cmd": "win"}


def _tokens(spec: str) -> list[str]:
    toks = (t.strip().lower() for t in str(spec or "").replace(" ", "").split("+"))
    return [_ALIASES.get(t, t) for t in toks if t]


def parse_hotkey(spec: str) -> frozenset[str]:
    mods = set(_tokens(spec)) & set(w.MODIFIER_VKS)
    return frozenset(mods) if mods else frozenset({"ctrl", "alt"})


def parse_combo(spec: str) -> Optional[frozenset[str]]:
    """Strict modifier-only combo ("ctrl+win"); None if empty or it names anything but modifiers."""
    toks = _tokens(spec)
    if not toks or any(t not in w.MODIFIER_VKS for t in toks):
        return None
    return frozenset(toks)


def _key_vk(name: str) -> Optional[int]:
    if len(name) == 1 and name.isascii() and name.isalnum():
        return ord(name.upper())
    if name.startswith("f") and name[1:].isdigit() and 1 <= int(name[1:]) <= 24:
        return 0x70 + int(name[1:]) - 1
    return _KEYNAME_TO_VK.get(name)


def parse_chord(spec: str) -> Optional[tuple[frozenset[str], int]]:
    """"alt+shift+z" -> (modifiers, vk); None unless it is 1+ modifiers plus exactly one key."""
    mods, vk = set(), None
    for t in _tokens(spec):
        if t in w.MODIFIER_VKS:
            mods.add(t)
        elif vk is None:
            vk = _key_vk(t)
            if vk is None:
                return None
        else:
            return None
    if not mods or vk is None:
        return None
    return frozenset(mods), vk


def _fmt(combo) -> str:
    return "+".join(sorted(combo))


class HotkeyManager:
    def __init__(self, hotkey: str = "ctrl+alt", handsfree_key: str = "space", *,
                 on_start: Callable[[str], None], on_stop: Callable[[], None],
                 on_cancel: Callable[[str], None], on_handsfree: Callable[[], None],
                 command_hotkey: Optional[str] = None, paste_hotkey: Optional[str] = None,
                 on_paste: Optional[Callable[[], None]] = None):
        self.required = parse_hotkey(hotkey)        # the dictation combo
        self.handsfree_vk = _KEYNAME_TO_VK.get(handsfree_key.lower(), w.VK_SPACE)
        self._cb_start, self._cb_stop = on_start, on_stop
        self._cb_cancel, self._cb_handsfree = on_cancel, on_handsfree
        self._cb_paste = on_paste
        self._command_spec, self._paste_spec = command_hotkey, paste_hotkey
        self.command: Optional[frozenset[str]] = None              # command-mode combo (None = off)
        self.paste: Optional[tuple[frozenset[str], int]] = None    # paste-last chord (None = off)
        self._validate()
        self.active_mode = "dictate"
        self._active = self.required    # combo of the current / last activation (hold, locked, suppressed)
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
    def combos(self) -> dict[str, frozenset[str]]:
        c = {"dictate": self.required}
        if self.command is not None:
            c["command"] = self.command
        return c

    def _validate(self) -> None:
        """(Re)derive the command combo + paste chord from their specs, rejecting ones that collide."""
        cmd = None
        if self._command_spec:
            cmd = parse_combo(self._command_spec)
            if cmd is None:
                log.warning("command mode hotkey %r is not a modifier combo - command mode off", self._command_spec)
            elif cmd <= self.required or self.required <= cmd:
                log.warning("command mode hotkey %s overlaps the dictation hotkey %s - command mode off",
                            _fmt(cmd), _fmt(self.required))
                cmd = None
        self.command = cmd
        chord = None
        if self._paste_spec:
            chord = parse_chord(self._paste_spec)
            if chord is None:
                log.warning("paste-last hotkey %r is not modifiers + one key - paste-last hotkey off", self._paste_spec)
            elif any(c <= chord[0] for c in self.combos().values()):
                log.warning("paste-last hotkey %r collides with a dictation/command hotkey - paste-last hotkey off",
                            self._paste_spec)
                chord = None
        self.paste = chord

    def _reset_after_change(self, why: str) -> None:
        if self.state in ("hold", "locked"):
            self._fire(self._cb_cancel, why)
        self.state = "suppressed" if self._held_mods() else "idle"

    def set_hotkey(self, spec: str) -> None:
        with self._lock:
            self.required = parse_hotkey(spec)
            self._validate()
            self._reset_after_change("hotkey-changed")
        log.info("hotkey set to %s", _fmt(self.required))

    def set_command_hotkey(self, spec: Optional[str]) -> None:
        """None / "" turns command mode off."""
        with self._lock:
            self._command_spec = spec or None
            old = self.command
            self._validate()
            if old != self.command and self.active_mode == "command":
                self._reset_after_change("hotkey-changed")
        log.info("command hotkey set to %s", _fmt(self.command) if self.command else "off")

    def set_paste_hotkey(self, spec: Optional[str]) -> None:
        """None / "" turns the paste-last hotkey off."""
        with self._lock:
            self._paste_spec = spec or None
            self._validate()
        log.info("paste-last hotkey set to %s", self._paste_spec if self.paste else "off")

    def set_handsfree_key(self, name: str) -> None:
        self.handsfree_vk = _KEYNAME_TO_VK.get(name.lower(), w.VK_SPACE)

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self.enabled = enabled
            if not enabled and self.state in ("hold", "locked"):
                self._fire(self._cb_cancel, "paused")
                self.state = "suppressed"
            elif enabled and self.state == "suppressed":
                self.state = "suppressed" if self._held_mods() else "idle"

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
            req = self._active
            st = self.state

            if mod:
                if down:
                    if st == "idle":
                        for mode, combo in self.combos().items():
                            if held == combo:
                                self.state = "hold"
                                self.active_mode = mode
                                self._active = req = combo
                                self._armed_stop = False
                                self._needs_mask = bool(req & {"alt", "win"})
                                self._fire(self._cb_start, mode)
                                break
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
                if mod in ("alt", "win") and self._needs_mask:
                    other_down = any(v != vk and _VK_TO_GROUP.get(v) == mod for v in self._down)
                    if not other_down:
                        self._inject_mask()
                        self._needs_mask = False
                return False

            # ---- non-modifier key
            if not down:
                return False
            if st == "idle" and self.paste is not None and self._cb_paste is not None \
                    and vk == self.paste[1] and held == self.paste[0]:
                # paste-last chord: swallow the key (+ its repeats / key-up) and mask the modifiers' release so
                # Windows neither opens the menu bar (Alt) nor switches the keyboard layout (Alt+Shift)
                self._swallow.add(vk)
                self._inject_mask()
                self._fire(self._cb_paste)
                return True
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
                held = self._held_mods()
                if self.enabled and self.state == "idle" and \
                        any(phys == set(c) and not c <= held for c in self.combos().values()):
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
