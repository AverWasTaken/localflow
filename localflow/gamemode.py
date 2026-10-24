"""Game mode watcher: polls running process names; fires on_enter as soon as a configured game is seen and on_exit
once it has been gone for ``exit_grace_s``. Only process NAMES are read (nothing else about other processes)."""
from __future__ import annotations

import ctypes
import logging
import threading
import time
from ctypes import wintypes
from typing import Callable, Iterable, Optional

log = logging.getLogger("localflow")


class _PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD), ("szExeFile", ctypes.c_wchar * 260)]


def list_processes() -> list[tuple[int, str]]:
    """(pid, exe name) of every running process from a Toolhelp snapshot. This opens NO handle to any process (psutil's
    name() calls OpenProcess on each one the first time), so anti-cheat processes are never touched."""
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    k32.Process32FirstW.argtypes = k32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PROCESSENTRY32W)]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    snap = k32.CreateToolhelp32Snapshot(0x2, 0)       # TH32CS_SNAPPROCESS
    if not snap or snap == ctypes.c_void_p(-1).value:
        raise OSError("CreateToolhelp32Snapshot failed")
    out: list[tuple[int, str]] = []
    try:
        e = _PROCESSENTRY32W()
        e.dwSize = ctypes.sizeof(e)
        ok = k32.Process32FirstW(snap, ctypes.byref(e))
        while ok:
            out.append((int(e.th32ProcessID), e.szExeFile))
            ok = k32.Process32NextW(snap, ctypes.byref(e))
    finally:
        k32.CloseHandle(snap)
    if not out:
        raise OSError("empty process snapshot")      # a real one always has System/Idle: treat as scan failure
    return out


def list_process_names() -> set[str]:
    """Lowercased names of all running processes."""
    return {n.lower() for _, n in list_processes() if n}


class GameWatcher:
    def __init__(self, cfg, on_enter: Callable[[], None], on_exit: Callable[[], None],
                 lister: Callable[[], Iterable[str]] = list_process_names, clock: Callable[[], float] = time.monotonic):
        self.cfg = cfg
        self.on_enter = on_enter
        self.on_exit = on_exit
        self.lister = lister
        self.clock = clock
        self.active = False
        self._absent_since: Optional[float] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # --- config (read live on every poll)
    def _enabled(self) -> bool:
        return bool(self.cfg.get("game_mode.enabled", True))

    def _targets(self) -> set[str]:
        return {str(p).strip().lower() for p in (self.cfg.get("game_mode.processes", []) or []) if str(p).strip()}

    def game_running(self) -> bool:
        targets = self._targets()
        if not targets:
            return False
        try:
            return bool(targets & {str(n).lower() for n in self.lister()})
        except Exception:
            log.exception("game mode: process scan failed")
            return self.active          # unknown: keep the current state

    def detect_initial(self) -> bool:
        """Synchronous first poll at startup. Sets ``active`` WITHOUT firing callbacks; the caller applies it."""
        if self._enabled() and self.game_running():
            self.active = True
        return self.active

    def poll_once(self) -> None:
        if not self._enabled():
            self._absent_since = None
            if self.active:
                log.info("game mode: disabled while active -> exiting")
                self.active = False
                self._fire(self.on_exit)
            return
        if self.game_running():
            self._absent_since = None
            if not self.active:
                log.info("game mode: game detected -> entering")
                self.active = True
                self._fire(self.on_enter)
        elif self.active:
            now = self.clock()
            if self._absent_since is None:
                self._absent_since = now
            if now - self._absent_since >= float(self.cfg.get("game_mode.exit_grace_s", 10)):
                log.info("game mode: game gone for %.0fs -> exiting", now - self._absent_since)
                self._absent_since = None
                self.active = False
                self._fire(self.on_exit)

    @staticmethod
    def _fire(cb: Callable[[], None]) -> None:
        try:
            cb()
        except Exception:
            log.exception("game mode callback failed")

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="game-watch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception:
                log.exception("game mode poll failed")
            self._stop.wait(max(1.0, float(self.cfg.get("game_mode.poll_s", 3))))
