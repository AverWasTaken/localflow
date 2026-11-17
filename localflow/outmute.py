"""Mute the speakers while dictating, so the microphone only hears your voice (setting audio.mute_output_while_dictating).

Only the Windows default playback device for the console role is muted, plus the multimedia-role default when it is a
different device. The communications default and every other endpoint are never touched. Devices that were already
muted are left alone, and only the devices this module muted are unmuted again, looked up by their saved device id
(so changing the default output mid-dictation still restores the right one).

All COM calls run on one dedicated worker thread, in order; callers never wait longer than a short bounded time.
Safety nets: a marker file listing the devices we muted is written BEFORE muting and removed after restoring (the next
start restores whatever it lists, e.g. after a crash), a liveness poll restores once the app is no longer recording,
and a maximum mute duration. A device that is unplugged while muted stays in the marker and is retried on the next
start / stop / app startup.

Plain ctypes COM (same helpers as devwatch.py), no comtypes / pycaw. Never raises into callers.
"""
from __future__ import annotations

import ctypes
import json
import logging
import os
import queue
import sys
import threading
import time
from ctypes import POINTER, byref, c_int, c_ulong, c_ushort, c_void_p
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger(__name__)

MARKER_NAME = "output_mute.json"

_E_RENDER = 0           # EDataFlow.eRender
_ROLE_CONSOLE = 0       # ERole.eConsole ("Default device" in Sound settings)
_ROLE_MULTIMEDIA = 1    # ERole.eMultimedia (eCommunications = 2 is never touched)
_CLSCTX_ALL = 23
_DEVICE_STATE_ACTIVE = 1
_STGM_READ = 0
_VT_LPWSTR = 31

# vtable slots, checked against the Windows SDK 10.0.26100 headers (mmdeviceapi.h, endpointvolume.h, propsys.h)
_ENUM_GET_DEFAULT = 4   # IMMDeviceEnumerator::GetDefaultAudioEndpoint
_ENUM_GET_DEVICE = 5    # IMMDeviceEnumerator::GetDevice
_DEV_ACTIVATE = 3       # IMMDevice::Activate
_DEV_OPEN_PROPS = 4     # IMMDevice::OpenPropertyStore
_DEV_GET_ID = 5         # IMMDevice::GetId
_DEV_GET_STATE = 6      # IMMDevice::GetState
_EPV_SET_MUTE = 14      # IAudioEndpointVolume::SetMute(BOOL, LPCGUID)
_EPV_GET_MUTE = 15      # IAudioEndpointVolume::GetMute(BOOL*)
_PS_GET_VALUE = 5       # IPropertyStore::GetValue(REFPROPERTYKEY, PROPVARIANT*)


class _PROPVARIANT(ctypes.Structure):
    _fields_ = [("vt", c_ushort), ("r1", c_ushort), ("r2", c_ushort), ("r3", c_ushort),
                ("p", c_void_p), ("p2", c_void_p)]


class WinEndpoints:
    """Real Windows backend. Every method must be called from the one thread that called init()."""

    def __init__(self):
        self._enum = c_void_p()
        self._inited = False
        self._dw = None

    def init(self) -> None:
        from . import devwatch as dw
        self._dw = dw
        hr = ctypes.windll.ole32.CoInitializeEx(None, 0)       # MTA; S_OK / S_FALSE need CoUninitialize
        self._inited = hr in (0, 1)
        self._ensure_enum()

    def close(self) -> None:
        if self._dw is not None:
            self._dw._release(self._enum)
        self._enum = c_void_p()
        if self._inited:
            self._inited = False
            ctypes.windll.ole32.CoUninitialize()

    def _ensure_enum(self):
        if not self._enum:
            dw = self._dw
            clsid = dw._guid("{BCDE0395-E52F-467C-8E3D-C4579291692E}")     # MMDeviceEnumerator
            iid = dw._guid("{A95664D2-9614-4F35-A746-DE8DB63617E6}")       # IMMDeviceEnumerator
            enum = c_void_p()
            ctypes.oledll.ole32.CoCreateInstance(byref(clsid), None, _CLSCTX_ALL, byref(iid), byref(enum))
            self._enum = enum
        return self._enum

    def _reset_enum(self) -> None:
        self._dw._release(self._enum)
        self._enum = c_void_p()

    def _dev_id(self, dev) -> Optional[str]:
        out = c_void_p()
        self._dw._vcall(dev, _DEV_GET_ID, (POINTER(c_void_p),), byref(out))
        try:
            return ctypes.wstring_at(out.value) if out.value else None
        finally:
            ole32 = ctypes.windll.ole32
            ole32.CoTaskMemFree.argtypes = [c_void_p]
            ole32.CoTaskMemFree(out.value)

    def _dev_name(self, dev) -> str:
        dw = self._dw
        store = c_void_p()
        try:
            dw._vcall(dev, _DEV_OPEN_PROPS, (c_ulong, POINTER(c_void_p)), _STGM_READ, byref(store))
            g = dw._guid("{A45C254E-DF1C-4EFD-8020-67D146A850E0}")         # PKEY_Device_FriendlyName

            class PROPERTYKEY(ctypes.Structure):
                _fields_ = [("fmtid", type(g)), ("pid", c_ulong)]
            key = PROPERTYKEY(g, 14)
            pv = _PROPVARIANT()
            dw._vcall(store, _PS_GET_VALUE, (POINTER(PROPERTYKEY), POINTER(_PROPVARIANT)), byref(key), byref(pv))
            try:
                return ctypes.wstring_at(pv.p) if pv.vt == _VT_LPWSTR and pv.p else ""
            finally:
                ctypes.windll.ole32.PropVariantClear(byref(pv))
        except Exception:
            return ""
        finally:
            dw._release(store)

    def _open(self, dev_id: str):
        """Active IMMDevice for this id, or None when it is gone / unplugged / disabled."""
        dw = self._dw
        for attempt in (0, 1):          # one retry with a fresh enumerator (audio service restarted)
            dev = c_void_p()
            try:
                dw._vcall(self._ensure_enum(), _ENUM_GET_DEVICE, (ctypes.c_wchar_p, POINTER(c_void_p)),
                          dev_id, byref(dev))
                st = c_ulong(0)
                dw._vcall(dev, _DEV_GET_STATE, (POINTER(c_ulong),), byref(st))
                if st.value != _DEVICE_STATE_ACTIVE:
                    dw._release(dev)
                    return None
                return dev
            except OSError:
                dw._release(dev)
                if attempt == 0:
                    self._reset_enum()
                    continue
                return None
        return None

    def _epv(self, dev):
        dw = self._dw
        iid = dw._guid("{5CDF2C82-841E-4546-9722-0CF74078229A}")           # IAudioEndpointVolume
        epv = c_void_p()
        dw._vcall(dev, _DEV_ACTIVATE, (POINTER(type(iid)), c_ulong, c_void_p, POINTER(c_void_p)),
                  byref(iid), _CLSCTX_ALL, None, byref(epv))
        return epv

    def defaults(self) -> list[tuple[str, str]]:
        """[(device id, friendly name)] of the default render endpoint (console role) and, if different, the
        multimedia-role default. Empty when there is no playback device."""
        dw = self._dw
        out: list[tuple[str, str]] = []
        for role in (_ROLE_CONSOLE, _ROLE_MULTIMEDIA):
            dev = c_void_p()
            try:
                dw._vcall(self._ensure_enum(), _ENUM_GET_DEFAULT, (c_int, c_int, POINTER(c_void_p)),
                          _E_RENDER, role, byref(dev))
                did = self._dev_id(dev)
                if did and all(did != d for d, _ in out):
                    out.append((did, self._dev_name(dev)))
            except OSError:
                pass                    # E_NOTFOUND: no playback device for this role
            finally:
                dw._release(dev)
        return out

    def get_mute(self, dev_id: str) -> Optional[bool]:
        """True / False, or None when the device is gone or cannot be read."""
        dev = self._open(dev_id)
        if dev is None:
            return None
        epv = c_void_p()
        try:
            epv = self._epv(dev)
            m = c_int(0)
            self._dw._vcall(epv, _EPV_GET_MUTE, (POINTER(c_int),), byref(m))
            return bool(m.value)
        except OSError:
            return None
        finally:
            self._dw._release(epv)
            self._dw._release(dev)

    def set_mute(self, dev_id: str, mute: bool) -> bool:
        dev = self._open(dev_id)
        if dev is None:
            return False
        epv = c_void_p()
        try:
            epv = self._epv(dev)
            self._dw._vcall(epv, _EPV_SET_MUTE, (c_int, c_void_p), 1 if mute else 0, None)   # S_FALSE = no change
            return True
        except OSError:
            return False
        finally:
            self._dw._release(epv)
            self._dw._release(dev)


class OutputMuter:
    """Serialized mute / restore of the default playback device(s). backend=None makes it fully inert.

    App flow: arm(rid) at hotkey-down, engage(rid, delay) once the hold is confirmed (mutes only if rid is still
    armed), release() whenever recording ends for any reason (disarms first, then restores). shutdown() on quit.
    """

    def __init__(self, backend=None, marker_path=None, is_recording: Optional[Callable[[], bool]] = None,
                 max_s: Optional[Callable[[], float]] = None, poll_s: float = 1.0):
        self._b = backend
        if marker_path is None and backend is not None:
            from .paths import DATA_DIR
            marker_path = DATA_DIR / MARKER_NAME
        self._marker = Path(marker_path) if marker_path is not None else None
        self._is_rec = is_recording or (lambda: False)
        self._max_s = max_s or (lambda: 330.0)
        self._poll_s = poll_s
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._armed: Optional[int] = None
        self._mutes_queued = 0                      # mute ops queued or running (lock)
        self._ours: dict[str, str] = {}             # device id -> name, muted by us and not restored yet (lock)
        self._engaged_at: Optional[float] = None    # worker only: when the current session muted something
        self._q: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._closed = False
        if backend is not None:
            self._ours = self._load_marker()

    @property
    def active(self) -> bool:
        return self._b is not None

    # ------------------------------------------------------------------ caller side (never blocks long, never raises)
    def arm(self, rid: int) -> None:
        if self._b is None:
            return
        with self._cv:
            self._armed = rid

    def engage(self, rid: int, delay: float = 0.0) -> None:
        """Mute (after ``delay`` seconds, e.g. to let the start cue play) if ``rid`` is still armed then."""
        if self._b is None:
            return
        try:
            with self._cv:
                if self._closed or self._armed != rid:
                    return
                self._mutes_queued += 1
            self._put(("mute", rid, max(0.0, float(delay))))
        except Exception:
            log.debug("output mute: engage failed", exc_info=True)

    def release(self, wait: float = 0.15) -> bool:
        """Disarm and restore whatever we muted. Waits up to ``wait`` seconds; True = restored (or nothing to do)."""
        if self._b is None:
            return True
        try:
            with self._cv:
                self._armed = None
                self._cv.notify_all()
                need = bool(self._ours) or self._mutes_queued > 0
            if not need:
                return True
            done = threading.Event()
            self._put(("restore", done))
            return done.wait(wait) if wait > 0 else False
        except Exception:
            log.debug("output mute: release failed", exc_info=True)
            return False

    def recover(self) -> None:
        """App startup: restore devices listed in a marker left behind by a crash / kill."""
        if self._b is None:
            return
        with self._cv:
            need = bool(self._ours)
        if need:
            log.info("output mute: restoring devices left muted by a previous run: %s",
                     ", ".join(self._ours.values()) or "?")
            self._put(("restore", None))

    def shutdown(self, wait: float = 1.0) -> None:
        if self._b is None:
            return
        try:
            with self._cv:
                self._armed = None
                self._closed = True
                self._cv.notify_all()
                running = self._thread is not None and self._thread.is_alive()
                need = bool(self._ours) or running
            if not need:
                return
            done = threading.Event()
            self._put(("stop", done))
            done.wait(wait)
        except Exception:
            log.debug("output mute: shutdown failed", exc_info=True)

    def _put(self, op) -> None:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="outmute", daemon=True)
                self._thread.start()
        self._q.put(op)

    # ------------------------------------------------------------------ worker thread
    def _run(self) -> None:
        try:
            if hasattr(self._b, "init"):
                self._b.init()
        except Exception:
            log.exception("output mute: audio endpoint API unavailable")
        try:
            while True:
                timeout = self._poll_s if self._engaged_at is not None else None
                try:
                    op = self._q.get(timeout=timeout)
                except queue.Empty:
                    self._check_alive()
                    continue
                kind = op[0]
                try:
                    if kind == "mute":
                        self._op_mute(op[1], op[2])
                    elif kind == "restore":
                        self._restore()
                    elif kind == "stop":
                        self._restore()
                except Exception:
                    log.exception("output mute: %s failed", kind)
                finally:
                    if kind in ("restore", "stop") and op[1] is not None:
                        op[1].set()
                if kind == "stop":
                    break
        finally:
            try:
                if hasattr(self._b, "close"):
                    self._b.close()
            except Exception:
                pass
            with self._lock:
                self._thread = None

    def _check_alive(self) -> None:
        if self._engaged_at is None:
            return
        try:
            rec = bool(self._is_rec())
        except Exception:
            rec = False
        try:
            limit = float(self._max_s())
        except Exception:
            limit = 330.0
        why = ""
        if not rec:
            why = "app is no longer recording"
        elif time.monotonic() - self._engaged_at > limit:
            why = f"muted for over {limit:.0f}s"
        if why:
            log.warning("output mute: safety restore (%s)", why)
            with self._cv:
                self._armed = None
            self._restore()

    def _op_mute(self, rid: int, delay: float) -> None:
        try:
            with self._cv:
                if delay > 0:
                    self._cv.wait_for(lambda: self._armed != rid or self._closed, timeout=delay)
                still = self._armed == rid and not self._closed
            if still:
                self._mute()
        finally:
            with self._cv:
                self._mutes_queued -= 1

    def _mute(self) -> None:
        b = self._b
        targets = b.defaults()
        target_ids = {d for d, _ in targets}
        with self._lock:
            leftovers = [(d, n) for d, n in self._ours.items() if d not in target_ids]
        if leftovers:                   # devices we muted earlier and could not restore (unplugged): retry now
            self._restore(only={d for d, _ in leftovers})
        muted_any = False
        for dev_id, name in targets:
            st = b.get_mute(dev_id)
            if st is None:
                continue
            with self._lock:
                ours = dev_id in self._ours
            if ours:                    # still ours from an earlier session: keep it muted, restore later
                if not st:
                    b.set_mute(dev_id, True)
                muted_any = True
                continue
            if st:
                log.info("output mute: %s was already muted, leaving it alone", name or dev_id)
                continue
            with self._lock:
                self._ours[dev_id] = name
            self._write_marker()        # BEFORE muting: a crash right after still gets restored next start
            if b.set_mute(dev_id, True):
                log.info("output muted while dictating: %s", name or dev_id)
                muted_any = True
            else:
                log.warning("output mute: could not mute %s", name or dev_id)
                with self._lock:
                    self._ours.pop(dev_id, None)
                self._write_marker()
        if muted_any:
            self._engaged_at = time.monotonic()

    def _restore(self, only: Optional[set] = None) -> None:
        b = self._b
        with self._lock:
            items = [(d, n) for d, n in self._ours.items() if only is None or d in only]
        for dev_id, name in items:
            st = b.get_mute(dev_id)
            if st is None:
                log.warning("output mute: %s is not available, will unmute it later", name or dev_id)
                continue
            if st and not b.set_mute(dev_id, False):
                log.warning("output mute: could not unmute %s, will retry", name or dev_id)
                continue
            log.info("output restored: %s", name or dev_id)
            with self._lock:
                self._ours.pop(dev_id, None)
        if only is None:
            self._engaged_at = None
        self._write_marker()

    # ------------------------------------------------------------------ marker file
    def _load_marker(self) -> dict[str, str]:
        p = self._marker
        if p is None or not p.exists():
            return {}
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            devs = data.get("devices") or {}
            return {str(k): str(v) for k, v in devs.items()}
        except Exception:
            log.warning("output mute: unreadable marker %s, ignoring it", p)
            try:
                p.unlink()
            except OSError:
                pass
            return {}

    def _write_marker(self) -> None:
        p = self._marker
        if p is None:
            return
        with self._lock:
            devs = dict(self._ours)
        try:
            if not devs:
                if p.exists():
                    p.unlink()
                return
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(p.name + ".tmp")
            tmp.write_text(json.dumps({"version": 1, "pid": os.getpid(), "time": time.time(), "devices": devs}),
                           encoding="utf-8")
            os.replace(tmp, p)
        except OSError:
            log.warning("output mute: could not write marker %s", p, exc_info=True)


def make_real(**kw) -> OutputMuter:
    """The real muter (Windows only; inert elsewhere). Only the app's real entry point uses this."""
    if sys.platform != "win32":
        return OutputMuter(None)
    return OutputMuter(WinEndpoints(), **kw)
