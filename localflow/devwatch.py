"""Cheap lookup of the Windows default recording endpoint id (MMDevice API, plain ctypes, no extra deps).

PortAudio only enumerates devices when it is initialised, so it never notices that you changed your default
microphone in Sound settings or plugged one in. Asking MMDevice for the default capture endpoint's id is a
single COM call (about 5 ms), which makes it a good thing to poll; when the id changes the caller re-initialises
PortAudio and reopens the stream.
"""
from __future__ import annotations

import ctypes
import logging
import sys
from ctypes import POINTER, byref, c_ulong, c_void_p

log = logging.getLogger(__name__)

_E_CAPTURE = 1       # EDataFlow.eCapture
_E_CONSOLE = 0       # ERole.eConsole (what "Default device" in Sound settings means)
_CLSCTX_ALL = 23


def _guid(s: str):
    from ctypes import wintypes

    class GUID(ctypes.Structure):
        _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD),
                    ("Data4", ctypes.c_ubyte * 8)]
    g = GUID()
    ctypes.oledll.ole32.CLSIDFromString(s, byref(g))
    return g


def _vcall(obj, index: int, argtypes: tuple, *args, restype=ctypes.HRESULT):
    """Call COM method ``index`` of the interface pointer ``obj`` (a c_void_p)."""
    vtbl = ctypes.cast(obj, POINTER(POINTER(c_void_p)))[0]
    fn = ctypes.WINFUNCTYPE(restype, c_void_p, *argtypes)(vtbl[index])
    return fn(obj, *args)


def _release(obj) -> None:
    if obj:
        try:
            _vcall(obj, 2, (), restype=c_ulong)
        except Exception:
            pass


def default_capture_id() -> str | None:
    """Endpoint id of the current default recording device, or None (no device / not Windows / error)."""
    if sys.platform != "win32":
        return None
    ole32 = ctypes.windll.ole32
    hr = ole32.CoInitializeEx(None, 0)          # MTA. S_OK / S_FALSE need a matching CoUninitialize
    inited = hr in (0, 1)
    enum = c_void_p()
    dev = c_void_p()
    try:
        clsid = _guid("{BCDE0395-E52F-467C-8E3D-C4579291692E}")     # MMDeviceEnumerator
        iid = _guid("{A95664D2-9614-4F35-A746-DE8DB63617E6}")       # IMMDeviceEnumerator
        ctypes.oledll.ole32.CoCreateInstance(byref(clsid), None, _CLSCTX_ALL, byref(iid), byref(enum))
        # IMMDeviceEnumerator::GetDefaultAudioEndpoint (vtable slot 4)
        _vcall(enum, 4, (ctypes.c_int, ctypes.c_int, POINTER(c_void_p)), _E_CAPTURE, _E_CONSOLE, byref(dev))
        out = c_void_p()
        # IMMDevice::GetId (vtable slot 5)
        _vcall(dev, 5, (POINTER(c_void_p),), byref(out))
        try:
            return ctypes.wstring_at(out.value) if out.value else None
        finally:
            ole32.CoTaskMemFree.argtypes = [c_void_p]
            ole32.CoTaskMemFree(out.value)
    except OSError:
        return None          # E_NOTFOUND when no recording device exists
    except Exception:
        log.debug("default capture id lookup failed", exc_info=True)
        return None
    finally:
        _release(dev)
        _release(enum)
        if inited:
            ole32.CoUninitialize()
