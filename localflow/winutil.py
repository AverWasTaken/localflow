"""Small Win32 helpers shared by several modules (ctypes, no pywin32 needed)."""
from __future__ import annotations

import ctypes
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

# marker placed in dwExtraInfo of every key event WE inject, so our hook can ignore them
INJECT_MARK = 0x4C46_4C57  # "LFLW"

ULONG_PTR = ctypes.c_size_t

INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_SCANCODE = 0x0008

VK_BACK = 0x08
VK_TAB = 0x09
VK_RETURN = 0x0D
VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12
VK_ESCAPE = 0x1B
VK_SPACE = 0x20
VK_LWIN = 0x5B
VK_RWIN = 0x5C
VK_LSHIFT, VK_RSHIFT = 0xA0, 0xA1
VK_LCONTROL, VK_RCONTROL = 0xA2, 0xA3
VK_LMENU, VK_RMENU = 0xA4, 0xA5
VK_UNASSIGNED = 0xE8  # unassigned VK used for the Alt/Win "menu mask" trick


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
user32.SendInput.restype = wintypes.UINT
user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
user32.GetAsyncKeyState.restype = ctypes.c_short


def _key_input(vk: int = 0, scan: int = 0, flags: int = 0) -> INPUT:
    i = INPUT()
    i.type = INPUT_KEYBOARD
    i.ki = KEYBDINPUT(vk, scan, flags, 0, INJECT_MARK)
    return i


def send_inputs(inputs: list[INPUT]) -> int:
    if not inputs:
        return 0
    arr = (INPUT * len(inputs))(*inputs)
    return user32.SendInput(len(inputs), arr, ctypes.sizeof(INPUT))


def send_vk(vk: int, up: bool = False, extended: bool = False) -> None:
    flags = (KEYEVENTF_KEYUP if up else 0) | (KEYEVENTF_EXTENDEDKEY if extended else 0)
    send_inputs([_key_input(vk=vk, flags=flags)])


def tap_vk(vk: int) -> None:
    send_inputs([_key_input(vk=vk), _key_input(vk=vk, flags=KEYEVENTF_KEYUP)])


def is_vk_down(vk: int) -> bool:
    return bool(user32.GetAsyncKeyState(vk) & 0x8000)


MODIFIER_VKS = {
    "ctrl": (VK_LCONTROL, VK_RCONTROL),
    "alt": (VK_LMENU, VK_RMENU),
    "shift": (VK_LSHIFT, VK_RSHIFT),
    "win": (VK_LWIN, VK_RWIN),
}


def physical_modifiers_down() -> set[str]:
    return {name for name, vks in MODIFIER_VKS.items() if any(is_vk_down(v) for v in vks)}


# ----- foreground window / process -----
user32.GetForegroundWindow.restype = wintypes.HWND
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user32.GetWindowThreadProcessId.restype = wintypes.DWORD
kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
                                                ctypes.POINTER(wintypes.DWORD)]
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
user32.GetWindowTextLengthW.restype = ctypes.c_int
user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
_WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
user32.EnumChildWindows.argtypes = [wintypes.HWND, _WNDENUMPROC, wintypes.LPARAM]

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def window_title(hwnd) -> str:
    """Full window title (browsers put the active tab's title here)."""
    try:
        n = user32.GetWindowTextLengthW(hwnd)
        if n <= 0:
            return ""
        buf = ctypes.create_unicode_buffer(min(n, 2048) + 1)
        user32.GetWindowTextW(hwnd, buf, len(buf))
        return buf.value
    except Exception:
        return ""


def pid_exe_name(pid: int) -> str:
    """Lower-cased exe file name for a pid via QueryFullProcessImageNameW (limited-rights handle, works for
    elevated targets when we are elevated); falls back to psutil for protected processes. '' if unknown."""
    h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if h:
        try:
            buf = ctypes.create_unicode_buffer(1024)
            size = wintypes.DWORD(1024)
            if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
                return buf.value.rsplit("\\", 1)[-1].lower()
        finally:
            kernel32.CloseHandle(h)
    try:   # access denied (protected / higher-integrity process): psutil can still read the image name
        import psutil
        return psutil.Process(pid).name().lower()
    except Exception:
        return ""


def _uwp_inner_exe(hwnd, frame_pid: int) -> str:
    """ApplicationFrameHost.exe hosts UWP windows; the real app owns a child CoreWindow in another process."""
    found: list[str] = []

    def cb(child, _lp):
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(child, ctypes.byref(pid))
        if pid.value and pid.value != frame_pid:
            name = pid_exe_name(pid.value)
            if name and name != "applicationframehost.exe":
                found.append(name)
                return False
        return True

    try:
        user32.EnumChildWindows(hwnd, _WNDENUMPROC(cb), 0)
    except Exception:
        pass
    return found[0] if found else ""


def window_info(hwnd) -> tuple[str, str]:
    """(exe name lowercased, title) for a window handle; ('', '') parts when unknown."""
    try:
        if not hwnd:
            return "", ""
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        title = window_title(hwnd)
        name = pid_exe_name(pid.value) if pid.value else ""
        if name == "applicationframehost.exe":
            name = _uwp_inner_exe(hwnd, pid.value) or name
        return name, title
    except Exception:
        return "", ""


def foreground_info() -> tuple[str, str]:
    """(process exe name lowercased, window title) of the focused window; ('','') if unknown. ~0.1 ms."""
    try:
        return window_info(user32.GetForegroundWindow())
    except Exception:
        return "", ""
