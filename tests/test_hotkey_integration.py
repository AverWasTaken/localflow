"""Integration test of the real keyboard hook + paste path + Alt-menu mask.

Starts LocalFlow as a subprocess with LOCALFLOW_TEST_WAV (a SAPI recording substitutes for the microphone),
then drives it with unmarked SendInput key events, exactly like a physical keyboard, into Notepad and into
a probe window that records which keys reach an application.

   .venv\\Scripts\\python.exe tests\\test_hotkey_integration.py [--no-mask]   (--no-mask is the negative control)
Stop any running LocalFlow instance first (single-instance mutex).
"""
import ctypes
import os
import subprocess
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path

import win32api
import win32clipboard
import win32con
import win32gui

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from localflow import winutil as w  # noqa: E402

NO_MASK = "--no-mask" in sys.argv
ALT_ONLY = "--alt-only" in sys.argv    # hotkey = Alt alone: the case where the menu mask really matters
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


# ---- raw keyboard injection WITHOUT our marker (looks like real hardware to the hook)
def key(vk, up=False):
    i = w.INPUT()
    i.type = 1
    i.ki = w.KEYBDINPUT(vk, 0, w.KEYEVENTF_KEYUP if up else 0, 0, 0)
    w.user32.SendInput(1, ctypes.byref(i), ctypes.sizeof(w.INPUT))
    time.sleep(0.03)


CTRL, ALT, SPACE, ESC, T, X = 0xA2, 0xA4, 0x20, 0x1B, 0x54, 0x58


def hold_dictation(seconds):
    key(CTRL)
    key(ALT)
    time.sleep(seconds)
    key(CTRL, up=True)
    key(ALT, up=True)


# ---- probe window recording received keys
received = []
keymenu = []      # WM_SYSCOMMAND/SC_KEYMENU == Windows would activate the menu bar after a lone Alt tap


def _wndproc(hwnd, msg, wp, lp):
    if msg in (win32con.WM_KEYDOWN, win32con.WM_SYSKEYDOWN):
        received.append(wp)
    elif msg == win32con.WM_SYSCOMMAND and (wp & 0xFFF0) == 0xF100:
        keymenu.append(wp)
    elif msg == win32con.WM_DESTROY:
        win32gui.PostQuitMessage(0)
    return win32gui.DefWindowProc(hwnd, msg, wp, lp)


probe_hwnd = [0]


def probe_thread():
    wc = win32gui.WNDCLASS()
    wc.lpszClassName = "LFProbe"
    wc.lpfnWndProc = _wndproc
    wc.hInstance = win32api.GetModuleHandle(None)
    win32gui.RegisterClass(wc)
    menu = win32gui.CreateMenu()
    sub = win32gui.CreatePopupMenu()
    win32gui.AppendMenu(sub, win32con.MF_STRING, 1, "Item")
    win32gui.AppendMenu(menu, win32con.MF_POPUP, sub, "&File")
    hwnd = win32gui.CreateWindow("LFProbe", "LF probe", win32con.WS_OVERLAPPEDWINDOW | win32con.WS_VISIBLE,
                                 100, 100, 400, 200, 0, menu, wc.hInstance, None)
    probe_hwnd[0] = hwnd
    win32gui.PumpMessages()


def focus(hwnd, trick=True):
    # Alt-tap trick lets SetForegroundWindow succeed from a background process (but it would itself
    # activate Notepad's access keys, so Notepad is only focused if it is not already foreground)
    if trick:
        w.user32.keybd_event(0x12, 0, 0, 0)
        w.user32.keybd_event(0x12, 0, 2, 0)
    for _ in range(3):
        if win32gui.GetForegroundWindow() == hwnd:
            break
        try:
            win32gui.SetForegroundWindow(hwnd)
        except Exception:
            time.sleep(0.3)
    time.sleep(0.3)


def clipboard_text():
    for _ in range(10):
        try:
            win32clipboard.OpenClipboard()
            try:
                return win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
            finally:
                win32clipboard.CloseClipboard()
        except Exception:
            time.sleep(0.1)
    return None


def notepad_text(hwnd):
    out = []

    def cb(h, _):
        n = win32gui.SendMessage(h, win32con.WM_GETTEXTLENGTH, 0, 0)
        if n:
            buf = win32gui.PyMakeBuffer(n + 1) if False else None
            out.append(win32gui.GetWindowText(h) or "")
            try:
                out.append(win32gui.SendMessage(h, win32con.WM_GETTEXT, n + 1, 0))
            except Exception:
                pass
        return True
    win32gui.EnumChildWindows(hwnd, cb, None)
    return out


def screenshot(path):
    from PySide6.QtGui import QGuiApplication
    app = QGuiApplication.instance() or QGuiApplication([])
    app.primaryScreen().grabWindow(0).save(str(path))


def finish():
    fails = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(fails)}/{len(results)} checks passed", ("; FAILED: " + ", ".join(fails)) if fails else "")
    return 1 if fails else 0


def main():
    env = dict(os.environ, LOCALFLOW_TEST_WAV=str(ROOT / "tests" / "audio" / "02_fillers.wav"),
               LOCALFLOW_CONSOLE="1")
    if NO_MASK:
        env["LOCALFLOW_NO_MASK"] = "1"
    if ALT_ONLY:
        env["LOCALFLOW_TEST_HOTKEY"] = "alt"
    log = open(ROOT / "tests" / "app_test.log", "w")
    app = subprocess.Popen([str(ROOT / ".venv" / "Scripts" / "python.exe"), "-m", "localflow"], cwd=str(ROOT),
                           env=env, stdout=log, stderr=subprocess.STDOUT)

    def applog():
        log.flush()
        return (ROOT / "tests" / "app_test.log").read_text(errors="ignore")

    try:
        t0 = time.time()
        while "LLM warm-up done" not in applog():
            if time.time() - t0 > 120:
                print(applog()[-2000:])
                raise SystemExit("app did not become ready")
            time.sleep(0.5)
        check("hook installed", "keyboard hook installed=True" in applog())
        check("microphone opened", "microphone:" in applog())
        check("STT on GPU", "device=cuda" in applog())

        threading.Thread(target=probe_thread, daemon=True).start()
        time.sleep(0.5)

        if ALT_ONLY:
            focus(probe_hwnd[0])
            keymenu.clear()
            key(ALT)
            time.sleep(0.8)
            key(ALT, up=True)
            time.sleep(3.0)
            if NO_MASK:
                check("negative control (Alt-only hotkey, no mask): menu bar IS activated", len(keymenu) >= 1, f"({keymenu})")
            else:
                check("Alt-only hotkey WITH mask: menu bar not activated", len(keymenu) == 0, f"({keymenu})")
            check("Alt-only hotkey dictated", "dictation:" in applog())
            raise SystemExit(finish())

        # ---------- 1a. Alt-menu mask on a classic Win32 menu window (SC_KEYMENU = menu bar would activate)
        focus(probe_hwnd[0])
        keymenu.clear()
        hold_dictation(0.8)
        time.sleep(3.0)
        check("Alt-menu mask: no SC_KEYMENU after Ctrl-then-Alt release", len(keymenu) == 0, f"({keymenu})")
        # Alt released BEFORE Ctrl as well
        keymenu.clear()
        key(CTRL)
        key(ALT)
        time.sleep(0.8)
        key(ALT, up=True)
        key(CTRL, up=True)
        time.sleep(3.0)
        if NO_MASK:
            check("negative control: WITHOUT mask a lone Alt release activates the menu (SC_KEYMENU)", len(keymenu) >= 1, f"({keymenu})")
        else:
            check("Alt-menu mask: no SC_KEYMENU when Alt is released first", len(keymenu) == 0, f"({keymenu})")

        # ---------- 1b. paste into Notepad (a throw-away temp file opens as a NEW tab; user's tabs untouched)
        import tempfile
        tmpf = Path(tempfile.gettempdir()) / "localflow_paste_test.txt"
        tmpf.write_text("", encoding="utf-8")
        subprocess.Popen(["notepad.exe", str(tmpf)])
        time.sleep(3.0)
        npad = win32gui.FindWindow("Notepad", None)
        title = win32gui.GetWindowText(npad) if npad else ""
        print("   notepad title:", title)
        ok_target = bool(npad) and "localflow_paste_test" in title
        check("notepad opened our temp file in the active tab", ok_target)
        if ok_target:
            focus(npad, trick=False)
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardText("CLIPBOARD-BEFORE")
            win32clipboard.CloseClipboard()
            hold_dictation(0.8)
            time.sleep(3.0)          # processing + paste + clipboard restore
            screenshot(ROOT / "tests" / "notepad_after_dictation.png")
            key(X)
            key(X, up=True)           # if the Alt menu was triggered, this 'x' will not be typed
            time.sleep(0.4)
            key(CTRL)
            key(0x53)
            key(0x53, up=True)        # Ctrl+S: save the temp file, then read it from disk
            key(CTRL, up=True)
            time.sleep(1.0)
            doc = tmpf.read_text(encoding="utf-8", errors="ignore")
            print("   file content:", repr(doc))
            check("dictation pasted into Notepad (cleaned, fillers removed)",
                  "ship the new build on Friday" in doc and " um" not in doc.lower() and "uh," not in doc.lower())
            check("Notepad: next keystroke typed normally after dictation", doc.rstrip().endswith("x"))
            check("clipboard restored after paste", clipboard_text() == "CLIPBOARD-BEFORE", f"({clipboard_text()!r})")
            key(CTRL)
            key(0x57)
            key(0x57, up=True)        # Ctrl+W: close only our temp tab
            key(CTRL, up=True)
            time.sleep(0.5)
        check("pipeline log shows paste", "dictation:" in applog())

        # ---------- 2. real shortcut passes through, no dictation
        n_before = applog().count("dictation:")
        focus(probe_hwnd[0])
        received.clear()
        key(CTRL)
        key(ALT)
        key(T)
        key(T, up=True)
        key(ALT, up=True)
        key(CTRL, up=True)
        time.sleep(1.0)
        check("Ctrl+Alt+T: key passed through to app", T in received, f"(received {received})")
        check("Ctrl+Alt+T: no dictation triggered", applog().count("dictation:") == n_before)
        check("Ctrl+Alt+T: recording cancelled silently", "cancelled (shortcut)" in applog())

        # ---------- 3. hands-free: Ctrl+Alt+Space locks, Ctrl+Alt again stops
        received.clear()
        key(CTRL)
        key(ALT)
        key(SPACE)
        key(SPACE, up=True)
        key(ALT, up=True)
        key(CTRL, up=True)
        time.sleep(0.8)          # still recording hands-free
        check("space swallowed when locking hands-free", SPACE not in received, f"(received {received})")
        key(CTRL)
        key(ALT)
        key(ALT, up=True)
        key(CTRL, up=True)
        time.sleep(3.0)
        check("hands-free: second Ctrl+Alt stopped + dictated", applog().count("dictation:") == n_before + 1)

        # ---------- 4. Esc cancels
        n_before = applog().count("dictation:")
        key(CTRL)
        key(ALT)
        time.sleep(0.4)
        key(ESC)
        key(ESC, up=True)
        key(ALT, up=True)
        key(CTRL, up=True)
        time.sleep(1.0)
        check("Esc cancels recording", "cancelled (esc)" in applog() and applog().count("dictation:") == n_before)
        check("Esc swallowed", ESC not in received)

        # ---------- 5. too-short tap discarded
        n_before = applog().count("dictation:")
        hold_dictation(0.15)
        time.sleep(1.0)
        check("very short tap produces no dictation", applog().count("dictation:") == n_before)
    finally:
        app.terminate()
        try:
            app.wait(10)
        except Exception:
            app.kill()
        print("\n--- app log tail ---")
        print("".join(applog().splitlines(True)[-14:]))
    return finish()


if __name__ == "__main__":
    sys.exit(main())
