"""Live test of command mode + paste-last through the real keyboard hook, clipboard and LLM.

Starts LocalFlow as a subprocess with LOCALFLOW_TEST_WAV = a SAPI recording of "Make this all uppercase." standing
in for the microphone, then drives it with unmarked SendInput key events (like a physical keyboard) into Notepad:
  1. select a line, hold Ctrl+Win, release -> the selection is replaced by its uppercase version,
     the user's clipboard is restored, the Start menu does not open (Win menu mask)
  2. Alt+Shift+Z -> the last result is pasted again, the Z is swallowed
  3. nothing selected + an edit instruction -> "Select some text first", nothing inserted, clipboard intact

   .venv\\Scripts\\python.exe tests\\test_command_integration.py
Stop any running LocalFlow instance first (single-instance mutex).
"""
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import win32clipboard
import win32gui

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import test_hotkey_integration as hi  # noqa: E402  (key injection, focus, clipboard helpers)

CTRL, WIN, ALT, SHIFT, Z = 0xA2, 0x5B, 0xA4, 0xA0, 0x5A
check, key = hi.check, hi.key


def chord(*vks):
    for v in vks:
        key(v)
    for v in reversed(vks):
        key(v, up=True)


def type_ascii(s):
    for ch in s:
        vk = ord(ch.upper()) if ch.isalnum() else 0x20
        key(vk)
        key(vk, up=True)


def set_clip(text):
    win32clipboard.OpenClipboard()
    win32clipboard.EmptyClipboard()
    win32clipboard.SetClipboardText(text)
    win32clipboard.CloseClipboard()


def save_and_read(tmpf):
    chord(CTRL, 0x53)       # Ctrl+S
    time.sleep(1.0)
    return tmpf.read_text(encoding="utf-8", errors="ignore")


def main():
    wav = ROOT / "tests" / "audio" / "cmd_uppercase.wav"
    env = dict(os.environ, LOCALFLOW_TEST_WAV=str(wav), LOCALFLOW_CONSOLE="1")
    logp = ROOT / "tests" / "app_cmd_test.log"
    log = open(logp, "w")
    app = subprocess.Popen([str(ROOT / ".venv" / "Scripts" / "python.exe"), "-m", "localflow"], cwd=str(ROOT),
                           env=env, stdout=log, stderr=subprocess.STDOUT)

    def applog():
        log.flush()
        return logp.read_text(errors="ignore")

    try:
        t0 = time.time()
        while "LLM warm-up done" not in applog():
            if time.time() - t0 > 180:
                print(applog()[-2000:])
                raise SystemExit("app did not become ready")
            time.sleep(0.5)
        check("hook installed", "keyboard hook installed=True" in applog())
        tmpf = Path(tempfile.gettempdir()) / "localflow_command_test.txt"
        tmpf.write_text("", encoding="utf-8")
        subprocess.Popen(["notepad.exe", str(tmpf)])
        time.sleep(3.0)
        npad = win32gui.FindWindow("Notepad", None)
        ok = bool(npad) and "localflow_command_test" in win32gui.GetWindowText(npad)
        check("notepad opened our temp file", ok)
        if not ok:
            raise SystemExit(hi.finish())
        hi.focus(npad, trick=False)
        time.sleep(0.3)

        # ---------- 1. rewrite the selection
        type_ascii("hello world from localflow")
        chord(CTRL, 0x41)           # Ctrl+A: select it (Shift+Home is numpad-Home for unmarked SendInput)
        set_clip("CLIPBOARD-BEFORE")
        n0 = applog().count("command done:")
        key(CTRL)
        key(WIN)
        time.sleep(1.0)
        key(WIN, up=True)
        key(CTRL, up=True)
        t1 = time.time()
        while applog().count("command done:") == n0 and "command failed" not in applog() and time.time() - t1 < 30:
            time.sleep(0.2)
        time.sleep(0.8)
        fg_cls = win32gui.GetClassName(win32gui.GetForegroundWindow())
        check("Start menu did not open (Notepad still focused)", win32gui.GetForegroundWindow() == npad,
              f"(foreground class {fg_cls})")
        doc = save_and_read(tmpf)
        print("   file after command:", repr(doc))
        check("command rewrote the selection", doc.strip() == "HELLO WORLD FROM LOCALFLOW", f"({doc!r})")
        check("clipboard restored after command", hi.clipboard_text() == "CLIPBOARD-BEFORE",
              f"({hi.clipboard_text()!r})")
        check("command logged", "command: selection 26 chars" in applog() and "command done:" in applog())

        # ---------- 2. paste last (Alt+Shift+Z) at the end of the document
        chord(CTRL, 0x23)           # Ctrl+End
        key(0x0D)
        key(0x0D, up=True)          # Enter
        key(ALT)
        key(SHIFT)
        key(Z)
        key(Z, up=True)
        key(SHIFT, up=True)
        key(ALT, up=True)
        time.sleep(2.0)
        check("no menu bar / app still focused after Alt+Shift+Z", win32gui.GetForegroundWindow() == npad)
        doc = save_and_read(tmpf)
        print("   file after paste-last:", repr(doc))
        lines = [l for l in doc.splitlines() if l.strip()]
        check("paste-last inserted the last result, Z swallowed",
              lines == ["HELLO WORLD FROM LOCALFLOW", "HELLO WORLD FROM LOCALFLOW"], f"({lines})")
        check("paste-last logged", "pasted last transcript" in applog())

        # ---------- 3. nothing selected: the edit instruction has nothing to work on -> nothing inserted
        chord(CTRL, 0x23)
        key(0x0D)
        key(0x0D, up=True)
        set_clip("STALE-CLIP")
        n0 = applog().count("command done:")
        key(CTRL)
        key(WIN)
        time.sleep(0.8)
        key(WIN, up=True)
        key(CTRL, up=True)
        t1 = time.time()
        while applog().count("command done:") == n0 and applog().count("command failed") == 0 and time.time() - t1 < 30:
            time.sleep(0.2)
        time.sleep(0.8)
        before = doc
        doc = save_and_read(tmpf)
        print("   file after no-selection command:", repr(doc))
        check("no-selection command: selection seen as none", "command: selection none" in applog())
        check("no-selection edit command: 'nothing selected', nothing inserted",
              "command failed (nothing selected)" in applog() and doc.rstrip() == before.rstrip(), f"({doc!r})")
        check("clipboard intact after no-selection command", hi.clipboard_text() == "STALE-CLIP")
        chord(CTRL, 0x57)           # Ctrl+W: close only our temp tab
        time.sleep(0.5)
    finally:
        app.terminate()
        try:
            app.wait(10)
        except Exception:
            app.kill()
        print("\n--- app log (command lines) ---")
        print("".join(l for l in applog().splitlines(True) if "command" in l or "paste" in l or "ERROR" in l)[-3000:])
    return hi.finish()


if __name__ == "__main__":
    sys.exit(main())
