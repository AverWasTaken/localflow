"""Autostart via a Task Scheduler logon task (run level Highest, so we can paste into elevated windows).

There is one task name ("LocalFlow") per user, so several LocalFlow copies (a git checkout and an installed copy,
say) would fight over it. This module only reports, creates or deletes the task when it runs *this* copy."""
from __future__ import annotations

import html
import logging
import os
import re
import subprocess

from .paths import PROJECT_DIR

log = logging.getLogger(__name__)
TASK_NAME = "LocalFlow"
CREATE_NO_WINDOW = 0x08000000


def _run(args: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, creationflags=CREATE_NO_WINDOW,
                          errors="replace")


def task_executable() -> str | None:
    """The program the LocalFlow task runs, or None when there is no task (or it can't be read)."""
    try:
        r = _run(["schtasks", "/Query", "/TN", TASK_NAME, "/XML"], 15)
    except Exception:
        return None
    if r.returncode != 0:
        return None
    m = re.search(r"<Command>([^<]+)</Command>", (r.stdout or "").replace("\x00", ""))
    if not m:
        return None
    return os.path.expandvars(html.unescape(m.group(1)).strip().strip('"'))   # XML text: &amp; -> &


def _is_inside(path: str, root) -> bool:
    try:
        p = os.path.normcase(os.path.abspath(path))
        r = os.path.normcase(os.path.abspath(str(root))).rstrip("\\/")
    except Exception:
        return False
    return p == r or p.startswith(r + os.sep)


def task_is_ours() -> bool:
    exe = task_executable()
    return bool(exe) and _is_inside(exe, PROJECT_DIR)


def is_enabled() -> bool:
    return task_is_ours()


def set_enabled(enable: bool) -> bool:
    try:
        exe = task_executable()
        if exe and not _is_inside(exe, PROJECT_DIR):
            if enable:
                log.error("autostart: the %s task runs another LocalFlow copy (%s); not replacing it", TASK_NAME, exe)
                return False
            log.info("autostart: the %s task runs another LocalFlow copy (%s); leaving it alone", TASK_NAME, exe)
            return True
        if enable:
            script = PROJECT_DIR / "scripts" / "register-task.ps1"
            r = _run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)])
        else:
            if not exe:
                return True
            r = _run(["schtasks", "/Delete", "/TN", TASK_NAME, "/F"])
        if r.returncode != 0:
            log.error("autostart change failed: %s %s", r.stdout, r.stderr)
        return r.returncode == 0
    except Exception:
        log.exception("autostart change failed")
        return False
