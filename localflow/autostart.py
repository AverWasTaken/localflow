"""Autostart via a Task Scheduler logon task (run level Highest, so we can paste into elevated windows)."""
from __future__ import annotations

import logging
import subprocess

from .paths import PROJECT_DIR

log = logging.getLogger(__name__)
TASK_NAME = "LocalFlow"
CREATE_NO_WINDOW = 0x08000000


def _run(args: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, creationflags=CREATE_NO_WINDOW)


def is_enabled() -> bool:
    try:
        return _run(["schtasks", "/Query", "/TN", TASK_NAME], 15).returncode == 0
    except Exception:
        return False


def set_enabled(enable: bool) -> bool:
    try:
        if enable:
            script = PROJECT_DIR / "scripts" / "register-task.ps1"
            r = _run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)])
        else:
            r = _run(["schtasks", "/Delete", "/TN", TASK_NAME, "/F"])
        if r.returncode != 0:
            log.error("autostart change failed: %s %s", r.stdout, r.stderr)
        return r.returncode == 0
    except Exception:
        log.exception("autostart change failed")
        return False
