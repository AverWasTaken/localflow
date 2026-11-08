"""Start Menu entry point: starts LocalFlow without a console window.

If the "LocalFlow" startup task runs this copy, start it through the task (elevated, no UAC prompt).
Otherwise start this copy directly. A second instance exits on its own (single-instance mutex)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CREATE_NO_WINDOW = 0x08000000


def main() -> int:
    try:
        from localflow import autostart
        if autostart.task_is_ours():
            r = subprocess.run(["schtasks", "/Run", "/TN", autostart.TASK_NAME], capture_output=True,
                               creationflags=CREATE_NO_WINDOW, timeout=30)
            if r.returncode == 0:
                return 0
    except Exception:
        pass
    pyw = ROOT / ".venv" / "Scripts" / "pythonw.exe"
    exe = str(pyw) if pyw.exists() else sys.executable
    subprocess.Popen([exe, "-m", "localflow"], cwd=str(ROOT), creationflags=CREATE_NO_WINDOW,
                     close_fds=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
