"""Rotating file logging + crash capture (the app runs under pythonw, so there is no console)."""
from __future__ import annotations

import faulthandler
import logging
import logging.handlers
import os
import sys
import threading

from .paths import DATA_DIR, LOG_PATH

_crash_file = None


def setup(debug: bool = False) -> None:
    global _crash_file
    # pythonw has no std streams; make sure libs that print/tqdm don't blow up
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if debug else logging.INFO)
    fh = logging.handlers.RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root.addHandler(fh)
    if os.environ.get("LOCALFLOW_CONSOLE"):
        sh = logging.StreamHandler(sys.__stderr__ or sys.stderr)
        sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
        root.addHandler(sh)
    for noisy in ("urllib3", "PIL", "httpx", "httpcore", "huggingface_hub", "faster_whisper", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.captureWarnings(True)
    try:
        _crash_file = open(DATA_DIR / "crash.log", "a", buffering=1)
        faulthandler.enable(_crash_file)
    except Exception:
        pass

    def excepthook(t, v, tb):
        logging.getLogger("crash").error("uncaught exception", exc_info=(t, v, tb))

    def thread_hook(args):
        logging.getLogger("crash").error("uncaught thread exception in %s", args.thread.name if args.thread else "?",
                                         exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

    sys.excepthook = excepthook
    threading.excepthook = thread_hook
    sys.setswitchinterval(0.002)   # keep the keyboard-hook thread snappy under GIL contention
