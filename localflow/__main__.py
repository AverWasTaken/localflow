"""Entry point: python -m localflow  (run with pythonw for a windowless app)."""
import ctypes
import sys


def _single_instance():
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateMutexW.restype = ctypes.c_void_p
    handle = k32.CreateMutexW(None, False, "Global\\LocalFlow_SingleInstance_Mutex")
    if ctypes.get_last_error() == 183:      # ERROR_ALREADY_EXISTS
        return None
    return handle


def main() -> int:
    handle = _single_instance()
    if handle is None:
        return 0
    from . import logsetup
    logsetup.setup(debug="--debug" in sys.argv)
    from .app import run
    return run()


if __name__ == "__main__":
    sys.exit(main())
