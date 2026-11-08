"""Foreground gate: live dictation / command mode always beats background jobs on the GPU.

Foreground work (a dictation or command, from key-down until its text is inserted, cancelled or
discarded) calls ``begin()`` / ``end()``. Background jobs (file + meeting transcription, transcript
cleanup, the transcript assistant) call ``wait_idle()`` before EACH unit of GPU work (one STT chunk,
one LLM request) and, for long streamed LLM requests, poll ``busy()`` to abort early.

A unit that is already running is never preempted, so keep background units short.
"""
from __future__ import annotations

import threading
import time


class ForegroundGate:
    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._active = 0
        self._last_end = 0.0

    def begin(self) -> None:
        """Foreground demand starts (call as early as possible, e.g. on hotkey down)."""
        with self._cv:
            self._active += 1

    def end(self) -> None:
        """Foreground work finished (inserted, cancelled, discarded or failed). Safe to over-call."""
        with self._cv:
            if self._active > 0:
                self._active -= 1
            self._last_end = time.monotonic()
            self._cv.notify_all()

    def busy(self) -> bool:
        with self._cv:
            return self._active > 0

    def wait_idle(self, timeout: float | None = None, settle_s: float = 0.5,
                  stop: threading.Event | None = None) -> bool:
        """Block until no foreground work is active and ``settle_s`` has passed since the last end.

        Returns False on timeout or when ``stop`` is set (job cancelled / app quitting)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cv:
            while True:
                if stop is not None and stop.is_set():
                    return False
                now = time.monotonic()
                if self._active == 0:
                    left = self._last_end + settle_s - now
                    if left <= 0:
                        return True
                else:
                    left = 0.25
                if deadline is not None:
                    if now >= deadline:
                        return False
                    left = min(left, deadline - now)
                self._cv.wait(max(0.01, min(left, 0.25)))
