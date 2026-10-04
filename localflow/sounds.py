"""Tiny generated start/stop/cancel/error blips (soft sine tones), played asynchronously via winsound."""
from __future__ import annotations

import logging
import threading
import wave
from pathlib import Path

import numpy as np

from .paths import DATA_DIR

log = logging.getLogger(__name__)

SR = 22050
_DIR = DATA_DIR / "sounds"
_lock = threading.Lock()
_volume_built = -1.0

# name -> list of (freq_hz, ms)
_PATTERNS = {
    "start": [(660, 55), (880, 70)],
    "stop": [(880, 50), (587, 80)],
    "cancel": [(440, 60), (330, 80)],
    "error": [(300, 90), (220, 140)],
}


def _tone(freq: float, ms: int, vol: float) -> np.ndarray:
    n = int(SR * ms / 1000)
    t = np.arange(n) / SR
    env = np.minimum(1, np.minimum(t / 0.008, (t[-1] - t) / 0.02 if n else 1))   # soft attack/release
    return (np.sin(2 * np.pi * freq * t) * env * vol).astype(np.float32)


def ensure_sounds(volume: float) -> None:
    global _volume_built
    with _lock:
        if abs(volume - _volume_built) < 1e-3 and _DIR.exists():
            return
        _DIR.mkdir(parents=True, exist_ok=True)
        for name, pat in _PATTERNS.items():
            data = np.concatenate([_tone(f, ms, max(0.0, min(1.0, volume))) for f, ms in pat])
            pcm = (data * 32767).astype(np.int16)
            with wave.open(str(_DIR / f"{name}.wav"), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SR)
                wf.writeframes(pcm.tobytes())
        _volume_built = volume


def play(name: str, cfg=None) -> None:
    try:
        if cfg is not None and not cfg.get("ui.sounds", True):
            return
        vol = float(cfg.get("ui.sound_volume", 0.25)) if cfg is not None else 0.25
        ensure_sounds(vol)
        import winsound
        winsound.PlaySound(str(_DIR / f"{name}.wav"), winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
    except Exception:
        log.debug("sound failed", exc_info=True)
