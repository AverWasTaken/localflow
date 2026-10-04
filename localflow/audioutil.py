"""Small numpy audio helpers shared by the pipeline and the streaming session (16 kHz mono float32)."""
from __future__ import annotations

import numpy as np


def peak_window_rms(audio: np.ndarray, sr: int = 16000, win_ms: int = 100) -> float:
    n = int(sr * win_ms / 1000)
    if audio.size < n:
        return float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
    m = (audio.size // n) * n
    frames = audio[:m].reshape(-1, n)
    return float(np.sqrt((frames ** 2).mean(axis=1)).max())


def normalize(audio: np.ndarray) -> np.ndarray:
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if 0 < peak < 0.3:
        audio = audio * min(8.0, 0.5 / peak)
    return audio
