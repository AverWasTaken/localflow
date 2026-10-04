"""Microphone capture: device selection, always-open stream + pre-roll ring buffer, resampling to 16 kHz."""
from __future__ import annotations

import collections
import logging
import os
import threading
import time
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from . import devwatch

log = logging.getLogger(__name__)

TARGET_SR = 16000
DEFAULT_POLL_S = 3.0      # how often to check whether the Windows default recording device changed


# ----------------------------------------------------------------------------- device helpers
def _hostapi_name(idx: int) -> str:
    try:
        return sd.query_hostapis(idx)["name"]
    except Exception:
        return ""


def list_input_devices() -> list[dict]:
    """All input-capable devices: dicts with index, name, hostapi, channels, default_sr."""
    out = []
    try:
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                out.append({"index": i, "name": d["name"], "hostapi": _hostapi_name(d["hostapi"]),
                            "channels": d["max_input_channels"], "default_sr": d["default_samplerate"]})
    except Exception:
        log.exception("query_devices failed")
    return out


def input_device_names() -> list[str]:
    """Unique input names (WASAPI preferred) for UI pickers."""
    seen: dict[str, dict] = {}
    for d in list_input_devices():
        if d["hostapi"] == "Windows WASAPI" or d["name"] not in seen:
            seen[d["name"]] = d
    return sorted(seen)


def _rank(d: dict) -> int:
    order = {"Windows WASAPI": 0, "Windows WDM-KS": 3, "Windows DirectSound": 2, "MME": 1}
    return order.get(d["hostapi"], 5)


def _device_dict(index: int, d: dict, hostapi: str) -> dict:
    return {"index": index, "name": d["name"], "hostapi": hostapi,
            "channels": d["max_input_channels"], "default_sr": d["default_samplerate"]}


def default_input_device() -> tuple[Optional[dict], str]:
    """The device Windows has selected as the default recording device.

    Asks the WASAPI host API first (its default input is the one chosen in Sound settings), then PortAudio's
    global default input. Returns (None, "system default") if neither knows an input device."""
    try:
        for h in sd.query_hostapis():
            idx = h.get("default_input_device", -1)
            if h["name"] == "Windows WASAPI" and idx is not None and idx >= 0:
                d = sd.query_devices(idx)
                if d["max_input_channels"] > 0:
                    return _device_dict(idx, d, h["name"]), "system default (WASAPI)"
    except Exception:
        log.exception("WASAPI default device lookup failed")
    try:
        idx = sd.default.device[0]
        if idx is None or idx < 0:
            idx = sd.query_devices(kind="input")["index"]
        d = sd.query_devices(idx)
        if d["max_input_channels"] > 0:
            return _device_dict(idx, d, _hostapi_name(d["hostapi"])), "system default (PortAudio)"
    except Exception:
        log.debug("PortAudio default input lookup failed", exc_info=True)
    return None, "system default"


def default_input_name() -> str:
    """Name of the Windows default recording device ("" if there is none) - for menus and labels."""
    dev, _ = default_input_device()
    return dev["name"] if dev else ""


def pick_device(match: str, exact: str = "") -> tuple[Optional[dict], str]:
    """Returns (device dict or None for system default, reason).

    Order: explicit ``exact`` name (tray / Settings pick), then ``match`` substring (power-user override),
    then the Windows default recording device. A name that is configured but not connected falls through to
    the default so dictation keeps working."""
    devs = list_input_devices()
    if exact:
        c = [d for d in devs if d["name"].lower() == exact.lower()] or             [d for d in devs if exact.lower() in d["name"].lower()]
        if c:
            return sorted(c, key=_rank)[0], f"selected '{exact}'"
        log.warning("configured microphone '%s' not found - falling back to the Windows default", exact)
    elif match:
        c = [d for d in devs if match.lower() in d["name"].lower()]
        if c:
            return sorted(c, key=_rank)[0], f"matched '{match}'"
        log.warning("no input device matching '%s' - falling back to the Windows default", match)
    return default_input_device()


def resample_to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SR or x.size == 0:
        return x.astype(np.float32, copy=False)
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(TARGET_SR, int(sr))
    return resample_poly(x, TARGET_SR // g, int(sr) // g).astype(np.float32)


# ----------------------------------------------------------------------------- recorder
class AudioInput:
    def __init__(self, cfg, on_level: Optional[Callable[[float], None]] = None):
        self.cfg = cfg                      # Config
        self.on_level = on_level
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.RLock()
        self._preroll: collections.deque = collections.deque()
        self._preroll_samples = 0
        self._rec: Optional[list[np.ndarray]] = None
        self._rec_samples = 0
        self._sr = TARGET_SR
        self.device: Optional[dict] = None
        self.device_name = "?"
        self.preferred_found = False
        self._last_cb = 0.0
        self._stop_monitor = threading.Event()
        self._monitor: Optional[threading.Thread] = None
        self._last_enum = 0.0
        self._last_default_check = 0.0
        self._default_id: Optional[str] = None     # Windows default capture endpoint id our stream was opened for
        self.on_device_change: Optional[Callable[[str], None]] = None   # called with the new device name
        self.max_samples_hit = False

    # --- config shortcuts
    @property
    def keep_open(self) -> bool:
        return bool(self.cfg.get("audio.keep_open", True))

    # --- stream management
    def _open(self) -> bool:
        match = self.cfg.get("mic.match", "")
        exact = self.cfg.get("mic.device", "")
        dev, why = pick_device(match, exact)
        self.preferred_found = bool(dev) and (why.startswith("matched") or why.startswith("selected"))
        idx = dev["index"] if dev else None
        name = dev["name"] if dev else "default"
        channels = 1
        attempts = []
        # try native-format first: WASAPI shared needs the mixer rate; we resample ourselves
        sr_native = int(dev["default_sr"]) if dev else 48000
        attempts.append((sr_native, min(2, dev["channels"]) if dev else 1))
        attempts.append((sr_native, 1))
        attempts.append((TARGET_SR, 1))
        last_err = None
        for sr, ch in attempts:
            try:
                extra = None
                if dev and dev["hostapi"] == "Windows WASAPI":
                    extra = sd.WasapiSettings(auto_convert=True)
                block = int(sr * 0.02)
                stream = sd.InputStream(device=idx, samplerate=sr, channels=ch, dtype="float32",
                                        blocksize=block, callback=self._callback, extra_settings=extra,
                                        latency="low")
                stream.start()
                with self._lock:
                    self._stream, self._sr, self._ch = stream, sr, ch
                    self.device, self.device_name = dev, name
                    self._preroll.clear()
                    self._preroll_samples = 0
                    self._last_cb = time.monotonic()
                log.info("microphone: %s [%s] %d Hz x%d ch (%s)", name,
                         dev["hostapi"] if dev else "default", sr, ch, why)
                return True
            except Exception as e:  # try the next config
                last_err = e
        log.error("could not open microphone '%s': %s", name, last_err)
        return False

    def _close(self) -> None:
        with self._lock:
            s, self._stream = self._stream, None
        if s is not None:
            try:
                s.abort(ignore_errors=True)
                s.close(ignore_errors=True)
            except Exception:
                pass

    def start(self) -> bool:
        ok = self._refresh_and_open() if self.keep_open else True
        if self._monitor is None:
            self._stop_monitor.clear()
            self._monitor = threading.Thread(target=self._watch, name="audio-watch", daemon=True)
            self._monitor.start()
        return ok

    def stop(self) -> None:
        self._stop_monitor.set()
        self._close()

    def reopen(self) -> None:
        """Re-pick the device (config change / hot-plug)."""
        with self._lock:
            recording = self._rec is not None
        if recording:
            return
        self._close()
        if self.keep_open:
            self._refresh_and_open()
        else:
            self._default_id = devwatch.default_capture_id()
            self._reinit_portaudio()

    def _refresh_and_open(self) -> bool:
        """Re-enumerate devices, then open the stream. The default endpoint id is read BEFORE the re-init so a
        default change that lands in between is noticed by the next poll instead of being missed."""
        self._default_id = devwatch.default_capture_id()
        self._reinit_portaudio()
        return self._open()

    @staticmethod
    def _reinit_portaudio() -> None:
        """Refresh PortAudio's cached device list (it only enumerates at init)."""
        try:
            sd._terminate()
            sd._initialize()
        except Exception:
            log.exception("PortAudio re-init failed")

    def _watch(self) -> None:
        """Detect disconnects (dead stream) and replug of the preferred mic."""
        while not self._stop_monitor.wait(1.5):
            try:
                if not self.keep_open:
                    continue
                with self._lock:
                    recording = self._rec is not None
                    s = self._stream
                if recording:
                    continue        # device changes wait until the dictation is over
                now = time.monotonic()
                if (not self.preferred_found) and now - self._last_default_check >= DEFAULT_POLL_S:
                    self._last_default_check = now
                    if self._default_changed():
                        self.reopen()
                        self._last_enum = now
                        if self.on_device_change:
                            self.on_device_change(self.device_name)
                        continue
                dead = s is None or not s.active or (now - self._last_cb > 2.0)
                want_scan = (not self.preferred_found) and (now - self._last_enum > 4.0)
                if dead:
                    log.warning("microphone stream lost - re-opening")
                    self.reopen()
                    self._last_enum = now
                elif want_scan:
                    self._last_enum = now
                    self._reinit_portaudio_if_preferred_appeared()
            except Exception:
                log.exception("audio watchdog error")

    def _default_changed(self) -> bool:
        """True when Windows' default recording device is no longer the one we opened (changed in Sound
        settings, unplugged, replugged). A failed lookup (None) never counts as a change."""
        cur = devwatch.default_capture_id()
        if cur is None or cur == self._default_id:
            return False
        log.info("default recording device changed - switching")
        return True

    def _reinit_portaudio_if_preferred_appeared(self) -> None:
        # Only refresh when we're on a fallback device and a configured mic might have appeared.
        # (re-init briefly interrupts the stream, so it is rate limited and skipped while recording.)
        match = self.cfg.get("mic.match", "")
        exact = self.cfg.get("mic.device", "")
        if not (match or exact):
            return
        with self._lock:
            if self._rec is not None:
                return
        self._close()
        self._refresh_and_open()
        if self.preferred_found:
            log.info("preferred microphone is back")

    # --- audio callback
    def _callback(self, indata, frames, time_info, status) -> None:
        self._last_cb = time.monotonic()
        x = indata[:, 0] if indata.shape[1] == 1 else indata.mean(axis=1)
        chunk = np.array(x, dtype=np.float32)  # copy: indata is reused by PortAudio
        if self.on_level is not None:
            try:
                self.on_level(float(np.sqrt(np.mean(chunk * chunk))) if chunk.size else 0.0)
            except Exception:
                pass
        with self._lock:
            if self._rec is not None:
                self._rec.append(chunk)
                self._rec_samples += chunk.size
                if self._rec_samples > self._sr * float(self.cfg.get("audio.max_duration_s", 300)):
                    self.max_samples_hit = True
            else:
                self._preroll.append(chunk)
                self._preroll_samples += chunk.size
                limit = int(self._sr * float(self.cfg.get("audio.preroll_ms", 400)) / 1000)
                while self._preroll_samples - self._preroll[0].size >= limit and len(self._preroll) > 1:
                    self._preroll_samples -= self._preroll.popleft().size

    # --- live access for streaming transcription
    @property
    def sample_rate(self) -> int:
        """Native sample rate of the open stream (what pull_chunks returns audio at)."""
        return int(self._sr)

    def pull_chunks(self, since: int) -> tuple[Optional[list], int]:
        """Chunks recorded since chunk index ``since`` (native rate, mono float32) and the new index.
        Returns (None, since) once the recording has ended. Cheap: copies list references only."""
        with self._lock:
            if self._rec is None:
                return None, since
            return self._rec[since:], len(self._rec)

    # --- recording control
    def begin(self) -> None:
        if not self.keep_open and self._stream is None:
            # no watcher in this mode, so check for a changed default device here before opening
            if not self.preferred_found and self._default_changed():
                self._refresh_and_open()
            else:
                self._open()
        with self._lock:
            self.max_samples_hit = False
            self._rec = list(self._preroll) if self.keep_open else []
            self._rec_samples = sum(c.size for c in self._rec)
            self._preroll.clear()
            self._preroll_samples = 0

    def end(self) -> np.ndarray:
        """Finish recording and return 16 kHz mono float32."""
        test_wav = os.environ.get("LOCALFLOW_TEST_WAV")   # test hook: substitute a file for the mic audio
        if test_wav and os.path.exists(test_wav):
            return self._end_test(test_wav)
        with self._lock:
            chunks, self._rec, self._rec_samples = self._rec or [], None, 0
            sr = self._sr
        if not self.keep_open:
            self._close()
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return resample_to_16k(np.concatenate(chunks), sr)

    def _end_test(self, path: str) -> np.ndarray:
        import wave
        with self._lock:
            self._rec, self._rec_samples = None, 0
        with wave.open(path, "rb") as w:
            x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
            sr = w.getframerate()
        x = np.concatenate([np.zeros(int(sr * 0.15), np.float32), x, np.zeros(int(sr * 0.2), np.float32)])
        return resample_to_16k(x, sr)

    def cancel(self) -> None:
        with self._lock:
            self._rec, self._rec_samples = None, 0
        if not self.keep_open:
            self._close()

    @property
    def recording(self) -> bool:
        return self._rec is not None
