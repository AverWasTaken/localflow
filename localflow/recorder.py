"""Meeting recorder: your microphone ("You") + WASAPI loopback of a playback device ("Others").

All three streams (mic, loopback, and a SILENT output stream on the same playback device) are opened with
PyAudioWPatch, which bundles its own PortAudio. That matters: the dictation mic (audio.py, sounddevice) re-inits
sounddevice's PortAudio (``sd._terminate()``) on device changes, which kills every sounddevice stream in the
process - PyAudioWPatch streams are unaffected. The silent output stream keeps the playback endpoint rendering,
because WASAPI loopback delivers no data at all while nothing plays; with it both tracks run continuously and
stay aligned. Wall-clock zero-fill remains as a fallback (gaps > 0.5 s are padded with silence).

Each track is written as raw int16 mono PCM at the device's native rate (``mic.pcm`` / ``system.pcm``) from a
writer thread fed by a bounded queue; ``meta.json`` (atomic replace) describes rates, devices, start offsets and
the state (recording | stopped | transcribing | done | interrupted | failed). On stop the PCM is converted to
FLAC; the PCM is only deleted after the FLAC was read back and verified.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import threading
import time
import zlib
from pathlib import Path
from typing import Callable, Optional

import numpy as np

log = logging.getLogger(__name__)

TRACKS = ("mic", "system")          # mic = "You", system = "Others"
SPEAKERS = {"mic": "You", "system": "Others"}
ZERO_FILL_S = 0.5                   # pad a track with silence when it falls this far behind the wall clock
QUEUE_BLOCKS = 3000                 # ~60 s of 20 ms blocks per track before blocks are dropped (and zero-filled)
FLUSH_S = 1.0
META_EVERY_S = 5.0
STALL_S = 3.0                       # no callbacks for this long -> try to re-open that stream


class RecorderError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- small helpers
def atomic_write_json(path: Path, data: dict) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_meta(rec_dir: Path) -> Optional[dict]:
    try:
        with open(Path(rec_dir) / "meta.json", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write_meta(rec_dir: Path, meta: dict) -> None:
    atomic_write_json(Path(rec_dir) / "meta.json", meta)


def set_state(rec_dir: Path, state: str, **extra) -> Optional[dict]:
    """Update meta.json's state (the job runner mirrors transcribing/done/failed here)."""
    meta = read_meta(rec_dir)
    if meta is None:
        return None
    meta["state"] = state
    meta.update(extra)
    write_meta(rec_dir, meta)
    return meta


def _downmix(a: np.ndarray, ch: int) -> np.ndarray:
    if ch <= 1:
        return a
    n = a.size // ch
    return (a[: n * ch].reshape(n, ch).astype(np.int32).sum(axis=1) // ch).astype(np.int16)


def _rms(a: np.ndarray) -> float:
    if a.size == 0:
        return 0.0
    x = a.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(x * x)))


# ----------------------------------------------------------------------------- FLAC conversion + recovery
def pcm_to_flac(pcm_path: Path, flac_path: Path, rate: int) -> int:
    """Convert raw int16 mono PCM to FLAC, verify the FLAC (frame count + CRC of every sample), then delete the
    PCM. Returns the sample count. On any failure the PCM is kept and the exception propagates."""
    import soundfile as sf
    pcm_path, flac_path = Path(pcm_path), Path(flac_path)
    tmp = flac_path.with_name(flac_path.name + ".tmp")
    size = os.path.getsize(pcm_path)
    n = size // 2
    block = max(1, int(rate)) * 60
    crc_in = 0
    with open(pcm_path, "rb") as fh, sf.SoundFile(str(tmp), "w", samplerate=int(rate), channels=1,
                                                  subtype="PCM_16", format="FLAC") as out:
        left = n
        while left > 0:
            take = min(block, left)
            buf = fh.read(take * 2)
            a = np.frombuffer(buf, dtype=np.int16)
            if a.size == 0:
                break
            crc_in = zlib.crc32(a.tobytes(), crc_in)
            out.write(a)
            left -= a.size
    crc_out = 0
    frames = 0
    with sf.SoundFile(str(tmp), "r") as f:
        if f.frames != n:
            raise RecorderError(f"FLAC verify failed: {f.frames} frames != {n}")
        for b in f.blocks(blocksize=block, dtype="int16"):
            b = np.ascontiguousarray(b).reshape(-1)
            frames += b.size
            crc_out = zlib.crc32(b.tobytes(), crc_out)
    if frames != n or crc_in != crc_out:
        raise RecorderError("FLAC verify failed: content mismatch")
    os.replace(tmp, flac_path)
    os.remove(pcm_path)
    return n


def finalize_dir(rec_dir: Path, state: Optional[str] = None) -> dict:
    """Convert every track still stored as PCM to FLAC (idempotent) and optionally set the state."""
    rec_dir = Path(rec_dir)
    meta = read_meta(rec_dir)
    if meta is None:
        raise RecorderError(f"no meta.json in {rec_dir}")
    for key, tr in (meta.get("tracks") or {}).items():
        pcm = rec_dir / f"{key}.pcm"
        flac = rec_dir / f"{key}.flac"
        if pcm.exists():
            rate = int(tr.get("rate") or 48000)
            try:
                if pcm.stat().st_size < 2:
                    pcm.unlink()
                    tr.update(file="", samples=0)
                    continue
                tr["samples"] = pcm_to_flac(pcm, flac, rate)
                tr["file"] = flac.name
                tr.pop("flac_error", None)
            except Exception as e:
                log.exception("FLAC conversion failed for %s", pcm)
                tr["file"] = pcm.name
                tr["samples"] = pcm.stat().st_size // 2 if pcm.exists() else 0
                tr["flac_error"] = f"{type(e).__name__}: {e}"
        elif flac.exists() and tr.get("file") != flac.name:
            tr["file"] = flac.name
    if state:
        meta["state"] = state
    rates = [(t.get("samples") or 0) / max(1, int(t.get("rate") or 1)) + float(t.get("offset_s") or 0)
             for t in (meta.get("tracks") or {}).values()]
    if rates:
        meta["duration_s"] = round(max(meta.get("duration_s") or 0, max(rates)), 3)
    write_meta(rec_dir, meta)
    return meta


def _pid_alive(pid: int) -> bool:
    if not pid or pid == os.getpid():
        return False
    try:
        import psutil
        return psutil.pid_exists(int(pid))
    except Exception:
        return False


def recover_interrupted(root: Path, skip=None) -> list[tuple[Path, dict]]:
    """Startup recovery. Recording dirs left in ``recording``/``stopping``/``transcribing`` (the app died) are
    converted to FLAC and marked ``interrupted``; stopped dirs that still hold PCM are converted. Returns the
    (dir, meta) pairs that were marked interrupted. Dirs whose writer process is still alive are left alone."""
    root = Path(root)
    out: list[tuple[Path, dict]] = []
    if not root.exists():
        return out
    if skip is None:
        skips = set()
    elif isinstance(skip, (str, Path)):
        skips = {Path(skip).resolve()}
    else:
        skips = {Path(x).resolve() for x in skip if x}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if d.resolve() in skips or (d / ".deleted").exists():     # live / in use / deleted by the user
            continue
        meta = read_meta(d)
        if meta is None:
            continue
        state = meta.get("state")
        if state in ("recording", "stopping") and _pid_alive(int(meta.get("pid") or 0)):
            log.info("recording %s still owned by live pid %s - skipped", d, meta.get("pid"))
            continue
        has_pcm = any((d / f"{k}.pcm").exists() for k in TRACKS)
        try:
            if state in ("recording", "stopping", "transcribing"):
                meta = finalize_dir(d, state="interrupted")
                out.append((d, meta))
                log.warning("recovered interrupted recording %s (was %s)", d, state)
            elif has_pcm:
                finalize_dir(d)
        except Exception:
            log.exception("recovery failed for %s", d)
    return out


# ----------------------------------------------------------------------------- device lookup (PyAudioWPatch)
def _pa():
    import pyaudiowpatch as pa
    return pa


def _wasapi_devices(p) -> tuple[dict, list[dict]]:
    pa = _pa()
    host = p.get_host_api_info_by_type(pa.paWASAPI)
    devs = []
    for i in range(p.get_device_count()):
        d = p.get_device_info_by_index(i)
        if d.get("hostApi") == host["index"]:
            devs.append(d)
    return host, devs


def _pick(devs: list[dict], name: str, match: str = "") -> Optional[dict]:
    if name:
        c = [d for d in devs if d["name"].lower() == name.lower()] or \
            [d for d in devs if name.lower() in d["name"].lower()]
        if c:
            return c[0]
        log.warning("meeting device '%s' not found - using the Windows default", name)
    if match:
        c = [d for d in devs if match.lower() in d["name"].lower()]
        if c:
            return c[0]
    return None


def list_devices() -> dict:
    """WASAPI device names for the Settings pickers: {"inputs": [...], "outputs": [...], "default_input": ..,
    "default_output": ..}. Opens and closes its own PyAudio instance."""
    pa = _pa()
    p = pa.PyAudio()
    try:
        host, devs = _wasapi_devices(p)
        ins = sorted({d["name"] for d in devs if d["maxInputChannels"] > 0 and not d.get("isLoopbackDevice")})
        outs = sorted({d["name"] for d in devs if d["maxOutputChannels"] > 0 and not d.get("isLoopbackDevice")})
        di = host.get("defaultInputDevice", -1)
        do = host.get("defaultOutputDevice", -1)
        return {"inputs": ins, "outputs": outs,
                "default_input": p.get_device_info_by_index(di)["name"] if di is not None and di >= 0 else "",
                "default_output": p.get_device_info_by_index(do)["name"] if do is not None and do >= 0 else ""}
    finally:
        p.terminate()


# ----------------------------------------------------------------------------- recorder
class _Track:
    def __init__(self, key: str, path: Path):
        self.key = key
        self.path = path
        self.fh = None
        self.rate = 0
        self.channels = 1
        self.device = ""
        self.stream = None
        self.first_mono: Optional[float] = None
        self.last_cb = 0.0
        self.written = 0
        self.dropped_pending = 0
        self.dropped = 0
        self.zero_filled = 0
        self.level = 0.0
        self.reopens = 0
        self.resync = False     # after a re-open: snap to the wall clock even for gaps < ZERO_FILL_S
        self.error = ""

    def info(self, t0_mono: float) -> dict:
        return {"file": self.path.name, "rate": self.rate, "device": self.device,
                "offset_s": round((self.first_mono - t0_mono) if self.first_mono is not None else 0.0, 4),
                "samples": self.written, "dropped": self.dropped, "zero_filled": self.zero_filled,
                "reopens": self.reopens, "no_data": self.first_mono is None, "error": self.error}


class MeetingRecorder:
    """``cfg`` is anything with ``.get(dotted, default)`` (the app Config). Keys used: ``meetings.mic`` ("" = the
    dictation mic: ``mic.device`` / ``mic.match`` / Windows default), ``meetings.output_device`` ("" = Windows
    default playback device), ``meetings.max_hours`` (4), ``meetings.min_free_gb`` (1.0).

    ``on_level(mic_rms, system_rms)`` ~10x/s from the writer thread. ``on_auto_stop(meta, reason)`` when the
    recording stopped by itself (max length, low disk space, write error)."""

    def __init__(self, cfg, on_level: Optional[Callable[[float, float], None]] = None,
                 on_auto_stop: Optional[Callable[[dict, str], None]] = None):
        self.cfg = cfg
        self.on_level = on_level
        self.on_auto_stop = on_auto_stop
        self._lock = threading.RLock()
        self._q: "queue.Queue" = queue.Queue(maxsize=QUEUE_BLOCKS * 2)
        self._tracks: dict[str, _Track] = {}
        self._p = None
        self._silent = None
        self._silent_ch = 2
        self._writer: Optional[threading.Thread] = None
        self._watchdog: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()      # stops the watchdog
        self._drain = threading.Event()         # writer: finish what is queued, then exit
        self.rec_dir: Optional[Path] = None
        self.meta: dict = {}
        self.t0 = 0.0
        self.t0_mono = 0.0
        self._t_stop: Optional[float] = None
        self.recording = False
        self.error = ""
        self.stop_reason = ""
        self._out_dev: Optional[dict] = None
        self._mic_dev: Optional[dict] = None
        self._last_reopen = 0.0
        self._stuck_writer: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ public state
    @property
    def elapsed(self) -> float:
        if not self.t0_mono:
            return 0.0
        end = self._t_stop if self._t_stop is not None else time.monotonic()
        return max(0.0, end - self.t0_mono)

    @property
    def levels(self) -> tuple[float, float]:
        m = self._tracks.get("mic")
        s = self._tracks.get("system")
        return (m.level if m else 0.0, s.level if s else 0.0)

    def devices(self) -> dict:
        return {k: t.device for k, t in self._tracks.items()}

    # ------------------------------------------------------------------ config
    def _cfg(self, key: str, default):
        try:
            v = self.cfg.get(key, default)
        except Exception:
            v = default
        return default if v is None else v

    def _max_s(self) -> float:
        return float(self._cfg("meetings.max_hours", 4)) * 3600.0

    def _min_free(self) -> float:
        return float(self._cfg("meetings.min_free_gb", 1.0)) * 1024 ** 3

    # ------------------------------------------------------------------ start / stop
    def start(self, rec_dir: Path) -> dict:
        with self._lock:
            if self.recording:
                raise RecorderError("already recording")
            if self._stuck_writer is not None and self._stuck_writer.is_alive():
                raise RecorderError("The previous recording is still being saved. Try again in a moment.")
            self._stuck_writer = None
            rec_dir = Path(rec_dir)
            rec_dir.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(rec_dir).free
            if free < self._min_free():
                raise RecorderError(f"Not enough free disk space to record ({free / 1024 ** 3:.1f} GB free, "
                                    f"need {self._min_free() / 1024 ** 3:.1f} GB).")
            self.rec_dir = rec_dir
            self.error = ""
            self.stop_reason = ""
            self._t_stop = None
            self._stop_evt.clear()
            self._drain.clear()
            self._q = queue.Queue(maxsize=QUEUE_BLOCKS * 2)
            self._tracks = {k: _Track(k, rec_dir / f"{k}.pcm") for k in TRACKS}
            for tr in self._tracks.values():
                tr.fh = open(tr.path, "ab")
            self.t0 = time.time()
            self.t0_mono = time.monotonic()
            try:
                self._open_streams()
            except Exception as e:
                self._close_streams()
                for tr in self._tracks.values():
                    try:
                        tr.fh.close()
                    except Exception:
                        pass
                shutil.rmtree(rec_dir, ignore_errors=True)
                self.t0_mono = 0.0
                raise RecorderError(f"Could not start recording: {e}") from e
            self.recording = True
            self.meta = {
                "version": 1, "kind": "meeting", "state": "recording", "pid": os.getpid(), "dir": str(rec_dir),
                "t0": self.t0, "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.t0)),
                "max_s": self._max_s(), "speakers": dict(SPEAKERS),
                "tracks": {k: t.info(self.t0_mono) for k, t in self._tracks.items()},
            }
            write_meta(rec_dir, self.meta)
            self._writer = threading.Thread(target=self._writer_loop, name="meeting-writer", daemon=True)
            self._writer.start()
            self._watchdog = threading.Thread(target=self._watchdog_loop, name="meeting-watchdog", daemon=True)
            self._watchdog.start()
            log.info("meeting recording started in %s (mic=%s, output=%s)", rec_dir,
                     self._tracks["mic"].device, self._tracks["system"].device)
            return dict(self.meta)

    def stop(self, finalize: bool = True, reason: str = "", expect_dir: Optional[Path] = None) -> dict:
        """Stop recording. With ``finalize`` the PCM is converted to FLAC here (can take a few seconds for long
        recordings); otherwise the caller converts later with ``finalize_dir`` (state is ``stopped`` either
        way; a leftover PCM is converted by recovery too).

        The returned meta has ``stopped_now`` = True only for the call that actually stopped the recording, and
        ``dir`` = that recording's directory. With ``expect_dir`` nothing happens unless that exact recording is
        still running (an auto-stop never hits a newer recording)."""
        with self._lock:
            if not self.recording or (expect_dir is not None and self.rec_dir != Path(expect_dir)):
                out = dict(self.meta)
                out["stopped_now"] = False
                return out
            self.recording = False
            self.stop_reason = reason
            self._t_stop = time.monotonic()
            self._stop_evt.set()
            self._close_streams()                # no more callbacks after this
            self._drain.set()
            writer_ok = True
            if self._writer is not None and self._writer is not threading.current_thread():
                self._writer.join(timeout=30)
                writer_ok = not self._writer.is_alive()
            if self._watchdog is not None and self._watchdog is not threading.current_thread():
                self._watchdog.join(timeout=5)
            rec_dir = self.rec_dir
            end = self._t_stop
            if not writer_ok:
                # never touch files a live writer still owns: meta.json stays "recording", nothing is queued, the
                # recorder refuses to start again until that writer is gone, and the next start's recovery
                # converts whatever reached the disk
                log.error("meeting writer did not finish - leaving the recording to startup recovery")
                self.error = self.error or "writer did not finish"
                self._stuck_writer = self._writer
                out = dict(self.meta)
                out.update(dir=str(rec_dir), stopped_now=True, writer_busy=True)
                return out
            for tr in self._tracks.values():
                try:
                    if tr.first_mono is not None:
                        self._pad_to(tr, end)
                    tr.fh.flush()
                    tr.fh.close()
                except Exception:
                    log.exception("closing track %s", tr.key)
            self.meta["tracks"] = {k: t.info(self.t0_mono) for k, t in self._tracks.items()}
            self.meta["duration_s"] = round(end - self.t0_mono, 3)
            self.meta["state"] = "stopped"
            self.meta["stop_reason"] = reason
            if self.error:
                self.meta["error"] = self.error
            write_meta(rec_dir, self.meta)
            out = dict(self.meta)
            log.info("meeting recording stopped (%.1f s%s)", self.meta["duration_s"],
                     f", {reason}" if reason else "")
        if finalize:
            try:
                out = finalize_dir(rec_dir)
                with self._lock:
                    if self.rec_dir == rec_dir:
                        self.meta = dict(out)
            except Exception:
                log.exception("finalize failed")
        out["dir"] = str(rec_dir)
        out["stopped_now"] = True
        return out

    # ------------------------------------------------------------------ streams
    def _open_streams(self) -> None:
        """Open the silent output, the loopback and the mic streams (overridden by tests)."""
        pa = _pa()
        self._p = pa.PyAudio()
        host, devs = _wasapi_devices(self._p)
        outs = [d for d in devs if d["maxOutputChannels"] > 0 and not d.get("isLoopbackDevice")]
        ins = [d for d in devs if d["maxInputChannels"] > 0 and not d.get("isLoopbackDevice")]
        out_name = str(self._cfg("meetings.output_device", "") or "")
        out = _pick(outs, out_name)
        if out is None:
            idx = host.get("defaultOutputDevice", -1)
            if idx is None or idx < 0:
                raise RecorderError("no playback device found")
            out = self._p.get_device_info_by_index(idx)
        mic_name = str(self._cfg("meetings.mic", "") or "")
        mic_match = ""
        if not mic_name:
            mic_name = str(self._cfg("mic.device", "") or "")
            mic_match = str(self._cfg("mic.match", "") or "")
        mic = _pick(ins, mic_name, mic_match)
        if mic is None:
            idx = host.get("defaultInputDevice", -1)
            if idx is None or idx < 0:
                raise RecorderError("no microphone found")
            mic = self._p.get_device_info_by_index(idx)
        self._out_dev, self._mic_dev = out, mic
        self._open_silent(out)
        self._open_loopback(out)
        self._open_mic(mic)

    def _open_silent(self, out: dict) -> None:
        pa = _pa()
        rate = int(out["defaultSampleRate"])
        ch = max(1, min(2, int(out["maxOutputChannels"])))
        self._silent_ch = ch

        def cb(data, frames, ti, status):
            return (bytes(2 * ch * frames), pa.paContinue)
        try:
            self._silent = self._p.open(format=pa.paInt16, channels=ch, rate=rate, output=True,
                                        output_device_index=out["index"], frames_per_buffer=int(rate * 0.02),
                                        stream_callback=cb)
        except Exception as e:
            # not fatal: loopback still works whenever something plays; gaps are zero-filled
            self._silent = None
            log.warning("silent output stream failed on %s (%s) - relying on zero-fill", out["name"], e)

    def _open_loopback(self, out: dict) -> None:
        pa = _pa()
        lb = self._p.get_wasapi_loopback_analogue_by_dict(out)
        tr = self._tracks["system"]
        rate = int(lb["defaultSampleRate"])
        ch = int(lb["maxInputChannels"]) or 2
        if tr.rate and tr.rate != rate:
            raise RecorderError(f"loopback rate changed ({tr.rate} -> {rate})")
        tr.rate, tr.channels, tr.device = rate, ch, out["name"]
        tr.stream = self._p.open(format=pa.paInt16, channels=ch, rate=rate, input=True,
                                 input_device_index=lb["index"], frames_per_buffer=int(rate * 0.02),
                                 stream_callback=self._make_cb(tr))
        tr.last_cb = time.monotonic()

    def _open_mic(self, mic: dict) -> None:
        pa = _pa()
        tr = self._tracks["mic"]
        rate = tr.rate or int(mic["defaultSampleRate"])
        last = None
        for ch in (1, max(1, min(2, int(mic["maxInputChannels"])))):
            try:
                stream = self._p.open(format=pa.paInt16, channels=ch, rate=rate, input=True,
                                      input_device_index=mic["index"], frames_per_buffer=int(rate * 0.02),
                                      stream_callback=self._make_cb(tr))
                tr.rate, tr.channels, tr.device, tr.stream = rate, ch, mic["name"], stream
                tr.last_cb = time.monotonic()
                return
            except Exception as e:
                last = e
        raise RecorderError(f"could not open microphone '{mic['name']}': {last}")

    def _make_cb(self, tr: _Track):
        pa = _pa()

        def cb(data, frames, ti, status):
            self._on_input(tr, data, frames)
            return (None, pa.paContinue)
        return cb

    def _on_input(self, tr: _Track, data: bytes, frames: int) -> None:
        """Audio callback body (PortAudio thread): downmix, level, enqueue. Never blocks."""
        now = time.monotonic()
        tr.last_cb = now
        a = _downmix(np.frombuffer(data, dtype=np.int16), tr.channels)
        tr.level = _rms(a)
        try:
            self._q.put_nowait((tr.key, now, a))
        except queue.Full:
            tr.dropped_pending += a.size

    def _close_streams(self) -> None:
        for tr in list(self._tracks.values()):
            s, tr.stream = tr.stream, None
            if s is not None:
                try:
                    s.stop_stream()
                    s.close()
                except Exception:
                    log.debug("closing %s stream", tr.key, exc_info=True)
        s, self._silent = self._silent, None
        if s is not None:
            try:
                s.stop_stream()
                s.close()
            except Exception:
                pass
        p, self._p = self._p, None
        if p is not None:
            try:
                p.terminate()
            except Exception:
                pass

    def _reopen_all(self, why: str) -> None:
        """A stream stopped delivering (device lost / endpoint reset). PortAudio only enumerates devices at init,
        so close everything, re-init our own PyAudio and re-open all streams (devices re-picked by name; the
        gap is zero-filled by the writer). Rate limited."""
        now = time.monotonic()
        if now - self._last_reopen < 5.0:
            return
        self._last_reopen = now
        log.warning("meeting stream stalled (%s) - re-opening all streams", why)
        with self._lock:
            if not self.recording:
                return
            self._close_streams()
            try:
                self._open_streams()
                for tr in self._tracks.values():
                    tr.error = ""
            except Exception as e:
                log.warning("meeting stream re-open failed: %s", e)
                for tr in self._tracks.values():
                    if tr.stream is None:
                        tr.reopens += 1
                        tr.error = f"{type(e).__name__}: {e}"
                        tr.last_cb = now    # wait another STALL_S before the next attempt
                return
            for tr in self._tracks.values():
                tr.reopens += 1
                tr.resync = True

    # ------------------------------------------------------------------ writer
    def _write(self, tr: _Track, a: np.ndarray) -> None:
        tr.fh.write(a.tobytes())
        tr.written += a.size

    def _zeros(self, tr: _Track, n: int) -> None:
        while n > 0:
            k = min(n, tr.rate * 10 or n)
            self._write(tr, np.zeros(k, dtype=np.int16))
            n -= k

    def _pad_to(self, tr: _Track, t_mono: float) -> None:
        """Zero-fill ``tr`` up to the wall-clock position ``t_mono`` if it is more than ZERO_FILL_S behind."""
        if tr.first_mono is None or not tr.rate:
            return
        behind = int((t_mono - tr.first_mono) * tr.rate) - tr.written
        if behind > tr.rate * ZERO_FILL_S:
            self._zeros(tr, behind)
            tr.zero_filled += behind

    def _writer_loop(self) -> None:
        last_flush = last_meta = last_level = time.monotonic()
        try:
            while True:
                try:
                    item = self._q.get(timeout=0.1)
                except queue.Empty:
                    item = None
                    if self._drain.is_set():      # streams are closed and the queue is empty: done
                        break
                if item is not None:
                    self._handle(*item)
                now = time.monotonic()
                if now - last_level >= 0.1:
                    last_level = now
                    if self.on_level:
                        try:
                            self.on_level(*self.levels)
                        except Exception:
                            log.debug("on_level failed", exc_info=True)
                if now - last_flush >= FLUSH_S:
                    last_flush = now
                    for tr in self._tracks.values():
                        tr.fh.flush()
                if now - last_meta >= META_EVERY_S:
                    last_meta = now
                    self.meta["tracks"] = {k: t.info(self.t0_mono) for k, t in self._tracks.items()}
                    self.meta["duration_s"] = round(now - self.t0_mono, 3)
                    try:
                        write_meta(self.rec_dir, self.meta)
                    except Exception:
                        log.debug("meta write failed", exc_info=True)
        except Exception as e:
            log.exception("meeting writer failed")
            self.error = f"write error: {type(e).__name__}: {e}"
            self._auto_stop(self.error)

    def _handle(self, key: str, t_cb: float, a: np.ndarray) -> None:
        tr = self._tracks[key]
        if tr.first_mono is None:
            tr.first_mono = t_cb - a.size / float(tr.rate or 48000)   # time of this block's first sample
        if tr.dropped_pending:
            n, tr.dropped_pending = tr.dropped_pending, 0
            tr.dropped += n
            self._zeros(tr, n)
        # fallback alignment: a stream that delivered nothing for a while (loopback starved, device hiccup)
        # gets the gap filled with silence so it stays on the wall clock
        behind = int((t_cb - tr.first_mono) * tr.rate) - (tr.written + a.size)
        limit = tr.rate * (0.03 if tr.resync else ZERO_FILL_S)
        tr.resync = False
        if behind > limit:
            self._zeros(tr, behind)
            tr.zero_filled += behind
        self._write(tr, a)

    # ------------------------------------------------------------------ watchdog
    def _watchdog_loop(self) -> None:
        last_disk = 0.0
        while not self._stop_evt.wait(0.5):
            try:
                now = time.monotonic()
                if now - self.t0_mono >= self._max_s():
                    self._auto_stop("maximum recording length reached")
                    return
                if now - last_disk >= 10.0:
                    last_disk = now
                    if shutil.disk_usage(self.rec_dir).free < self._min_free():
                        self.error = "low disk space"
                        self._auto_stop("low disk space - recording stopped")
                        return
                for tr in self._tracks.values():
                    stalled = now - tr.last_cb > STALL_S
                    # without the silent output stream an idle loopback is expected to be quiet
                    if stalled and (tr.key == "mic" or self._silent is not None):
                        self._reopen_all(f"{tr.key}: no audio for {now - tr.last_cb:.1f} s")
                        break
            except Exception:
                log.exception("meeting watchdog error")

    def _auto_stop(self, reason: str) -> None:
        d = self.rec_dir

        def run():
            meta = self.stop(finalize=False, reason=reason, expect_dir=d)
            if not meta.get("stopped_now"):
                return                      # the user stopped it first (or it is a different recording now)
            if self.on_auto_stop:
                try:
                    self.on_auto_stop(meta, reason)
                except Exception:
                    log.exception("on_auto_stop failed")
        threading.Thread(target=run, name="meeting-autostop", daemon=True).start()
