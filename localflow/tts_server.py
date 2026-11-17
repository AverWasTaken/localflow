"""LocalFlow text-to-speech server (spoken command-mode answers).

Runs in its own Python environment (.venv-tts: torch + faster-qwen3-tts), started by localflow/tts.py as
    .venv-tts\\Scripts\\python.exe tts_server.py --model <dir> --log <file>
It never imports the localflow package (the app's .venv has no torch, this venv has no PySide6).

Protocol: one JSON object per line.
  stdin  (app -> server): {"cmd": "speak", "id": 7, "text": "...", "voice": "ryan", "device": "", "volume": 1.0}
                          {"cmd": "stop"}            stop speaking right away (generation and playback)
                          {"cmd": "quit"}            exit; EOF on stdin does the same
  events (server -> app): {"ev": "ready", "voices": [...], "sr": 24000, "load_s": .., "warmup_s": ..}
                          {"ev": "started", "id": 7}            generation began
                          {"ev": "playing", "id": 7, "ttfa_ms": 350}   first audio handed to the speakers
                          {"ev": "done", "id": 7, "audio_s": 3.2, "reason": "end" | "stopped" | "cap" | "error"}
                          {"ev": "error", "id": 7 | null, "msg": "..."}
The model libraries print to stdout on import, so the event channel is a duplicate of the original stdout handle
and fd 1 / fd 2 are pointed at the log file before anything heavy is imported.

Playback happens here, not in the app: the app's microphone code re-initialises PortAudio when devices change,
which would kill an output stream living in the app process.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# the script's folder (localflow/) is on sys.path when run as a script: its modules (audio, config, streaming,
# ...) must never shadow third-party ones
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]

import argparse  # noqa: E402
import collections  # noqa: E402
import json  # noqa: E402
import queue  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import wave  # noqa: E402

import numpy as np  # noqa: E402

SR = 24000
FRAME_HZ = 12.5            # codec frames per second of audio (Qwen3-TTS 12 Hz tokenizer)
MAX_AUDIO_S = 30.0         # runaway guard: never more than this much speech per answer
MAX_GEN_S = 60.0           # and never spend longer than this generating one answer
CHUNK_FRAMES = 8           # frames per streamed chunk (~0.64 s of audio; first audio ~0.35 s on an RTX 5060 Ti)
MAX_SEQ_LEN = 1024         # static KV cache: text prompt (<= ~200 tokens after the app's trim) + 375 codec frames
LANGUAGE = "english"
DEFAULT_VOICE = "ryan"


def max_new_tokens(cap_s: float = MAX_AUDIO_S) -> int:
    return int(cap_s * FRAME_HZ)


# ---------------------------------------------------------------------------------------------- event channel
class Events:
    """Thread-safe JSON-lines writer on the duplicated stdout handle."""

    def __init__(self, stream):
        self._s = stream
        self._lock = threading.Lock()

    def emit(self, ev: str, **kw) -> None:
        line = json.dumps({"ev": ev, **kw}, ensure_ascii=True)
        with self._lock:
            try:
                self._s.write(line + "\n")
                self._s.flush()
            except (OSError, ValueError):
                pass        # app gone: the stdin EOF will end us


def setup_channels(log_path: str):
    """Duplicate fd 1 for events, then send fd 1 and fd 2 (and sys.stdout / sys.stderr) to the log file."""
    evt_fd = os.dup(1)
    evt = os.fdopen(evt_fd, "w", encoding="utf-8", buffering=1, newline="\n")
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    logf = open(log_path, "a", encoding="utf-8", buffering=1, errors="replace")
    os.dup2(logf.fileno(), 1)
    os.dup2(logf.fileno(), 2)
    sys.stdout = logf
    sys.stderr = logf
    return Events(evt), logf


def log(msg: str) -> None:
    try:
        sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")
        sys.stderr.flush()
    except Exception:
        pass


def parse_command(line: str):
    """One stdin line -> dict with a "cmd", or None for blank / malformed lines."""
    line = (line or "").strip()
    if not line:
        return None
    try:
        d = json.loads(line)
    except ValueError:
        return None
    if not isinstance(d, dict) or d.get("cmd") not in ("speak", "stop", "quit"):
        return None
    if d["cmd"] == "speak":
        if not isinstance(d.get("text"), str) or not d["text"].strip():
            return None
        try:
            d["id"] = int(d.get("id", 0))
        except (TypeError, ValueError):
            d["id"] = 0
        try:
            d["volume"] = min(1.0, max(0.0, float(d.get("volume", 1.0))))
        except (TypeError, ValueError):
            d["volume"] = 1.0
        d["voice"] = str(d.get("voice") or DEFAULT_VOICE).strip().lower().replace(" ", "_")
        d["device"] = str(d.get("device") or "")
    return d


# ---------------------------------------------------------------------------------------------- audio output
def pick_output(sd, name: str = ""):
    """(device index, is_wasapi) for ``name`` ("" = the Windows default output). Prefers WASAPI devices."""
    try:
        apis = sd.query_hostapis()
        devs = sd.query_devices()
    except Exception:
        return None, False
    was = next((i for i, a in enumerate(apis) if "WASAPI" in str(a.get("name", ""))), None)

    def outs(api):
        return [i for i, d in enumerate(devs) if d.get("max_output_channels", 0) > 0 and
                (api is None or d.get("hostapi") == api)]
    if name:
        want = name.strip().lower()
        for api in ([was] if was is not None else []) + [None]:
            cand = outs(api)
            for i in cand:              # exact name first, then "contains" (names get truncated by some APIs)
                if str(devs[i].get("name", "")).strip().lower() == want:
                    return i, devs[i].get("hostapi") == was
            for i in cand:
                n = str(devs[i].get("name", "")).strip().lower()
                if n and (want in n or n in want):
                    return i, devs[i].get("hostapi") == was
    if was is not None:
        d = apis[was].get("default_output_device", -1)
        if isinstance(d, int) and d >= 0:
            return d, True
    try:
        d = sd.default.device[1]
        return (d if isinstance(d, int) and d >= 0 else None), False
    except Exception:
        return None, False


class SdPlayer:
    """Callback output stream fed from a queue of float32 chunks; outputs silence while the queue is empty."""

    def __init__(self, sd, device_name: str = "", volume: float = 1.0):
        self.sd = sd
        self.volume = float(volume)
        self._buf: collections.deque = collections.deque()
        self._lock = threading.Lock()
        self._cur = np.zeros(0, np.float32)
        self._pos = 0
        self.played = 0             # samples handed to the device
        self.queued = 0
        self.stream = None
        self.device_name = device_name
        self.muted = False

    def _callback(self, outdata, frames, _time, _status):
        out = outdata[:, 0] if outdata.ndim == 2 else outdata
        n = 0
        with self._lock:
            while n < frames and not self.muted:
                if self._pos >= self._cur.size:
                    if not self._buf:
                        break
                    self._cur, self._pos = self._buf.popleft(), 0
                take = min(frames - n, self._cur.size - self._pos)
                out[n:n + take] = self._cur[self._pos:self._pos + take]
                self._pos += take
                n += take
            self.played += n
        if n < frames:
            out[n:] = 0
        if outdata.ndim == 2 and outdata.shape[1] > 1:
            outdata[:, 1:] = outdata[:, :1]

    def open(self) -> None:
        sd = self.sd
        try:        # fresh device list for every answer (headset plugged in, default device changed)
            sd._terminate()
            sd._initialize()
        except Exception:
            pass
        dev, wasapi = pick_output(sd, self.device_name)
        extra = None
        if wasapi:
            try:
                extra = sd.WasapiSettings(auto_convert=True)     # 24 kHz on a 48 kHz shared-mode device
            except Exception:
                extra = None
        self.stream = sd.OutputStream(samplerate=SR, channels=1, dtype="float32", device=dev,
                                      callback=self._callback, extra_settings=extra, latency="low")
        self.stream.start()

    def write(self, chunk) -> None:
        a = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if self.volume != 1.0:
            a = a * self.volume
        with self._lock:
            if self.muted:
                return
            self._buf.append(np.clip(a, -1.0, 1.0))
            self.queued += a.size

    def silence(self) -> None:
        """Any thread: stop the sound now (the callback plays zeros from here on; close() follows)."""
        with self._lock:
            self.muted = True
            self._buf.clear()
            self._cur = np.zeros(0, np.float32)
            self._pos = 0
            self.queued = self.played

    def pending(self) -> int:
        with self._lock:
            return self.queued - self.played

    def drain(self, cancel: threading.Event, timeout: float) -> bool:
        """Wait until everything queued has been played (+ the device latency). False = cancelled."""
        end = time.monotonic() + timeout
        while self.pending() > 0 and time.monotonic() < end:
            if cancel.wait(0.02):
                return False
        lat = 0.1
        try:
            lat = float(self.stream.latency or 0.1)
        except Exception:
            pass
        return not cancel.wait(min(0.5, lat + 0.05))

    def close(self, abort: bool = False) -> None:
        with self._lock:
            self._buf.clear()
            self._cur = np.zeros(0, np.float32)
            self._pos = 0
        s, self.stream = self.stream, None
        if s is None:
            return
        try:
            if abort:
                s.abort()
            else:
                s.stop()
        except Exception:
            pass
        try:
            s.close()
        except Exception:
            pass


class WavPlayer:
    """Test output: writes each answer to <dir>/tts-<id>.wav instead of the speakers (real-time pacing optional)."""

    def __init__(self, out_dir: str, rid: int, volume: float = 1.0, realtime: bool = False):
        self.path = os.path.join(out_dir, f"tts-{rid}.wav")
        self.volume = float(volume)
        self.realtime = realtime
        self._chunks: list = []
        self.played = 0
        self._t0 = None

    def open(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._t0 = time.monotonic()

    def write(self, chunk) -> None:
        a = np.asarray(chunk, dtype=np.float32).reshape(-1) * self.volume
        self._chunks.append(np.clip(a, -1.0, 1.0))
        self.played += a.size

    def drain(self, cancel: threading.Event, timeout: float) -> bool:
        if self.realtime and self._t0 is not None:
            left = self.played / SR - (time.monotonic() - self._t0)
            if left > 0 and cancel.wait(min(left, timeout)):
                return False
        return not cancel.is_set()

    def close(self, abort: bool = False) -> None:
        a = np.concatenate(self._chunks) if self._chunks else np.zeros(0, np.float32)
        tmp = self.path + ".part"
        with wave.open(tmp, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes((a * 32767).astype("<i2").tobytes())
        os.replace(tmp, self.path)


# ---------------------------------------------------------------------------------------------- server
class Server:
    """``generate(text, voice, cancel)`` yields float32 chunks (SR Hz); ``make_player(req)`` builds the output."""

    def __init__(self, events: Events, generate, make_player, voices=()):
        self.events = events
        self.generate = generate
        self.make_player = make_player
        self.voices = [str(v).lower() for v in voices]
        self.jobs: queue.Queue = queue.Queue()
        self.cancel = threading.Event()
        self.quit = threading.Event()
        self._lock = threading.Lock()
        self._latest = 0        # id of the newest speak request: an older queued one is skipped
        self._player = None     # output of the answer being spoken (silenced right away by stop / a newer answer)

    # stdin reader thread
    def read_commands(self, stream) -> None:
        try:
            for line in stream:
                cmd = parse_command(line)
                if cmd is None:
                    continue
                self.handle(cmd)
                if cmd["cmd"] == "quit":
                    return
        except (OSError, ValueError):
            pass
        finally:
            self.handle({"cmd": "quit"})     # EOF: the app is gone

    def _silence_current(self) -> None:
        """Called on the stdin thread: the generator only notices cancel between chunks (~0.3 s), the speakers
        must go quiet now."""
        with self._lock:
            p = self._player
        fn = getattr(p, "silence", None)
        if fn is not None:
            try:
                fn()
            except Exception:
                pass

    def handle(self, cmd: dict) -> None:
        c = cmd["cmd"]
        if c in ("speak", "stop", "quit"):
            self._silence_current()
        if c == "speak":
            with self._lock:
                self._latest = cmd["id"]
            self.cancel.set()           # a newer answer replaces the one being spoken
            self.jobs.put(cmd)
        elif c == "stop":
            with self._lock:
                self._latest = -1       # drops queued requests too
            self.cancel.set()
        elif c == "quit":
            self.quit.set()
            self.cancel.set()
            self.jobs.put(None)

    def run(self) -> None:
        while not self.quit.is_set():
            try:
                req = self.jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            if req is None or self.quit.is_set():
                break
            with self._lock:
                if req["id"] != self._latest:
                    continue            # superseded or stopped before it began
                self.cancel.clear()
            self.speak(req)

    def speak(self, req: dict) -> str:
        rid = req["id"]
        voice = req["voice"] if (not self.voices or req["voice"] in self.voices) else DEFAULT_VOICE
        self.events.emit("started", id=rid)
        t0 = time.monotonic()
        player = None
        samples = 0
        reason = "end"
        gen = None
        opened: dict = {}
        opener = None
        try:
            player = self.make_player(req)
            with self._lock:
                self._player = player
                if self.cancel.is_set():        # stopped while the player was being made
                    reason = "stopped"
            if reason == "stopped":
                return reason

            def _open():        # device list refresh + stream open (~0.2 s) overlaps the first chunk's generation
                try:
                    player.open()
                    opened["ok"] = True
                except Exception as e:      # noqa: BLE001 - re-raised on the main thread
                    opened["err"] = e
            opener = threading.Thread(target=_open, name="tts-open", daemon=True)
            opener.start()
            gen = self.generate(req["text"], voice, self.cancel)
            cap = int(MAX_AUDIO_S * SR)
            for chunk in gen:
                if self.cancel.is_set():
                    reason = "stopped"
                    break
                a = np.asarray(chunk, dtype=np.float32).reshape(-1)
                if a.size == 0:
                    continue
                if samples + a.size > cap:
                    a = a[:max(0, cap - samples)]
                    reason = "cap"
                if samples == 0:
                    opener.join()
                    if "err" in opened:
                        raise opened["err"]
                    player.write(a)
                    self.events.emit("playing", id=rid, ttfa_ms=int((time.monotonic() - t0) * 1000))
                else:
                    player.write(a)
                samples += a.size
                if reason == "cap":
                    break
                if time.monotonic() - t0 > MAX_GEN_S:
                    reason = "cap"
                    break
            if reason != "stopped" and samples and not player.drain(self.cancel, timeout=MAX_AUDIO_S + 5):
                reason = "stopped"
        except Exception as e:
            reason = "error"
            log(f"speak {rid} failed: {type(e).__name__}: {e}")
            self.events.emit("error", id=rid, msg=f"{type(e).__name__}: {e}")
        finally:
            if gen is not None and hasattr(gen, "close"):
                try:
                    gen.close()
                except Exception:
                    pass
            with self._lock:
                self._player = None
            if opener is not None:
                opener.join()
            if player is not None and opened.get("ok"):
                player.close(abort=reason in ("stopped", "error") or not samples)
            self.events.emit("done", id=rid, audio_s=round(samples / SR, 2), reason=reason)
            log(f"speak {rid}: {samples / SR:.1f}s audio, {reason}, {time.monotonic() - t0:.1f}s")
        return reason


# ---------------------------------------------------------------------------------------------- model
def load_model(model_dir: str):
    """(generate, voices, load_s, warmup_s). generate(text, voice, cancel) -> iterator of float32 chunks."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    t = time.perf_counter()
    import torch
    from faster_qwen3_tts import FasterQwen3TTS
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    model = FasterQwen3TTS.from_pretrained(model_dir, device="cuda", dtype=torch.bfloat16, local_files_only=True,
                                           max_seq_len=MAX_SEQ_LEN)
    load_s = time.perf_counter() - t
    t = time.perf_counter()
    model.warmup()
    voices = [str(v).lower() for v in (model.model.get_supported_speakers() or [])]

    def generate(text: str, voice: str, cancel: threading.Event):
        with torch.inference_mode():
            for audio, sr, _timing in model.generate_custom_voice_streaming(
                    text=text, speaker=voice, language=LANGUAGE, chunk_size=CHUNK_FRAMES,
                    max_new_tokens=max_new_tokens()):
                if cancel.is_set():
                    return
                a = np.asarray(audio, dtype=np.float32).reshape(-1)
                if sr != SR:        # not expected (the 12 Hz tokenizer decodes at 24 kHz)
                    n = int(round(a.size * SR / float(sr)))
                    a = np.interp(np.linspace(0, a.size - 1, n), np.arange(a.size), a).astype(np.float32)
                yield a
    # the first real generation after the graph capture is slow (~1.7 s to first audio): do it now
    never = threading.Event()
    for _ in generate("Ready.", DEFAULT_VOICE if DEFAULT_VOICE in voices or not voices else voices[0], never):
        pass
    torch.cuda.synchronize()
    return generate, voices, load_s, time.perf_counter() - t


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--wav-out", default="", help="test hook: write answers to WAV files here, no speakers")
    args = ap.parse_args(argv)
    events, _logf = setup_channels(args.log)
    log(f"tts server starting (pid {os.getpid()}), model {args.model}")
    try:
        generate, voices, load_s, warm_s = load_model(args.model)
    except Exception as e:
        import traceback
        traceback.print_exc()
        events.emit("error", id=None, msg=f"load failed: {type(e).__name__}: {e}")
        return 2
    if args.wav_out:
        def make_player(req):
            return WavPlayer(args.wav_out, req["id"], req["volume"], realtime=True)
    else:
        import sounddevice as sd

        def make_player(req):
            return SdPlayer(sd, req["device"], req["volume"])
    srv = Server(events, generate, make_player, voices)
    threading.Thread(target=srv.read_commands, args=(sys.stdin,), name="stdin", daemon=True).start()
    log(f"ready: load {load_s:.1f}s, warmup {warm_s:.1f}s, voices {voices}")
    events.emit("ready", voices=voices, sr=SR, load_s=round(load_s, 2), warmup_s=round(warm_s, 2))
    srv.run()
    log("tts server exiting")
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)      # skip slow CUDA / interpreter teardown; the OS frees the GPU memory
