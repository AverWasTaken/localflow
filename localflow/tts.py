"""Spoken command-mode answers: lifecycle of the TTS server process (localflow/tts_server.py, run by .venv-tts)
and the text clean-up that makes an answer speakable.

The server holds Qwen3-TTS 0.6B on the GPU (~2.6 GB) while the app runs. It is never running in game mode, while
paused, or with command_mode.speak_answers off. Nothing here raises into the app: a missing install or a crashing
server only means answers are not spoken.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from .paths import DATA_DIR, MODELS_DIR, PROJECT_DIR

log = logging.getLogger(__name__)

TTS_VENV_PY = PROJECT_DIR / ".venv-tts" / "Scripts" / "python.exe"
TTS_MODEL_DIR = MODELS_DIR / "tts" / "Qwen3-TTS-12Hz-0.6B-CustomVoice"
TTS_SERVER = Path(__file__).resolve().parent / "tts_server.py"
TTS_LOG = DATA_DIR / "tts-server.log"

# (config value, label) for the settings dropdown; the model's built-in speakers
VOICES = [("Ryan", "Ryan (English, male)"), ("Aiden", "Aiden (American English, male)"),
          ("Vivian", "Vivian (Chinese, female)"), ("Serena", "Serena (Chinese, female)"),
          ("Uncle_Fu", "Uncle Fu (Chinese, male)"), ("Dylan", "Dylan (Beijing Chinese, male)"),
          ("Eric", "Eric (Sichuan Chinese, male)"), ("Ono_Anna", "Ono Anna (Japanese, female)"),
          ("Sohee", "Sohee (Korean, female)")]

MAX_SPEECH_S = 30.0         # the server stops after this much audio too
WORDS_PER_S = 2.6           # measured: Ryan reads ~76 words in ~28 s
MAX_WORDS = int(MAX_SPEECH_S * WORDS_PER_S) - 4
PREROLL_SKIP_S = 0.5        # a recording this soon after audible speech starts without the mic pre-roll
RESTART_DELAYS = (3.0, 10.0, 30.0)     # crash restarts (then give up until the next start())
CREATE_NO_WINDOW = 0x08000000


# ---------------------------------------------------------------------------------------------- text
_NUM = r"\d+(?:,\d{3})*(?:\.\d+)?"
_UNITS = [   # unit after a number -> spoken; only units that can't be an ordinary word right after a number
    (r"km/h|kph", "kilometers per hour"), (r"mph", "miles per hour"), (r"m/s", "meters per second"),
    (r"km", "kilometers"), (r"cm", "centimeters"), (r"mm", "millimeters"), (r"kg", "kilograms"),
    (r"mg", "milligrams"), (r"lbs?", "pounds"), (r"oz", "ounces"), (r"ml|mL", "milliliters"),
    (r"ft", "feet"), (r"sq ft", "square feet"), (r"kWh", "kilowatt hours"), (r"GB", "gigabytes"),
    (r"MB", "megabytes"), (r"TB", "terabytes"), (r"GHz", "gigahertz"), (r"MHz", "megahertz"),
]
_UNIT_RE = [(re.compile(rf"({_NUM})\s?(?:{pat})(?![A-Za-z0-9/])"), word) for pat, word in
            sorted(_UNITS, key=lambda u: -len(u[0]))]
_SCALE = r"(?:\s(?:thousand|million|billion|trillion))?"


def _strip_markdown(t: str) -> str:
    t = re.sub(r"```.*?(?:```|$)", " ", t, flags=re.S)             # code blocks: not read out
    t = re.sub(r"`([^`]*)`", r"\1", t)
    t = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", t)                 # images / links -> their text
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"https?://\S+|www\.\S+", "a link", t)
    t = re.sub(r"(\*\*|__)(.+?)\1", r"\2", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)([^*\n]+?)\*(?!\w)", r"\1", t)
    t = re.sub(r"(?<![\w_])_(?!\s)([^_\n]+?)_(?!\w)", r"\1", t)
    t = re.sub(r"~~(.+?)~~", r"\1", t)
    lines = []
    # "1." / "2)" are list markers only in a real list (2+ such lines): "42. That is the answer." keeps its 42
    # (or a numbered line inside a bulleted list, but never the first line: that is the answer itself)
    raw = t.splitlines()
    n_num = sum(1 for ln in raw if re.match(r"\s*\d{1,3}[.)]\s", ln))
    n_list = sum(1 for ln in raw if re.match(r"\s*(?:\d{1,3}[.)]|[-*+•])\s", ln))
    first = next((i for i, ln in enumerate(raw) if ln.strip()), -1)
    for i, ln in enumerate(raw):
        numbered = n_num >= 2 or (n_list >= 2 and i != first)
        ln = ln.strip()
        if re.fullmatch(r"[|:\-\s]+", ln) and ("-" in ln or "|" in ln):   # table rule / horizontal rule
            continue
        ln = re.sub(r"^#{1,6}\s+", "", ln)
        ln = re.sub(r"^>\s?", "", ln)
        ln = re.sub(r"^[-*+•]\s+", "", ln)
        if numbered:
            ln = re.sub(r"^\d{1,3}[.)]\s+", "", ln)
        if ln.startswith("|") or ln.count("|") >= 2:
            ln = ", ".join(c.strip() for c in ln.strip("|").split("|") if c.strip())
        if not ln:
            continue
        if not re.search(r"[.!?:;,]$", ln):     # list items / headings become their own sentence
            ln += "."
        lines.append(ln)
    return " ".join(lines)


_OPEN_QUOTE = re.compile(r"(?:^|(?<=[\s(\[]))['‘](?=\w)")


def _feet(m: "re.Match") -> str:
    """6' -> "6 feet", unless the quote closes a quotation opened earlier in the sentence ('Windows 11', 'F5')."""
    before = m.string[:m.start()]
    sentence = re.split(r"[.!?]\s", before)[-1]
    if _OPEN_QUOTE.search(sentence):
        return m.group(0)
    return f"{m.group(1)} feet"


def _expand_symbols(t: str) -> str:
    t = re.sub(r"(\d)\s?°\s?F\b", r"\1 degrees Fahrenheit", t)
    t = re.sub(r"(\d)\s?°\s?C\b", r"\1 degrees Celsius", t)
    t = re.sub(r"(\d)\s?°", r"\1 degrees", t)
    t = re.sub(r"(\d)\s?%", r"\1 percent", t)
    # $12.50, $3 million, -$5 -> "12.50 dollars", "3 million dollars"
    t = re.sub(rf"\$\s?({_NUM}{_SCALE})", r"\1 dollars", t)
    t = re.sub(rf"€\s?({_NUM}{_SCALE})", r"\1 euros", t)
    t = re.sub(rf"£\s?({_NUM}{_SCALE})", r"\1 pounds", t)
    # 5'11" -> 5 feet 11 inches; 6' tall -> 6 feet (a quote right after a number, never before one: '90s)
    t = re.sub(r"(\d+)\s?['′]\s?(\d{1,2})\s?(?:\"|″|'')", r"\1 feet \2 inches", t)
    t = re.sub(r"(?<![\w'‘’\"])(\d+)['′](?=\s|$|[,.;!?])", _feet, t)
    # "12 in." / "12in" (an abbreviation), never "12 in total"
    t = re.sub(r"(\d)\s?in\.(?=\s+[a-z0-9])", r"\1 inches", t)     # "12 in. wide": the dot was the abbreviation
    t = re.sub(r"(\d)\s?in\.(?=\s|$)", r"\1 inches.", t)
    t = re.sub(r"(\d)in\b", r"\1 inches", t)
    for rx, word in _UNIT_RE:
        t = rx.sub(lambda m, w=word: f"{m.group(1)} {w}", t)
    t = re.sub(r"\s&\s", " and ", t)
    t = re.sub(r"\s(?:->|→)\s", " to ", t)
    t = re.sub(r"(\d)\s?×\s?(?=\d)", r"\1 times ", t)
    t = re.sub(r"(\d)\s?÷\s?(?=\d)", r"\1 divided by ", t)
    t = re.sub(r"(?:^|(?<=[\s(]))[-−](?=\d)", "minus ", t)     # "-3" (not "3-5")
    t = re.sub(r"(\d)\s+\*\s+(?=\d)", r"\1 times ", t)
    t = re.sub(r"\s=\s", " equals ", t)
    # comparisons only next to a number; between words ("Settings > Privacy") it is a menu path
    num = r"(?:minus )?\d"
    t = re.sub(rf"(?<=\d)\s<=\s(?={num})|(?<=\w)\s<=\s(?={num})", " is at most ", t)
    t = re.sub(rf"(?<=\w)\s>=\s(?={num})", " is at least ", t)
    t = re.sub(rf"(?<=\w)\s<\s(?={num})|(?<=\d)\s<\s", " is less than ", t)
    t = re.sub(rf"(?<=\w)\s>\s(?={num})|(?<=\d)\s>\s", " is greater than ", t)
    t = re.sub(r"(?<=\w)\s>\s(?=\w)", ", then ", t)
    t = re.sub(r"~\s?(\d)", r"about \1", t)
    t = re.sub(r"\be\.g\.", "for example", t)
    t = re.sub(r"\bi\.e\.", "that is", t)
    t = re.sub(r"\bvs\.?(?=\s)", "versus", t)
    return t


def _trim_words(t: str, max_words: int) -> str:
    words = t.split()
    if len(words) <= max_words:
        return t
    cut = " ".join(words[:max_words])
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if cut.endswith((".", "!", "?")):
        end = len(cut) - 1
    if end >= len(cut) // 3:           # end on a sentence when one ends reasonably late
        return cut[:end + 1]
    return cut.rstrip(",;:") + "."


def speech_text(text: str, max_words: int = MAX_WORDS) -> str:
    """Answer text -> what the voice reads: markdown stripped, unambiguous symbols / units spelled out, trimmed to
    about MAX_SPEECH_S of speech (on a sentence end when possible)."""
    t = str(text or "")
    t = _strip_markdown(t)
    t = _expand_symbols(t)
    t = re.sub(r"[\U0001F000-\U0001FAFF☀-➿️‍]", "", t)    # emoji
    t = re.sub(r"[*_#`|<>\[\]{}]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = re.sub(r"([.!?])\.+", r"\1", t)
    return _trim_words(t, max_words) if t else ""


def estimate_seconds(spoken: str) -> float:
    """Generous guess of the speech length (short answers with numbers read slower than WORDS_PER_S)."""
    n = len(spoken.split())
    return min(MAX_SPEECH_S, 1.5 + n / 2.3) if n else 0.0


# ---------------------------------------------------------------------------------------------- manager
def installed() -> bool:
    return TTS_VENV_PY.exists() and (TTS_MODEL_DIR / "config.json").exists() and TTS_SERVER.exists()


class TtsManager:
    """Owns the TTS server process. Thread-safe; every method returns quickly except shutdown(wait=True).

    start()/shutdown() say whether the server should run; speak()/stop() talk to it. ``blocked`` (game mode,
    paused) keeps it from starting. ``popen`` and ``installed`` are injectable for tests."""

    def __init__(self, cfg, blocked: Callable[[], bool] = lambda: False, inert: bool = False,
                 popen: Callable = subprocess.Popen, is_installed: Callable[[], bool] = installed,
                 wav_out: str = "", restart_delays=RESTART_DELAYS, log_path: Path = TTS_LOG):
        self.cfg = cfg
        self.blocked = blocked
        self.inert = inert                  # tests that build a whole app never spawn a server
        self._popen = popen
        self._installed = is_installed
        self.wav_out = wav_out              # test hook: the server writes WAV files instead of playing
        self.restart_delays = tuple(restart_delays)
        self.log_path = Path(log_path)
        self._lock = threading.RLock()
        self._want = False
        self._proc = None
        self._job = None
        self._gen = 0                       # bumped by every start/shutdown: stale threads see it and stop
        self.state = "off"                  # off | starting | ready | failed
        self.voices: list[str] = []
        self._crashes = 0
        self._sid = 0                       # id of the latest speak request
        self._speaking: Optional[int] = None
        self._audible = False
        self._audible_end = 0.0
        self._min_rid = 0                   # answers of recordings older than this are dropped
        self._said_missing = False
        self.on_event: Optional[Callable[[dict], None]] = None     # tests / diagnostics

    # ------------------------------------------------------------------ config
    def enabled(self) -> bool:
        g = self.cfg.get
        return bool(g("command_mode.speak_answers", True) and g("command_mode.enabled", True)
                    and g("command_mode.answer_questions", True))

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """Make sure the server runs (if enabled, installed and not blocked). Resets the crash budget."""
        if self.inert or not self.enabled():
            return
        if not self._installed():
            if not self._said_missing:
                self._said_missing = True
                log.info("tts: not installed (%s / %s missing) - answers are not spoken", TTS_VENV_PY.parent.parent,
                         TTS_MODEL_DIR)
            return
        try:
            if self.blocked():
                return
        except Exception:
            return
        with self._lock:
            self._want = True
            self._crashes = 0
            if self._proc is not None or self.state == "starting":
                return
            self._spawn_locked()

    def _spawn_locked(self) -> None:
        self._gen += 1
        gen = self._gen
        self.state = "starting"
        threading.Thread(target=self._run, args=(gen,), name="tts-server", daemon=True).start()

    def shutdown(self, why: str = "", wait: bool = True) -> None:
        """Stop the server and free its GPU memory (kills the whole process tree)."""
        with self._lock:
            self._want = False
            self._gen += 1
            proc, job = self._proc, self._job
            self._proc, self._job = None, None
            self._speaking = None
            if self._audible:
                self._audible_end = time.monotonic()
            self._audible = False
            was = self.state
            self.state = "off"
        if proc is None:
            return
        log.info("tts: stopping server (%s)", why or "shutdown")
        self._send_to(proc, {"cmd": "quit"})
        if wait and was == "ready":     # a loading server reads stdin only after the load: just kill it
            try:
                proc.wait(timeout=1.5)
            except Exception:
                pass
        self._kill_tree(proc)
        if job is not None:
            try:
                job.Close()
            except Exception:
                pass
        log.info("tts: server stopped (was %s)", was)

    def _run(self, gen: int) -> None:
        """Spawn the server and read its events until it exits; restart it on a crash (bounded)."""
        proc = None
        try:
            cmd = [str(TTS_VENV_PY), "-X", "utf8", str(TTS_SERVER), "--model", str(TTS_MODEL_DIR),
                   "--log", str(self.log_path)]
            if self.wav_out:
                cmd += ["--wav-out", self.wav_out]
            env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}
            env.update({"PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                        "PYTHONUNBUFFERED": "1"})
            try:
                errf = open(self.log_path, "wb")      # fresh log per start; the server appends after its own setup
            except OSError:
                errf = subprocess.DEVNULL
            t0 = time.perf_counter()
            proc = self._popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errf,
                               cwd=str(PROJECT_DIR), env=env, creationflags=CREATE_NO_WINDOW)
            if errf is not subprocess.DEVNULL:
                try:
                    errf.close()
                except Exception:
                    pass
            job = self._assign_job(proc)
            with self._lock:
                if gen != self._gen or not self._want:      # shut down while spawning: no GPU process left behind
                    stale = True
                else:
                    stale = False
                    self._proc, self._job = proc, job
            if stale:
                self._send_to(proc, {"cmd": "quit"})
                self._kill_tree(proc)
                if job is not None:
                    try:
                        job.Close()
                    except Exception:
                        pass
                return
            log.info("tts: server started (pid %s)", getattr(proc, "pid", "?"))
            for raw in proc.stdout:
                try:
                    ev = json.loads(raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw)
                except ValueError:
                    continue
                if isinstance(ev, dict):
                    self._on_event(gen, ev, t0)
        except Exception:
            log.exception("tts: server failed")
        finally:
            self._exited(gen, proc)

    def _exited(self, gen: int, proc) -> None:
        code = None
        if proc is not None:
            try:
                code = proc.wait(timeout=5)
            except Exception:
                self._kill_tree(proc)
        with self._lock:
            if gen != self._gen:            # an intentional shutdown / restart: nothing to do
                return
            if self._proc is proc:
                self._proc = None
                job, self._job = self._job, None
                if job is not None:
                    try:
                        job.Close()
                    except Exception:
                        pass
            self._speaking = None
            if self._audible:
                self._audible_end = time.monotonic()
            self._audible = False
            if not self._want:
                self.state = "off"
                return
            if self._crashes >= len(self.restart_delays):
                self.state = "failed"
                log.error("tts: server exited (code %s) %d times - giving up until it is turned on again; see %s",
                          code, self._crashes + 1, self.log_path)
                return
            delay = self.restart_delays[self._crashes]
            self._crashes += 1
            self.state = "starting"
            self._gen += 1
            ngen = self._gen
            log.warning("tts: server exited (code %s) - restarting in %.0fs (%d/%d)", code, delay, self._crashes,
                        len(self.restart_delays))
        threading.Thread(target=self._restart_later, args=(ngen, delay), name="tts-restart", daemon=True).start()

    def _restart_later(self, gen: int, delay: float) -> None:
        time.sleep(delay)
        try:
            if self.blocked():
                with self._lock:
                    if gen == self._gen:
                        self.state = "off"
                        self._want = False
                return
        except Exception:
            return
        with self._lock:
            if gen != self._gen or not self._want or self._proc is not None:
                return
            self.state = "starting"
        self._run(gen)

    def _on_event(self, gen: int, ev: dict, t0: float) -> None:
        kind = ev.get("ev")
        with self._lock:
            if gen != self._gen:
                return
            if kind == "ready":
                self.state = "ready"
                self.voices = [str(v) for v in ev.get("voices") or []]
                log.info("tts: READY in %.1fs (load %.1fs, warmup %.1fs)", time.perf_counter() - t0,
                         float(ev.get("load_s") or 0), float(ev.get("warmup_s") or 0))
            elif kind == "playing" and ev.get("id") == self._speaking:
                self._audible = True
                log.info("tts: speaking (first audio %s ms after the request)", ev.get("ttfa_ms"))
            elif kind == "done" and ev.get("id") == self._speaking:
                self._speaking = None
                if self._audible:
                    self._audible_end = time.monotonic()
                self._audible = False
                log.info("tts: done (%ss of audio, %s)", ev.get("audio_s"), ev.get("reason"))
            elif kind == "error":
                log.warning("tts: server error: %s", ev.get("msg"))
        cb = self.on_event
        if cb is not None:
            try:
                cb(ev)
            except Exception:
                pass

    # ------------------------------------------------------------------ speaking
    def speak(self, text: str, rid: Optional[int] = None) -> float:
        """Speak an answer. Returns the estimated speech length in seconds (0 = not spoken: off / not ready /
        a newer recording started since ``rid``)."""
        if self.inert or not self.enabled():
            return 0.0
        spoken = speech_text(text)
        if not spoken:
            return 0.0
        g = self.cfg.get
        with self._lock:
            if self.state != "ready" or self._proc is None:
                log.info("tts: answer not spoken (server %s)", self.state)
                return 0.0
            if rid is not None and rid < self._min_rid:
                log.info("tts: answer not spoken (a newer recording started)")
                return 0.0
            self._sid += 1
            self._speaking = self._sid
            ok = self._send_to(self._proc, {
                "cmd": "speak", "id": self._sid, "text": spoken,
                "voice": str(g("command_mode.tts_voice", "Ryan") or "Ryan"),
                "device": str(g("command_mode.tts_device", "") or ""),
                "volume": float(g("command_mode.tts_volume", 1.0) or 0.0)})
            if not ok:
                self._speaking = None
                return 0.0
        return estimate_seconds(spoken)

    def stop(self, why: str = "") -> None:
        with self._lock:
            if self._speaking is None:
                return
            self._speaking = None
            if self._audible:
                self._audible_end = time.monotonic()
            self._audible = False
            if self._proc is not None:
                self._send_to(self._proc, {"cmd": "stop"})
        log.info("tts: stopped (%s)", why or "stop")

    def recording_started(self, rid: int) -> bool:
        """A recording begins: silence any answer and drop answers of older recordings. True = speech was
        audible within PREROLL_SKIP_S (the mic pre-roll may contain the voice: skip it)."""
        with self._lock:
            self._min_rid = max(self._min_rid, int(rid))
            recent = self._audible or (self._audible_end and time.monotonic() - self._audible_end < PREROLL_SKIP_S)
            if self._speaking is not None:
                self.stop("recording")
        return bool(recent)

    @property
    def speaking(self) -> bool:
        return self._speaking is not None

    # ------------------------------------------------------------------ process plumbing
    def _send_to(self, proc, obj: dict) -> bool:
        try:
            proc.stdin.write((json.dumps(obj) + "\n").encode("utf-8"))
            proc.stdin.flush()
            return True
        except Exception:
            return False

    @staticmethod
    def _assign_job(proc):
        """Kill-on-close job object: the server dies with the app even if the app is killed."""
        if os.name != "nt" or not isinstance(getattr(proc, "pid", None), int):
            return None
        try:
            import win32api
            import win32job
            job = win32job.CreateJobObject(None, "")
            info = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
            info["BasicLimitInformation"]["LimitFlags"] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, info)
            h = win32api.OpenProcess(0x1F0FFF, False, proc.pid)
            win32job.AssignProcessToJobObject(job, h)
            return job
        except Exception:
            log.exception("tts: job object failed (the server may outlive a killed app)")
            return None

    @staticmethod
    def _kill_tree(proc) -> None:
        pid = getattr(proc, "pid", None)
        try:
            if isinstance(pid, int):
                import psutil
                try:
                    kids = psutil.Process(pid).children(recursive=True)
                except psutil.Error:
                    kids = []
                for k in kids:
                    try:
                        k.kill()
                    except psutil.Error:
                        pass
        except Exception:
            pass
        try:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass
        for s in (getattr(proc, "stdin", None), getattr(proc, "stdout", None)):
            try:
                if s is not None:
                    s.close()
            except Exception:
                pass
