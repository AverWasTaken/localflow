"""Chunked ("streaming") transcription for long dictations.

While the user is still talking a background worker watches the incoming audio with the Silero VAD that ships
with faster-whisper (CPU/onnx, ~0.3 ms per 32 ms frame). At a natural pause (>= ``pause_ms`` of silence) after
>= ``min_segment_s`` of new speech it finalizes that segment and transcribes it on the GPU straight away; with no
pause for ``max_segment_s`` it cuts at the lowest-energy point of the last 2.5 s. On release only the tail after
the last cut is transcribed, and the segments are joined.

Cleanup can run incrementally too: finished sentences are cleaned in the background (the previous cleaned text is
shown to the LLM as read-only context) so at release only the last unit is cleaned. A segment that starts with a
self-correction cue ("actually no, 4") is merged into the previous unit and that unit is re-cleaned.

Everything here is best-effort: if anything fails the pipeline falls back to the normal single pass over the full
recording, which it always has.
"""
from __future__ import annotations

import logging
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from math import ceil, gcd
from typing import Optional

import numpy as np

from . import styles, textproc
from .audioutil import normalize, peak_window_rms

log = logging.getLogger(__name__)

SR = 16000
FRAME = 512                       # Silero window (32 ms)
FRAME_S = FRAME / SR

# a segment that opens with one of these may be a spoken correction of the previous segment
_CUE_RE = re.compile(
    r"^\W*(?:(?:um+|uh+|so|and|well|oh|okay|ok)\b[\s,]*)?"
    r"(?:actually|no[,.]?\s+wait|wait|sorry|scratch that|i mean|i meant|no[,.]?\s+no|no[,.]\s|or rather|make that|"
    r"correction|let me rephrase|let me correct|never mind that|hold on|oops)\b", re.I)
_TERMINAL_RE = re.compile(r"[.!?][\"')\]]*\s*$")


def correction_cue(text: str) -> bool:
    return bool(_CUE_RE.match(text or ""))


def ends_sentence(text: str) -> bool:
    return bool(_TERMINAL_RE.search(text or ""))


_BOUNDARY_RE = re.compile(r"([.!?][\"')\]]*)\s+(?=[A-Z0-9\"'(\[])")
_ABBREV_END = ("a.m.", "p.m.", "e.g.", "i.e.", "mr.", "mrs.", "ms.", "dr.", "vs.", "etc.", "st.", "inc.", "no.",
               "approx.", "u.s.", "jr.", "sr.", "prof.")


def split_complete(text: str, min_words: int = 6) -> tuple[str, str]:
    """Split ``text`` after its LAST real sentence boundary -> (complete sentences, unfinished fragment).
    A boundary is .?! followed by a capital/digit that is not an abbreviation (p.m., Dr., e.g. ...).
    Returns ("", text) when there is no usable boundary or the head would be shorter than ``min_words`` words."""
    best = -1
    for m in _BOUNDARY_RE.finditer(text):
        word = text[:m.end(1)].rsplit(None, 1)[-1].lower().strip("\"'([")
        if m.group(1)[0] == "." and word.endswith(_ABBREV_END):
            continue
        best = m.end(1)
    if best < 0:
        return "", text
    head, rest = text[:best].strip(), text[best:].strip()
    if len(head.split()) < min_words or not rest:
        return "", text
    return head, rest


# ----------------------------------------------------------------------------- resampling
class StreamResampler:
    """Incremental native-rate -> 16 kHz resampler without block-edge artefacts (overlap-and-trim around
    scipy's resample_poly; output sample n lines up with sample n of a one-shot resample of the whole signal)."""

    def __init__(self, sr: int):
        self.sr = int(sr)
        self.passthrough = self.sr == SR
        if self.passthrough:
            return
        g = gcd(SR, self.sr)
        self.up, self.down = SR // g, self.sr // g
        half_in = ceil(10 * max(self.up, self.down) / self.up)      # filter half-length in input samples
        self.ctx = ((half_in + self.down - 1) // self.down + 1) * self.down   # multiple of `down`
        self.pend = np.zeros(0, np.float32)
        self.left = np.zeros(0, np.float32)

    def push(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.passthrough:
            return x
        from scipy.signal import resample_poly
        self.pend = np.concatenate([self.pend, x])
        usable = ((len(self.pend) - self.ctx) // self.down) * self.down
        if usable <= 0:
            return np.zeros(0, np.float32)
        block = self.pend[:usable]
        right = self.pend[usable:usable + self.ctx]
        inp = np.concatenate([self.left, block, right])
        y = resample_poly(inp, self.up, self.down).astype(np.float32)
        l0 = len(self.left) * self.up // self.down
        out = y[l0:l0 + usable * self.up // self.down]
        self.left = block[-self.ctx:].copy() if usable >= self.ctx else np.concatenate([self.left, block])[-self.ctx:]
        self.pend = self.pend[usable:]
        return out


# ----------------------------------------------------------------------------- VAD
class VadStream:
    """Per-frame speech probabilities for a growing 16 kHz signal (Silero v6 via faster-whisper's onnx session).

    The model is recurrent across frames, so the LSTM state (h, c) and the 64-sample context are carried between
    calls; output is bit-identical to running the whole recording through faster-whisper's wrapper at once."""

    def __init__(self):
        self.rem = np.zeros(0, np.float32)
        self.ctx = np.zeros(64, np.float32)
        self.h = np.zeros((1, 1, 128), np.float32)
        self.c = np.zeros((1, 1, 128), np.float32)
        self.model = None
        try:
            from faster_whisper.vad import get_vad_model
            self.model = get_vad_model()
        except Exception as e:  # pragma: no cover
            log.warning("silero VAD unavailable (%s) - using an energy VAD for chunking", e)
        self.noise = 0.004

    def push(self, x: np.ndarray) -> np.ndarray:
        buf = np.concatenate([self.rem, x]) if self.rem.size else np.asarray(x, np.float32)
        n = len(buf) // FRAME
        if n == 0:
            self.rem = buf
            return np.zeros(0, np.float32)
        fr = buf[:n * FRAME].reshape(n, FRAME)
        self.rem = buf[n * FRAME:].copy()
        if self.model is not None:
            ctx = np.concatenate([self.ctx[None, :], fr[:-1, -64:]], axis=0)
            self.ctx = fr[-1, -64:].copy()
            inp = np.concatenate([ctx, fr], axis=1).astype(np.float32)
            out, self.h, self.c = self.model.session.run(None, {"input": inp, "h": self.h, "c": self.c})
            return np.asarray(out, dtype=np.float32).reshape(-1)
        rms = np.sqrt((fr ** 2).mean(axis=1))
        self.noise = float(min(self.noise, np.percentile(rms, 20) + 1e-5)) if rms.size else self.noise
        return np.clip((rms - 2.5 * self.noise) / max(0.01, 3 * self.noise), 0, 1).astype(np.float32)


def _runs(flags: np.ndarray) -> list[tuple[bool, int, int]]:
    """Run-length encode a bool array -> [(value, start, length)]."""
    if flags.size == 0:
        return []
    idx = np.flatnonzero(np.diff(flags.astype(np.int8))) + 1
    starts = np.concatenate([[0], idx])
    ends = np.concatenate([idx, [flags.size]])
    return [(bool(flags[s]), int(s), int(e - s)) for s, e in zip(starts, ends)]


# ----------------------------------------------------------------------------- results
@dataclass
class Unit:
    """A piece of text that is cleaned by the LLM in one request (one or more sentences)."""
    raw: str
    cleaned: Optional[str] = None
    ms: int = 0
    llm: bool = False
    reason: str = ""
    version: int = 0
    pause_ms: int = 0              # silence before the first segment of this unit (paragraph heuristic)


@dataclass
class StreamResult:
    raw: str = ""
    text: str = ""
    n_segments: int = 0            # segments finalized before release (the tail is extra)
    tail_stt_ms: int = 0           # STT time spent after release (excludes waiting for in-flight work)
    stt_total_ms: int = 0
    wait_ms: int = 0               # release -> background work drained
    cleanup_ms: int = 0            # LLM/regex time after release
    cleanup_bg_ms: int = 0         # LLM/regex time hidden behind recording
    llm: bool = False
    reason: str = ""
    units: int = 0
    merged: int = 0                # correction-cue merges
    segments: list = field(default_factory=list)


# ----------------------------------------------------------------------------- session
class StreamSession:
    def __init__(self, cfg, stt, cleaner, ctx: styles.StyleContext, sample_rate: int = SR):
        self.cfg, self.stt, self.cleaner, self.ctx = cfg, stt, cleaner, ctx
        sc = cfg.get("streaming", {}) or {}
        self.min_seg = float(sc.get("min_segment_s", 4.0))
        self.min_record = float(sc.get("min_record_s", 7.0))
        self.pause_frames = max(2, int(ceil(float(sc.get("pause_ms", 500)) / 1000.0 / FRAME_S)))
        self.max_seg = float(sc.get("max_segment_s", 22.0))
        self.prompt_chars = int(sc.get("prompt_chars", 220))
        self.incremental = bool(sc.get("incremental_cleanup", True))
        self.context_chars = int(sc.get("context_chars", 300))
        self.max_unit_words = int(sc.get("max_unit_words", 140))
        self.para_pause_ms = int(sc.get("paragraph_pause_ms", 900))

        dic = cfg.get("dictionary", {}) or {}
        self.words = [w for w in dic.get("words", []) if w.strip()]
        self.repl = dic.get("replacements", {}) or {}
        self.sd = styles.get_style(cfg, ctx.style)
        self.paragraphs = bool(self.sd.get("paragraphs", False)) and self.para_pause_ms > 0
        self.abbr = cfg.get("abbreviations", styles.DEFAULT_ABBREVIATIONS)
        self.silence_rms = float(cfg.get("audio.silence_rms", 0.006))

        # audio store (16 kHz)
        self._buf = np.zeros(SR * 40, np.float32)
        self._n = 0
        self._lk = threading.Lock()
        self._wake = threading.Event()
        self._poll_stop = threading.Event()   # stops the microphone poller
        self._vad_stop = threading.Event()    # stops the VAD/cutting thread (on release or cancel)
        self._closed = threading.Event()      # session over: STT/cleanup workers exit
        self._resampler = StreamResampler(sample_rate)
        # VAD state
        self._vad = None
        self._vad_pos = 0
        self._probs = np.zeros(0, np.float32)
        self._flags = np.zeros(0, bool)
        self._in_speech = False
        self._cut = 0                       # sample where the current (unfinalized) segment starts
        self.n_segments = 0
        self.failed: Optional[str] = None
        self._seg_log: list[tuple[float, float, str, int]] = []
        # STT worker
        self._stt_q: queue.Queue = queue.Queue()
        self._texts: list[str] = []
        self._prompt_src = ""
        self._stt_total = 0
        self._tail_ms = 0
        self._tail_pause = 0
        # cleanup state
        self._ulk = threading.RLock()
        self._pending: list[tuple[str, int]] = []      # (segment text, pause before it in ms)
        self._units: list[Unit] = []
        self._clean_q: queue.Queue = queue.Queue()
        self._merged = 0
        self._bg_clean_ms = 0
        self._threads: list[threading.Thread] = []
        self._poll_thread: Optional[threading.Thread] = None
        self._vad_thread: Optional[threading.Thread] = None
        self._started = False

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._vad_thread = threading.Thread(target=self._vad_loop, name="stream-vad", daemon=True)
        self._vad_thread.start()
        self._threads.append(self._vad_thread)
        t = threading.Thread(target=self._stt_loop, name="stream-stt", daemon=True)
        t.start()
        self._threads.append(t)
        if self.incremental:
            t = threading.Thread(target=self._clean_loop, name="stream-clean", daemon=True)
            t.start()
            self._threads.append(t)

    def attach(self, audio_input, poll_s: float = 0.05) -> None:
        """Pull the live microphone chunks from an AudioInput on a poller thread."""
        self._resampler = StreamResampler(audio_input.sample_rate)

        def run():
            idx = 0
            while not self._poll_stop.wait(poll_s):
                chunks, idx = audio_input.pull_chunks(idx)
                if chunks is None:      # recording ended
                    return
                if chunks:
                    self.feed(self._resampler.push(np.concatenate(chunks)))

        self._poll_thread = threading.Thread(target=run, name="stream-poll", daemon=True)
        self._poll_thread.start()

    def _stop_poll(self) -> None:
        self._poll_stop.set()
        t, self._poll_thread = self._poll_thread, None
        if t is not None and t is not threading.current_thread():
            t.join(timeout=0.5)

    def cancel(self) -> None:
        """Abandon the session (Esc / too short / shortcut): stop all threads, drop the work."""
        self._poll_stop.set()
        self._vad_stop.set()
        self._closed.set()
        self._wake.set()
        self._stt_q.put(None)
        self._clean_q.put(None)

    def feed(self, x: np.ndarray) -> None:
        """Append 16 kHz mono float32 samples (any thread)."""
        n = len(x)
        if not n:
            return
        with self._lk:
            if self._n + n > len(self._buf):
                nb = np.zeros(max(len(self._buf) * 2, self._n + n), np.float32)
                nb[:self._n] = self._buf[:self._n]
                self._buf = nb
            self._buf[self._n:self._n + n] = x
            self._n += n
        self._wake.set()

    # ------------------------------------------------------------------ VAD / cutting
    def _vad_loop(self) -> None:
        try:
            self._vad = VadStream()
        except Exception as e:
            self.failed = f"vad init: {e}"
            log.exception("stream VAD init failed")
            return
        while not self._vad_stop.is_set():
            self._wake.wait(0.1)
            self._wake.clear()
            if self._vad_stop.is_set():
                break
            try:
                self._vad_step()
            except Exception as e:
                self.failed = f"{type(e).__name__}: {e}"
                log.exception("stream VAD step failed - falling back to single pass")
                return

    def _vad_step(self) -> None:
        with self._lk:
            n = self._n
            if n <= self._vad_pos:
                return
            new = self._buf[self._vad_pos:n].copy()
        self._vad_pos = n
        p = self._vad.push(new)
        if p.size:
            self._probs = np.concatenate([self._probs, p])
            flags = np.empty(p.size, bool)
            st = self._in_speech
            for i, v in enumerate(p):           # hysteresis: on at >=0.5, off below 0.35
                st = (v >= 0.35) if st else (v >= 0.5)
                flags[i] = st
            self._in_speech = st
            self._flags = np.concatenate([self._flags, flags])
            self._maybe_cut()

    def _maybe_cut(self) -> None:
        F = self._flags.size
        rs = int(ceil(self._cut / FRAME))
        if F - rs < self.pause_frames + 4:
            return
        elapsed = F * FRAME_S
        seg_s = (F - rs) * FRAME_S
        flags = self._flags[rs:F].copy()
        # ignore speech blips shorter than ~100 ms (key clicks) inside silence
        runs = _runs(flags)
        for val, s, ln in runs:
            if val and ln < 3:
                flags[s:s + ln] = False
        runs = _runs(flags)
        need = self.pause_frames if seg_s < 12.0 else max(2, int(self.pause_frames * 0.6))
        min_frames = int(self.min_seg / FRAME_S)
        speech = 0
        cands = []
        for val, s, ln in runs:
            if val:
                speech += ln
            elif speech >= min_frames and ln >= need:
                cands.append((s, ln))
        cut_sample = None
        why = ""
        if cands and elapsed >= self.min_record:
            s, ln = cands[-1]
            cut_sample = (rs + s) * FRAME + min(int(0.30 * SR), (ln * FRAME) // 2)
            why = f"pause {ln * FRAME_S * 1000:.0f} ms"
        elif seg_s > self.max_seg:
            cut_sample = self._lowest_energy_cut(rs * FRAME, F * FRAME)
            why = "forced (lowest energy)"
        if cut_sample is not None and cut_sample - self._cut >= SR:
            self._finalize(self._cut, cut_sample, why)

    def _lowest_energy_cut(self, lo: int, hi: int) -> int:
        with self._lk:
            x = self._buf[max(lo, hi - int(2.5 * SR)):hi].copy()
        base = max(lo, hi - int(2.5 * SR))
        hop = 160
        n = len(x) // hop
        if n < 8:
            return hi
        e = np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(axis=1))
        e = np.convolve(e, np.ones(5) / 5.0, mode="same")
        e[:2] = e[-2:] = np.inf               # keep 100 ms from the edges
        i = int(np.argmin(e))
        return base + i * hop + hop // 2

    def _finalize(self, a: int, b: int, why: str) -> None:
        self._cut = b
        self.n_segments += 1
        log.info("stream: cut #%d at %.1fs (%s), segment %.1f-%.1fs", self.n_segments, b / SR, why, a / SR, b / SR)
        self._stt_q.put(("seg", a, b, None, self._lead_pause_ms(a)))

    def _lead_pause_ms(self, a: int) -> int:
        """Silence before the speech that starts at/after sample ``a`` (includes the ~0.3 s the previous segment
        kept as trailing padding). 0 for the very first segment."""
        if a <= 0:
            return 0
        f0 = int(ceil(a / FRAME))
        fl = self._flags[f0:]
        if fl.size == 0 or not fl.any():
            return 0
        return int(((int(np.argmax(fl)) * FRAME) / SR + 0.30) * 1000)

    # ------------------------------------------------------------------ STT worker
    def _prompt(self) -> str:
        s = self._prompt_src[-self.prompt_chars:]
        if len(self._prompt_src) > self.prompt_chars and " " in s:
            s = s.split(" ", 1)[1]
        return s.strip()

    def _stt_loop(self) -> None:
        while True:
            item = self._stt_q.get()
            try:
                if item is None or self._closed.is_set():
                    return
                self._transcribe_piece(*item)
            except Exception as e:
                self.failed = f"{type(e).__name__}: {e}"
                log.exception("stream STT failed - falling back to single pass")
            finally:
                self._stt_q.task_done()

    def _transcribe_piece(self, kind: str, a: int, b: int, audio: Optional[np.ndarray] = None,
                          pause_ms: int = 0) -> None:
        if audio is None:
            with self._lk:
                audio = self._buf[a:b].copy()
        text, ms = "", 0
        if len(audio) >= int(0.25 * SR) and peak_window_rms(audio) >= self.silence_rms:
            res = self.stt.transcribe(normalize(audio), hotwords=self.words, initial_prompt=self._prompt() or None)
            text, ms = res.text.strip(), res.ms
        text = textproc.apply_replacements(text, self.repl)
        self._stt_total += ms
        if kind == "tail":
            self._tail_ms = ms
            self._tail_pause = pause_ms
        self._texts.append(text)
        self._seg_log.append((a / SR, b / SR, text, ms))
        if text:
            self._prompt_src = (self._prompt_src + " " + text).strip()
        log.info("stream: %s %.1f-%.1fs stt %d ms -> %r", kind, a / SR, b / SR, ms, text[:80])
        if kind == "seg" and text:
            self._on_text(text, background=True, pause_ms=pause_ms)

    # ------------------------------------------------------------------ cleanup units
    def _pending_text(self) -> str:
        return " ".join(t for t, _ in self._pending)

    def _on_text(self, text: str, background: bool, pause_ms: int = 0) -> None:
        with self._ulk:
            if self._pending == [] and self._units and correction_cue(text):
                u = self._units[-1]                    # spoken correction of the previous unit: re-clean together
                u.raw = (u.raw + " " + text).strip()
                u.version += 1
                u.cleaned = None
                self._merged += 1
                log.info("stream: correction cue %r - merging into previous unit", text[:40])
                if background and self.incremental:
                    self._clean_q.put(len(self._units) - 1)
                return
            self._pending.append((text, pause_ms))
            if background and self.incremental:
                joined = self._pending_text()
                if ends_sentence(joined) or len(joined.split()) >= self.max_unit_words:
                    self._flush_pending(submit=True)
                else:                                   # clean the finished sentences now, hold the fragment
                    head, rest = split_complete(joined)
                    if head:
                        pause = self._pending[0][1]
                        self._pending = [(head, pause)]
                        self._flush_pending(submit=True)
                        self._pending = [(rest, 0)]

    def _flush_pending(self, submit: bool) -> None:
        with self._ulk:
            if not self._pending:
                return
            self._units.append(Unit(raw=self._pending_text().strip(), pause_ms=self._pending[0][1]))
            self._pending = []
            if submit:
                self._clean_q.put(len(self._units) - 1)

    def _context_before(self, i: int) -> str:
        with self._ulk:
            prev = " ".join((u.cleaned if u.cleaned is not None else u.raw) for u in self._units[:i])
        prev = prev.strip()
        if len(prev) > self.context_chars:
            prev = prev[-self.context_chars:]
            if " " in prev:
                prev = prev.split(" ", 1)[1]
        return prev

    def _clean_unit(self, i: int) -> None:
        with self._ulk:
            u = self._units[i]
            ver, raw = u.version, u.raw
        ctx = self._context_before(i)
        text, info = self.cleaner.clean(raw, self.sd["name"], self.words, context=ctx, final=False)
        with self._ulk:
            if u.version == ver:
                u.cleaned, u.ms, u.llm, u.reason = text, info["ms"], info["llm"], info["reason"]
        self._bg_clean_ms += info["ms"]

    def _clean_loop(self) -> None:
        while True:
            i = self._clean_q.get()
            try:
                if i is None or self._closed.is_set():
                    return
                self._clean_unit(i)
            except Exception:
                log.exception("stream cleanup failed")
            finally:
                self._clean_q.task_done()

    # ------------------------------------------------------------------ release
    def finish(self, full_audio: np.ndarray) -> StreamResult:
        """Called on release with the complete 16 kHz recording. Transcribes only the tail after the last cut,
        joins the segments and cleans whatever has not been cleaned yet. Raises if the session failed (caller
        falls back to the single pass)."""
        t0 = time.perf_counter()
        self._stop_poll()
        # stop cutting; remember where the last segment ended
        self._vad_stop.set()
        self._wake.set()
        if self._vad_thread is not None:
            self._vad_thread.join(timeout=1.0)
        cut = self._cut
        tail = np.asarray(full_audio[cut:], np.float32)
        self._stt_q.put(("tail", cut, len(full_audio), tail, self._lead_pause_ms(cut)))
        self._stt_q.join()                       # in-flight segment(s), then the tail
        t_stt = time.perf_counter()
        if self.failed:
            self._shutdown_workers()
            raise RuntimeError(f"stream session failed: {self.failed}")
        tail_text = self._texts[-1] if self._texts else ""
        # the segment texts before the tail were handed to cleanup as they arrived; the tail is handled here
        if self.incremental:
            self._clean_q.join()                 # background cleans of earlier units finish first
        with self._ulk:
            if tail_text:
                self._on_text(tail_text, background=False, pause_ms=self._tail_pause)
            self._flush_pending(submit=False)
            todo = [i for i, u in enumerate(self._units) if u.cleaned is None]
        t_clean0 = time.perf_counter()
        for i in todo:
            self._clean_unit_final(i)
        t_end = time.perf_counter()
        self._shutdown_workers()
        with self._ulk:
            parts = [u.cleaned if u.cleaned is not None else u.raw for u in self._units]
            pauses = [u.pause_ms for u in self._units]
            llm = bool(self._units) and all(u.llm for u in self._units)
            reasons = sorted({u.reason for u in self._units if u.reason})
            units = len(self._units)
        joined = join_pieces(parts, pauses if self.paragraphs else None, self.para_pause_ms)
        text = textproc.apply_style(joined, self.sd, self.abbr, self.words)
        raw = " ".join(t for t in self._texts if t).strip()
        res = StreamResult(
            raw=raw, text=text, n_segments=self.n_segments, tail_stt_ms=self._tail_ms, stt_total_ms=self._stt_total,
            wait_ms=int((t_stt - t0) * 1000) - self._tail_ms, cleanup_ms=int((t_end - t_stt) * 1000),
            cleanup_bg_ms=self._bg_clean_ms, llm=llm, reason=", ".join(reasons), units=units, merged=self._merged,
            segments=list(self._seg_log))
        return res

    def _clean_unit_final(self, i: int) -> None:
        self._clean_unit(i)

    def _shutdown_workers(self) -> None:
        self._closed.set()
        self._stt_q.put(None)
        self._clean_q.put(None)


def join_pieces(parts: list[str], pauses: Optional[list[int]] = None, para_pause_ms: int = 900) -> str:
    """Join cleaned pieces. With ``pauses`` (ms of silence before each piece) a long pause after a paragraph that
    already holds >= 3 sentences starts a new paragraph - what the LLM does on its own in a single long pass."""
    out, sentences, words = "", 0, 0
    for k, p in enumerate(parts):
        p = p.strip(" ")
        if not p:
            continue
        brk = bool(pauses) and k > 0 and pauses[k] >= para_pause_ms and sentences >= 3 and words >= 35
        if not out:
            out = p
        elif out.endswith("\n") or p.startswith("\n"):
            out += p
        elif brk:
            out += "\n\n" + p
            sentences = words = 0
        else:
            out += " " + p
        if out.endswith("\n"):
            sentences = words = 0
        sentences += len(re.findall(r"[.!?](?:\s|$)", p))
        words += len(p.split())
    return out
