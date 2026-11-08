"""Background transcription jobs (audio files + meeting recordings) and the TranscriptService façade.

Everything here yields to live dictation: before EVERY unit of GPU work (one <= 30 s STT chunk, one cleanup
LLM request) the job calls ``gate.wait_idle()`` and waits while ``is_paused()`` (game mode / STT not ready / STT
on the CPU fallback). Per-chunk results are persisted in the store, so a crash or quit resumes the job instead
of starting over.
"""
from __future__ import annotations

import bisect
import collections
import difflib
import logging
import os
import re
import threading
import time
from math import gcd
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from . import recorder as rec
from .transcripts import ACTIVE_STATUSES, TranscriptStore, fmt_ts

log = logging.getLogger(__name__)

SR = 16000
CHUNK_MAX_S = 30.0
VAD_BLOCK_S = 600            # VAD runs on 10-minute blocks so a long recording never needs a float32 copy
PARA_GAP_S = 1.5
PARA_WORDS = 80
TURN_GAP_S = 10.0            # same speaker after a longer pause starts a new turn
CLEAN_PIECE_WORDS = 110      # cleanup requests are split to about this size (short GPU units)
ECHO_SLACK_S = 0.3           # timestamp jitter allowance for echo overlap (short replies get none)


LLM_UNAVAILABLE = "llm unavailable"     # _clean_one fallback reason: the AI model never became usable
CLEANUP_UNAVAILABLE_NOTE = "AI cleanup unavailable for part of this transcript"


class JobError(RuntimeError):
    """A job failed for a reason worth showing to the user."""


class _Stopped(Exception):
    """The job was cancelled or the runner is shutting down."""


def _cfg(cfg, key: str, default):
    if cfg is None:
        return default
    try:
        v = cfg.get(key, default)
    except Exception:
        return default
    return default if v is None else v


# ----------------------------------------------------------------------------- audio loading
def probe_duration(path: str) -> Optional[float]:
    try:
        import av
        with av.open(str(path)) as c:
            if c.duration:
                return c.duration / 1_000_000.0
            st = c.streams.audio[0] if c.streams.audio else None
            if st is not None and st.duration and st.time_base:
                return float(st.duration * st.time_base)
    except Exception:
        log.debug("probe failed for %s", path, exc_info=True)
    return None


def decode_file_16k(path: str, max_s: float) -> np.ndarray:
    """Decode any audio/video file to 16 kHz mono int16 (half the RAM of faster_whisper.decode_audio's float32;
    same PyAV pipeline). Refuses files longer than ``max_s``."""
    import av
    dur = probe_duration(path)
    if dur and dur > max_s:
        raise JobError(f"File is too long ({fmt_ts(dur)}); the limit is {fmt_ts(max_s)}.")
    parts: list[np.ndarray] = []
    total = 0
    limit = int(max_s * SR) + SR
    try:
        with av.open(str(path), mode="r") as c:
            if not c.streams.audio:
                raise JobError("The file has no audio track.")
            res = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=SR)
            frames = c.decode(audio=0)
            while True:
                try:
                    frame = next(frames)
                except StopIteration:
                    frame = None
                except av.error.InvalidDataError:
                    continue
                if frame is not None:
                    frame.pts = None
                for f in res.resample(frame):
                    a = f.to_ndarray().reshape(-1).astype(np.int16, copy=True)
                    parts.append(a)
                    total += a.size
                if total > limit:
                    raise JobError(f"File is too long; the limit is {fmt_ts(max_s)}.")
                if frame is None:
                    break
    except JobError:
        raise
    except Exception as e:
        raise JobError(f"Could not read the audio file ({type(e).__name__}: {e}).") from e
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int16)


class _TrackReader:
    """Random access int16 reads from a meeting track (FLAC or raw PCM)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._sf = None
        self._fh = None
        if self.path.suffix.lower() == ".flac":
            import soundfile as sf
            self._sf = sf.SoundFile(str(self.path), "r")
            self.frames = int(self._sf.frames)
        else:
            self._fh = open(self.path, "rb")
            self.frames = os.path.getsize(self.path) // 2

    def read(self, start: int, count: int) -> np.ndarray:
        start = max(0, start)
        count = max(0, min(count, self.frames - start))
        if count == 0:
            return np.zeros(0, dtype=np.int16)
        if self._sf is not None:
            self._sf.seek(start)
            return np.ascontiguousarray(self._sf.read(count, dtype="int16")).reshape(-1)
        self._fh.seek(start * 2)
        return np.frombuffer(self._fh.read(count * 2), dtype=np.int16)

    def close(self) -> None:
        for h in (self._sf, self._fh):
            if h is not None:
                try:
                    h.close()
                except Exception:
                    pass


def load_track_16k(path: Path, rate: int, stop: Optional[threading.Event] = None) -> np.ndarray:
    """Read a meeting track at its native rate and resample to 16 kHz mono int16, in 60 s blocks with a small
    overlap on each side (no seams), so a 4 h recording never sits in RAM at 48 kHz."""
    from scipy.signal import resample_poly
    rate = int(rate)
    r = _TrackReader(path)
    try:
        n = r.frames
        if rate == SR:
            return r.read(0, n).copy()
        g = gcd(SR, rate)
        up, down = SR // g, rate // g
        block = (rate * 60) // down * down
        ctx = max(down, (rate // 10) // down * down)
        out = np.zeros(int(np.ceil(n * up / down)), dtype=np.int16)
        o = 0
        for i in range(0, n, block):
            if stop is not None and stop.is_set():
                raise _Stopped()
            s0 = max(0, i - ctx)
            a = r.read(s0, (i - s0) + block + ctx).astype(np.float32)
            y = resample_poly(a, up, down)
            front = (i - s0) * up // down
            want = min(block, n - i) * up // down
            seg = y[front:front + want]
            k = min(seg.size, out.size - o)
            out[o:o + k] = np.clip(np.round(seg[:k]), -32768, 32767).astype(np.int16)
            o += k
        return out[:o]
    finally:
        r.close()


# ----------------------------------------------------------------------------- chunk planning
def plan_chunks(audio: np.ndarray, max_s: float = CHUNK_MAX_S, vad_fn: Optional[Callable] = None) -> list[list[int]]:
    """Split 16 kHz audio (int16 or float32) into speech chunks of at most ``max_s`` seconds.

    Silero VAD (faster_whisper.vad) finds speech regions (forced to split before ``max_s``); neighbouring regions
    are merged while the chunk stays within ``max_s``. Returns [[start_sample, end_sample], ...] (absolute)."""
    n = int(audio.size)
    if n == 0:
        return []
    max_n = int(max_s * SR)
    if vad_fn is None:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
        opts = VadOptions(threshold=0.5, min_speech_duration_ms=150, max_speech_duration_s=max_s - 1.0,
                          min_silence_duration_ms=500, speech_pad_ms=300)

        def vad_fn(x):
            return get_speech_timestamps(x, opts, sampling_rate=SR)
    regions: list[tuple[int, int]] = []
    blk = VAD_BLOCK_S * SR
    for b0 in range(0, n, blk):
        x = audio[b0:b0 + blk]
        if x.dtype != np.float32:
            x = x.astype(np.float32) / 32768.0
        for t in vad_fn(x):
            s, e = b0 + int(t["start"]), b0 + int(t["end"])
            if e > s:
                regions.append((s, min(e, n)))
    regions.sort()
    # hard-split anything VAD left longer than max_s (guard)
    split: list[tuple[int, int]] = []
    for s, e in regions:
        while e - s > max_n:
            split.append((s, s + max_n))
            s += max_n
        split.append((s, e))
    chunks: list[list[int]] = []
    for s, e in split:
        if chunks and e <= chunks[-1][1]:
            continue                                  # fully inside the previous chunk
        if chunks and s < chunks[-1][1]:
            s = chunks[-1][1]                         # padded regions overlap: never transcribe audio twice
        if chunks and e - chunks[-1][0] <= max_n:
            chunks[-1][1] = e
        else:
            chunks.append([s, e])
    return chunks


# ----------------------------------------------------------------------------- text helpers
_WORD_RE = re.compile(r"[\w']+")


def norm_words(text: str) -> list[str]:
    return [w.strip("'") for w in _WORD_RE.findall((text or "").lower()) if w.strip("'")]


def containment(a: list[str], b: list[str]) -> float:
    """Share of the words of ``a`` that appear, in order, in ``b`` (1.0 = a is a copy of part of b)."""
    if not a or not b:
        return 0.0
    m = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return sum(bl.size for bl in m.get_matching_blocks()) / len(a)


def mark_echo(segments: list[dict], threshold: float = 0.75, slack: float = ECHO_SLACK_S) -> int:
    """Mark (never delete) "You" segments that are just the "Others" audio picked up by the microphone: they
    overlap an Others segment in time (± ``slack``) and their words are >= ``threshold`` contained in the
    overlapping Others words. Sets ``echo`` + ``echo_sim``; returns how many were marked."""
    others = sorted((s for s in segments if s.get("speaker") == "Others"), key=lambda s: s["start"])
    starts = [o["start"] for o in others]
    marked = 0
    for y in segments:
        if y.get("speaker") != "You":
            continue
        y.pop("echo", None)
        y.pop("echo_sim", None)
        yw = norm_words(y["text"])
        if not yw:
            continue
        sl = slack if len(yw) >= 3 else 0.0   # "Yes." right after their "Yes." is a reply, not an echo
        lo = bisect.bisect_left(starts, y["start"] - sl - 60.0)
        hi = bisect.bisect_right(starts, y["end"] + sl)
        ov = [o for o in others[lo:hi] if o["start"] < y["end"] + sl and o["end"] > y["start"] - sl]
        if not ov:
            continue
        ow: list[str] = []
        for o in ov:
            ow += norm_words(o["text"])
        sim = containment(yw, ow)
        if sim >= threshold:
            y["echo"] = True
            y["echo_sim"] = round(sim, 3)
            marked += 1
    return marked


def merge_tracks(you: list, others: list) -> list[dict]:
    segs = [{"start": s, "end": e, "text": t, "speaker": "You"} for s, e, t in you]
    segs += [{"start": s, "end": e, "text": t, "speaker": "Others"} for s, e, t in others]
    segs.sort(key=lambda s: (s["start"], s["end"]))
    return segs


def build_turns(segments: list[dict], include_echo: bool = False, gap_s: float = TURN_GAP_S) -> list[dict]:
    """Consecutive segments of the same speaker -> one turn (echo-marked ones skipped unless included)."""
    turns: list[dict] = []
    for s in sorted(segments, key=lambda s: (s["start"], s["end"])):
        if s.get("echo") and not include_echo:
            continue
        t = turns[-1] if turns else None
        if t and t["speaker"] == s["speaker"] and s["start"] - t["end"] <= gap_s:
            t["end"] = max(t["end"], s["end"])
            t["raw"] = (t["raw"] + " " + s["text"]).strip()
        else:
            turns.append({"speaker": s["speaker"], "start": s["start"], "end": s["end"], "raw": s["text"].strip()})
    return turns


def render_turns(turns: list[dict], cleaned: bool = True) -> str:
    out = []
    for t in turns:
        txt = (t.get("text") if cleaned and t.get("text") else t["raw"]).strip()
        if txt:
            out.append(f"[{fmt_ts(t['start'])}] {t['speaker']}: {txt}")
    return "\n\n".join(out)


def group_paragraphs(segs: list, gap_s: float = PARA_GAP_S, max_words: int = PARA_WORDS) -> list[dict]:
    """File transcripts: start a new paragraph after a pause > ``gap_s`` or at a sentence end once a paragraph
    has ~``max_words`` words (hard break at 1.6x)."""
    paras: list[dict] = []
    for s, e, t in segs:
        t = t.strip()
        if not t:
            continue
        cur = paras[-1] if paras else None
        new = cur is None or s - cur["end"] > gap_s
        if cur and not new:
            w = len(cur["raw"].split())
            new = (w >= max_words and cur["raw"].rstrip()[-1:] in ".?!") or w >= max_words * 1.6
        if new:
            paras.append({"start": s, "end": e, "raw": t})
        else:
            cur["end"] = max(cur["end"], e)
            cur["raw"] += " " + t
    return paras


_SENT_RE = re.compile(r"(?<=[.?!])\s+")


def split_for_cleanup(text: str, max_words: int = CLEAN_PIECE_WORDS) -> list[str]:
    """Split a long turn/paragraph at sentence ends into pieces of about ``max_words`` words."""
    words = text.split()
    if len(words) <= max_words * 1.3:
        return [text]
    pieces, cur = [], []
    for sent in _SENT_RE.split(text):
        sw = sent.split()
        if cur and len(cur) + len(sw) > max_words:
            pieces.append(" ".join(cur))
            cur = []
        cur += sw
        while len(cur) > max_words * 1.6:            # one enormous "sentence" without punctuation
            pieces.append(" ".join(cur[:max_words]))
            cur = cur[max_words:]
    if cur:
        pieces.append(" ".join(cur))
    return pieces


# ----------------------------------------------------------------------------- job runner
class _Job:
    def __init__(self, tid: int):
        self.tid = tid
        self.stop = threading.Event()
        self.status = "queued"
        self.llm_waited = 0.0           # seconds spent waiting for a not-ready LLM (capped by assistant.wait_ready_s)
        self.llm_gave_up = False        # cap reached: later pieces use the regex fallback without waiting
        self.llm_unavailable = 0        # pieces cleaned by the regex fallback because the LLM was unavailable


class JobRunner:
    """Single background thread, FIFO queue of transcript ids (the job spec lives in the store row)."""

    def __init__(self, stt, cleaner, gate, store: TranscriptStore, is_paused: Callable[[], bool],
                 on_progress: Optional[Callable[[int, str, float, str], None]] = None,
                 on_done: Optional[Callable[[int, str], None]] = None, cfg=None):
        self.stt = stt
        self.cleaner = cleaner
        self.gate = gate
        self.store = store
        self.is_paused = is_paused or (lambda: False)
        self.on_progress = on_progress
        self.on_done = on_done
        self.cfg = cfg
        self._cv = threading.Condition()
        self._queue: collections.deque[int] = collections.deque()
        self._current: Optional[_Job] = None
        self._shutdown = threading.Event()
        self.max_chunk_s = 0.0          # worst-case STT chunk wall time seen (for tuning)
        self.retry_sleep_s = 2.0
        self._thread = threading.Thread(target=self._loop, name="transcript-jobs", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ queue API
    def submit(self, tid: int) -> None:
        with self._cv:
            if tid in self._queue or (self._current and self._current.tid == tid):
                return
            self._queue.append(int(tid))
            self._cv.notify_all()

    def pending(self) -> list[int]:
        with self._cv:
            cur = [self._current.tid] if self._current else []
            return cur + list(self._queue)

    def current(self) -> Optional[int]:
        with self._cv:
            return self._current.tid if self._current else None

    def cancel(self, tid: int) -> bool:
        with self._cv:
            if tid in self._queue:
                self._queue.remove(tid)
                queued = True
            else:
                queued = False
            cur = self._current
        if queued:
            self.store.update(tid, status="cancelled")
            self.store.clear_progress(tid)
            self._emit_done(tid, "cancelled")
            return True
        if cur and cur.tid == tid:
            cur.stop.set()
            return True
        return False

    def shutdown(self, timeout: float = 15.0) -> None:
        self._shutdown.set()
        with self._cv:
            if self._current:
                self._current.stop.set()
            self._cv.notify_all()
        self._thread.join(timeout)

    # ------------------------------------------------------------------ loop
    def _loop(self) -> None:
        while not self._shutdown.is_set():
            with self._cv:
                while not self._queue and not self._shutdown.is_set():
                    self._cv.wait(0.5)
                if self._shutdown.is_set():
                    return
                job = _Job(self._queue.popleft())
                self._current = job
            try:
                self._run(job)
            except Exception:
                log.exception("job runner crashed on %s", job.tid)
            finally:
                with self._cv:
                    self._current = None

    def _emit(self, job: _Job, status: str, progress: float, message: str = "") -> None:
        job.status = status
        if self.on_progress:
            try:
                self.on_progress(job.tid, status, float(progress), message)
            except Exception:
                log.debug("on_progress failed", exc_info=True)

    def _emit_done(self, tid: int, status: str) -> None:
        if self.on_done:
            try:
                self.on_done(tid, status)
            except Exception:
                log.debug("on_done failed", exc_info=True)

    def _run(self, job: _Job) -> None:
        row = self.store.get(job.tid)
        if row is None or row["status"] in ("done", "cancelled"):
            # deleted / finished meanwhile: still report, so deferred clean-ups (pending deletes) run
            self._emit_done(job.tid, row["status"] if row else "deleted")
            return
        audio_dir = row.get("audio_dir") or ""
        try:
            self.store.update(job.tid, status="transcribing", error="")
            if audio_dir and row["kind"] == "meeting":
                rec.set_state(Path(audio_dir), "transcribing")
            self._emit(job, "transcribing", row.get("progress") or 0.0, "Starting")
            if row["kind"] == "file":
                self._run_file(job, row)
            else:
                self._run_meeting(job, row)
            if job.stop.is_set():
                raise _Stopped()
            if audio_dir and row["kind"] == "meeting":
                rec.set_state(Path(audio_dir), "done")
            self.store.clear_progress(job.tid)
            note = CLEANUP_UNAVAILABLE_NOTE if job.llm_unavailable else ""
            self.store.update(job.tid, status="done", progress=1.0, error=note)
            self._emit(job, "done", 1.0, note)
            self._emit_done(job.tid, "done")
        except _Stopped:
            if self._shutdown.is_set():
                self.store.update(job.tid, status="interrupted")      # resumes on next start
                status = "interrupted"
            else:
                self.store.update(job.tid, status="cancelled")
                self.store.clear_progress(job.tid)
                status = "cancelled"
            if audio_dir and row["kind"] == "meeting":
                rec.set_state(Path(audio_dir), "interrupted")
            self._emit(job, status, 0.0, "")
            self._emit_done(job.tid, status)
        except Exception as e:
            msg = str(e) if isinstance(e, JobError) else f"{type(e).__name__}: {e}"
            log.exception("transcript job %s failed", job.tid)
            self.store.update(job.tid, status="failed", error=msg)
            if audio_dir and row["kind"] == "meeting":
                rec.set_state(Path(audio_dir), "failed")
            self._emit(job, "failed", 0.0, msg)
            self._emit_done(job.tid, "failed")

    # ------------------------------------------------------------------ waiting
    def _sleep(self, job: _Job, s: float) -> None:
        if job.stop.wait(s):
            raise _Stopped()

    def _wait_turn(self, job: _Job, status: str, progress: float) -> None:
        """Block until this job may use the GPU: not paused, and no foreground dictation (plus settle time)."""
        announced = False
        while True:
            if job.stop.is_set():
                raise _Stopped()
            if self._paused():
                if not announced:
                    self._emit(job, status, progress, "Paused (speech model busy or unavailable)")
                    announced = True
                self._sleep(job, 1.0)
                continue
            if not self.gate.wait_idle(stop=job.stop):
                raise _Stopped()
            if not self._paused():         # may have flipped while we waited for the dictation
                return

    def _paused(self) -> bool:
        try:
            return bool(self.is_paused())
        except Exception:
            return False

    # ------------------------------------------------------------------ STT
    def _stt_chunk(self, job: _Job, chunk: np.ndarray, offset_s: float, status: str, progress: float) -> list:
        from .stt import ModelUnloaded
        lang = str(_cfg(self.cfg, "transcripts.language", "") or "") or None
        failures = 0
        while True:
            self._wait_turn(job, status, progress)
            try:
                t0 = time.perf_counter()
                out = self.stt.transcribe_segments(chunk, offset_s, language=lang)
                dt = time.perf_counter() - t0
                if dt > self.max_chunk_s:
                    self.max_chunk_s = dt
                    log.info("transcript job: slowest STT chunk so far %.2f s (%.1f s audio)", dt, chunk.size / SR)
                return [list(s) for s in out]
            except ModelUnloaded:
                self._emit(job, status, progress, "Waiting for the speech model")
                self._sleep(job, self.retry_sleep_s)
            except _Stopped:
                raise
            except Exception as e:
                failures += 1
                log.warning("STT chunk failed (%d): %s: %s", failures, type(e).__name__, e)
                if failures >= 3:
                    raise JobError(f"Speech recognition failed: {type(e).__name__}: {e}") from e
                self._sleep(job, self.retry_sleep_s)

    def _transcribe(self, job: _Job, key: str, audio: np.ndarray, offset_s: float, done: dict,
                    p_lo: float, p_hi: float, label: str) -> list:
        """Transcribe one 16 kHz track chunk by chunk, persisting each chunk's segments under ``stt:<key>:<i>``."""
        plan = done.get(f"plan:{key}")
        if plan is None:
            self._emit(job, "transcribing", p_lo, f"Finding speech{label}")
            plan = plan_chunks(audio)
            self.store.save_progress(job.tid, f"plan:{key}", plan)
        segs: list = []
        n = max(1, len(plan))
        for i, (s, e) in enumerate(plan):
            k = f"stt:{key}:{i}"
            prog = p_lo + (p_hi - p_lo) * i / n
            if k in done:
                segs += done[k]
                continue
            self._emit(job, "transcribing", prog, f"Transcribing{label} {i + 1}/{n}")
            chunk = audio[s:e].astype(np.float32) / 32768.0
            out = self._stt_chunk(job, chunk, offset_s + s / SR, "transcribing", prog)
            self.store.save_progress(job.tid, k, out)
            self.store.update(job.tid, progress=round(p_lo + (p_hi - p_lo) * (i + 1) / n, 4))
            segs += out
        segs.sort(key=lambda x: (x[0], x[1]))
        return segs

    # ------------------------------------------------------------------ cleanup (LLM)
    def _clean_one(self, job: _Job, raw: str, style: str, context: str, progress: float) -> tuple[str, str]:
        """Returns (text, fallback_reason or ''). Waits while the LLM is suspended / not ready instead of
        silently accepting the regex fallback, but at most ``assistant.wait_ready_s`` (default 900 s) per job:
        after that this piece and every later not-ready piece of the job get the regex fallback right away
        (reason ``LLM_UNAVAILABLE``). Connection errors retry a few times. Time spent paused in ``_wait_turn``
        (game mode, dictation, speech model loading) does not count toward the cap."""
        errors = 0
        cap = float(_cfg(self.cfg, "assistant.wait_ready_s", 900) or 0)
        while True:
            self._wait_turn(job, "cleaning", progress)
            text, info = self.cleaner.clean(raw, style=style, context=context, final=True)
            if info.get("llm"):
                return text, ""
            reason = str(info.get("reason", ""))
            if reason in ("game mode", "llm not ready"):
                if job.llm_gave_up or job.llm_waited >= cap:
                    if not job.llm_gave_up:
                        log.warning("transcript %s: AI model not ready after %.0f s - using basic cleanup for the "
                                    "rest of this job", job.tid, job.llm_waited)
                    job.llm_gave_up = True
                    return text, LLM_UNAVAILABLE
                self._emit(job, "cleaning", progress, "Waiting for the AI model")
                step = max(0.0, min(3.0, cap - job.llm_waited))
                self._sleep(job, step)
                job.llm_waited += step
                continue
            if reason.startswith("error"):
                errors += 1
                # connection refused/reset = llama-server (re)starting: give it ~60 s; other errors 3 tries.
                # Retry sleeps share the job's wait budget, so a dead server never costs more than the cap.
                if not job.llm_gave_up and job.llm_waited < cap and \
                        errors < (20 if "ConnectionError" in reason else 3):
                    if errors == 1:
                        self._emit(job, "cleaning", progress, "Waiting for the AI model")
                    step = min(3.0, cap - job.llm_waited)
                    self._sleep(job, step)
                    job.llm_waited += step
                    continue
                if job.llm_waited >= cap:
                    job.llm_gave_up = True
                return text, LLM_UNAVAILABLE
            return text, reason or "fallback"

    def _clean_pieces(self, job: _Job, texts: list[str], style: str, done: dict,
                      p_lo: float, p_hi: float) -> tuple[list[str], int]:
        """Clean each text (split into short requests), persisting results as ``clean:<i>``."""
        out: list[str] = []
        fallbacks = 0
        prev = ""
        n = max(1, len(texts))
        for i, raw in enumerate(texts):
            k = f"clean:{i}"
            prog = p_lo + (p_hi - p_lo) * i / n
            if k in done:
                text, fb = done[k][0], done[k][1]
                job.llm_unavailable += int(done[k][2]) if len(done[k]) > 2 else 0
            else:
                self._emit(job, "cleaning", prog, f"Cleaning up {i + 1}/{n}")
                parts, fb, unavail = [], 0, 0
                for piece in split_for_cleanup(raw):
                    t, why = self._clean_one(job, piece, style, prev[-400:], prog)
                    fb += 1 if why and why not in ("short utterance", "cleanup disabled") else 0
                    unavail += 1 if why == LLM_UNAVAILABLE else 0
                    parts.append(t.strip())
                    prev = t
                text = " ".join(p for p in parts if p)
                job.llm_unavailable += unavail
                self.store.save_progress(job.tid, k, [text, fb, unavail])
                self.store.update(job.tid, progress=round(p_lo + (p_hi - p_lo) * (i + 1) / n, 4))
            prev = text or prev
            fallbacks += fb
            out.append(text)
        return out, fallbacks

    # ------------------------------------------------------------------ file job
    def _run_file(self, job: _Job, row: dict) -> None:
        path = row["source_path"]
        if not path or not os.path.isfile(path):
            raise JobError(f"Audio file not found: {path}")
        max_s = float(_cfg(self.cfg, "transcripts.max_file_hours", 4)) * 3600
        done = self.store.load_progress(job.tid)
        st = os.stat(path)
        sig = [st.st_size, st.st_mtime_ns]
        if done and done.get("source") != sig:
            log.info("transcript %s: source file changed since the last run - starting over", job.tid)
            self.store.clear_progress(job.tid)
            done = {}
        if not done:
            self.store.save_progress(job.tid, "source", sig)
        self._emit(job, "transcribing", 0.0, "Reading audio")
        audio = decode_file_16k(path, max_s)
        if job.stop.is_set():
            raise _Stopped()
        duration = audio.size / SR
        self.store.update(job.tid, duration_s=round(duration, 2))
        style = row.get("style") or "raw"
        p_hi = 0.95 if style == "raw" else 0.6
        segs = self._transcribe(job, "file", audio, 0.0, done, 0.0, p_hi, "")
        del audio
        paras = group_paragraphs(segs)
        raw_text = "\n\n".join(p["raw"] for p in paras)
        fallbacks = 0
        if style != "raw" and paras:
            self.store.update(job.tid, status="cleaning", raw_text=raw_text)
            cleaned, fallbacks = self._clean_pieces(job, [p["raw"] for p in paras], style, done, p_hi, 0.99)
            for p, c in zip(paras, cleaned):
                p["text"] = c
            text = "\n\n".join(p.get("text") or p["raw"] for p in paras)
        else:
            text = raw_text
        seg_json = {"version": 1, "kind": "file", "style": style,
                    "segments": [{"start": s, "end": e, "text": t} for s, e, t in segs],
                    "paragraphs": paras}
        if job.stop.is_set():
            raise _Stopped()
        self.store.update(job.tid, raw_text=raw_text, text=text, segments_json=seg_json)
        self.store.update_info(job.tid, cleanup_fallbacks=fallbacks or None,
                               no_speech=True if not segs else None)

    # ------------------------------------------------------------------ meeting job
    def _run_meeting(self, job: _Job, row: dict) -> None:
        d = Path(row["audio_dir"])
        meta = rec.read_meta(d)
        if meta is None:
            raise JobError(f"Recording not found: {d}")
        if any((d / f"{k}.pcm").exists() for k in rec.TRACKS):
            self._emit(job, "transcribing", 0.0, "Saving audio")
            meta = rec.finalize_dir(d)
            rec.set_state(d, "transcribing")
        done = self.store.load_progress(job.tid)
        tracks = meta.get("tracks") or {}
        self.store.update(job.tid, duration_s=float(meta.get("duration_s") or 0))
        style = row.get("style") or "raw"
        p_end = 0.95 if style == "raw" else 0.7
        results: dict[str, list] = {}
        keys = [k for k in rec.TRACKS if k in tracks]
        for ti, key in enumerate(keys):
            tr = tracks[key]
            f = tr.get("file") or ""
            path = d / f if f else None
            lo = p_end * ti / max(1, len(keys))
            hi = p_end * (ti + 1) / max(1, len(keys))
            label = f" ({rec.SPEAKERS.get(key, key)})"
            if path is None or not path.exists() or not (tr.get("samples") or path.stat().st_size > 64):
                results[key] = []
                continue
            plan_key = f"plan:{key}"
            all_done = plan_key in done and all(f"stt:{key}:{i}" in done for i in range(len(done[plan_key])))
            if all_done:
                audio = np.zeros(0, dtype=np.int16)      # everything cached: no need to load the audio
            else:
                self._emit(job, "transcribing", lo, f"Loading audio{label}")
                audio = load_track_16k(path, int(tr.get("rate") or 48000), stop=job.stop)
            results[key] = self._transcribe(job, key, audio, float(tr.get("offset_s") or 0.0), done, lo, hi, label)
            del audio
        segs = merge_tracks(results.get("mic", []), results.get("system", []))
        thr = float(_cfg(self.cfg, "meetings.echo_similarity", 0.75))
        echoes = mark_echo(segs, thr)
        turns = build_turns(segs)
        raw_text = render_turns(turns, cleaned=False)
        fallbacks = 0
        if style != "raw" and turns:
            self.store.update(job.tid, status="cleaning", raw_text=raw_text)
            cleaned, fallbacks = self._clean_pieces(job, [t["raw"] for t in turns], style, done, p_end, 0.99)
            for t, c in zip(turns, cleaned):
                t["text"] = c
        text = render_turns(turns, cleaned=True)
        seg_json = {"version": 1, "kind": "meeting", "style": style, "echo_threshold": thr,
                    "segments": segs, "turns": turns}
        if job.stop.is_set():
            raise _Stopped()
        self.store.update(job.tid, raw_text=raw_text, text=text, segments_json=seg_json, include_echo=0)
        self.store.update_info(job.tid, echo_marked=echoes, cleanup_fallbacks=fallbacks or None,
                               no_speech=True if not segs else None)


def render_meeting(seg_json: dict, include_echo: bool) -> str:
    """Re-render a meeting transcript with or without the echo-marked "You" segments. The cleaned turns are
    used when echo is hidden (that is what was cleaned); with echo shown the raw text is rendered."""
    if not seg_json:
        return ""
    if not include_echo and seg_json.get("turns"):
        return render_turns(seg_json["turns"], cleaned=True)
    return render_turns(build_turns(seg_json.get("segments") or [], include_echo=include_echo), cleaned=False)


# ----------------------------------------------------------------------------- service façade
class AskHandle:
    def __init__(self, tid: int = 0):
        self.tid = int(tid)
        self.stop = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.result: Optional[dict] = None
        self.error: str = ""

    def cancel(self) -> None:
        self.stop.set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        if self.thread is None:
            return True
        self.thread.join(timeout)
        return not self.thread.is_alive()


class TranscriptService:
    """What the UI talks to. All callbacks are plain callables invoked from background threads (bridge them to
    Qt signals): ``on_progress(tid, status, progress, message)``, ``on_done(tid, status)``,
    ``on_meeting(state_dict)``, ``on_levels(mic_rms, system_rms)``."""

    def __init__(self, cfg, stt, cleaner, gate, is_paused: Callable[[], bool] = lambda: False, *,
                 on_progress=None, on_done=None, on_meeting=None, on_levels=None,
                 db_path: Optional[Path] = None, recordings_root: Optional[Path] = None,
                 base_url_fn: Optional[Callable[[], str]] = None,
                 llm_ready_fn: Optional[Callable[[], bool]] = None,
                 request_extra_fn: Optional[Callable[[], dict]] = None):
        from .assistant import Assistant
        self.cfg = cfg
        self.on_progress = on_progress
        self.on_done = on_done
        self.on_meeting = on_meeting
        self.on_levels = on_levels
        self.store = TranscriptStore(db_path, recordings_root)
        self.root = self.store.recordings_root
        self.root.mkdir(parents=True, exist_ok=True)
        self.runner = JobRunner(stt, cleaner, gate, self.store, is_paused,
                                on_progress=self._progress, on_done=self._done, cfg=cfg)
        self.recorder = rec.MeetingRecorder(cfg, on_level=self._levels, on_auto_stop=self._auto_stopped)
        if base_url_fn is None:
            base_url_fn = lambda: cleaner.server.base_url          # noqa: E731
        if llm_ready_fn is None:
            def llm_ready_fn():
                s = cleaner.server
                return bool(s.ready) and not bool(getattr(s, "suspended", False))
        self.assistant = Assistant(cfg.get, gate, self.store, base_url_fn, llm_ready_fn,
                                   request_extra_fn=request_extra_fn)
        self._ask_lock = threading.Lock()
        self._meeting_lock = threading.Lock()
        self._asks: set = set()
        self._asks_lock = threading.Lock()
        self._pending_delete: dict[int, str] = {}       # tid -> audio dir to remove once its job has ended
        self._del_lock = threading.Lock()
        self.last_stop_reason = ""

    # ------------------------------------------------------------------ callbacks
    def _progress(self, tid, status, progress, message):
        if self.on_progress:
            self.on_progress(tid, status, progress, message)

    def _finish_pending_delete(self, tid: int) -> bool:
        with self._del_lock:
            d = self._pending_delete.pop(int(tid), None)
        if d is None:
            return False
        self.store.clear_progress(tid)
        self.store.remove_audio_dir(d)
        return True

    def _done(self, tid, status):
        if self._finish_pending_delete(tid):      # deleted while its job was still running
            return
        if self.on_done:
            self.on_done(tid, status)

    def _levels(self, m, s):
        if self.on_levels:
            self.on_levels(m, s)

    def _notify_meeting(self) -> None:
        if self.on_meeting:
            try:
                self.on_meeting(self.meeting_state())
            except Exception:
                log.debug("on_meeting failed", exc_info=True)

    # ------------------------------------------------------------------ meetings
    def meeting_state(self) -> dict:
        r = self.recorder
        m, s = r.levels
        return {"recording": r.recording, "elapsed": r.elapsed if r.recording else 0.0,
                "levels": (m, s), "dir": str(r.rec_dir) if r.rec_dir else "", "devices": r.devices(),
                "error": r.error, "stop_reason": self.last_stop_reason}

    def _new_rec_dir(self) -> Path:
        base = time.strftime("%Y%m%d-%H%M%S")
        d = self.root / base
        i = 2
        while d.exists():
            d = self.root / f"{base}-{i}"
            i += 1
        return d

    def start_meeting(self) -> dict:
        """Raises recorder.RecorderError (message is user-presentable)."""
        with self._meeting_lock:
            self.last_stop_reason = ""
            self.recorder.start(self._new_rec_dir())
        self._notify_meeting()
        return self.meeting_state()

    def _create_meeting_row(self, meta: dict, status: str = "queued") -> Optional[int]:
        """One transcript row per recording dir (``meta["dir"]``); returns the new id, or None if that
        recording already has a row."""
        d = meta.get("dir")
        if not d:
            return None
        if self.store.find_by_audio_dir(str(d)) is not None:
            return None
        t0 = float(meta.get("t0") or time.time())
        title = "Meeting " + time.strftime("%Y-%m-%d %H:%M", time.localtime(t0))
        return self.store.create("meeting", title, audio_dir=str(d), duration_s=float(meta.get("duration_s") or 0),
                                 style=str(_cfg(self.cfg, "meetings.style", "raw") or "raw"), status=status,
                                 created=t0)

    def stop_meeting(self) -> Optional[int]:
        """Stop and queue the transcription. Returns the new transcript id (None if nothing was recording).
        FLAC conversion happens on the job thread (first step of the meeting job)."""
        with self._meeting_lock:
            if not self.recorder.recording:
                return None
            meta = self.recorder.stop(finalize=False, reason="stopped by user")
            if not meta.get("stopped_now") or meta.get("writer_busy"):
                return None
            tid = self._create_meeting_row(meta)
        if tid is not None:
            self.runner.submit(tid)
        self._notify_meeting()
        return tid

    def _auto_stopped(self, meta: dict, reason: str) -> None:
        self.last_stop_reason = reason
        if meta.get("writer_busy"):
            self._notify_meeting()
            return
        with self._meeting_lock:
            tid = self._create_meeting_row(meta)
        if tid is not None:
            self.runner.submit(tid)
        self._notify_meeting()

    # ------------------------------------------------------------------ files
    def import_file(self, path: str, style: Optional[str] = None) -> int:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(path)
        style = style or str(_cfg(self.cfg, "transcripts.file_style", "raw") or "raw")
        dur = probe_duration(str(p)) or 0.0
        max_s = float(_cfg(self.cfg, "transcripts.max_file_hours", 4)) * 3600
        if dur > max_s:
            raise JobError(f"File is too long ({fmt_ts(dur)}); the limit is {fmt_ts(max_s)}.")
        tid = self.store.create("file", p.stem, source_path=str(p.resolve()), style=style, duration_s=dur)
        self.runner.submit(tid)
        return tid

    # ------------------------------------------------------------------ jobs
    def cancel(self, tid: int) -> bool:
        return self.runner.cancel(int(tid))

    def resume(self, tid: int) -> bool:
        row = self.store.get(tid)
        if row is None or row["status"] not in ("interrupted", "failed", "cancelled", "queued"):
            return False
        self.store.update(tid, status="queued", error="")
        self.runner.submit(int(tid))
        return True

    def recover(self, auto_resume: Optional[bool] = None) -> list[int]:
        """Startup recovery (call once, from a background thread: it may convert PCM to FLAC). Interrupted
        recordings get a transcript row; jobs that were running when the app died become ``interrupted`` and,
        with ``transcripts.auto_resume`` (default on), are queued again (they continue from saved progress)."""
        with self._meeting_lock:              # no meeting can start/stop while recordings are reconciled
            self._recover_dirs()
        running = set(self.runner.pending())
        for tid in self.store.ids_with_status(ACTIVE_STATUSES):
            if tid not in running:
                self.store.update(tid, status="interrupted")
        if auto_resume is None:
            auto_resume = bool(_cfg(self.cfg, "transcripts.auto_resume", True))
        ids = self.store.ids_with_status(["interrupted"])
        if auto_resume:
            for tid in ids:
                self.resume(tid)
        return ids

    def _recover_dirs(self) -> None:
        skip = self.recorder.rec_dir if self.recorder.recording else None
        busy = [skip] if skip else []
        for tid in self.runner.pending():         # never touch a recording a job is working on
            row = self.store.get(tid)
            if row and row.get("audio_dir"):
                busy.append(row["audio_dir"])
        for d in [x for x in self.root.iterdir() if x.is_dir() and (x / TranscriptStore.DELETED_MARKER).exists()]:
            self.store.remove_audio_dir(str(d))            # a delete that Windows file locking blocked earlier
        for d, meta in rec.recover_interrupted(self.root, skip=busy):
            row = self.store.find_by_audio_dir(str(d))
            if row is not None and row["status"] in ACTIVE_STATUSES:
                self.store.update(row["id"], status="interrupted")
        # every finished recording needs a row (e.g. the app died between stopping and creating the row)
        for d in sorted(x for x in self.root.iterdir() if x.is_dir()):
            if any(d.resolve() == Path(b).resolve() for b in busy):
                continue
            if (d / TranscriptStore.DELETED_MARKER).exists():
                continue
            meta = rec.read_meta(d)
            if meta is None or meta.get("state") in ("recording", "stopping"):
                continue
            if self.store.find_by_audio_dir(str(d)) is None:
                meta = dict(meta)
                meta["dir"] = str(d)
                self._create_meeting_row(meta, status="interrupted")

    # ------------------------------------------------------------------ store passthrough
    def list(self, search: str = "") -> list[dict]:
        return self.store.list(search)

    def get(self, tid: int) -> Optional[dict]:
        return self.store.get(tid)

    def rename(self, tid: int, title: str) -> None:
        self.store.rename(tid, title)

    def delete(self, tid: int) -> bool:
        """Delete the transcript, its AI notes and (for meetings) its audio. If its job is still running, the
        row goes now and the audio is removed when the job has let go of the files."""
        tid = int(tid)
        row = self.store.get(tid)
        if row is None:
            return False
        with self._asks_lock:
            for h in self._asks:
                if h.tid == tid:
                    h.cancel()
        self.runner.cancel(tid)
        for _ in range(50):                       # give a running job a moment to notice
            if self.runner.current() != tid:
                break
            time.sleep(0.1)
        if self.runner.current() == tid:
            with self._del_lock:
                self._pending_delete[tid] = row.get("audio_dir") or ""
            ok = self.store.delete(tid, delete_audio=False)     # marks the dir deleted (durable) first

            def watch():                          # the job may have finished between the check and now
                while self.runner.current() == tid:
                    time.sleep(0.2)
                self._finish_pending_delete(tid)
            threading.Thread(target=watch, name="transcript-delete", daemon=True).start()
            return ok
        return self.store.delete(tid, delete_audio=True)

    def export(self, tid: int, fmt: str = "txt", path: Optional[str] = None) -> str:
        text = self.store.export_text(tid, fmt)
        if path:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
        return text

    def set_include_echo(self, tid: int, include: bool) -> str:
        row = self.store.get(tid)
        if row is None or row["kind"] != "meeting":
            return ""
        text = render_meeting(self.store.segments(tid), bool(include))
        self.store.update(tid, include_echo=1 if include else 0, text=text)
        return text

    def list_ai_results(self, tid: int) -> list[dict]:
        return self.store.list_ai_results(tid)

    def delete_ai_result(self, result_id: int) -> None:
        self.store.delete_ai_result(result_id)

    # ------------------------------------------------------------------ assistant
    def ask(self, tid: int, preset: str, prompt: Optional[str] = None,
            on_partial: Optional[Callable[[str], None]] = None,
            on_done: Optional[Callable[[Optional[dict], str], None]] = None,
            on_status: Optional[Callable[[str], None]] = None) -> AskHandle:
        """Run an AI preset (summary / bullets / key_points / action_items) or ``custom`` with ``prompt`` in the
        background. ``on_partial(text_so_far)`` streams the answer; ``on_done(result_row or None, error)``."""
        h = AskHandle(tid)
        with self._asks_lock:
            self._asks.add(h)

        def run():
            with self._ask_lock:            # one AI request at a time (llama-server runs --parallel 1)
                try:
                    what = prompt if preset == "custom" else preset
                    out = self.assistant.ask(tid, what, on_partial=on_partial, stop=h.stop, on_status=on_status)
                    rows = self.store.list_ai_results(tid)
                    h.result = rows[0] if rows else {"output": out}
                except Exception as e:
                    h.error = str(e) or type(e).__name__
                    if not h.stop.is_set():
                        log.warning("assistant failed: %s", h.error)
            with self._asks_lock:
                self._asks.discard(h)
            if on_done:
                try:
                    on_done(h.result, h.error)
                except Exception:
                    log.debug("ask on_done failed", exc_info=True)
        h.thread = threading.Thread(target=run, name="transcript-ask", daemon=True)
        h.thread.start()
        return h

    # ------------------------------------------------------------------ lifecycle
    FINALIZE_ON_QUIT_S = 20 * 60      # longer recordings are converted to FLAC by recovery on the next start

    def shutdown(self, timeout: float = 15.0) -> None:
        """App quit: stop + save a running recording (queued for transcription on next start; converted to FLAC
        now when it is short enough to do quickly), cancel AI requests, interrupt the running job (it resumes
        from saved progress)."""
        with self._meeting_lock:
            if self.recorder.recording:
                try:
                    quick = self.recorder.elapsed <= self.FINALIZE_ON_QUIT_S
                    meta = self.recorder.stop(finalize=quick, reason="app closed")
                    if meta.get("stopped_now") and not meta.get("writer_busy"):
                        self._create_meeting_row(meta)
                except Exception:
                    log.exception("stopping the meeting recording on quit failed")
        with self._asks_lock:
            asks = list(self._asks)
        for h in asks:
            h.cancel()
        self.runner.shutdown(timeout)
        for h in asks:
            h.wait(5)
        try:
            self.assistant.http.close()
        except Exception:
            pass
        if self.runner._thread.is_alive() or any(h.thread and h.thread.is_alive() for h in asks):
            log.warning("transcript workers still busy at shutdown - leaving the database open")
            return
        self.store.close()
