"""The dictation pipeline: audio -> STT -> dictionary/snippets -> AI cleanup -> insertion -> history.

Qt-free so it can be exercised from tests and scripts.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Optional

import numpy as np

from . import styles, textproc
from .audioutil import normalize, peak_window_rms   # noqa: F401  (re-exported for older callers/tests)
from .cleanup import Cleaner
from .history import History
from .stt import Transcriber

log = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, cfg, stt: Transcriber, cleaner: Cleaner, history: Optional[History],
                 inserter: Callable[..., dict]):
        self.cfg, self.stt, self.cleaner, self.history, self.inserter = cfg, stt, cleaner, history, inserter

    def process(self, audio: np.ndarray, app_exe: str = "", app_title: str = "",
                t_release: Optional[float] = None, insert: bool = True,
                style_ctx: Optional[styles.StyleContext] = None, session=None) -> dict:
        """Run one dictation. ``style_ctx`` is the style resolved when recording STARTED (falls back to resolving
        from app_exe/app_title). ``session`` is an optional streaming.StreamSession that has been transcribing
        while the user spoke; when it already finalized segments only the tail is transcribed here."""
        t_release = t_release or time.perf_counter()
        cfg = self.cfg
        sctx = style_ctx or styles.resolve(app_exe, app_title, cfg)
        app_exe = app_exe or sctx.exe
        app_title = app_title or sctx.title
        r: dict = {"status": "ok", "raw": "", "text": "", "app": app_exe, "title": app_title, "style": sctx.style,
                   "rule": sctx.rule, "stt_ms": 0, "cleanup_ms": 0, "insert_ms": 0, "total_ms": 0, "llm": False,
                   "reason": "", "segments": 1, "streamed": False}
        dur = len(audio) / 16000.0
        r["duration_s"] = dur
        if dur < float(cfg.get("audio.min_duration_s", 0.3)):
            self._drop_session(session)
            r.update(status="discard", reason=f"too short ({dur:.2f}s)")
            return r
        if peak_window_rms(audio) < float(cfg.get("audio.silence_rms", 0.006)):
            self._drop_session(session)
            r.update(status="discard", reason="silence")
            return r
        dic = cfg.get("dictionary", {})
        words = [w for w in dic.get("words", []) if w.strip()]
        snippets = cfg.get("snippets", {})
        abbr = cfg.get("abbreviations", styles.DEFAULT_ABBREVIATIONS)

        streamed = None
        if session is not None:
            if session.n_segments > 0 and not session.failed:
                try:
                    streamed = session.finish(audio)
                except Exception:
                    log.exception("streaming finish failed - redoing as a single pass")
                    self._drop_session(session)
            else:
                self._drop_session(session)

        if streamed is not None:
            r.update(streamed=True, segments=streamed.n_segments + 1, stt_ms=streamed.tail_stt_ms,
                     stt_total_ms=streamed.stt_total_ms, cleanup_ms=streamed.cleanup_ms,
                     wait_ms=streamed.wait_ms, cleanup_bg_ms=streamed.cleanup_bg_ms, llm=streamed.llm,
                     reason=streamed.reason, merged=streamed.merged, units=streamed.units)
            raw = streamed.raw
            r["raw"] = raw
            if not raw:
                r.update(status="discard", reason="empty transcript")
                return r
            text = textproc.apply_snippets_inline(streamed.text, snippets)
            text = textproc.apply_replacements(text, dic.get("replacements", {}))
        else:
            audio = normalize(audio)
            res = self.stt.transcribe(audio, hotwords=words)
            r["stt_ms"] = r["stt_total_ms"] = res.ms
            if not res.text:
                r.update(status="discard", reason=res.dropped or "empty transcript")
                return r
            raw = res.text.strip()
            r["raw"] = raw
            raw_fixed = textproc.apply_replacements(raw, dic.get("replacements", {}))
            whole = textproc.match_snippet_whole(raw_fixed, snippets)
            if whole is not None:
                text, info = whole, {"llm": False, "ms": 0, "reason": "snippet"}
            else:
                text, info = self.cleaner.clean(raw_fixed, sctx.style, words)
                text = textproc.apply_snippets_inline(text, snippets)
                text = textproc.apply_replacements(text, dic.get("replacements", {}))
            r["cleanup_ms"], r["llm"], r["reason"] = info["ms"], info["llm"], info["reason"]
        text = text.strip()
        if not text:
            r.update(status="discard", reason="empty after cleanup")
            return r
        if cfg.get("insert.trailing_space", False):
            text += " "
        r["text"] = text

        if insert:
            t_ins = time.perf_counter()
            try:
                self.inserter(text, mode=cfg.get("insert.mode", "paste"),
                              restore=bool(cfg.get("insert.restore_clipboard", True)),
                              restore_delay_ms=int(cfg.get("insert.restore_delay_ms", 300)))
            except Exception:
                log.exception("insertion failed")
                r["status"] = "insert-error"
            r["insert_ms"] = int((time.perf_counter() - t_ins) * 1000)
        # latency to the moment the paste keystroke went out (excludes the deliberate clipboard-restore wait)
        restore_wait = int(cfg.get("insert.restore_delay_ms", 300)) if (insert and cfg.get("insert.mode") == "paste"
                                                                        and cfg.get("insert.restore_clipboard")) else 0
        r["total_ms"] = int((time.perf_counter() - t_release) * 1000) - restore_wait
        if self.history is not None and cfg.get("history.enabled", True):
            try:
                self.history.add(raw, text, app_exe, dur, r["stt_ms"], r["cleanup_ms"], r["total_ms"],
                                 title=app_title, style=sctx.style, segments=r["segments"])
            except Exception:
                log.exception("history write failed")
        log.info("dictation: %.1fs audio | stt %d ms | cleanup %d ms (%s%s) | insert %d ms | total %d ms | "
                 "style %s%s%s | %s -> %r",
                 dur, r["stt_ms"], r["cleanup_ms"], "llm" if r["llm"] else "regex",
                 f", {r['reason']}" if r["reason"] else "", r["insert_ms"], r["total_ms"], sctx.style,
                 f" ({sctx.rule})" if sctx.rule else "",
                 f" | {r['segments']} segments, stt total {r.get('stt_total_ms', r['stt_ms'])} ms" if streamed else "",
                 app_exe, text[:120])
        return r

    @staticmethod
    def _drop_session(session) -> None:
        if session is not None:
            try:
                session.cancel()
            except Exception:
                log.exception("closing streaming session failed")
