"""Transcript assistant: summary / bullet points / key points / action items / custom questions over a stored
transcript, using the local llama-server.

Uses its own requests.Session (never Cleaner.http). Requests stream; while a dictation is in progress
(``gate.busy()``) the running request is dropped and retried once the dictation is done, so live dictation never
waits behind a long AI answer. Transcripts longer than the model context are map-reduced: each part is turned
into notes for the request, then the notes are combined into the answer.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Callable, Optional

import requests

from .cleanup import strip_think

log = logging.getLogger(__name__)

CHARS_PER_TOKEN = 3.5          # conservative (English is ~4): over-estimates tokens, so chunks always fit
SAFETY_TOKENS = 256

PRESETS: dict[str, tuple[str, str]] = {
    "summary": ("Summary",
                "Write a concise summary of this transcript: one short overview paragraph, then the main topics "
                "discussed as a few short paragraphs or bullets. Mention decisions and conclusions."),
    "bullets": ("Bullet points",
                "Rewrite the content of this transcript as clear, compact bullet points in the order it was "
                "discussed. One idea per bullet, each starting with \"- \"."),
    "key_points": ("Key points",
                   "List the most important key points, facts and decisions from this transcript as a numbered "
                   "list, most important first. Only include what was actually said."),
    "action_items": ("Action items",
                     "Extract every action item, task, follow-up and commitment from this transcript as a "
                     "checklist: \"- [ ] task (owner, if mentioned; due date, if mentioned)\". If there are none, "
                     "answer exactly: No action items found."),
}

SYSTEM = ("You are LocalFlow's transcript assistant. You work only with the transcript you are given (a recording "
          "of speech; meeting transcripts label speakers as \"You\" and \"Others\"). Base every statement on the "
          "transcript; never invent facts, names or dates. Answer in the language of the transcript unless asked "
          "otherwise. Output only the answer: no preamble, no closing remarks.")

MAP_INSTRUCTION = ("This is part {i} of {n} of a long transcript. Write compact notes from this part that will "
                   "later be combined to answer the request below. Keep every detail relevant to the request "
                   "(facts, decisions, tasks, owners, dates, timestamps if present); skip small talk. Output only "
                   "the notes.\n\nRequest: {request}")

REDUCE_INSTRUCTION = ("Below are notes taken, in order, from consecutive parts of one long transcript. Using only "
                      "these notes, do this: {request}")


class AssistantError(RuntimeError):
    pass


class AssistantCancelled(AssistantError):
    pass


MIN_BODY_TOKENS = 200          # a request must leave at least this much room for transcript text


def budget_chars(ctx: int, prompt_chars: int, max_tokens: int) -> int:
    """How many transcript characters fit in one request: context minus prompt minus answer minus safety.
    Raises AssistantError when the prompt + answer alone (nearly) fill the context."""
    avail = int(ctx) - int(prompt_chars / CHARS_PER_TOKEN) - int(max_tokens) - SAFETY_TOKENS
    if avail < MIN_BODY_TOKENS:
        raise AssistantError("The request is too long for the AI model's context. Use a shorter question.")
    return int(avail * CHARS_PER_TOKEN)


_SENT_SPLIT = re.compile(r"(?<=[.?!])\s+")


def chunk_text(text: str, limit: int) -> list[str]:
    """Split at paragraph breaks, then sentences, then words, so every piece is <= ``limit`` characters."""
    pieces: list[str] = []
    for para in [p for p in text.split("\n\n") if p.strip()]:
        if len(para) <= limit:
            pieces.append(para)
            continue
        cur = ""
        for sent in _SENT_SPLIT.split(para):
            while len(sent) > limit:                       # no sentence ends: cut at a word boundary
                cut = sent.rfind(" ", 0, limit)
                cut = cut if cut > limit // 2 else limit
                if cur:
                    pieces.append(cur)
                    cur = ""
                pieces.append(sent[:cut])
                sent = sent[cut:].lstrip()
            if cur and len(cur) + 1 + len(sent) > limit:
                pieces.append(cur)
                cur = sent
            else:
                cur = f"{cur} {sent}" if cur else sent
        if cur:
            pieces.append(cur)
    chunks: list[str] = []
    for p in pieces:
        if chunks and len(chunks[-1]) + 2 + len(p) <= limit:
            chunks[-1] += "\n\n" + p
        else:
            chunks.append(p)
    return chunks


def _visible(acc: str) -> str:
    """What to show while streaming: nothing while an unterminated <think> block is open."""
    if acc.lstrip().startswith("<think>") and "</think>" not in acc:
        return ""
    return strip_think(acc).strip()


class Assistant:
    def __init__(self, cfg_get: Callable, gate, store, base_url_fn: Callable[[], str], is_ready_fn: Callable[[], bool],
                 request_extra_fn: Optional[Callable[[], dict]] = None, session: Optional[requests.Session] = None):
        self.cfg_get = cfg_get
        self.gate = gate
        self.store = store
        self.base_url_fn = base_url_fn
        self.is_ready_fn = is_ready_fn
        self.request_extra_fn = request_extra_fn
        self.http = session or requests.Session()
        self.retry_sleep_s = 2.0
        self.ready_poll_s = 1.0
        self.requests_made = 0          # for tests / diagnostics

    def _get(self, key: str, default):
        try:
            v = self.cfg_get(key, default)
        except Exception:
            v = default
        return default if v is None else v

    # ------------------------------------------------------------------ public
    def ask(self, transcript_id: int, preset_or_prompt: Optional[str], on_partial: Optional[Callable[[str], None]] = None,
            stop: Optional[threading.Event] = None, on_status: Optional[Callable[[str], None]] = None) -> str:
        """Run a preset (``summary``/``bullets``/``key_points``/``action_items``) or a custom prompt over the
        transcript. Streams the final answer to ``on_partial(text_so_far)``; saves it to ai_results and returns
        it. Raises AssistantError (AssistantCancelled when ``stop`` is set)."""
        stop = stop or threading.Event()
        row = self.store.get(transcript_id)
        if row is None:
            raise AssistantError("Transcript not found.")
        text = (row.get("text") or row.get("raw_text") or "").strip()
        if not text:
            raise AssistantError("This transcript is empty.")
        key = (preset_or_prompt or "").strip()
        if key in PRESETS:
            preset, prompt, request = key, "", PRESETS[key][1]
        else:
            preset, prompt, request = "custom", key, key
            if not prompt:
                raise AssistantError("Type a question or instruction first.")
        out = self.run(text, request, on_partial=on_partial, stop=stop, on_status=on_status)
        if not out:
            raise AssistantError("The AI model returned an empty answer.")
        if stop.is_set():
            raise AssistantCancelled("cancelled")
        if not self.store.add_ai_result(transcript_id, preset, prompt, out):
            raise AssistantCancelled("transcript was deleted")
        return out

    def run(self, text: str, request: str, on_partial=None, stop: Optional[threading.Event] = None,
            on_status=None) -> str:
        stop = stop or threading.Event()
        ctx = int(self._get("cleanup.ctx", 8192))
        max_tokens = int(self._get("assistant.max_tokens", 1024))
        max_tokens = max(64, min(max_tokens, ctx // 3))
        overhead = len(SYSTEM) + len(REDUCE_INSTRUCTION) + len(MAP_INSTRUCTION) + len(request) + 64
        single = budget_chars(ctx, len(SYSTEM) + len(request) + 64, max_tokens)
        if len(text) <= single:
            return self._complete(self._messages(text, request), max_tokens, stop, on_partial, on_status)
        reduce_limit = budget_chars(ctx, overhead, max_tokens)
        # notes are capped so that at least 3 of them (at ~4 chars/token, worst case) fit in one combine request:
        # every combine round therefore shrinks the notes and nothing is ever cut off
        note_tokens = int(self._get("assistant.note_tokens", 600))
        note_tokens = max(48, min(note_tokens, int(reduce_limit / 4.0 / 3)))
        limit = budget_chars(ctx, overhead, note_tokens)
        chunks = chunk_text(text, limit)
        notes: list[str] = []
        for i, ch in enumerate(chunks, 1):
            self._status(on_status, f"Reading part {i} of {len(chunks)}")
            instr = MAP_INSTRUCTION.format(i=i, n=len(chunks), request=request)
            notes.append(self._complete(self._messages(ch, instr), note_tokens, stop, None, on_status))
        # combine; if the notes themselves are too long, merge them in groups first (each round shrinks them)
        level = 0
        while len("\n\n".join(notes)) > reduce_limit and level < 8:
            level += 1
            groups = chunk_text("\n\n".join(notes), reduce_limit)
            merged = []
            for gi, g in enumerate(groups, 1):
                self._status(on_status, f"Combining notes {gi} of {len(groups)}")
                instr = REDUCE_INSTRUCTION.format(request="merge these notes into one compact set of notes, "
                                                          "keeping every relevant detail for: " + request)
                merged.append(self._complete(self._messages(g, instr, tag="notes"), note_tokens, stop, None,
                                             on_status))
            notes = merged
        joined = "\n\n".join(notes)
        if len(joined) > reduce_limit:            # practically unreachable (see note_tokens); never overflow
            log.warning("assistant notes still too long after %d combine rounds - truncating", level)
            joined = joined[:reduce_limit]
        self._status(on_status, "Writing the answer")
        return self._complete(self._messages(joined, REDUCE_INSTRUCTION.format(request=request), tag="notes"),
                              max_tokens, stop, on_partial, on_status)

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _messages(body: str, instruction: str, tag: str = "transcript") -> list[dict]:
        return [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"<{tag}>\n{body}\n</{tag}>\n\n{instruction}"}]

    @staticmethod
    def _status(cb, msg: str) -> None:
        if cb:
            try:
                cb(msg)
            except Exception:
                pass

    def _wait_ready(self, stop: threading.Event, on_status) -> None:
        """Wait until the LLM server is up and no dictation is running."""
        waited = 0.0
        limit = float(self._get("assistant.wait_ready_s", 900))
        announced = False
        while True:
            if stop.is_set():
                raise AssistantCancelled("cancelled")
            ready = False
            try:
                ready = bool(self.is_ready_fn())
            except Exception:
                ready = False
            if not ready:
                if not bool(self._get("cleanup.enabled", True)):
                    raise AssistantError("The AI model is turned off (Settings > Cleanup). Turn it on to use AI notes.")
                if waited >= limit:
                    raise AssistantError("The AI model is not running.")
                if not announced:
                    self._status(on_status, "Waiting for the AI model")
                    announced = True
                if stop.wait(self.ready_poll_s):
                    raise AssistantCancelled("cancelled")
                waited += self.ready_poll_s
                continue
            if not self.gate.wait_idle(stop=stop):
                raise AssistantCancelled("cancelled")
            try:
                if self.is_ready_fn():
                    return
            except Exception:
                pass

    def _body(self, messages: list[dict], max_tokens: int) -> dict:
        body = {"messages": messages, "stream": True, "max_tokens": int(max_tokens),
                "temperature": float(self._get("assistant.temperature", 0.3)), "top_p": 0.9,
                "cache_prompt": True, "chat_template_kwargs": {"enable_thinking": False}}
        if self.request_extra_fn:
            try:
                extra = self.request_extra_fn() or {}
            except Exception:
                extra = {}
            for k, v in extra.items():
                if k not in ("messages", "stream", "max_tokens", "temperature", "stop", "n_predict"):
                    body[k] = v
        return body

    def _complete(self, messages: list[dict], max_tokens: int, stop: threading.Event,
                  on_partial: Optional[Callable[[str], None]], on_status) -> str:
        errors = 0
        while True:
            self._wait_ready(stop, on_status)
            preempted = False
            acc = ""
            try:
                self.requests_made += 1
                with self.http.post(self.base_url_fn() + "/v1/chat/completions", json=self._body(messages, max_tokens),
                                    stream=True, timeout=(5, 120)) as r:
                    r.raise_for_status()
                    finish = None
                    for line in r.iter_lines():
                        if stop.is_set():
                            raise AssistantCancelled("cancelled")
                        if self.gate.busy():
                            preempted = True          # closing the response makes llama-server stop generating
                            break
                        if not line or not line.startswith(b"data:"):
                            continue
                        data = line[5:].strip()
                        if data == b"[DONE]":
                            break
                        try:
                            j = json.loads(data)
                        except ValueError:
                            continue
                        ch = (j.get("choices") or [{}])[0]
                        delta = (ch.get("delta") or {}).get("content") or ""
                        finish = ch.get("finish_reason") or finish
                        if delta:
                            acc += delta
                            if on_partial:
                                try:
                                    on_partial(_visible(acc))
                                except Exception:
                                    pass
                if preempted:
                    log.info("assistant request paused for a dictation - will retry")
                    self._status(on_status, "Paused for dictation")
                    if on_partial:
                        try:
                            on_partial("")
                        except Exception:
                            pass
                    continue
                if finish == "length":
                    log.warning("assistant answer hit max_tokens (%d)", max_tokens)
                return strip_think(acc).strip()
            except AssistantCancelled:
                raise
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
                errors += 1
                log.warning("assistant request failed (%d): %s: %s", errors, type(e).__name__, e)
                if errors >= 4:
                    raise AssistantError(f"The AI model did not answer ({type(e).__name__}).") from e
                if stop.wait(self.retry_sleep_s):
                    raise AssistantCancelled("cancelled")
