"""JSON configuration with defaults, deep-merge, live reload and change callbacks.

The file lives at %APPDATA%\\LocalFlow\\config.json. Unknown keys are preserved,
missing keys fall back to DEFAULTS. Editing the file by hand is picked up live
(the app polls the mtime); the Settings window writes through Config.update().
"""
from __future__ import annotations

import copy
import json
import logging
import os
import threading
from typing import Any, Callable

from . import styles as _styles
from .paths import CONFIG_PATH

log = logging.getLogger(__name__)

# keys whose dict values are user maps: replaced wholesale on update, not merged
_MAP_KEYS = ("replacements", "snippets", "abbreviations", "process_groups")

DEFAULTS: dict[str, Any] = {
    # Hold these modifiers together to dictate. Any of: ctrl, alt, shift, win
    "hotkey": "ctrl+alt",
    # While holding the hotkey, tap this key to lock hands-free recording on
    "handsfree_key": "space",
    "enabled": True,  # False == paused
    "mic": {
        # Both empty = follow the Windows default recording device (Sound settings), switching live when it changes.
        "match": "",         # optional: case-insensitive substring of an input device name to prefer over the default
        "device": "",        # exact device name override (set by the tray / Settings picker); wins over match
    },
    "audio": {
        "keep_open": True,       # keep the input stream running + pre-roll ring buffer
        "preroll_ms": 400,
        "min_duration_s": 0.3,   # shorter recordings are discarded
        "silence_rms": 0.006,    # peak-window RMS below this == silence (pre-STT gate)
        "max_duration_s": 300,
    },
    "stt": {
        "model": "large-v3-turbo",   # or small.en / base.en / a local path
        "device": "cuda",
        "compute_type": "float16",
        "language": "en",            # "auto" to detect
        "beam_size": 1,
        "cpu_fallback_model": "small.en",
        "no_speech_threshold": 0.6,
        "log_prob_threshold": -1.0,
    },
    "cleanup": {
        "enabled": True,
        "min_words": 4,              # skip the LLM for utterances shorter than this
        "model_path": "models/Qwen3.5-4B-Q4_K_M.gguf",
        "server_exe": "bin/llama/llama-server.exe",
        "port": 8765,
        "gpu_layers": 99,
        "ctx": 4096,
        "timeout_s": 5.0,
        "max_len_ratio": 1.6,        # sanity guard on output/input length
    },
    "insert": {
        "mode": "paste",             # "paste" (clipboard + Ctrl+V) or "type" (SendInput unicode)
        "restore_clipboard": True,
        "restore_delay_ms": 300,
        "trailing_space": False,
    },
    "ui": {
        "overlay": True,
        "sounds": True,
        "sound_volume": 0.25,
    },
    # ----- per-app / per-tab writing styles (see README "Styles and profiles") -----
    "auto_style": True,         # False == always use "neutral" (tray: "Auto style by app")
    # Ordered rules, FIRST MATCH WINS. process = exe names (case-insens.; "@browsers" expands process_groups),
    # title = regex searched in the window title (browser tab titles live there). process + title are ANDed.
    # Slack/Teams are "neutral" (work chat); set their style to "casual" if you text-chat there.
    "profiles": _styles.DEFAULT_PROFILES,
    "process_groups": _styles.DEFAULT_PROCESS_GROUPS,
    # Style definitions: prompt (LLM instructions) + deterministic flags (lowercase, strip_trailing_period
    # true/"single", abbreviations, drop_apostrophes, capitalize) + optional examples [[spoken, written], ...]
    "styles": _styles.DEFAULT_STYLES,
    "abbreviations": _styles.DEFAULT_ABBREVIATIONS,   # phrase -> short form, for styles with abbreviations=true
    # Chunked transcription while you are still talking (long dictations paste sooner after release)
    "streaming": {
        "enabled": True,
        "min_segment_s": 4.0,        # speech needed before a pause may finalize a segment
        "min_record_s": 7.0,         # recordings shorter than this never stream (single pass)
        "pause_ms": 500,             # silence that counts as a natural pause
        "max_segment_s": 22.0,       # no pause by here -> cut at the lowest-energy point of the last 2.5 s
        "prompt_chars": 220,         # tail of previous text passed to Whisper as initial_prompt
        "incremental_cleanup": True, # also clean finished sentences in the background
        "context_chars": 300,        # previous cleaned text shown to the LLM as read-only context
        "paragraph_pause_ms": 900,   # long pause after 3+ sentences -> new paragraph (neutral/professional); 0 = off
    },
    "dictionary": {
        "words": [],            # custom words/spellings -> Whisper hotwords + LLM hints
        "replacements": {},     # {"heard": "wanted"} applied after transcription (whole words, case-insens.)
    },
    "snippets": {},             # {"spoken trigger": "expansion text"}
    "history": {"enabled": True, "max_entries": 5000},
}


# top-level sections introduced after v1: written into an existing config.json the first time
PERSISTED_KEYS = {"auto_style", "profiles", "process_groups", "styles", "abbreviations", "streaming"}


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k not in _MAP_KEYS:
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class Config:
    def __init__(self, path=CONFIG_PATH):
        self.path = path
        self._lock = threading.RLock()
        self._callbacks: list[Callable[[dict, dict], None]] = []
        self._mtime = 0.0
        self.data: dict[str, Any] = copy.deepcopy(DEFAULTS)
        self._user_keys: set[str] = set()
        self.load()
        if not self.path.exists():
            self.save()
        elif PERSISTED_KEYS - self._user_keys:
            # upgrade: write newly introduced sections (profiles, styles, ...) into the file so they are editable
            log.info("config: adding new default sections %s", sorted(PERSISTED_KEYS - self._user_keys))
            self.save()

    # ----- access -----
    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def __getitem__(self, k):
        return self.data[k]

    def set(self, dotted: str, value: Any, save: bool = True) -> None:
        with self._lock:
            old = copy.deepcopy(self.data)
            cur = self.data
            parts = dotted.split(".")
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            cur[parts[-1]] = value
        if save:
            self.save()
        self._notify(old)

    def update(self, new_data: dict) -> None:
        """Deep-merge new_data (full replace for replacements/snippets) and persist."""
        with self._lock:
            old = copy.deepcopy(self.data)
            self.data = deep_merge(self.data, new_data)
        self.save()
        self._notify(old)

    # ----- persistence -----
    def load(self) -> bool:
        with self._lock:
            try:
                if not self.path.exists():
                    return False
                self._mtime = self.path.stat().st_mtime
                with open(self.path, "r", encoding="utf-8") as f:
                    user = json.load(f)
                self._user_keys = set(user) if isinstance(user, dict) else set()
                self.data = deep_merge(DEFAULTS, user)
                return True
            except Exception:
                log.exception("failed to read config; keeping previous values")
                return False

    def save(self) -> None:
        with self._lock:
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, self.path)
            self._mtime = self.path.stat().st_mtime

    def reload_if_changed(self) -> bool:
        try:
            m = self.path.stat().st_mtime
        except OSError:
            return False
        if m == self._mtime:
            return False
        old = copy.deepcopy(self.data)
        if self.load():
            log.info("config reloaded from disk")
            self._notify(old)
            return True
        return False

    # ----- change notification -----
    def on_change(self, cb: Callable[[dict, dict], None]) -> None:
        """cb(old_data, new_data) - called on whichever thread changed the config."""
        self._callbacks.append(cb)

    def _notify(self, old: dict) -> None:
        for cb in list(self._callbacks):
            try:
                cb(old, self.data)
            except Exception:
                log.exception("config callback failed")
