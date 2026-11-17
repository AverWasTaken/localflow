"""Tab autocomplete: after a short typing pause, read the text around the caret of the focused text field (UI
Automation, IAccessible2 fallback), ask the local LLM (llama-server raw /completion, slot 1) for the next few words and
show them as a gray "ghost" right of the caret. Tab accepts (types the words), Backspace dismisses and keeps quiet for
the rest of that sentence, any other key just hides it.

Threading
- The keyboard hook (hotkey.py) only calls ``note_key`` (a counter + timestamp, no characters) and ``want_swallow``
  (Tab / Backspace decision from a snapshot, no I/O except GetForegroundWindow).
- ONE worker thread does everything else: COM (MTA) + every UIA / IA2 call, the HTTP stream, accept typing.
- The ghost window lives on the Qt GUI thread (ghost.py); it is driven through signals and reports back what it
  actually shows (``ghost_shown``), which is what Tab accepts.

Privacy: field text, page text and suggestions are never logged or written anywhere (only lengths, timings and
method names).
"""
from __future__ import annotations

import ctypes
import fnmatch
import hashlib
import json
import logging
import os
import queue
import re
import sys
import threading
import time
import unicodedata
from collections import OrderedDict
from ctypes import wintypes
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from . import winutil as w
from .uia import _GTI, _fg_hwnd, _import_comtypes_mta, _window_pid, gui_thread_caret  # noqa: F401  (shared, patched by tests)

log = logging.getLogger(__name__)

PRIVATE_TITLE_RE = re.compile(r"inprivate|incognito|private browsing", re.I)
SENTENCE_END_RE = re.compile(r"[.!?…]+[\"')\]]*(?=\s|$)")
_STREAM_SENTENCE_RE = re.compile(r"\w[.!?…]+[\"')\]]*\s")     # a finished sentence followed by more text
_GARBAGE_RE = re.compile(r"<\|?/?(think|im_start|im_end|endoftext|end|start)|\|>|^\s*(Me|User|Assistant)\s*:", re.I)
# an open @mention (Teams keeps the people picker open across one space: "@John Sm"), #channel, :emoji, /command
PICKER_RE = re.compile(r"(?:^|\s)(?:@\S+(?: \S+)?|[@#:/]\S*)$")
ZWSP = "​‌‍﻿"
OBJ_REPLACEMENT = "￼"

VK_TAB, VK_BACK = w.VK_TAB, w.VK_BACK
_MOUSE_VKS = (0x01, 0x02, 0x04)          # left / right / middle button

AC_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "debounce_ms": 350,         # typing pause before a guess is requested
    "min_chars": 3,             # text needed before the caret
    "max_chars": 80,            # longest guess shown / typed
    "max_tokens": 24,
    "temperature": 0.2,
    "before_chars": 1500,       # field text before the caret sent to the model
    "after_chars": 300,         # field text after the caret read (mid-line check)
    "context_chars": 2000,      # other visible text of the page / window sent to the model
    "ttl_s": 8.0,               # the guess disappears after this long
    "min_visible_ms": 250,      # Tab / Backspace only act on a guess shown at least this long
    "timeout_s": 3.0,
    "exclude_apps": [],
    "debug_log": False,         # log every trigger's outcome (reasons / lengths only), not once a minute
}
DIAG_EVERY_S = 60.0             # the same (app, reason) is logged at INFO at most this often

_CONTROL_TRANS = {c: None for c in range(0x20) if c not in (0x20,)}


# ----------------------------------------------------------------------------- pure helpers (unit tested)
def cfg_value(cfg, key: str):
    v = cfg.get(f"autocomplete.{key}", AC_DEFAULTS[key]) if cfg is not None else AC_DEFAULTS[key]
    return AC_DEFAULTS[key] if v is None else v


def strip_controls(text: str) -> str:
    """Drop every control / format character (Tab, newline, CR, zero-width, bidi marks, U+2028/9 ...)."""
    out = []
    for ch in text or "":
        cat = unicodedata.category(ch)
        if cat in ("Cc", "Cf", "Zl", "Zp") or ch in ZWSP or ch == OBJ_REPLACEMENT:
            continue
        out.append(ch)
    return "".join(out)


def is_private_title(title: str) -> bool:
    return bool(PRIVATE_TITLE_RE.search(title or ""))


def app_excluded(exe: str, patterns) -> bool:
    exe = (exe or "").lower()
    if not exe:
        return False
    for p in patterns or []:
        p = str(p).strip().lower()
        if p and (exe == p or fnmatch.fnmatchcase(exe, p)):
            return True
    return False


def current_sentence(before: str) -> str:
    """The sentence the caret is in (from the last sentence end / line break to the caret), left-stripped."""
    s = before or ""
    cut = max(s.rfind("\n"), s.rfind("\r"))
    for m in re.finditer(r"[.!?…][\"')\]]*\s", s):
        cut = max(cut, m.end() - 1)
    return s[cut + 1:].lstrip()


_ANCHOR_CHARS = 40          # text before the sentence that identifies where it starts
_HEAD_CHARS = 8             # first characters of the sentence


def _h(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "replace")).hexdigest()


_ENDS_SENTENCE_RE = re.compile(r"[.!?…][\"')\]]*$")


def sentence_parts(before: str) -> tuple[str, str]:
    """(anchor, sentence): the text before the sentence the caret is in (right-stripped) and that sentence so far.
    Right after a sentence end ("All good." / "All good. ") the caret is in the NEXT, still empty, sentence."""
    b = before or ""
    if _ENDS_SENTENCE_RE.search(b.rstrip()):
        return b.rstrip(), ""
    sent = current_sentence(b)
    start = len(b) - len(sent)
    return b[:start].rstrip(), sent


MIN_WORDS_NO_PAGE = 4      # with no page text, fewer words than this give "please provide the rest of your text"
SUPPRESS_LIFT_WORDS = 8    # a Backspace dismissal also ends after this many more words (chat text has no periods)


def _word_count(s: str) -> int:
    return len(re.findall(r"\S+", s or ""))


def sentence_key(hwnd: int, el_id: Any, before: str) -> tuple:
    """Suppression key for the sentence being typed: (window, element, hash of the text before the sentence, hashes
    of the sentence's first 1..8 characters, words in the sentence so far). Survives edits to the end of the
    sentence; no text is kept."""
    anchor, sent = sentence_parts(before)
    head = sent[:_HEAD_CHARS]
    return (int(hwnd or 0), el_id, _h(anchor[-_ANCHOR_CHARS:]),
            tuple(_h(head[:i]) for i in range(1, len(head) + 1)), _word_count(sent))


def sentence_suppressed(suppressed, hwnd: int, el_id: Any, before: str) -> bool:
    """True when the caret is still in a sentence the user dismissed with Backspace (same field, same start) and
    fewer than SUPPRESS_LIFT_WORDS words were typed since."""
    anchor, sent = sentence_parts(before)
    a = _h(anchor[-_ANCHOR_CHARS:])
    words = None
    for key in suppressed:
        h, e, anchor_h, heads = key[:4]
        if h != int(hwnd or 0) or e != el_id or anchor_h != a:
            continue
        n = min(len(sent), len(heads))
        if n == 0 or heads[n - 1] == _h(sent[:n]):
            if len(key) > 4:
                words = _word_count(sent) if words is None else words
                if words - int(key[4]) >= SUPPRESS_LIFT_WORDS:
                    continue
            return True
    return False


def tail_hash(before: str, n: int = 200) -> str:
    """Hash of the text just before the caret, normalised so that re-reads of the same text compare equal
    (trailing line breaks / no-break spaces / zero-width and object chars differ between UIA read paths)."""
    t = "".join(ch for ch in (before or "") if ch not in ZWSP and ch != OBJ_REPLACEMENT)
    t = t.replace(NBSP, " ").rstrip()
    return hashlib.sha1(t[-n:].encode("utf-8", "replace")).hexdigest()


def caret_mid_line(after: str) -> bool:
    """True when there is real text between the caret and the end of its line."""
    line = (after or "").split("\n", 1)[0].split("\r", 1)[0]
    return bool(line.strip(" \t " + ZWSP))


_SCAFFOLD_RE = re.compile(r"the user is typing|visible on screen|text on screen|continue my text|"
                          r"^\s*(window|app|title)\s*:", re.I)
_ODD_LEAD = set(")]}|#*=/\\<>~^_@&%+")
_DOTTED_DIGITS_RE = re.compile(r"^\d+(?:\s*\.\s*\d+){2,}|\d+(?:\s*\.\s*\d+){3,}")   # "3.1.2..." / "1.2.3.4"
_NUM_LIST_RE = re.compile(r"(?:\b\d{1,3}[.)]\s*){3,}")                 # "1) 2) 3)"
_LONE_NUMBER_RE = re.compile(r"^[$€£]?\d{1,4}(?:[,.:]\d{1,3})*\s*(?:am|pm|k|%)?[.!?]?$", re.I)  # "10?", "5,880."
_ECHO_SPAN = 40             # a verbatim span this long copied from the page / field = echo
NBSP = " "
TRAIL_WS = " \t "      # trailing spaces trimmed from the prompt (Chromium boxes often end in a no-break space)
# the model talking about the task instead of continuing ("provide the rest of your text so I can continue it")
_META_RE = re.compile(r"rest of your|your (text|sentence)\b|you were (writing|saying|typing)|continue (it|your|the "
                      r"text|writing)|more context|\bliquid (ai|foundation)|\bas an ai\b", re.I)
_CURLY = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})
_CLOSERS = {")": "(", "]": "[", '"': '"', "'": "'"}


def _has_long_span(core: str, src: str, n: int = _ECHO_SPAN, words: int = 0) -> bool:
    """True when ``core`` (lowercased guess) contains a span of ``n`` characters that also appears in ``src``
    (and, with ``words``, holds at least that many spaces: several words, not one long path / URL)."""
    if len(core) < n or not src:
        return False
    for i in range(0, len(core) - n + 1):
        w = core[i:i + n]
        if (not words or w.count(" ") >= words) and w in src:
            return True
    return False


def _repeats(words: list[str]) -> bool:
    """Loops: few distinct words, one word 3x in a row, a bigram / trigram 3+ times, or a 2-3 word phrase repeated
    back to back. (A word 3x apart is normal: "to go to the store to get milk".)"""
    if len(words) >= 4 and len(set(words)) <= len(words) * 0.4:
        return True
    for i in range(len(words) - 2):
        if words[i] == words[i + 1] == words[i + 2]:
            return True
    for n in (2, 3):
        grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
        counts: dict = {}
        for g in grams:
            counts[g] = counts.get(g, 0) + 1
            if counts[g] >= 3:
                return True
        if n >= 2:
            for i in range(len(words) - 2 * n + 1):
                if words[i:i + n] == words[i + n:i + 2 * n]:
                    return True
    return False


def _strip_unbalanced_tail(s: str, before: str) -> str:
    """Drop trailing ) ] " ' that close nothing in the guess or the caret's line
    ('instructions.")])])' -> 'instructions.')."""
    line = (before or "").rsplit("\n", 1)[-1]
    while s and s[-1] in _CLOSERS:
        ch = s[-1]
        body = line + s[:-1]
        if ch in "\"'":
            closes = body.count(ch) % 2 == 1
        else:
            closes = body.count(_CLOSERS[ch]) > body.count(ch)
        if closes:
            break
        s = s[:-1]
    return s


def postprocess(raw: str, before: str, after: str = "", max_chars: int = 80, page: str = "") -> Optional[str]:
    """Turn a raw model continuation into the ghost text, or None to show nothing."""
    return postprocess_ex(raw, before, after, max_chars, page)[0]


def postprocess_ex(raw: str, before: str, after: str = "", max_chars: int = 80,
                   page: str = "") -> tuple[Optional[str], str]:
    """(ghost text, '') or (None, rule that rejected it). ``page`` only feeds the echo check."""
    if not raw:
        return None, "empty"
    s = raw.replace("\r\n", "\n")
    s = s.split("\n", 1)[0]
    if _GARBAGE_RE.search(s):
        return None, "special-token"
    s = strip_controls(s)
    if not s.strip():
        return None, "empty"
    if _SCAFFOLD_RE.search(s):
        return None, "scaffold"
    if _NUM_LIST_RE.search(s) or _DOTTED_DIGITS_RE.search(s.strip()):
        return None, "digit-run"            # checked before the sentence cut: " 1. 2. 3." would become " 1."
    b = before or ""
    s = s.replace(NBSP, " ")                # never type no-break spaces
    if not any(q in b for q in "’‘“”"):
        s = s.translate(_CURLY)             # the user types straight quotes: "isn’t" -> "isn't"
    trimmed = b.rstrip(TRAIL_WS)
    if b and b[-1] in TRAIL_WS and trimmed and not trimmed[-1].isspace() and not s[:1].isspace():
        # the prompt drops the field's trailing space, so a new word must start with the model's own space;
        # "you " + "'d like" / "be " + "autiful" would glue onto the typed word (an indented new line is fine)
        return None, "word-join"
    if _META_RE.search(s):
        return None, "meta"                 # the model answered the instruction instead of continuing
    if not b or b[-1].isspace() or b[-1] in "([{\"'“‘":
        s = s.lstrip()                      # the space (or opening bracket / quote) is already there
    elif b[-1] in ",;:.!?" and s[:1].isalnum():
        s = " " + s                         # "Thanks," + "see you" -> "Thanks, see you"
    s = re.sub(r"[ \t ]{2,}", " ", s)
    m = SENTENCE_END_RE.search(s)
    if m:
        s = s[:m.end()]
    s = _strip_unbalanced_tail(s.rstrip(), b)
    if len(s) > max_chars:
        cut = s[:max_chars]
        sp = cut.rfind(" ")
        s = cut[:sp] if sp >= max_chars // 3 else cut
    s = s.rstrip() if s.strip() else ""
    if not s.strip():
        return None, "empty"
    st = s.strip()
    if st[0] in _ODD_LEAD:
        return None, "odd-lead"
    letters = sum(ch.isalpha() for ch in st)
    spaces = sum(ch.isspace() for ch in st)
    lone_number = bool(_LONE_NUMBER_RE.match(st)) and not _DOTTED_DIGITS_RE.search(st)
    if not lone_number and (letters == 0 or letters + spaces < len(st) * 0.6):
        return None, "few-letters"          # "3.1.2.4.5.6.", "| --- |", "(1)"
    if _DOTTED_DIGITS_RE.search(st) or _NUM_LIST_RE.search(st):
        return None, "digit-run"
    words = re.findall(r"\w+", st.lower())
    if _repeats(words):
        return None, "repeat"
    # echo: the model repeated what is already there (before the caret, the field, the page)
    core = re.sub(r"\s+", " ", st.lower())
    bl = re.sub(r"\s+", " ", b.lower())
    if len(core) >= 4 and core in bl[-max(len(core) * 3, 60):]:
        return None, "echo"
    if _has_long_span(core, bl):
        return None, "echo"
    a = (after or "").strip().lower()
    if a and len(core) >= 3 and a.startswith(core):
        return None, "echo-after"
    pl = re.sub(r"\s+", " ", (page or "").lower())
    if (len(core) >= 25 and core.count(" ") >= 3 and core in pl) or _has_long_span(core, pl, words=3):
        return None, "echo-page"            # copied page lines; a quoted path / URL (no spaces) is fine
    if _has_long_span(core, _PROMPT_TEXT, 28):
        return None, "echo-prompt"          # "few more words they are most likely to type."
    return s, ""


AC_SYSTEM = ("You autocomplete text. Continue the user's text with the next few words they are most likely to type. "
             "Reply with the continuation only.")
_PROMPT_TEXT = (AC_SYSTEM + " Continue my text. Text on screen:").lower()     # echo check in postprocess_ex
SYS_MARK, USER_MARK = "@@LF_SYSTEM@@", "@@LF_USER@@"
# the chat template with the two marks, up to and including the assistant turn opener (LFM3 / ChatML, no thinking);
# the live one comes from the server's /apply-template (Completer.template), this is the fallback
FALLBACK_TEMPLATE = (f"<|im_start|>system\n{SYS_MARK}<|im_end|>\n<|im_start|>user\n{USER_MARK}<|im_end|>\n"
                     "<|im_start|>assistant\n<think></think>")


def _clean_text(s: str) -> str:
    s = (s or "").replace(OBJ_REPLACEMENT, " ")
    s = "".join(ch for ch in s if ch in "\n\t" or unicodedata.category(ch) not in ("Cc", "Cf"))
    return s.replace("<|", "< |").replace("|>", "| >")     # typed / page text never forms <|im_end|> & co.


def build_prompt(before: str, page: str, template: Optional[str] = None) -> str:
    """Raw completion prompt: a chat turn whose assistant reply is pre-filled with the field text, so the model
    continues the user's own words. The page text goes in the user turn; no window title / app name (the model
    copied those, and scaffold labels, into its guesses). Trailing spaces are trimmed: the model then starts its
    guess with its own space token, which postprocess drops again."""
    tpl = template if template and SYS_MARK in template and USER_MARK in template else FALLBACK_TEMPLATE
    page = _clean_text(page).strip()
    user = (f"Text on screen:\n{page}\n\n" if page else "") + "Continue my text."
    body = _clean_text(before).rstrip(TRAIL_WS)
    return tpl.replace(SYS_MARK, AC_SYSTEM).replace(USER_MARK, user) + body


def trim_page(text: str, field_text: str, cap: int) -> str:
    """Visible page text without the field's own text, newest (bottom) part kept."""
    t = (text or "").replace("\r", "")
    ft = (field_text or "").strip()
    if ft and len(ft) >= 3:
        t = t.replace(ft, " ")
    t = re.sub(r"[ \t ]+", " ", t)
    t = re.sub(r"\n\s*\n+", "\n", t).strip()
    if len(t) > cap:
        t = t[-cap:]
        nl = t.find("\n")
        if 0 <= nl < 200:
            t = t[nl + 1:]
    return t


@dataclass
class Field:
    """What the reader found at the caret (screen rects are physical pixels)."""
    method: str = ""
    before: str = ""
    after: str = ""
    caret: Optional[tuple[int, int, int]] = None     # x, y, height
    field_rect: Optional[tuple[int, int, int, int]] = None   # left, top, right, bottom
    el_id: Any = None
    multiline: Optional[bool] = None
    editable: bool = True
    password: bool = False
    selection: bool = False
    pid: int = 0
    ml_method: str = ""
    page_root: Any = None                            # UIA document element for the page context (worker only)
    focus_el: Any = None                             # the focused UIA element (worker only)
    at_start: bool = False                           # caret at the very start of the field
    font_pt: float = 0.0                             # text font at the caret (UIA / IA2 text attributes), 0 = unknown
    font_name: str = ""
    placeholder: bool = False                        # the "text" read was the empty box's placeholder (Teams)


@dataclass
class Ghost:
    """A suggestion that is (being) shown. Written by the Qt thread (shown_at/text), read by the hook thread."""
    token: int
    hwnd: int
    key_count: int
    text: str
    supp_key: tuple
    before_hash: str
    el_id: Any
    caret: tuple
    field_rect: Optional[tuple]
    shown_at: float = 0.0
    expires_at: float = 0.0
    exe: str = ""
    caret_src: str = ""                              # Field.method when the guess was made (watch compares like)
    font: Optional[tuple] = None                     # (pt, family) at the caret, for the ghost font
    misses: int = 0                                  # failed watch re-reads in a row
    before_len: int = 0                              # diagnostics only


WATCH_GRACE_S = 0.3         # caret position noise never hides a ghost younger than this


def watch_verdict(g: Ghost, fld: Optional[Field], read_err: str, now: float) -> str:
    """Re-read check while a ghost is up: '' = keep it, else why to hide it. One failed / rect-less re-read is
    tolerated (UIA times out now and then); positions are only compared when both came from the same caret source
    (UIA vs. the system-caret fallback differ by a few px) and beyond rounding jitter."""
    if fld is None or read_err or fld.caret is None:
        g.misses += 1
        if g.misses < 2:
            return ""
        return f"caret: re-read failed ({read_err or ('no rect' if fld is not None else 'none')})"
    g.misses = 0
    if fld.el_id != g.el_id:
        return "caret: other element"
    if tail_hash(fld.before) != g.before_hash:
        return f"caret: text changed (len {g.before_len}->{len(fld.before)}, {g.caret_src}->{fld.method})"
    if now - g.shown_at >= WATCH_GRACE_S and fld.method == g.caret_src and caret_moved(g.caret, fld.caret):
        return (f"caret: moved {fld.caret[0] - g.caret[0]:+d},{fld.caret[1] - g.caret[1]:+d} "
                f"({fld.method})")
    return ""


def swallow_decision(ghost: Optional[Ghost], vk: int, key_count: int, fg_hwnd: int, now: float,
                     min_visible_s: float) -> bool:
    """Tab / Backspace act on the ghost only if nothing changed since it was shown and it is really visible."""
    if ghost is None or vk not in (VK_TAB, VK_BACK) or not ghost.text or not ghost.shown_at:
        return False
    if key_count != ghost.key_count:
        return False
    if now - ghost.shown_at < min_visible_s or now > ghost.expires_at:
        return False
    return bool(fg_hwnd) and fg_hwnd == ghost.hwnd


def skip_reason(fld: Field, min_chars: int) -> str:
    """Why this field / caret position gets no suggestion ('' = go)."""
    if fld.password:
        return "password"
    if not fld.editable:
        return "read-only"
    if fld.multiline is not True:
        return "single-line"
    if fld.selection:
        return "selection"
    if caret_mid_line(fld.after):
        return "mid-line"
    b = "".join(ch for ch in (fld.before or "") if ch != OBJ_REPLACEMENT and ch not in ZWSP)   # Discord: "﻿"
    if len(b.strip()) < min_chars:
        return "too short"
    if b.endswith(("\n", "\r")):
        return "line start"
    if PICKER_RE.search(b):
        return "picker"                     # @mention / #channel / :emoji / /command popup: Tab belongs to it
    if fld.caret is None:
        return "no caret rect"
    return ""


def is_placeholder(text: str, *labels: str) -> bool:
    """True when the field's whole text is just its label / placeholder (Teams reports "Type a message" as the
    text of the empty compose box; that text equals the element's Name)."""
    t = " ".join((text or "").replace(OBJ_REPLACEMENT, " ").split())
    if not t:
        return False
    return any(t == " ".join((lb or "").split()) for lb in labels if lb)


def parse_ia2_font(attrs: str) -> tuple[float, str]:
    """(size in pt, family) from IAccessible2 text attributes ("font-family:Inter;font-size:12pt;..."), 0/'' when
    absent. px sizes are converted at 96 px per 72 pt."""
    pt, fam = 0.0, ""
    for part in re.split(r"(?<!\\);", attrs or ""):
        k, _, v = part.partition(":")
        k, v = k.strip().lower(), v.strip().replace("\\", "")
        if k == "font-size":
            m = re.match(r"([\d.]+)\s*(pt|px)?", v, re.I)
            if m:
                try:
                    n = float(m.group(1))
                except ValueError:
                    continue
                pt = n * 72.0 / 96.0 if (m.group(2) or "").lower() == "px" else n
        elif k == "font-family":
            fam = v.strip("'\" ")
    return (pt if 3.0 <= pt <= 96.0 else 0.0), fam


MIN_CARET_H = 6             # px; smaller "caret" rects are bogus (Chromium reports 2x2 boxes for some ranges)


def refine_caret(caret: Optional[tuple], prev: Optional[tuple]) -> Optional[tuple]:
    """Caret (x, y, h) from the degenerate caret range rect and the previous character's box (x, y, w, h).
    - no usable caret rect: right edge of the previous character;
    - caret rect much taller than the character box on the same line (Notepad reports ~33 px for 15 px text):
      keep the caret x, take the character's top / height, so the ghost sits on the text line at its size."""
    if caret is not None and caret[2] < MIN_CARET_H:
        caret = None
    if prev is not None and prev[3] < MIN_CARET_H:
        prev = None
    if caret is None:
        if prev is None:
            return None
        x, y, wd, h = prev
        return x + wd, y, h
    if prev is None:
        return caret
    cx, cy, ch = caret
    px_, py, pw, ph = prev
    same_line = py >= cy - 2 and py + ph <= cy + ch + 2 and abs((px_ + pw) - cx) <= max(4, ph)
    if same_line and ph < 0.8 * ch:
        return cx, py, ph
    return caret


def caret_moved(old: tuple, new: tuple) -> bool:
    """Caret really moved (vs. sub-pixel / rounding jitter between reads of the same source): more than 6 px
    sideways, or onto another line."""
    dx = abs(new[0] - old[0])
    dy = abs(new[1] - old[1])
    return dx > 6 or dy > max(6, 0.5 * max(old[2], new[2], 1))


def aria_multiline(aria: str) -> Optional[bool]:
    for part in (aria or "").split(";"):
        k, _, v = part.partition("=")
        if k.strip().lower() == "multiline":
            return v.strip().lower() == "true"
    return None


class SuppressionSet:
    """Bounded set of dismissed-sentence keys. An entry lives only while its field is still in that sentence:
    ``prune`` drops it as soon as a later read of the same field shows another sentence, and after ``max_age_s``."""

    def __init__(self, cap: int = 128, max_age_s: float = 600.0):
        self._d: "OrderedDict[tuple, float]" = OrderedDict()
        self.cap = cap
        self.max_age_s = max_age_s

    def prune(self, hwnd: int, el_id: Any, before: str, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        for key, t in list(self._d.items()):
            if now - t > self.max_age_s:
                self._d.pop(key, None)
            elif key[0] == int(hwnd or 0) and key[1] == el_id and not sentence_suppressed([key], hwnd, el_id, before):
                self._d.pop(key, None)          # the user moved on to another sentence in this field

    def add(self, key: tuple) -> None:
        self._d[key] = time.monotonic()
        self._d.move_to_end(key)
        while len(self._d) > self.cap:
            self._d.popitem(last=False)

    def __iter__(self):
        return iter(list(self._d.keys()))

    def __len__(self):
        return len(self._d)

    def clear(self) -> None:
        self._d.clear()


# ----------------------------------------------------------------------------- Win32 helpers
u32 = w.user32


def _mouse_down() -> bool:
    try:
        return any(u32.GetAsyncKeyState(v) & 0x8000 for v in _MOUSE_VKS)
    except Exception:
        return False


GWL_STYLE = -16
ES_MULTILINE = 0x0004


def _win32_multiline(hwnd: int) -> Optional[bool]:
    try:
        buf = ctypes.create_unicode_buffer(128)
        u32.GetClassNameW(hwnd, buf, 128)
        cls = buf.value.lower()
        if "edit" not in cls:
            return None
        get = getattr(u32, "GetWindowLongPtrW", None) or u32.GetWindowLongW
        get.restype = ctypes.c_ssize_t
        get.argtypes = [wintypes.HWND, ctypes.c_int]
        return bool(get(hwnd, GWL_STYLE) & ES_MULTILINE)
    except Exception:
        return None


# ----------------------------------------------------------------------------- COM / UIA reader (worker thread only)
class _IA2:
    """Minimal IAccessible2 / IAccessibleText definitions (built lazily, needs comtypes)."""
    _built = None

    @classmethod
    def get(cls):
        if cls._built is not None:
            return cls._built
        import comtypes.client
        from ctypes import POINTER, byref, c_long
        from comtypes import BSTR, COMMETHOD, GUID, HRESULT, IUnknown
        oleacc_mod = comtypes.client.GetModule("oleacc.dll")
        IAccessible = oleacc_mod.IAccessible

        class IServiceProvider(IUnknown):
            _iid_ = GUID("{6D5140C1-7436-11CE-8034-00AA006009FA}")
            _methods_ = [COMMETHOD([], HRESULT, "QueryService", (["in"], POINTER(GUID), "guidService"),
                                   (["in"], POINTER(GUID), "riid"), (["out"], POINTER(POINTER(IUnknown)), "ppv"))]

        class IAccessible2(IAccessible):
            _iid_ = GUID("{E89F726E-C4F4-4c19-BB19-B647D7FA8478}")
            _methods_ = [
                COMMETHOD(["propget"], HRESULT, "nRelations", (["out", "retval"], POINTER(c_long), "n")),
                COMMETHOD(["propget"], HRESULT, "relation", (["in"], c_long, "i"),
                          (["out", "retval"], POINTER(POINTER(IUnknown)), "r")),
                COMMETHOD(["propget"], HRESULT, "relations", (["in"], c_long, "max"),
                          (["out"], POINTER(POINTER(IUnknown)), "r"), (["out", "retval"], POINTER(c_long), "n")),
                COMMETHOD([], HRESULT, "role", (["out", "retval"], POINTER(c_long), "role")),
                COMMETHOD([], HRESULT, "scrollTo", (["in"], c_long, "t")),
                COMMETHOD([], HRESULT, "scrollToPoint", (["in"], c_long, "c"), (["in"], c_long, "x"),
                          (["in"], c_long, "y")),
                COMMETHOD(["propget"], HRESULT, "groupPosition", (["out"], POINTER(c_long), "a"),
                          (["out"], POINTER(c_long), "b"), (["out", "retval"], POINTER(c_long), "c")),
                COMMETHOD(["propget"], HRESULT, "states", (["out", "retval"], POINTER(c_long), "states")),
            ]

        class IAccessibleText(IUnknown):
            _iid_ = GUID("{24FD2FFB-3AAD-4a08-8335-A3AD89C0FB4B}")
            _methods_ = [
                COMMETHOD([], HRESULT, "addSelection", (["in"], c_long, "s"), (["in"], c_long, "e")),
                COMMETHOD(["propget"], HRESULT, "attributes", (["in"], c_long, "o"), (["out"], POINTER(c_long), "s"),
                          (["out"], POINTER(c_long), "e"), (["out", "retval"], POINTER(BSTR), "a")),
                COMMETHOD(["propget"], HRESULT, "caretOffset", (["out", "retval"], POINTER(c_long), "o")),
                COMMETHOD(["propget"], HRESULT, "characterExtents", (["in"], c_long, "o"), (["in"], c_long, "ct"),
                          (["out"], POINTER(c_long), "x"), (["out"], POINTER(c_long), "y"),
                          (["out"], POINTER(c_long), "w"), (["out", "retval"], POINTER(c_long), "h")),
                COMMETHOD(["propget"], HRESULT, "nSelections", (["out", "retval"], POINTER(c_long), "n")),
                COMMETHOD(["propget"], HRESULT, "offsetAtPoint", (["in"], c_long, "x"), (["in"], c_long, "y"),
                          (["in"], c_long, "ct"), (["out", "retval"], POINTER(c_long), "o")),
                COMMETHOD(["propget"], HRESULT, "selection", (["in"], c_long, "i"), (["out"], POINTER(c_long), "s"),
                          (["out", "retval"], POINTER(c_long), "e")),
                COMMETHOD(["propget"], HRESULT, "text", (["in"], c_long, "s"), (["in"], c_long, "e"),
                          (["out", "retval"], POINTER(BSTR), "t")),
            ]

        oleacc = ctypes.oledll.oleacc

        def acc_from_window(hwnd, objid):
            p = POINTER(IAccessible)()
            oleacc.AccessibleObjectFromWindow(wintypes.HWND(hwnd), ctypes.c_long(objid), byref(IAccessible._iid_),
                                              byref(p))
            return p

        def focused_acc(hwnd, max_depth=40):
            acc = acc_from_window(hwnd, -4)       # OBJID_CLIENT
            for _ in range(max_depth):
                f = acc.accFocus
                if f is None:
                    return acc, 0
                if isinstance(f, int):
                    return acc, f
                try:
                    nxt = f.QueryInterface(IAccessible)
                except Exception:
                    return acc, 0
                if nxt == acc:
                    return acc, 0
                acc = nxt
            return acc, 0

        def ia2_of(acc):
            sp = acc.QueryInterface(IServiceProvider)
            unk = sp.QueryService(byref(IAccessible._iid_), byref(IAccessible2._iid_))
            return unk.QueryInterface(IAccessible2)

        cls._built = {"IAccessible": IAccessible, "IAccessible2": IAccessible2, "IAccessibleText": IAccessibleText,
                      "acc_from_window": acc_from_window, "focused_acc": focused_acc, "ia2_of": ia2_of}
        return cls._built


IA2_STATE_EDITABLE, IA2_STATE_MULTI_LINE, IA2_STATE_SINGLE_LINE = 0x8, 0x200, 0x2000
STATE_SYSTEM_READONLY, STATE_SYSTEM_PROTECTED = 0x40, 0x20000000


class UiaReader:
    """Reads the focused field. Must be created and used on ONE thread (the autocomplete worker)."""

    def __init__(self, before_chars: int = 1500, after_chars: int = 300):
        self.before_chars, self.after_chars = before_chars, after_chars
        self.comtypes, self.client, self.U = _import_comtypes_mta()
        U = self.U
        self.uia = self.client.CreateObject(U.CUIAutomation8, interface=U.IUIAutomation2)
        try:
            self.uia.ConnectionTimeout = 400
            self.uia.TransactionTimeout = 400
        except Exception:
            pass
        self._walker = self.uia.ControlViewWalker
        self.own_pid = os.getpid()

    # ---- focus / identity
    def focused(self):
        return self.uia.GetFocusedElement()

    @staticmethod
    def runtime_id(el) -> Any:
        try:
            return tuple(el.GetRuntimeId())
        except Exception:
            return None

    # ---- main entry
    def read(self, hwnd: int) -> tuple[Optional[Field], str]:
        """(Field, '') or (None, reason). Never raises."""
        U = self.U
        try:
            el = self.focused()
        except Exception as e:
            return None, f"no focus ({type(e).__name__})"
        if el is None:
            return None, "no focus"
        try:
            pid = int(el.CurrentProcessId)
        except Exception:
            pid = 0
        if pid == self.own_pid:
            return None, "own process"
        fld = Field(pid=pid, el_id=self.runtime_id(el))
        try:
            fld.password = bool(el.CurrentIsPassword)
        except Exception:
            pass
        if fld.password:
            return fld, ""
        try:
            ctype = int(el.CurrentControlType)
        except Exception:
            ctype = 0
        try:
            r = el.CurrentBoundingRectangle
            fld.field_rect = (int(r.left), int(r.top), int(r.right), int(r.bottom))
        except Exception:
            pass
        self._editable_multiline(el, ctype, hwnd, fld)
        if not fld.editable or fld.multiline is False:
            return fld, ""
        ok = False
        try:
            ok = self._uia_text(el, fld)
        except Exception as e:
            log.debug("autocomplete: UIA text failed (%s)", type(e).__name__)
        if not ok:
            try:
                ok = self._ia2_text(hwnd, fld)
            except Exception as e:
                log.debug("autocomplete: IA2 text failed (%s)", type(e).__name__)
        if not ok:
            return fld, "no text access"
        try:
            labels = tuple(x for x in (el.CurrentName, el.CurrentHelpText) if isinstance(x, str))
        except Exception:
            labels = ()
        if is_placeholder(fld.before + fld.after, *labels):
            fld.before, fld.after, fld.placeholder = "", "", True
        if not fld.font_pt and fld.ml_method == "ia2" and fld.method.startswith("uia") and not fld.placeholder:
            try:
                fld.font_pt, name = self._ia2_font(hwnd)
                fld.font_name = fld.font_name or name
            except Exception:
                pass
        if fld.caret is None:
            fld.caret = self._msaa_caret(hwnd) or gui_thread_caret(hwnd)
            if fld.caret is not None:
                fld.method += "+caretfallback"
        if fld.multiline is None and fld.caret and fld.field_rect:
            h = fld.field_rect[3] - fld.field_rect[1]
            fld.multiline = h >= fld.caret[2] * 2.2
            fld.ml_method = "height"
        fld.page_root = el if ctype == U.UIA_DocumentControlTypeId else None
        fld.focus_el = el
        return fld, ""

    def _editable_multiline(self, el, ctype: int, hwnd: int, fld: Field) -> None:
        U = self.U
        try:
            vp = el.GetCurrentPattern(U.UIA_ValuePatternId)
            if vp:
                vp = vp.QueryInterface(U.IUIAutomationValuePattern)
                if vp.CurrentIsReadOnly:
                    fld.editable = False
                    return
        except Exception:
            pass
        if ctype not in (U.UIA_EditControlTypeId, U.UIA_DocumentControlTypeId):
            # some editors expose a Pane / Custom with a TextPattern; let the text pattern + IA2 decide
            pass
        try:
            ml = aria_multiline(el.CurrentAriaProperties)
        except Exception:
            ml = None
        if ml is not None:
            fld.multiline, fld.ml_method = ml, "aria"
            return
        st = self._ia2_states(el, hwnd)
        if st is not None:
            states, msaa = st
            if msaa & STATE_SYSTEM_PROTECTED:
                fld.password = True
                return
            if states & IA2_STATE_MULTI_LINE:
                fld.multiline, fld.ml_method = True, "ia2"
                return
            if states & IA2_STATE_SINGLE_LINE:
                fld.multiline, fld.ml_method = False, "ia2"
                return
        try:
            nh = int(el.CurrentNativeWindowHandle or 0)
        except Exception:
            nh = 0
        if nh:
            m = _win32_multiline(nh)
            if m is not None:
                fld.multiline, fld.ml_method = m, "win32"
                return
        if ctype == U.UIA_DocumentControlTypeId:
            fld.multiline, fld.ml_method = True, "document"

    def _ia2_states(self, el, hwnd: int) -> Optional[tuple[int, int]]:
        """(IA2 states, MSAA state) of the focused accessible: via the UIA legacy pattern (Chromium), else via the
        window's MSAA focus chain (Firefox / Gecko native UIA gives no IAccessible)."""
        U = self.U
        ia = _IA2.get()
        acc, child = None, 0
        try:
            lp = el.GetCurrentPattern(U.UIA_LegacyIAccessiblePatternId)
            if lp:
                a = lp.QueryInterface(U.IUIAutomationLegacyIAccessiblePattern).GetIAccessible()
                if a:
                    acc = a.QueryInterface(ia["IAccessible"])
        except Exception:
            acc = None
        if acc is None:
            try:
                acc, child = ia["focused_acc"](hwnd)
            except Exception:
                return None
        try:
            msaa = int(acc.accState(child) or 0)
        except Exception:
            msaa = 0
        if child:
            return 0, msaa
        try:
            return int(ia["ia2_of"](acc).states), msaa
        except Exception:
            return None

    def _uia_text(self, el, fld: Field) -> bool:
        U = self.U
        S, E = U.TextPatternRangeEndpoint_Start, U.TextPatternRangeEndpoint_End
        p = el.GetCurrentPattern(U.UIA_TextPatternId)
        if not p:
            return False
        tp = p.QueryInterface(U.IUIAutomationTextPattern)
        r = None
        method = "uia"
        try:
            p2 = el.GetCurrentPattern(U.UIA_TextPattern2Id)
            if p2:
                active, caret = p2.QueryInterface(U.IUIAutomationTextPattern2).GetCaretRange()
                if caret is not None:
                    r, method = caret, "uia2"
        except Exception:
            r = None
        sel = None
        try:
            sel = tp.GetSelection()
        except Exception:
            sel = None
        if sel is not None and sel.Length:
            s0 = sel.GetElement(0)
            fld.selection = s0.CompareEndpoints(S, s0, E) != 0
            if r is None:
                r = s0
        if r is None:
            return False
        d = r.Clone()
        d.MoveEndpointByRange(E, d, S)          # degenerate at the caret
        rect_rng = d                            # where the caret really is (its rect), even when d is clamped below
        # Gecko's ranges are document-wide: clamp everything to the field's own range
        doc = None
        try:
            doc = tp.DocumentRange
        except Exception:
            doc = None
        if doc is not None and d.CompareEndpoints(S, doc, S) < 0:
            return False                        # the caret is not inside this element
        if doc is not None and d.CompareEndpoints(S, doc, E) > 0:
            # Chromium (T3 Code, Discord): with the caret at the end of the text its position compares one step
            # past the DocumentRange end. Same element when nothing but line breaks lies in between: read from
            # the document end. Otherwise (Gecko's document-wide ranges) the caret is elsewhere.
            gap = doc.Clone()
            gap.MoveEndpointByRange(S, doc, E)
            gap.MoveEndpointByRange(E, d, S)
            if (gap.GetText(64) or "").strip(" \t\r\n" + ZWSP):
                return False
            d = doc.Clone()
            d.MoveEndpointByRange(S, doc, E)
            method += "+docend"
        b = d.Clone()
        b.MoveEndpointByUnit(S, U.TextUnit_Character, -int(self.before_chars))
        if doc is not None and b.CompareEndpoints(S, doc, S) < 0:
            b.MoveEndpointByRange(S, doc, S)
        fld.before = b.GetText(-1) or ""
        a = d.Clone()
        a.MoveEndpointByUnit(E, U.TextUnit_Character, int(self.after_chars))
        if doc is not None and a.CompareEndpoints(E, doc, E) > 0:
            a.MoveEndpointByRange(E, doc, E)
        fld.after = a.GetText(-1) or ""
        fld.at_start = doc is not None and d.CompareEndpoints(S, doc, S) == 0
        fld.caret = self._caret_rect(rect_rng, fld)
        fld.method = method
        try:
            fld.font_pt, fld.font_name = self._range_font(d)
            if not fld.font_pt and not fld.at_start:
                p = d.Clone()
                if p.MoveEndpointByUnit(S, U.TextUnit_Character, -1):
                    pt, name = self._range_font(p)
                    fld.font_pt, fld.font_name = pt, fld.font_name or name
        except Exception:
            pass
        return True

    def _range_font(self, rng) -> tuple[float, str]:
        """(font size in pt, font name) of a text range; 0 / '' when not supported or mixed."""
        U = self.U
        pt, name = 0.0, ""
        try:
            v = rng.GetAttributeValue(U.UIA_FontSizeAttributeId)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and 3.0 <= float(v) <= 96.0:
                pt = float(v)
        except Exception:
            pass
        try:
            v = rng.GetAttributeValue(U.UIA_FontNameAttributeId)
            if isinstance(v, str):
                name = v.strip()
        except Exception:
            pass
        return pt, name

    def _ia2_font(self, hwnd: int) -> tuple[float, str]:
        """Font at the caret from IAccessible2 text attributes (Gecko gives none through UIA)."""
        ia = _IA2.get()
        acc, child = ia["focused_acc"](hwnd)
        if child:
            return 0.0, ""
        tx = ia["ia2_of"](acc).QueryInterface(ia["IAccessibleText"])
        off = int(tx.caretOffset)
        for o in ((off - 1, off) if off > 0 else (off,)):
            try:
                res = tx.attributes(max(o, 0))
            except Exception:
                continue
            attrs = res[-1] if isinstance(res, (tuple, list)) else res
            pt, name = parse_ia2_font(attrs if isinstance(attrs, str) else "")
            if pt:
                return pt, name
        return 0.0, ""

    def _caret_rect(self, d, fld: Field) -> Optional[tuple[int, int, int]]:
        U = self.U
        S = U.TextPatternRangeEndpoint_Start
        caret = None
        try:
            rects = list(d.GetBoundingRectangles() or [])
            if len(rects) >= 4 and rects[3] > 0:
                caret = int(rects[0]), int(rects[1]), int(rects[3])
        except Exception:
            pass
        if fld.at_start:
            return caret
        prev = None
        try:
            p = d.Clone()
            if p.MoveEndpointByUnit(S, U.TextUnit_Character, -1):
                rects = list(p.GetBoundingRectangles() or [])
                if len(rects) >= 4 and rects[-1] > 0:
                    prev = tuple(int(v) for v in rects[-4:])
        except Exception:
            pass
        return refine_caret(caret, prev)

    def _ia2_text(self, hwnd: int, fld: Field) -> bool:
        ia = _IA2.get()
        acc, child = ia["focused_acc"](hwnd)
        if child:
            return False
        a2 = ia["ia2_of"](acc)
        if fld.multiline is None:
            st = int(a2.states)
            if st & IA2_STATE_MULTI_LINE:
                fld.multiline, fld.ml_method = True, "ia2"
            elif st & IA2_STATE_SINGLE_LINE:
                fld.multiline, fld.ml_method = False, "ia2"
        tx = a2.QueryInterface(ia["IAccessibleText"])
        off = int(tx.caretOffset)
        if off < 0:
            return False
        try:
            if int(tx.nSelections) > 0:
                s, e = tx.selection(0)
                fld.selection = int(s) != int(e)
        except Exception:
            pass
        fld.before = tx.text(max(0, off - int(self.before_chars)), off) or ""
        try:
            fld.after = tx.text(off, off + int(self.after_chars)) or ""
        except Exception:
            fld.after = tx.text(off, -1) or ""          # IA2_TEXT_OFFSET_LENGTH
            fld.after = fld.after[:int(self.after_chars)]
        if fld.after.startswith(OBJ_REPLACEMENT) and not fld.before.replace(OBJ_REPLACEMENT, "").strip():
            # the caret is inside an embedded child (Chromium contenteditable: one per paragraph); this container's
            # text is only placeholders, so it says nothing about the caret's line (was a false "mid-line")
            fld.before = fld.after = ""
            return False
        if off > 0:
            try:
                x, y, wd, h = tx.characterExtents(off - 1, 0)     # IA2_COORDTYPE_SCREEN_RELATIVE
                if h > 0:
                    fld.caret = (int(x + wd), int(y), int(h))
            except Exception:
                pass
        try:
            res = tx.attributes(max(off - 1, 0))
            attrs = res[-1] if isinstance(res, (tuple, list)) else res
            fld.font_pt, fld.font_name = parse_ia2_font(attrs if isinstance(attrs, str) else "")
        except Exception:
            pass
        fld.method = "ia2"
        return True

    def _msaa_caret(self, hwnd: int) -> Optional[tuple[int, int, int]]:
        try:
            ia = _IA2.get()
            c = ia["acc_from_window"](hwnd, -8)      # OBJID_CARET
            x, y, wd, h = c.accLocation(0)
            if h > 0 and (x or y):
                return int(x), int(y), int(h)
        except Exception:
            pass
        return None

    def _document_of(self, el, max_up: int = 40):
        U = self.U
        cur = el
        try:
            for _ in range(max_up):
                cur = self._walker.GetParentElement(cur)
                if cur is None:
                    return None
                if int(cur.CurrentControlType) == U.UIA_DocumentControlTypeId:
                    return cur
        except Exception:
            return None
        return None

    # ---- page context
    def page_text(self, hwnd: int, fld: Field, cap: int) -> tuple[str, str]:
        """(visible text of the page / window, method). Never raises."""
        U = self.U
        S, E = U.TextPatternRangeEndpoint_Start, U.TextPatternRangeEndpoint_End
        raw = ""
        method = ""
        if fld.page_root is not None:
            return "", "field is the document"      # Notepad / Word: the field text already is the page
        root = self._document_of(fld.focus_el) if fld.focus_el is not None else None
        tp = None
        if root is not None:
            try:
                p = root.GetCurrentPattern(U.UIA_TextPatternId)
                tp = p.QueryInterface(U.IUIAutomationTextPattern) if p else None
            except Exception:
                tp = None
        if tp is not None:
            # text just above the field in document order (chat messages above the compose box): cheap and nearest
            try:
                fr = tp.RangeFromChild(fld.focus_el)
                b = fr.Clone()
                b.MoveEndpointByRange(E, fr, S)
                b.MoveEndpointByUnit(S, U.TextUnit_Character, -int(cap * 1.5))
                raw = b.GetText(-1) or ""
                method = "before-field"
            except Exception as e:
                log.debug("autocomplete: before-field range failed (%s)", type(e).__name__)
                raw = ""
        if tp is not None and not raw.strip():
            try:
                vis = tp.GetVisibleRanges()
                chunks, total = [], 0
                for i in range(vis.Length):
                    t = vis.GetElement(i).GetText(max(cap * 4 - total, 0)) or ""
                    chunks.append(t)
                    total += len(t)
                    if total >= cap * 4:
                        break
                raw = "\n".join(chunks)
                method = "visible-ranges"
            except Exception as e:
                log.debug("autocomplete: visible ranges failed (%s)", type(e).__name__)
        if not raw.strip() and root is None:        # no document (classic Win32 app): visible text elements
            try:
                raw = self._names_text(hwnd, cap * 4)
                method = "names"
            except Exception as e:
                log.debug("autocomplete: names fallback failed (%s)", type(e).__name__)
                raw = ""
        return trim_page(raw, (fld.before or "") + (fld.after or ""), cap), method

    def _names_text(self, hwnd: int, cap: int) -> str:
        U = self.U
        root = self.uia.ElementFromHandle(wintypes.HWND(hwnd))
        cr = self.uia.CreateCacheRequest()
        cr.AddProperty(U.UIA_NamePropertyId)
        cond = self.uia.CreatePropertyCondition(U.UIA_ControlTypePropertyId, U.UIA_TextControlTypeId)
        els = root.FindAllBuildCache(U.TreeScope_Descendants, cond, cr)
        out, total = [], 0
        for i in range(min(els.Length, 400)):
            try:
                t = els.GetElement(i).CachedName or ""
            except Exception:
                continue
            if t.strip():
                out.append(t)
                total += len(t) + 1
                if total >= cap:
                    break
        return "\n".join(out)


# ----------------------------------------------------------------------------- LLM
class Completer:
    """Streams a raw /completion from llama-server on slot 1. Own HTTP session (Cleaner.http is replaced on restart)."""

    def __init__(self, base_url: Callable[[], str]):
        import requests
        self._requests = requests
        self.session = requests.Session()
        self._base_url = base_url
        self._tpl: tuple = ("", 0.0, None)          # (base_url, fetched at, template)

    def template(self) -> str:
        """The loaded model's own chat template (system + user + assistant opener, thinking off) with SYS_MARK /
        USER_MARK in it, from llama-server /apply-template; re-fetched every 60 s (model switch) and falling
        back to FALLBACK_TEMPLATE."""
        url = self._base_url()
        now = time.monotonic()
        u, t, tpl = self._tpl
        if u == url and tpl is not None and now - t < 60.0:
            return tpl
        tpl = FALLBACK_TEMPLATE
        try:
            r = self.session.post(url + "/apply-template", timeout=(0.5, 1.5), json={
                "messages": [{"role": "system", "content": SYS_MARK}, {"role": "user", "content": USER_MARK}],
                "add_generation_prompt": True, "chat_template_kwargs": {"enable_thinking": False}})
            if r.status_code == 200:
                p = str(r.json().get("prompt") or "")
                for bos in ("<|startoftext|>", "<|begin_of_text|>", "<s>", "<bos>"):
                    if p.startswith(bos):
                        p = p[len(bos):]        # /completion adds the BOS itself
                        break
                if SYS_MARK in p and USER_MARK in p and p.index(USER_MARK) > p.index(SYS_MARK):
                    tpl = p
        except (self._requests.RequestException, ValueError) as e:
            log.debug("autocomplete: apply-template failed (%s)", type(e).__name__)
        self._tpl = (url, now, tpl)
        return tpl

    def complete(self, prompt: str, *, max_tokens: int, temperature: float, timeout: float,
                 cancelled: Callable[[], bool], max_chars: int) -> Optional[str]:
        """Partial text as it stood when a stop condition hit; None when cancelled or failed."""
        body = {"prompt": prompt, "n_predict": int(max_tokens), "temperature": float(temperature), "top_k": 20,
                "top_p": 0.9, "stop": ["\n", "<|im_end|>", "<think>", "<|endoftext|>"], "cache_prompt": True,
                "id_slot": 1, "stream": True}
        out = []
        resp = None
        try:
            resp = self.session.post(self._base_url() + "/completion", json=body, stream=True,
                                     timeout=(0.5, float(timeout)))
            if resp.status_code != 200:
                log.debug("autocomplete: completion HTTP %s", resp.status_code)
                return None
            for line in resp.iter_lines(chunk_size=64):
                if cancelled():
                    return None
                if not line or not line.startswith(b"data:"):
                    continue
                try:
                    d = json.loads(line[5:].strip() or b"{}")
                except ValueError:
                    continue
                out.append(d.get("content") or "")
                text = "".join(out)
                if d.get("stop") or "\n" in text or len(text) > max_chars + 20 or _STREAM_SENTENCE_RE.search(text):
                    break
            if cancelled():
                return None
            return "".join(out)
        except self._requests.RequestException as e:
            log.debug("autocomplete: completion failed (%s)", type(e).__name__)
            return None
        finally:
            if resp is not None:
                try:
                    resp.close()        # stream=True: closing frees the slot at once
                except Exception:
                    pass

    def close(self) -> None:
        try:
            self.session.close()
        except Exception:
            pass


# ----------------------------------------------------------------------------- controller
class Autocomplete:
    """Owns the worker thread. ``host`` supplies app state (see LocalFlowApp wiring in app.py):
    host.ac_blocked() -> str ('' = free; else why: paused / game / busy), host.ac_server() -> (ready, base_url)."""

    def __init__(self, cfg, host, ghost=None, reader_factory=None, completer_factory=None,
                 typer: Optional[Callable[[str], None]] = None, tapper: Optional[Callable[[int], None]] = None):
        self.cfg = cfg
        self.host = host
        self.ghost_ui = ghost                       # ghost.GhostOverlay (Qt thread) or a test double
        self._reader_factory = reader_factory or (lambda: UiaReader(int(cfg_value(cfg, "before_chars")),
                                                                     int(cfg_value(cfg, "after_chars"))))
        self._completer_factory = completer_factory or (lambda: Completer(lambda: self.host.ac_server()[1]))
        if typer is None:
            from .inserter import type_text as typer
        self._type = typer
        self._tap = tapper or w.tap_vk
        self._lock = threading.Lock()
        self._keys = 0                  # non-modifier key-downs seen by the hook (no characters)
        self._last_key = 0.0
        self._cancel_gen = 0
        self._token = 0
        self._ghost: Optional[Ghost] = None       # shown (or being shown) ghost
        self._pending: Optional[Ghost] = None     # sent to the Qt thread, not yet confirmed shown
        self._accepts: "queue.Queue[Ghost]" = queue.Queue()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.suppressed = SuppressionSet()
        self._ctx_cache: dict = {}                # (hwnd, el_id) -> (time, text)
        self._ctx_refresh: Optional[tuple] = None
        self._diag_seen: dict = {}                # (exe, reason) -> last INFO time
        self.stats = {"shown": 0, "accepted": 0, "dismissed": 0}
        self.running = False
        self._min_visible_s = float(cfg_value(cfg, "min_visible_ms")) / 1000.0
        if ghost is not None:
            try:
                ghost.set_listener(self)
            except Exception:
                log.exception("autocomplete: ghost listener wiring failed")

    # ---------------------------------------------------------------- lifecycle
    def enabled(self) -> bool:
        return bool(cfg_value(self.cfg, "enabled"))

    def apply_config(self) -> None:
        self._min_visible_s = float(cfg_value(self.cfg, "min_visible_ms")) / 1000.0
        if self.enabled():
            self.start()
        else:
            self.stop()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="autocomplete", daemon=True)
        self.running = True
        self._thread.start()
        log.info("autocomplete on (debounce %d ms)", int(cfg_value(self.cfg, "debounce_ms")))

    def stop(self) -> None:
        t = self._thread
        if t is None:
            return
        self._stop.set()
        self.cancel("stopped")
        self._wake.set()
        t.join(timeout=3)
        self._thread = None
        self.running = False
        log.info("autocomplete off")

    # ---------------------------------------------------------------- hook thread API (must stay O(1))
    def note_key(self, vk: int) -> None:
        self._keys += 1
        self._last_key = time.monotonic()
        if self._ghost is not None or self._pending is not None:
            self._drop_ghost("key")
        self._wake.set()

    def want_swallow(self, vk: int) -> bool:
        """Hook thread, hotkey idle + no modifiers held. True = swallow this Tab / Backspace."""
        if not self.running:
            return False
        g = self._ghost
        if not swallow_decision(g, vk, self._keys, _fg_hwnd(), time.monotonic(), self._min_visible_s):
            return False
        with self._lock:
            if self._ghost is not g:
                return False
            self._ghost = None
        self._hide_ui(g.token)
        if vk == VK_TAB:
            self._accepts.put(g)
        else:
            self.suppressed.add(g.supp_key)
            self.stats["dismissed"] += 1
        self._wake.set()
        return True

    # ---------------------------------------------------------------- any thread
    def cancel(self, why: str = "") -> None:
        """New dictation / command, focus change, shutdown: stop the stream and hide the ghost."""
        self._cancel_gen += 1
        self._drop_ghost(why)
        self._wake.set()

    def _drop_ghost(self, why: str) -> None:
        with self._lock:
            g, self._ghost = self._ghost, None
            p, self._pending = self._pending, None
        for x in (g, p):
            if x is not None:
                self._hide_ui(x.token)

    def _hide_ui(self, token: int) -> None:
        if self.ghost_ui is not None:
            try:
                self.ghost_ui.hide_ghost(token)
            except Exception:
                pass

    # ---------------------------------------------------------------- Qt thread callbacks (ghost.py)
    def ghost_shown(self, token: int, text: str, ttl_s: float) -> bool:
        """The ghost is on screen showing ``text`` (possibly shortened to fit). False = it is stale, hide it."""
        with self._lock:
            p = self._pending
            if p is None or p.token != token or self._keys != p.key_count:
                return False
            self._pending = None
            now = time.monotonic()
            full = len(p.text)
            p.text, p.shown_at, p.expires_at = text, now, now + ttl_s
            self._ghost = p
        self.stats["shown"] += 1
        log.info("autocomplete: ghost shown (%d of %d chars) app=%s", len(text), full, p.exe or "?")
        self._wake.set()
        return True

    def ghost_failed(self, token: int, reason: str = "") -> None:
        with self._lock:
            p = self._pending
            if p is not None and p.token == token:
                self._pending = None
            else:
                p = None
        if p is not None:
            log.info("autocomplete: ghost layout failed (%s) app=%s", reason or "?", p.exe or "?")

    # ---------------------------------------------------------------- worker
    def _run(self) -> None:
        reader = completer = None
        try:
            try:
                reader = self._reader_factory()
                completer = self._completer_factory()
            except Exception:
                log.exception("autocomplete: could not start UI Automation - autocomplete off")
                self.running = False
                return
            handled = self._keys
            while not self._stop.is_set():
                self._wake.wait(0.05 if (self._ghost or self._pending) else 0.5)
                self._wake.clear()
                if self._stop.is_set():
                    break
                self._drain_accepts(reader)
                if self._ghost is not None:
                    self._watch_ghost(reader)
                k = self._keys
                if k == handled:
                    continue
                debounce = float(cfg_value(self.cfg, "debounce_ms")) / 1000.0
                quiet = time.monotonic() - self._last_key
                if quiet < debounce:
                    self._stop.wait(min(debounce - quiet, 0.5))
                    self._wake.set()
                    continue
                handled = k
                try:
                    self._predict(reader, completer, k)
                    if self._keys == k:
                        self._refresh_context(reader)
                except Exception:
                    log.exception("autocomplete: prediction failed")
        finally:
            self._drop_ghost("stop")
            if completer is not None:
                completer.close()
            self.running = False

    def _drain_accepts(self, reader) -> None:
        while True:
            try:
                g = self._accepts.get_nowait()
            except queue.Empty:
                return
            try:
                self._accept(reader, g)
            except Exception:
                log.exception("autocomplete: accept failed")

    def _accept(self, reader, g: Ghost) -> None:
        """Tab was swallowed for ``g``: type it if the field still looks the same, else give the app its Tab."""
        ok = False
        why = ""
        if _fg_hwnd() != g.hwnd:
            why = "window changed"
        else:
            fld, r = reader.read(g.hwnd)
            if fld is None or r:
                why = f"re-read failed ({r or 'none'})"
            elif fld.el_id != g.el_id or tail_hash(fld.before) != g.before_hash or fld.selection:
                why = "text changed"
            else:
                ok = True
        if not ok:
            log.info("autocomplete: Tab passed through (%s)", why)
            self._tap(VK_TAB)
            return
        text = strip_controls(g.text)[:int(cfg_value(self.cfg, "max_chars"))]
        if not text:
            self._tap(VK_TAB)
            return
        self._type(text)
        self.stats["accepted"] += 1
        log.info("autocomplete: accepted (%d chars)", len(text))
        # chain: guess the next words after the usual pause, as if a key was typed (one guess per Tab, no loop:
        # the typed text itself is injected and never reaches note_key)
        self._last_key = time.monotonic()
        self._keys += 1

    def _watch_ghost(self, reader) -> None:
        g = self._ghost
        if g is None:
            return
        now = time.monotonic()
        why = ""
        if now > g.expires_at:
            why = "ttl"
        elif _fg_hwnd() != g.hwnd:
            why = "focus"
        elif _mouse_down():
            why = "mouse"
        elif self._keys != g.key_count:
            why = "key"
        elif now - getattr(g, "_checked", g.shown_at) > 0.3:
            g._checked = now  # type: ignore[attr-defined]
            fld, r = reader.read(g.hwnd)
            why = watch_verdict(g, fld, r, time.monotonic())
        if why:
            with self._lock:
                if self._ghost is g:
                    self._ghost = None
            self._hide_ui(g.token)
            log.info("autocomplete: ghost hidden (%s) after %d ms app=%s", why, int((now - g.shown_at) * 1000),
                     g.exe or "?")

    def _predict(self, reader, completer, k: int) -> None:
        hwnd = _fg_hwnd()
        if not hwnd or _window_pid(hwnd) == os.getpid():
            return
        try:
            exe, title = w.window_info(hwnd)
        except Exception:
            exe, title = "", ""
        blocked = self.host.ac_blocked()
        if blocked:
            self._diag(exe, f"blocked: {blocked}")
            return
        ready, _url = self.host.ac_server()
        if not ready:
            self._diag(exe, "model server not ready")
            return
        if app_excluded(exe, cfg_value(self.cfg, "exclude_apps")):
            self._diag(exe, "excluded app")
            return
        if is_private_title(title):
            self._diag(exe, "private window")
            return
        t0 = time.perf_counter()
        fld, why = reader.read(hwnd)
        t_read = (time.perf_counter() - t0) * 1000
        if fld is None:
            self._diag(exe, why or "no field")
            return
        why = why or skip_reason(fld, int(cfg_value(self.cfg, "min_chars")))
        if why:
            self._diag(exe, f"{why} via {fld.method or '-'}/{fld.ml_method or '-'}"
                            f"{' (placeholder text)' if fld.placeholder else ''}")
            return
        self.suppressed.prune(hwnd, fld.el_id, fld.before)
        if sentence_suppressed(self.suppressed, hwnd, fld.el_id, fld.before):
            self._diag(exe, "dismissed with Backspace (this sentence)")
            return
        if self._keys != k or _fg_hwnd() != hwnd:
            self._diag(exe, "stale: typed / switched during the read")
            return
        page, pmethod = self._page_context(reader, hwnd, fld)
        gen = self._cancel_gen
        cancelled = lambda: self._keys != k or self._cancel_gen != gen or self._stop.is_set()
        if cancelled():
            self._diag(exe, "stale: typed during the page read")
            return
        if not page and _word_count(strip_controls(fld.before)) < MIN_WORDS_NO_PAGE:
            self._diag(exe, "too little context (no page text, few words)")   # the model answers the instruction
            return
        blocked = self.host.ac_blocked()
        if blocked:
            self._diag(exe, f"blocked: {blocked}")
            return
        tpl_fn = getattr(completer, "template", None)
        prompt = build_prompt(fld.before[-int(cfg_value(self.cfg, "before_chars")):], page,
                              tpl_fn() if callable(tpl_fn) else None)
        t1 = time.perf_counter()
        max_chars = int(cfg_value(self.cfg, "max_chars"))
        raw = completer.complete(prompt, max_tokens=int(cfg_value(self.cfg, "max_tokens")),
                                 temperature=float(cfg_value(self.cfg, "temperature")),
                                 timeout=float(cfg_value(self.cfg, "timeout_s")), cancelled=cancelled,
                                 max_chars=max_chars)
        t_llm = (time.perf_counter() - t1) * 1000
        if raw is None or cancelled():
            self._diag(exe, "stale: typed during the guess" if cancelled() else "model request failed")
            return
        text, rule = postprocess_ex(raw, fld.before, fld.after, max_chars, page)
        log.info("autocomplete: %s via %s/%s read %.0f ms llm %.0f ms before=%d page=%d(%s) raw=%d out=%d "
                 "font=%.1fpt app=%s",
                 "guess" if text else f"no guess ({rule})", fld.method, fld.ml_method, t_read, t_llm,
                 len(fld.before), len(page), pmethod, len(raw), len(text or ""), fld.font_pt, exe)
        if not text:
            return
        if cancelled() or _fg_hwnd() != hwnd:
            self._diag(exe, "stale: typed / switched during the guess")
            return
        with self._lock:
            if self._keys != k:
                return
            self._token += 1
            g = Ghost(token=self._token, hwnd=hwnd, key_count=k, text=text,
                      supp_key=sentence_key(hwnd, fld.el_id, fld.before), before_hash=tail_hash(fld.before),
                      el_id=fld.el_id, caret=fld.caret, field_rect=fld.field_rect, exe=exe,
                      caret_src=fld.method, before_len=len(fld.before), font=(fld.font_pt, fld.font_name) if (fld.font_pt or fld.font_name) else None)
            self._pending = g
        if self.ghost_ui is not None:
            self.ghost_ui.show_ghost(g.token, text, g.caret, g.field_rect, float(cfg_value(self.cfg, "ttl_s")),
                                     g.font)

    def _page_context(self, reader, hwnd: int, fld: Field) -> tuple[str, str]:
        """Visible page text, cached per field; a fresh read only when the field changes or the cache is old
        (new chat messages). Read before the prediction for a new field, after it (refresh) otherwise."""
        cap = int(cfg_value(self.cfg, "context_chars"))
        if cap <= 0:
            return "", "off"
        key = (hwnd, fld.el_id)
        hit = self._ctx_cache.get(key)
        if hit is not None:
            if time.monotonic() - hit[0] >= 10.0:
                self._ctx_refresh = (hwnd, fld)     # stale: use it now, re-read after this prediction
            return hit[1], "cache"
        return self._read_context(reader, hwnd, fld, cap)

    def _read_context(self, reader, hwnd: int, fld: Field, cap: int) -> tuple[str, str]:
        text, method = reader.page_text(hwnd, fld, cap)
        if len(self._ctx_cache) > 16:
            self._ctx_cache.clear()
        self._ctx_cache[(hwnd, fld.el_id)] = (time.monotonic(), text)
        return text, method

    def _refresh_context(self, reader) -> None:
        job, self._ctx_refresh = self._ctx_refresh, None
        if job is None or self._stop.is_set():
            return
        hwnd, fld = job
        if _fg_hwnd() != hwnd:
            return
        try:
            self._read_context(reader, hwnd, fld, int(cfg_value(self.cfg, "context_chars")))
        except Exception as e:
            log.debug("autocomplete: context refresh failed (%s)", type(e).__name__)

    def _diag(self, exe: str, why: str) -> None:
        """Why a pause got no ghost: INFO once per (app, reason) per minute, every time with autocomplete.debug_log
        (helps 'why nothing in X?'). Reasons / method names only, never text."""
        key = (exe or "?", why)
        now = time.monotonic()
        last = self._diag_seen.get(key)
        if bool(cfg_value(self.cfg, "debug_log")) or last is None or now - last >= DIAG_EVERY_S:
            if len(self._diag_seen) > 500:
                self._diag_seen.clear()
            self._diag_seen[key] = now
            log.info("autocomplete: no suggestion in %s (%s)", exe or "?", why)
        else:
            log.debug("autocomplete: skip (%s) app=%s", why, exe or "?")
