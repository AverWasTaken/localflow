"""Pure helpers for the UI (no Qt): number / duration formatting, hotkey validation, transcript parsing."""
from __future__ import annotations

import datetime as _dt
import re

MODIFIERS = ("ctrl", "alt", "shift", "win")


def fmt_int(n) -> str:
    try:
        return f"{int(n):,}"
    except Exception:
        return "0"


def fmt_saved(seconds) -> str:
    """Time saved: "3h 12m", "45m", "under 1m" (0 -> "0m")."""
    try:
        s = max(0.0, float(seconds))
    except Exception:
        s = 0.0
    if s <= 0:
        return "0m"
    m = int(s // 60)
    if m < 1:
        return "under 1m"
    h, m = divmod(m, 60)
    return f"{h}h {m}m" if h else f"{m}m"


def fmt_streak(days) -> str:
    try:
        d = max(0, int(days))
    except Exception:
        d = 0
    return "1 day" if d == 1 else f"{d} days"


def fmt_wpm(wpm) -> str:
    try:
        v = float(wpm)
    except Exception:
        v = 0.0
    return str(int(round(v))) if v > 0 else "-"


def fmt_duration(seconds) -> str:
    """Clock-style length: "4:05", "1:02:09"; 0 / unknown -> "-"."""
    try:
        s = int(round(float(seconds)))
    except Exception:
        return "-"
    if s <= 0:
        return "-"
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def fmt_elapsed(seconds) -> str:
    """Recording timer: always "MM:SS" or "H:MM:SS"."""
    s = max(0, int(seconds or 0))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def fmt_when(ts, now: _dt.datetime | None = None) -> str:
    """"Today, 2:04 PM" / "Yesterday, 9:10 AM" / "Mon Nov 3, 4:00 PM" / "Nov 3, 2025, 4:00 PM"."""
    try:
        d = _dt.datetime.fromtimestamp(float(ts))
    except Exception:
        return ""
    now = now or _dt.datetime.now()
    clock = d.strftime("%I:%M %p").lstrip("0")
    if d.date() == now.date():
        return f"Today, {clock}"
    if d.date() == (now - _dt.timedelta(days=1)).date():
        return f"Yesterday, {clock}"
    if d.year == now.year:
        return f"{d.strftime('%a %b')} {d.day}, {clock}"
    return f"{d.strftime('%b')} {d.day}, {d.year}, {clock}"


def fmt_when_short(ts, now: _dt.datetime | None = None) -> str:
    """Compact list date: "2:04 PM" today, "Yesterday", "Nov 3", "Nov 3, 2025"."""
    try:
        d = _dt.datetime.fromtimestamp(float(ts))
    except Exception:
        return ""
    now = now or _dt.datetime.now()
    if d.date() == now.date():
        return d.strftime("%I:%M %p").lstrip("0")
    if d.date() == (now - _dt.timedelta(days=1)).date():
        return "Yesterday"
    if d.year == now.year:
        return f"{d.strftime('%b')} {d.day}"
    return f"{d.strftime('%b')} {d.day}, {d.year}"


def parse_mods(spec: str) -> frozenset[str]:
    return frozenset(t for t in (p.strip().lower() for p in str(spec or "").split("+")) if t in MODIFIERS)


def fmt_combo(mods) -> str:
    names = {"ctrl": "Ctrl", "alt": "Alt", "shift": "Shift", "win": "Win"}
    return "+".join(names[m] for m in MODIFIERS if m in set(mods))


def fmt_chord(spec: str) -> str:
    """"alt+shift+z" -> "Alt+Shift+Z"."""
    out = []
    for t in (p.strip() for p in str(spec or "").split("+")):
        if not t:
            continue
        low = t.lower()
        out.append({"ctrl": "Ctrl", "alt": "Alt", "shift": "Shift", "win": "Win"}.get(low, t.upper() if len(t) == 1
                                                                                         else t.capitalize()))
    return "+".join(out)


def command_combo_error(dictate, command) -> str:
    """Mirror of the hotkey manager's rule: the command combo must be non-empty and neither equal to, nor a subset
    or superset of, the dictation combo. Returns "" when fine, else a short message for the UI."""
    d, c = frozenset(dictate), frozenset(command)
    if not c:
        return "Pick at least one key for command mode."
    if c == d:
        return "Command mode needs a different key combination than dictation."
    if c < d:
        return f"{fmt_combo(c)} is part of the dictation keys ({fmt_combo(d)}). Pick a combination that isn't."
    if d < c:
        return f"{fmt_combo(c)} contains the dictation keys ({fmt_combo(d)}). Pick a combination that doesn't."
    return ""


RETENTION_CHOICES = [(0, "Forever"), (7, "7 days"), (30, "30 days"), (90, "90 days")]

PRESET_LABELS = {"summary": "Summary", "bullets": "Bullet points", "key_points": "Key points",
                 "action_items": "Action items"}


def preset_label(preset: str, prompt: str = "") -> str:
    if preset == "custom":
        p = (prompt or "").strip().replace("\n", " ")
        return (p[:60] + "...") if len(p) > 60 else (p or "Custom question")
    return PRESET_LABELS.get(preset, (preset or "").replace("_", " ").capitalize())


_TURN = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\]\s+([^:\n]{1,40}):\s?(.*)$", re.S)


def parse_turns(text: str) -> list[tuple[str, str, str]]:
    """Meeting transcript text ("[00:01:02] You: words" paragraphs) -> [(timestamp, speaker, text)]. Paragraphs
    without the prefix are kept as ("", "", text)."""
    out = []
    for para in (text or "").split("\n\n"):
        p = para.strip()
        if not p:
            continue
        m = _TURN.match(p)
        if m:
            ts, who, body = m.groups()
            if ts.startswith("00:"):
                ts = ts[3:]
            out.append((ts, who.strip(), body.strip()))
        else:
            out.append(("", "", p))
    return out


def level_unit(rms) -> float:
    """Raw RMS (speech ~0.02-0.2) -> 0..1 meter fill on a dB scale (-52 dB -> 0, -12 dB -> 1), like the overlay."""
    import math
    try:
        raw = float(rms)
    except Exception:
        return 0.0
    if raw <= 1e-5:
        return 0.0
    v = (20.0 * math.log10(raw) + 52.0) / 40.0
    return max(0.0, min(1.0, v))


STATUS_LABELS = {"queued": "Waiting", "transcribing": "Transcribing", "cleaning": "Cleaning up", "done": "Done",
                 "failed": "Failed", "interrupted": "Interrupted", "cancelled": "Cancelled"}
ACTIVE = ("queued", "transcribing", "cleaning")
RESUMABLE = ("interrupted", "failed", "cancelled")


def status_text(status: str, progress=None, msg: str = "") -> str:
    """"Transcribing 42%", "Waiting", "Cleaning up 80%, Loading audio (You)"."""
    base = STATUS_LABELS.get(status or "", (status or "").capitalize())
    try:
        pct = int(round(float(progress) * 100))
    except Exception:
        pct = None
    if status in ("transcribing", "cleaning") and pct is not None and pct > 0:
        base = f"{base} {min(pct, 99)}%"
    if msg and status in ACTIVE:
        word = STATUS_LABELS.get(status, "")
        if word and msg.lower().startswith(word.lower()):
            rest = msg[len(word):].strip()          # "Transcribing (You)" -> "Transcribing 42% (You)"
            base = f"{base} {rest}" if rest else base
        else:
            base = f"{base}. {msg}"
    return base
