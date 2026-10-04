"""Deterministic text processing: dictionary replacements, snippets, regex cleaner, style post-pass."""
from __future__ import annotations

import re

from . import styles

# ----------------------------------------------------------------------------- dictionary / snippets
def apply_replacements(text: str, rules: dict[str, str]) -> str:
    """Whole-word, case-insensitive 'heard -> wanted' replacements (longest first)."""
    for src in sorted((s for s in rules if s.strip()), key=len, reverse=True):
        pat = r"(?<![\w])" + re.escape(src.strip()) + r"(?![\w])"
        text = re.sub(pat, lambda m, r=rules[src]: r, text, flags=re.IGNORECASE)
    return text


def _squash(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s']", " ", s.lower()).split())


def match_snippet_whole(text: str, snippets: dict[str, str]) -> str | None:
    """If the whole utterance is exactly a trigger phrase, return its expansion."""
    q = _squash(text)
    if not q:
        return None
    for trig, exp in snippets.items():
        if trig.strip() and _squash(trig) == q:
            return exp
    return None


def apply_snippets_inline(text: str, snippets: dict[str, str]) -> str:
    """Replace trigger phrases appearing inside a longer sentence."""
    for trig in sorted((t for t in snippets if t.strip()), key=len, reverse=True):
        words = [re.escape(w) for w in trig.split()]
        pat = r"(?<![\w])" + r"[\s,;:]*".join(words) + r"(?![\w])"
        exp = snippets[trig]
        text = re.sub(pat, lambda m, e=exp: e, text, flags=re.IGNORECASE)
    return text


# ----------------------------------------------------------------------------- spoken punctuation (fallback)
_SPOKEN = [
    (r"\s*\bnew paragraph\b[\s,.]*", "\n\n"),
    (r"\s*\b(?:new line|newline|next line)\b[\s,.]*", "\n"),
    (r"\s*\bquestion mark\b", "?"),
    (r"\s*\bexclamation (?:point|mark)\b", "!"),
    (r"\s*\bfull stop\b", "."),
    (r"\s*\bsemicolon\b", ";"),
    (r"\s*\bcomma\b", ","),
]
_FILLER_RE = re.compile(
    r"(?:(?<=\s)|^)(?:um+|uh+|uhm+|erm|er|ah+|hmm+|mm+|you know|i mean)(?=[\s,.!?]|$)[,\s]*",
    re.IGNORECASE)
_STUTTER_RE = re.compile(r"\b(\w+)(?:[\s,]+\1\b)+", re.IGNORECASE)


def apply_spoken_punctuation(text: str, final_period: bool = True) -> str:
    for pat, rep in _SPOKEN:
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)
    if final_period:
        text = re.sub(r"\s*\bperiod\s*[.]?\s*$", ".", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text


def capitalize_sentences(text: str) -> str:
    def cap(m):
        return m.group(1) + m.group(2).upper()
    text = re.sub(r"^(\s*)([a-z])", cap, text)
    text = re.sub(r"([.!?]\s+|\n)([a-z])", cap, text)
    text = re.sub(r"\bi\b(?=['\s,.!?]|$)", "I", text)
    return text


def _style_def(style) -> dict:
    if isinstance(style, dict):
        return style
    return styles.DEFAULT_STYLES.get(styles.canon_style(style), styles.DEFAULT_STYLES[styles.NEUTRAL])


def regex_clean(text: str, style="neutral", abbr: dict | None = None, preserve: list | None = None,
                post: bool = True) -> str:
    """Deterministic fallback cleaner: fillers, stutters, spoken punctuation, capitalisation, then the
    style's deterministic pass. ``style`` is a style name or a style definition dict."""
    sd = _style_def(style)
    t = text.strip()
    t = _FILLER_RE.sub("", t)
    t = _STUTTER_RE.sub(lambda m: m.group(1), t)
    t = apply_spoken_punctuation(t)
    t = re.sub(r"[ \t]{2,}", " ", t).strip(" ,")
    if sd.get("capitalize", True):
        t = capitalize_sentences(t)
        if t and t[-1] not in ".!?\n" and not t.endswith(("...", ":")):
            t += "."
    return apply_style(t, sd, abbr, preserve) if post else t


def apply_style(text: str, style="neutral", abbr: dict | None = None, preserve: list | None = None) -> str:
    """Final per-context tweaks (deterministic, applied after the LLM too). See styles.apply_style_def."""
    return styles.apply_style_def(text, _style_def(style), abbr, preserve)


def simple_words(s: str) -> list[str]:
    s = re.sub(r"\b([ap])\.\s?m\b\.?", r"\1m", s.lower())
    return re.findall(r"[a-z]+(?:'[a-z]+)*|\d+", s)


def norm_words(s: str) -> list[str]:
    """simple_words with apostrophes dropped, so don't/dont and let's/lets compare equal."""
    return [w.replace("'", "") for w in simple_words(s.replace("\u2019", "'"))]
