"""keep_words guard: undo word swaps the cleanup model made on its own.

Small cleanup models sometimes "correct" a word they don't recognise into a different, wrong one ("unhelped
technicians" -> "helpers", "pitch yaw" -> "YAPAC", "yaw is" -> "voice are"). The speech-to-text words are usually
right, so after the LLM output passed the guard, ``repair(raw, out, ...)`` aligns the two word sequences and puts
the spoken words back wherever the output holds a word nothing supports.

An output word is supported when
  - it was spoken (a raw word not already matched elsewhere; counted, so a word said once supports one use; this
    keeps self-corrections: "monday at 3 i mean tuesday" -> "Tuesday at 3"),
  - it is a formatting / contraction / number / abbreviation form of the spoken words (both sides are canonicalized
    the same way: lowercase, no apostrophes, "do not"/"don't"/"dont", "gonna"/"going to", "five"/"5",
    "sub agents"/"subagents", the user's abbreviations and the style's word_map),
  - it is one of the light grammar fixes the prompt allows: a homophone (their/there/they're, then/than, ...),
    subject-verb agreement (show/shows, was/were, is/are, has/have, do/does), a/an,
  - it is a dictionary word replacing a similar-sounding spoken word (sky -> Skye, "a j" -> AJ; NOT an unrelated
    word: slash -> Fifine), or
  - the user's own settings ask for it (replacement / snippet targets, abbreviations, word_map values, the words of a
    custom style prompt or custom style examples), so a rule like "replace AJ with Aiden Morales" keeps working.

Changes are judged per aligned region, all or nothing: if any output word in a region is unsupported, the whole
region goes back to the spoken words ("voice are" -> "yaw is", not "yaw are"); an unsupported insertion is removed.
Deletions are never touched (removing fillers and retracted words is the model's job). Never raises.
"""
from __future__ import annotations

import difflib
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional


log = logging.getLogger(__name__)

# styles whose job is to keep the speaker's wording (professional / email are meant to reword)
STYLES = frozenset({"neutral", "casual", "code"})

_TOKEN_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?!\d)|[^\W_]+(?:['’][^\W_]+)*")

# --------------------------------------------------------------------------- canonical forms
_EXPAND: dict[str, tuple[str, ...]] = {
    "dont": ("do", "not"), "doesnt": ("does", "not"), "didnt": ("did", "not"), "cant": ("can", "not"),
    "cannot": ("can", "not"), "wont": ("will", "not"), "isnt": ("is", "not"), "arent": ("are", "not"),
    "wasnt": ("was", "not"), "werent": ("were", "not"), "wouldnt": ("would", "not"), "couldnt": ("could", "not"),
    "shouldnt": ("should", "not"), "havent": ("have", "not"), "hasnt": ("has", "not"), "hadnt": ("had", "not"),
    "im": ("i", "am"), "ive": ("i", "have"), "youre": ("you", "are"), "theyre": ("they", "are"),
    "youve": ("you", "have"), "weve": ("we", "have"), "theyve": ("they", "have"), "thats": ("that", "is"),
    "whats": ("what", "is"), "theres": ("there", "is"), "heres": ("here", "is"), "yall": ("you", "all"),
    "gonna": ("going", "to"), "wanna": ("want", "to"), "gotta": ("got", "to"), "kinda": ("kind", "of"),
    "sorta": ("sort", "of"), "lemme": ("let", "me"), "gimme": ("give", "me"), "dunno": ("do", "not", "know"),
    "cuz": ("because",), "coz": ("because",), "cause": ("because",), "ok": ("okay",), "til": ("until",),
    "till": ("until",), "thru": ("through",), "tho": ("though",), "u": ("you",), "ur": ("your",),
    "pls": ("please",), "plz": ("please",), "thx": ("thanks",), "alright": ("all", "right"),
}
# X's -> X is only for these (John's stays johns)
_IS_BASES = frozenset({"it", "that", "he", "she", "who", "what", "where", "how", "there", "here", "when", "why"})
_NT_BASES = {"ca": "can", "wo": "will", "sha": "shall", "ai": "aint"}

_UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                    "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
_ORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
         "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
         "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19,
         "twentieth": 20, "thirtieth": 30, "fortieth": 40, "fiftieth": 50, "sixtieth": 60, "seventieth": 70,
         "eightieth": 80, "ninetieth": 90}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 10 ** 6, "billion": 10 ** 9}
_ORD_DIGITS_RE = re.compile(r"(\d+)(?:st|nd|rd|th)")
_GROUPED_RE = re.compile(r"\d{1,3}(?:,\d{3})+")          # 12,000
_THOUSANDS_RE = re.compile(r"(\d+)k")                     # 12k

# raw-side words left out of the alignment (still count as spoken, so an output that keeps them is fine)
_HESITATIONS = frozenset({"um", "umm", "uh", "uhh", "uhm", "er", "erm", "ah", "ahh", "hmm", "hm", "mm", "mhm"})
_FILLERS = frozenset({"like"})
_FILLER_PAIRS = frozenset({("you", "know"), ("i", "mean")})
# spoken symbol / punctuation names: the output writes the symbol instead ("document slash coding" -> "document/coding")
_SYMBOLS = frozenset({"paren", "parens", "parenthesis", "bracket", "brackets", "brace", "braces", "colon", "dash",
                      "hyphen", "quote", "quotes", "slash", "backslash", "asterisk", "dot", "period", "comma",
                      "semicolon", "underscore"})
_SYMBOL_PAIRS = frozenset({("question", "mark"), ("exclamation", "point"), ("exclamation", "mark"), ("new", "line"),
                           ("new", "paragraph"), ("full", "stop")})

# light grammar fixes the cleanup prompt allows
_HOMOPHONES = [frozenset(g) for g in (("their", "there", "theyre"), ("your", "youre"), ("then", "than"),
                                       ("to", "too"), ("were", "where"))]
_AGREE = [frozenset(g) for g in (("is", "are", "am", "was", "were"), ("has", "have"), ("do", "does"),
                                  ("a", "an"))]
# custom prompt text: words too common to count as "the settings asked for this word"
_STOP = frozenset("""a an the to of and or but if then than so as at by for from in into on onto off out over with
without is are was were be been being am it its this that these those there here i you he she we they me him her
us them my your his our their not no yes do does did done have has had will would can could should may might must
use using used say said write written replace replaced always never instead when what which who how word words
style text make like just keep only any all every each some more most less very also too please don't dont""".split())


def _canon_word(surface: str, abbr_rev: dict, word_map: dict, depth: int = 0) -> list[str]:
    """Canonical words for one token (usually one, sometimes an expansion)."""
    s = surface.lower().replace("’", "'")
    if "'" in s:
        if s.endswith("n't"):
            base = s[:-3]
            return [_NT_BASES.get(base, base), "not"] if base else [s.replace("'", "")]
        base, _, suf = s.rpartition("'")
        if suf == "re":
            return [base, "are"]
        if suf == "ve":
            return [base, "have"]
        if suf == "m" and base == "i":
            return ["i", "am"]
        if suf == "ll":
            return [base, "will"]
        if suf == "s" and base in _IS_BASES:
            return [base, "is"]
        s = s.replace("'", "")
    m = _ORD_DIGITS_RE.fullmatch(s)
    if m:
        return [str(int(m.group(1)))]
    if _GROUPED_RE.fullmatch(s):
        return [str(int(s.replace(",", "")))]
    m = _THOUSANDS_RE.fullmatch(s)
    if m:
        return [str(int(m.group(1)) * 1000)]
    if s.isdigit():
        return [str(int(s))] if s.strip("0") or len(s) == 1 else [s]
    if depth < 2:
        if s in word_map:
            return _canon_phrase(word_map[s], abbr_rev, word_map, depth + 1)
        if s in abbr_rev:
            return _canon_phrase(abbr_rev[s], abbr_rev, word_map, depth + 1)
    if s in _EXPAND:
        return list(_EXPAND[s])
    return [s]


def _canon_phrase(text: str, abbr_rev: dict, word_map: dict, depth: int = 0) -> list[str]:
    out: list[str] = []
    for m in _TOKEN_RE.finditer(text or ""):
        out += _canon_word(m.group(0), abbr_rev, word_map, depth)
    return out


def _parse_numbers(words: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    """Runs of number words -> one digit token each ("twenty three" -> "23", "three thirty" -> "3", "30").
    Items are (word, first source token, last source token)."""
    out: list[tuple[str, int, int]] = []
    i, n = 0, len(words)
    while i < n:
        w = words[i][0]
        digit_scale = w.isdigit() and i + 1 < n and words[i + 1][0] in _SCALES       # "5 million"
        if not (w in _UNITS or w in _TENS or w in _ORDS or w in _SCALES or digit_scale):
            out.append(words[i])
            i += 1
            continue
        # parse one number starting at i
        start = i
        total, cur, last = 0, 0, None       # last: unit / teen / tens / hundred / scale
        seen = False
        if digit_scale:
            cur, last, seen = int(w), "unit", True
            i += 1
        while i < n:
            w = words[i][0]
            if w in _ORDS:
                v = _ORDS[w]
                if last in (None, "hundred", "scale") or (last == "tens" and cur % 10 == 0 and v < 10):
                    cur += v
                    seen = True
                    i += 1
                break
            if w in _UNITS:
                v = _UNITS[w]
                if last in (None, "hundred", "scale") or (last == "tens" and v < 10 and v != 0):
                    cur += v
                    last = "teen" if v >= 10 else "unit"
                    seen = True
                    i += 1
                    continue
                break
            if w in _TENS:
                if last in (None, "hundred", "scale"):
                    cur += _TENS[w]
                    last = "tens"
                    seen = True
                    i += 1
                    continue
                break
            if w == "hundred":
                if last in ("unit", "teen", "tens") and (cur < 100 or digit_scale):
                    cur = cur * 100
                    last = "hundred"
                    i += 1
                    continue
                if last is None:
                    cur = 100
                    last = "hundred"
                    seen = True
                    i += 1
                    continue
                break
            if w in _SCALES:
                if last in ("unit", "teen", "tens", "hundred"):
                    total += cur * _SCALES[w]
                    cur = 0
                    last = "scale"
                    i += 1
                    continue
                if last is None:
                    total += _SCALES[w]
                    last = "scale"
                    seen = True
                    i += 1
                    continue
                break
            if w == "and" and last in ("hundred", "scale") and i + 1 < n and (
                    words[i + 1][0] in _UNITS or words[i + 1][0] in _TENS):
                i += 1
                continue
            break
        if not seen or i == start:
            out.append(words[start])
            i = start + 1
            continue
        out.append((str(total + cur), words[start][1], words[i - 1][2]))
    return out


@dataclass
class _Side:
    text: str
    toks: list = field(default_factory=list)       # (surface, start, end)
    canon: list = field(default_factory=list)      # (word, first tok, last tok)
    dropped: list = field(default_factory=list)    # (word, tok) left out of the alignment (raw side only)


def _tokenize(text: str, abbr_rev: dict, word_map: dict, drop: bool) -> _Side:
    side = _Side(text)
    side.toks = [(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(text or "")]
    words: list[tuple[str, int, int]] = []
    for ti, (surf, s, e) in enumerate(side.toks):
        low = surf.lower()
        # "4:00" -> the ":00" is formatting, not a word
        if not drop and low.strip("0") == "" and len(low) == 2 and s > 0 and text[s - 1] == ":":
            continue
        for w in _canon_word(surf, abbr_rev, word_map):
            words.append((w, ti, ti))
    # "p m" / "a m" -> pm / am
    merged: list[tuple[str, int, int]] = []
    k = 0
    while k < len(words):
        if k + 1 < len(words) and words[k][0] in ("a", "p") and words[k + 1][0] == "m" and \
                side.toks[words[k][1]][0][:1].isalpha() and len(side.toks[words[k][1]][0]) == 1:
            merged.append((words[k][0] + "m", words[k][1], words[k + 1][2]))
            k += 2
            continue
        merged.append(words[k])
        k += 1
    words = _parse_numbers(merged)
    if drop:
        kept: list[tuple[str, int, int]] = []
        k = 0
        while k < len(words):
            w = words[k][0]
            pair = (w, words[k + 1][0]) if k + 1 < len(words) else None
            if pair and (pair in _FILLER_PAIRS or pair in _SYMBOL_PAIRS):
                side.dropped += [(w, words[k][1]), (pair[1], words[k + 1][1])]
                k += 2
                continue
            if w in _HESITATIONS or w in _FILLERS or w in _SYMBOLS:
                side.dropped.append((w, words[k][1]))
                k += 1
                continue
            kept.append(words[k])
            k += 1
        words = kept
    side.canon = words
    return side


# --------------------------------------------------------------------------- support rules
def _alnum(s: str) -> str:
    return re.sub(r"[^0-9a-z]", "", s)


def _agree(r: str, o: str) -> bool:
    if r == o:
        return True
    for g in _AGREE:
        if r in g and o in g:
            return True
    a, b = (r, o) if len(r) <= len(o) else (o, r)
    if len(a) >= 3 and (b == a + "s" or b == a + "es" or (a.endswith("y") and b == a[:-1] + "ies")):
        return True
    return False


def _pair_ok(r: str, o: str) -> bool:
    if r == o:
        return True
    if o.isdigit() or r.isdigit():
        return False
    if _agree(r, o):
        return True
    return any(r in g and o in g for g in _HOMOPHONES)


@dataclass
class _Ctx:
    vocab: dict          # canonical joined vocab entry -> True
    vocab_words: set
    settings_words: set


def _similar_dict_word(o_run: list[str], r_words: list[str]) -> bool:
    """A dictionary entry (o_run joined) replacing a spelling-similar stretch of the spoken words."""
    if not r_words:
        return False
    target = "".join(o_run)
    best = 0.0
    for i in range(len(r_words)):
        for j in range(i + 1, min(len(r_words), i + 4) + 1):
            cand = "".join(r_words[i:j])
            best = max(best, difflib.SequenceMatcher(None, cand, target).ratio())
    return best >= 0.5


def _region_ok(r_c: list[str], o_c: list[str], r_surf: list[str], o_surf: list[str], pool: Counter,
               ctx: _Ctx) -> Optional[Counter]:
    """Counter of pool words this region uses if every output word is supported, else None."""
    if not o_c:
        return Counter()
    if r_c and _alnum("".join(r_c)) == _alnum("".join(o_c)):
        return Counter()                    # compound / split / formatting ("sub agents" -> "subagents", 330 -> 3:30)
    if r_c and len(r_c) != len(o_c) and not any(w.isdigit() for w in r_c + o_c):
        jr, jo = _alnum("".join(r_c)), _alnum("".join(o_c))
        if min(len(jr), len(jo)) >= 8 and difflib.SequenceMatcher(None, jr, jo).ratio() >= 0.85:
            return Counter()                # a long word the speech-to-text split or misspelled ("underperform it")
    if r_surf and len(r_surf) == len(o_surf) and all(_pair_ok(a, b) for a, b in zip(r_surf, o_surf)):
        return Counter()                    # they're / their, its / it's at the surface
    take: Counter = Counter()
    k = 0
    while k < len(o_c):
        o = o_c[k]
        if pool[o] - take[o] > 0:
            take[o] += 1
            k += 1
            continue
        if o.isdigit():
            return None                     # a number is only ever what was said
        if any(_pair_ok(r, o) for r in r_c) or any(_pair_ok(r, o) for r in r_surf):
            k += 1
            continue
        if o in ctx.settings_words:
            k += 1
            continue
        if o in ctx.vocab_words:
            run_end = k
            while run_end < len(o_c) and o_c[run_end] in ctx.vocab_words:
                run_end += 1
            run = o_c[k:run_end]
            if _similar_dict_word(run, r_c) or _similar_dict_word([o], r_c):
                k = run_end
                continue
        return None
    return take


# --------------------------------------------------------------------------- text edits
def _sentence_start(text: str, pos: int) -> bool:
    j = pos - 1
    while j >= 0 and text[j] in " \t\"'“‘(*-`":
        j -= 1
    return j < 0 or text[j] in ".!?\n:"


def _restore_words(raw: _Side, toks: list[int], keep_case: set) -> str:
    """Raw tokens ``toks`` joined, with sentence-start capitals of the raw undone."""
    parts: list[str] = []
    prev_end = None
    for ti in toks:
        surf, s, e = raw.toks[ti]
        if surf[:1].isupper() and surf[1:] == surf[1:].lower() and surf.lower() not in keep_case and \
                surf.lower() not in ("i", "i'm", "i’m", "i'll", "i've", "i'd") and _sentence_start(raw.text, s):
            surf = surf[:1].lower() + surf[1:]
        if parts:
            gap = raw.text[prev_end:s]       # "3.5", "release.js", "pitch-yaw", "4:30" keep their joiner
            parts.append(gap if gap and not any(c.isspace() for c in gap) and len(gap) <= 2 else " ")
        parts.append(surf)
        prev_end = e
    return "".join(parts)


_PAIRS = {"(": ")", "[": "]", '"': '"', "“": "”", "'": "'", "‘": "’", "`": "`"}
# the character(s) a spoken symbol name stands for (a dropped name is "written" when one of them is in the output)
_SYMBOL_CHARS = {"paren": "()", "parens": "()", "parenthesis": "()", "bracket": "[]", "brackets": "[]", "brace": "{}",
                 "braces": "{}", "colon": ":", "dash": "-–—", "hyphen": "-", "quote": "\"“”'", "quotes": "\"“”'",
                 "slash": "/", "backslash": "\\", "asterisk": "*", "dot": ".", "period": ".", "comma": ",",
                 "semicolon": ";", "underscore": "_", "question": "?", "exclamation": "!", "mark": "?!",
                 "point": "!", "new": "\n", "line": "\n", "paragraph": "\n", "full": ".", "stop": "."}
# raw symbols that stand for words the output may write out ("$5" -> "5 dollars", "50%" -> "50 percent")
_RAW_SYMBOL_WORDS = {"$": ("dollar", "dollars"), "%": ("percent",), "&": ("and",), "@": ("at",), "#": ("number",),
                     "+": ("plus",), "=": ("equals",)}


def _glued(text: str, s: int, e: int) -> bool:
    """The span s..e is joined to another word/number by . : / with no space ("2.5", "4:30", "1/2", "$1.50"):
    replacing only part of such a chunk would mangle it."""
    if s >= 2 and text[s - 1] in ".:/" and text[s - 2].isdigit():
        return True
    if e + 1 < len(text) and text[e] in ".:/" and text[e + 1].isdigit():
        return True
    return False


_NO_SENTENCE_START = frozenset("me to of for with than and or nor but at by from into onto is are was were".split())


def _cap_first(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


@dataclass
class Revert:
    out_words: str
    raw_words: str
    pos: int


def repair(raw: str, out: str, vocab: Iterable[str] = (), abbreviations: Optional[dict] = None,
           word_map: Optional[dict] = None, settings_text: Iterable[str] = ()) -> tuple[str, list[Revert]]:
    """(repaired output, reverted regions). ``settings_text``: strings from the user's settings whose words count as
    asked for (replacement / snippet targets, custom style prompt, ...)."""
    try:
        return _repair(raw, out, vocab, abbreviations, word_map, settings_text)
    except Exception:
        log.exception("keep_words repair failed - output left unchanged")
        return out, []


def _repair(raw, out, vocab, abbreviations, word_map, settings_text):
    if not raw or not out:
        return out, []
    abbr_rev = {}
    for ph, ab in (abbreviations or {}).items():
        a = str(ab or "").strip().lower()
        if a and str(ph).strip() and " " not in a:
            abbr_rev[a] = str(ph)
    wm = {str(k).strip().lower(): str(v) for k, v in (word_map or {}).items() if str(k).strip()}
    r = _tokenize(raw, abbr_rev, wm, drop=True)
    o = _tokenize(out, abbr_rev, wm, drop=False)
    if not o.canon:
        return out, []

    vocab = [str(v) for v in vocab or [] if str(v).strip()]
    vocab_words: set = set()
    for v in vocab:
        vocab_words.update(_canon_phrase(v, {}, {}))
    settings_words: set = set()
    for t in settings_text or []:
        settings_words.update(w for w in _canon_phrase(str(t), {}, {}) if w not in _STOP and not w.isdigit())
    ctx = _Ctx({}, vocab_words, settings_words)
    keep_case = {v.lower() for v in vocab} | {t[0].lower() for t in r.toks
                                              if t[0][:1].isupper() and not _sentence_start(raw, t[1])}

    rw = [c[0] for c in r.canon]
    ow = [c[0] for c in o.canon]
    ops = difflib.SequenceMatcher(None, rw, ow, autojunk=False).get_opcodes()
    pool: Counter = Counter()
    for tag, i1, i2, _j1, _j2 in ops:
        if tag in ("replace", "delete"):
            pool.update(rw[i1:i2])
    pool.update(w for w, _t in r.dropped)
    for sym, words in _RAW_SYMBOL_WORDS.items():
        for w in words:
            pool[w] += raw.count(sym)

    # output tokens referenced by each canonical output word (a token can expand to several words)
    tok_users: dict[int, set] = {}
    for ci, (_w, a, b) in enumerate(o.canon):
        for ti in range(a, b + 1):
            tok_users.setdefault(ti, set()).add(ci)
    raw_tok_users: dict[int, set] = {}
    for ci, (_w, a, b) in enumerate(r.canon):
        for ti in range(a, b + 1):
            raw_tok_users.setdefault(ti, set()).add(ci)

    # does this output capitalize its sentences (neutral) or not (casual / code)?
    caps = any(t[0][:1].isupper() for t in o.toks if t[0][:1].isalpha() and _sentence_start(out, t[1]))
    edits: list[tuple[int, int, str, Revert]] = []      # (start, end, replacement) on out
    n_del = 0
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            continue
        if tag == "delete":
            n_del += 1
            continue
        r_c, o_c = rw[i1:i2], ow[j1:j2]
        olo, ohi = o.canon[j1][1], o.canon[j2 - 1][2]
        r_surf = []
        if i2 > i1:
            rlo, rhi = r.canon[i1][1], r.canon[i2 - 1][2]
            r_surf = [r.toks[t][0].lower().replace("’", "'").replace("'", "") for t in range(rlo, rhi + 1)
                      if t in raw_tok_users]
        o_surf = [o.toks[t][0].lower().replace("’", "'").replace("'", "") for t in range(olo, ohi + 1)
                  if t in tok_users]
        take = _region_ok(r_c, o_c, r_surf, o_surf, pool, ctx)
        if take is not None:
            pool.subtract(take)
            continue
        # a token shared with a neighbouring region (e.g. "dont" split across two regions): leave it alone
        if any(not tok_users.get(t, set()) <= set(range(j1, j2)) for t in range(olo, ohi + 1)):
            log.debug("keep_words: region shares a token with its neighbour - skipped")
            continue
        s, e = o.toks[olo][1], o.toks[ohi][2]
        if _glued(out, s, e):
            log.debug("keep_words: region is glued to another token (2.5, 4:30, 1/2) - skipped")
            continue
        prev_end = o.toks[olo - 1][2] if olo > 0 else 0
        next_start = o.toks[ohi + 1][1] if ohi + 1 < len(o.toks) else len(out)
        cap = _sentence_start(out, s) and (out[s:s + 1].isupper() or (not out[s:s + 1].isalpha() and caps))
        # everything spoken between the neighbouring matched words
        glo = r.canon[i1 - 1][2] + 1 if i1 > 0 else 0
        ghi = r.canon[i1][1] - 1 if i1 < len(r.canon) else len(r.toks) - 1
        if i2 > i1:
            ghi = r.canon[i2][1] - 1 if i2 < len(r.canon) else len(r.toks) - 1
            if any(not raw_tok_users.get(t, set()) <= set(range(i1, i2)) for t in range(rlo, rhi + 1)):
                log.debug("keep_words: raw region shares a token with its neighbour - skipped")
                continue
            # minus hesitations; spoken symbol names only when the output kept them as words; "you know" / "i mean"
            # only inside the replaced stretch; "like" can be a real word ("be like")
            skip = {t for w, t in r.dropped if w in _HESITATIONS or
                    ((w in _SYMBOLS or any(w in p for p in _SYMBOL_PAIRS)) and w not in o_c) or
                    (any(w in p for p in _FILLER_PAIRS) and w not in o_c and not rlo < t < rhi)}
        else:
            # an "insertion" may really be a swap of a word the aligner leaves out ("i like pizza" -> "I love pizza",
            # "the quote was" -> "The price was"): put that word back; a symbol name only when no symbol was written
            ogap = re.sub(r"(?<=\w)['’-](?=\w)", "", out[prev_end:next_start])     # client's, well-known
            skip = {t for w, t in r.dropped if w in _HESITATIONS or
                    (w in _SYMBOL_CHARS and any(c in ogap for c in _SYMBOL_CHARS[w]))}
        toks = [t for t in range(glo, ghi + 1) if t not in skip and t not in raw_tok_users]
        if i2 > i1:
            toks = [t for t in range(glo, ghi + 1) if t not in skip]
        if toks:
            rep = _restore_words(r, toks, keep_case)
            a, b = s, e
            if cap:
                j = s - 1
                while j >= 0 and out[j] == " ":
                    j -= 1
                first = r.toks[toks[0]]
                # only where the raw shows there was no break: a punctuated raw, or a word that can't start one
                if (0 < j < s - 1 and out[j] == "." and out[j - 1].isalpha()
                        and not _sentence_start(raw, first[1])
                        and (re.search(r"[.!?]", raw) or first[0].lower() in _NO_SENTENCE_START)):
                    a, rep = j, " " + rep        # the model split the sentence here; the spoken words continue it
                else:
                    rep = _cap_first(rep)
            # the first / last restored word was joined to its matched neighbour in the raw ("release.js"): join again
            if a == s and toks[0] > 0 and out[s - 1:s] == " ":
                jg = raw[r.toks[toks[0] - 1][2]:r.toks[toks[0]][1]]
                if jg and len(jg) <= 2 and not any(c.isspace() for c in jg):
                    a, rep = s - 1, jg + rep
            if toks[-1] + 1 < len(r.toks) and out[e:e + 1] == " ":
                jg = raw[r.toks[toks[-1]][2]:r.toks[toks[-1] + 1][1]]
                if jg and len(jg) <= 2 and not any(c.isspace() for c in jg):
                    b, rep = e + 1, rep + jg
            if s > 0 and out[s - 1] in "$€£" and not rep[:1].isdigit():
                a = min(a, s - 1)                # "$five" -> the currency sign belonged to the model's digits
            if out[e:e + 1] == "%" and not rep[-1:].isdigit():
                b = e + 1
            edits.append((a, b, rep, Revert(out[s:e], rep.strip(), s)))
        else:
            before, after = out[prev_end:s], out[e:next_start]
            cap_next = _sentence_start(out, s) and out[s:s + 1].isupper()
            fill = "\x00CAP" if cap_next else ""
            if before[-1:] in _PAIRS and after[:1] == _PAIRS.get(before[-1:]):
                # bracket / quote touching both ends of the insertion: it goes with them ("store (the big one)." ->
                # "store."); a quote that belongs to a neighbouring word ('"Deep Sea" or "Clawed Code"') is not this
                a, b = s - 1, e + 1
                if out[a - 1:a] == " " and (b >= len(out) or out[b] in " .,;:!?\n"):
                    a -= 1
                elif a == 0:
                    while b < len(out) and out[b] == " ":
                        b += 1
            elif olo == 0:
                a, b = s, next_start             # leading insertion: drop it and what separates it from the next word
            elif after.strip() == "" or after == before:
                a, b = s, next_start
                # "deep sea, clawed code" -> "Deep Sea or Clawed Code": put the spoken comma back with the "or"
                if before.strip() == "" and after.strip() == "" and not cap_next and 0 < i1 < len(r.canon):
                    rgap = raw[r.toks[r.canon[i1 - 1][2]][2]:r.toks[r.canon[i1][1]][1]]
                    if "," in rgap and not re.search(r"[.!?]", rgap):
                        a, b, fill = prev_end, next_start, ", "
            elif before.strip(" ,;:") == "":
                a, b = prev_end, e               # "know, Reyna players" -> drop the word and the separator before it
            elif out[e:e + 1] == " ":
                a, b = s, e + 1                  # a neighbour's quote / bracket / backtick stays: delete word + 1 space
            elif out[s - 1:s] == " ":
                a, b = s - 1, e
            else:
                a, b = s, e
            edits.append((a, b, fill, Revert(out[s:e], "", s)))
    if not edits:
        return out, []
    text = out
    applied: list[Revert] = []
    floor = len(out) + 1                    # start of the edit applied last (edits go right to left)
    for a, b, rep, rv in sorted(edits, key=lambda x: x[0], reverse=True):
        if b > floor:
            continue                        # would overlap the edit to its right (shared separator): skip it
        floor = a
        applied.append(rv)
        if rep == "\x00CAP":
            rest = text[b:]
            m = re.match(r"[\s\"'“‘(*`-]*", rest)
            k = m.end() if m else 0
            rest = rest[:k] + rest[k:k + 1].upper() + rest[k + 1:]
            text = text[:a] + rest
        else:
            text = text[:a] + rep + text[b:]
    text = re.sub(r"(?<=\S) {2,}(?=\S)", " ", text)
    return text, applied[::-1]
