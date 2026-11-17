"""Command mode questions: decide whether a spoken command is a question (answered in the overlay card, never typed),
work out simple time spans / arithmetic exactly without the LLM, and tidy the LLM's answer for the card."""
from __future__ import annotations

import re
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal, localcontext
from fractions import Fraction
from typing import Optional

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

_FILLERS = frozenset({"hey", "ok", "okay", "so", "um", "umm", "uh", "uhh", "please", "just"})
_MODALS = frozenset({"can", "could", "would", "will"})
_MODAL_SUBJECTS = frozenset({"you", "u", "we"})
# multi-word polite prefixes (tuples of tokens); the modal + you/u/we forms are handled separately
_POLITE = (("would", "you", "mind"), ("can", "i", "get"), ("can", "i", "have"), ("could", "i", "get"),
           ("could", "i", "have"), ("help", "me"), ("go", "ahead", "and"), ("i", "want", "you", "to"),
           ("i", "need", "you", "to"))

_EDIT_VERBS = ("make", "rewrite", "write", "draft", "reply", "respond", "translate", "fix", "correct", "change", "turn",
               "convert", "summarize", "summarise", "shorten", "lengthen", "expand", "format", "add", "remove", "delete",
               "put", "insert", "type", "paste", "list", "bullet", "capitalize", "capitalise", "reword", "rephrase",
               "improve", "polish", "simplify", "clean", "do", "edit", "proofread", "replace")


def _verb_forms(v: str) -> set[str]:
    out = {v, v + "ing"}
    if v.endswith("e"):
        out.add(v[:-1] + "ing")
    if re.fullmatch(r"[^aeiou]*[aeiou][^aeiouwy]", v):      # put -> putting
        out.add(v + v[-1] + "ing")
    return out


EDIT_WORDS = frozenset(f for v in _EDIT_VERBS for f in _verb_forms(v))
_WH = frozenset({"what", "what's", "whats", "how", "how's", "hows", "when", "when's", "where", "where's", "who",
                 "who's", "whom", "whose", "why", "which"})
_BE_MODAL = frozenset({"is", "are", "am", "was", "were", "can", "could", "should", "would", "will", "isn't", "aren't",
                       "wasn't", "weren't", "shouldn't", "wouldn't", "couldn't", "explain", "define", "calculate",
                       "compute"})
_AUX = frozenset({"do", "does", "did", "have", "has", "had", "doesn't", "don't", "didn't", "hasn't", "haven't"})
# "do the same for ...", "do these in bullet points": an edit when one of these follows do + the/these/those;
# otherwise "do the numbers add up?" is a question
_DO_EDIT_NEXT = frozenset({"same", "rest", "first", "second", "last", "next", "other", "opposite", "math", "in", "as",
                           "again", "over", "with", "for", "instead"})
_DETERMINERS = frozenset({"the", "a", "an", "my", "your", "this", "that", "these", "those", "his", "her", "our",
                          "their", "its"})
_SUBJECTS = frozenset({"i", "you", "we", "they", "it", "he", "she", "this", "that", "these", "those", "the", "a", "an",
                       "my", "your"})
# after a "can/could/would/will you" opener only these make a question ("can you explain this"); any other verb is
# a request to do something ("can you sort these") and stays a command
_MODAL_ASK = frozenset(f for v in ("tell", "explain", "define", "calculate", "compute", "describe", "clarify", "check",
                                   "confirm", "estimate", "guess", "know", "understand", "figure", "remind", "answer",
                                   "solve", "count", "spell")
                       for f in _verb_forms(v)) | _WH
_TELL_WORDS = frozenset(_verb_forms("tell") | _verb_forms("remind"))     # only "tell me" / "remind us" ask something
_ADVERBS = frozenset({"also", "quickly", "briefly", "maybe", "kindly", "actually", "real", "quick"})
_FORCE_ASK = frozenset({"ask", "question"})
_FORCE_INSERT = frozenset({"write", "type", "insert"})

_ASK_PREFIX_RE = re.compile(r"^\W*(?:(?:hey|ok|okay|so|um|uh)\b\W*)*(?:ask|question)\b[\s,:;.\-]*", re.I)
_ASK_PUNCT_RE = re.compile(r"^\W*(?:(?:hey|ok|okay|so|um|uh)\b\W*)*(?:ask|question)\s*[,:;\-]", re.I)
# bare "it" is left out: "what time is it" should not press Ctrl+C
_REFERS_RE = re.compile(r"\b(?:this|that|these|those|here|selected|selection|highlighted|above|the text|my text|"
                        r"grammar|spelling|typos?|wording|sentence|paragraph|tone)\b", re.I)


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower().replace("’", "'"))


def _strip_prefixes(toks: list[str]) -> tuple[list[str], bool, bool]:
    """Repeatedly drop fillers and polite openers. Returns (rest, polite, modal) where modal = a "can/could/would/
    will you|u|we" opener was dropped."""
    polite = modal = False
    changed = True
    while toks and changed:
        changed = False
        if toks[0] in _FILLERS:
            polite = polite or toks[0] in ("please", "just")       # "hey" / "so" / "um" are not requests
            toks, changed = toks[1:], True
            continue
        for p in _POLITE:
            if tuple(toks[:len(p)]) == p:
                toks, changed, polite = toks[len(p):], True, True
                modal = modal or p in (("would", "you", "mind"), ("help", "me"))
                break
        else:
            if len(toks) >= 2 and toks[0] in _MODALS and toks[1] in _MODAL_SUBJECTS:
                toks, changed, polite, modal = toks[2:], True, True, True
    return toks, polite, modal


def _explains_to_other(toks: list[str]) -> bool:
    """ "explain to the client that ..." drafts a message; "explain to me ..." asks."""
    return bool(toks) and toks[0] in ("explain", "explains", "explaining") and toks[1:2] == ["to"] and \
        toks[2:3] not in (["me"], ["us"])


def is_question(instruction: str) -> bool:
    """True when a spoken command is a question to answer in the overlay; False (the default) keeps the normal
    rewrite / write-at-cursor behavior. Needs a strong signal: a question word, a "can you <not an edit>" opener or
    a leading "ask" / "question"."""
    toks = _tokens(instruction)
    head = toks
    while head and head[0] in _FILLERS - {"please", "just"}:
        head = head[1:]
    if not head:
        return False
    # "ask: ..." / "ask what the capital of France is" force the card; "ask them to reschedule" drafts a message
    if head[0] in _FORCE_ASK and len(head) > 1 and (head[1] in _WH | _BE_MODAL | _AUX or
                                                    _ASK_PUNCT_RE.match(instruction or "")):
        return True
    if head[0] in _FORCE_INSERT:
        return False
    rest, polite, modal = _strip_prefixes(toks)
    if not rest:
        return False
    if rest[0] in EDIT_WORDS:
        # "do this in bullet points" / "do a spell check" / "do the same" are edits; "do you know ..." is a question
        nxt = rest[1] if len(rest) > 1 else ""
        if rest[0] != "do" or nxt not in _SUBJECTS or nxt in ("this", "that", "it", "a", "an") or \
                (nxt in ("the", "these", "those") and (len(rest) < 3 or rest[2] in _DO_EDIT_NEXT)):
            return False
    if polite:
        # "can you also make this shorter" is an edit; "can you explain the change" is not (a noun after a determiner)
        for t in rest[:3]:
            if t in _DETERMINERS:
                break
            if t in EDIT_WORDS:
                return False
    if modal:
        i = next((k for k, t in enumerate(rest) if t not in _ADVERBS), len(rest))
        w = rest[i] if i < len(rest) else ""
        if w in _TELL_WORDS:                # "can you tell me ..." asks; "can you tell him I'm late" drafts a message
            return rest[i + 1:i + 2] in (["me"], ["us"])
        if _explains_to_other(rest[i:]):
            return False
        return w in _MODAL_ASK
    w = rest[0]
    if _explains_to_other(rest):
        return False
    if w in _WH or w in _BE_MODAL:
        return True
    if w in _AUX:
        return len(rest) > 1 and rest[1] in _SUBJECTS
    return rest[:2] == ["tell", "me"]


def question_text(instruction: str) -> str:
    """The question to show / answer: a leading "ask" / "question" override word is dropped."""
    s = (instruction or "").strip()
    m = _ASK_PREFIX_RE.match(s)
    if m and s[m.end():].strip():
        s = s[m.end():].strip()
        s = s[:1].upper() + s[1:]
    return s


def references_text(question: str) -> bool:
    """The question is about the selected text ("what does this mean"), so the selection is worth copying."""
    return bool(_REFERS_RE.search(question or ""))


# ----------------------------------------------------------------------------- quick (exact) answers
# a colon-less 3-4 digit time ("730am") needs am/pm, so "from 2019 to 2023" is not read as 20:19 to 20:23
_TIME = r"(noon|midnight|\d{1,2}(?:[:.]\d{2})?\s*(?:am|pm)?|\d{3,4}\s*(?:am|pm))(?:\s*o'?clock)?"
_SPAN_RE = re.compile(
    r"(how many (?:hours|minutes|hours and minutes)|how long|how much time)\s+"
    r"(?:(?:is it|it is|is there|are there|is that|are|is)\s+)?(?:in\s+)?(?:(?:from|between)\s+)?" + _TIME +
    r"\s+(?:to|and|until|till|til|through|thru)\s+" + _TIME)
_LEAD_RE = re.compile(r"^(?:(?:hey|ok|okay|so|um|uh|please|just)\b[\s,]*)+")
_TELL_RE = re.compile(r"^(?:(?:can|could|would|will) (?:you|u) (?:please )?)?(?:(?:tell me|do you know)\b[\s,]*)?")


def _norm(question: str) -> str:
    s = (question or "").lower().replace("’", "'").strip()
    s = re.sub(r"(?<![a-z])([ap])\.\s?m(?![a-z])\.?", r"\1m", s)
    s = re.sub(r"[?.!\s]+$", "", s)
    s = _LEAD_RE.sub("", s)
    s = _TELL_RE.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def _parse_time(t: str) -> Optional[tuple[int, Optional[str]]]:
    """(minutes after midnight on a 12-hour or 24-hour clock, meridiem or None). Meridiem "24" = already a 24-hour
    reading (13:00, 0:30, noon, midnight)."""
    t = t.replace(" ", "")
    if t == "noon":
        return 12 * 60, "24"
    if t == "midnight":
        return 0, "24"
    mer = None
    if t.endswith(("am", "pm")):
        t, mer = t[:-2], t[-2:]
    if ":" in t or "." in t:
        h, m = re.split(r"[:.]", t)
    elif len(t) >= 3:
        h, m = t[:-2], t[-2:]
    else:
        h, m = t, "0"
    h, m = int(h), int(m)
    if m > 59 or h > 23:
        return None
    if mer:
        if h == 0 or h > 12:
            return None
        return (h % 12 + (12 if mer == "pm" else 0)) * 60 + m, mer
    if h == 0 or h > 12:
        return h * 60 + m, "24"
    return h * 60 + m, None


def _readings(t: tuple[int, Optional[str]]) -> list[int]:
    mins, mer = t
    if mer is not None:
        return [mins]
    base = mins % (12 * 60)         # 12:15 -> 0:15 (am) / 12:15 (pm)
    return [base, base + 12 * 60]


def _plural(n: int, word: str) -> str:
    return f"{n:,} {word}" + ("" if n == 1 else "s")


def _hours_minutes(total: int) -> str:
    h, m = divmod(total, 60)
    if not h:
        return _plural(m, "minute")
    if not m:
        return _plural(h, "hour")
    return f"{_plural(h, 'hour')} {_plural(m, 'minute')}"


def _time_span(s: str) -> Optional[str]:
    m = _SPAN_RE.fullmatch(s)
    if not m:
        return None
    a, b = _parse_time(m.group(2)), _parse_time(m.group(3))
    if a is None or b is None:
        return None
    spans = []
    for x in _readings(a):
        for y in _readings(b):
            d = (y - x) % (24 * 60)
            spans.append(d or 24 * 60)
    total = min(spans)
    if m.group(1) == "how many minutes":
        return _plural(total, "minute") + ("" if total < 60 else f" ({_hours_minutes(total)})")
    out = _hours_minutes(total)
    if total >= 60 and total % 60 and (total % 60 * 100) % 60 == 0:      # exact decimal hours (x.25, x.5, x.05 ...)
        out += f" ({_fmt(Fraction(total, 60))} hours)"
    return out


_CALC_RE = re.compile(r"(?:what's|whats|what is|how much is|calculate|compute)\s+(.+)")
_OPS = ((r"multiplied by", "*"), (r"divided by", "/"), (r"plus", "+"), (r"minus", "-"), (r"times", "*"),
        (r"over", "/"), (r"x", "*"))
_PERCENT_OF_RE = re.compile(r"(\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)\s*(?:%|percent|per cent)\s+of\b")
_EXPR_TOKEN_RE = re.compile(r"\s*(\d+(?:\.\d+)?|\.\d+|[-+*/()])")


def _expr_tokens(expr: str) -> Optional[list]:
    s = _PERCENT_OF_RE.sub(lambda m: f"({m.group(1)}/100)*", expr)
    s = s.replace("×", "*").replace("÷", "/").replace("−", "-")
    for word, op in _OPS:
        s = re.sub(rf"(?<![a-z]){word}(?![a-z])", f" {op} ", s)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    out, pos = [], 0
    s = s.rstrip()
    while pos < len(s):
        m = _EXPR_TOKEN_RE.match(s, pos)
        if not m:
            return None
        tok = m.group(1)
        if tok[0].isdigit() or tok[0] == ".":
            if len(tok.replace(".", "")) > 20:
                return None
            out.append(Fraction(tok))
        else:
            out.append(tok)
        pos = m.end()
    if not any(isinstance(t, str) and t in "+-*/" for t in out) or not any(isinstance(t, Fraction) for t in out):
        return None
    return out


class _Calc:
    """expr := term (+|- term)* ; term := unary (*|/ unary)* ; unary := -unary | +unary | NUMBER | ( expr )"""

    def __init__(self, toks: list):
        self.t, self.i = toks, 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def take(self):
        tok = self.peek()
        self.i += 1
        return tok

    def expr(self) -> Fraction:
        v = self.term()
        while self.peek() in ("+", "-"):
            v = v + self.term() if self.take() == "+" else v - self.term()
        return v

    def term(self) -> Fraction:
        v = self.unary()
        while self.peek() in ("*", "/"):
            if self.take() == "*":
                v = v * self.unary()
            else:
                d = self.unary()
                if d == 0:
                    raise ZeroDivisionError
                v = v / d
        return v

    def unary(self) -> Fraction:
        tok = self.take()
        if tok == "-":
            return -self.unary()
        if tok == "+":
            return self.unary()
        if tok == "(":
            v = self.expr()
            if self.take() != ")":
                raise ValueError("unbalanced")
            return v
        if isinstance(tok, Fraction):
            return tok
        raise ValueError("bad token")


def _fmt(x: Fraction) -> str:
    """Exact decimal for terminating values, otherwise rounded to 6 places; trailing zeros dropped; thousands
    separators from 10,000 up."""
    neg = x < 0
    x = abs(x)
    d = x.denominator
    for p in (2, 5):
        while d % p == 0:
            d //= p
    with localcontext() as ctx:
        ctx.prec = 100
        q = Decimal(x.numerator) / Decimal(x.denominator)
        if d != 1:
            q = q.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
        s = format(q, "f")
    whole, _, frac = s.partition(".")
    frac = frac.rstrip("0")
    whole_i = int(whole)
    whole = f"{whole_i:,}" if x >= 10000 else str(whole_i)
    out = whole + ("." + frac if frac else "")
    return ("-" + out) if neg and out.strip("0.,") else out


def _arithmetic(s: str) -> Optional[str]:
    m = _CALC_RE.fullmatch(s)
    if not m:
        return None
    if s.startswith("what") and re.fullmatch(r"\d+(?:/\d+)+|\d+(?:-\d+)+", m.group(1)):
        return None                     # "what's 9/11", "what is 24/7", "what is 2026-11-10": not maths
    toks = _expr_tokens(m.group(1))
    if not toks:
        return None
    c = _Calc(toks)
    try:
        v = c.expr()
    except (ZeroDivisionError, ValueError):
        return None
    if c.i != len(toks):
        return None
    return _fmt(v)


def quick_answer(question: str, now: Optional[datetime] = None) -> Optional[str]:
    """Exact local answer for a time span ("how many hours is it from 7:30am to 5pm") or plain arithmetic ("what's
    15% of 240"); None when the whole question isn't one of those (the LLM answers it)."""
    s = _norm(question)
    if not s:
        return None
    return _time_span(s) or _arithmetic(s)


# ----------------------------------------------------------------------------- LLM answer clean-up
_ANS_TAG_RE = re.compile(r"</?(?:question|now|selection|app)>", re.I)
_FENCE_LINE_RE = re.compile(r"^\s*```[\w+.-]*\s*$", re.M)
_PREFIX_RE = re.compile(r"^\s*(?:answer|a)\s*:\s*", re.I)
_BULLET_RE = re.compile(r"^(?:[-*•]\s+|#{1,6}\s+)")


def tidy_answer(text: str) -> str:
    """One clean line for the answer card: no reasoning tags, markdown, leaked prompt tags or "Answer:" preface."""
    from .cleanup import strip_think
    s = strip_think(text or "")
    s = _ANS_TAG_RE.sub("", s)
    s = _FENCE_LINE_RE.sub("", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s, flags=re.S)
    s = re.sub(r"__(.+?)__", r"\1", s, flags=re.S)
    s = re.sub(r"(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])", r"\1", s)
    s = s.replace("`", "")
    lines = [_BULLET_RE.sub("", ln.strip()) for ln in s.splitlines()]
    s = " ".join(ln for ln in lines if ln)
    s = re.sub(r"\s+", " ", s).strip()
    return _PREFIX_RE.sub("", s).strip()
