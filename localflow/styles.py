"""Per-app / per-tab writing styles.

Two pieces of config drive this module:

* ``profiles``  - ordered rule list, FIRST MATCH WINS. A rule is
  ``{"name": "Discord", "process": ["discord.exe"], "title": "regex", "style": "casual"}``.
  ``process`` (exe names, case-insensitive; ``"@group"`` expands ``process_groups``) and ``title``
  (regex, case-insensitive ``search``) are ANDed when both are given; at least one must be present.
* ``styles``    - name -> definition: LLM ``prompt`` instructions, deterministic post-processing flags
  (``lowercase``, ``strip_trailing_period``, ``abbreviations``, ``drop_apostrophes``, ``capitalize``)
  and optional few-shot ``examples`` ([[spoken, written], ...]).

Detection (:func:`resolve`) is a handful of dict/regex lookups (well under 1 ms).
"""
from __future__ import annotations

import copy
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

log = logging.getLogger(__name__)

NEUTRAL = "neutral"
_ALIASES = {"general": NEUTRAL, "default": NEUTRAL, "chat": "casual"}

BROWSERS = [
    "chrome.exe", "msedge.exe", "firefox.exe", "brave.exe", "opera.exe", "opera_gx.exe", "vivaldi.exe",
    "arc.exe", "zen.exe", "librewolf.exe", "waterfox.exe", "floorp.exe", "thorium.exe", "chromium.exe",
    "iridium.exe", "tor.exe", "duckduckgo.exe", "sidekick.exe", "comet.exe", "dia.exe",
]

# --------------------------------------------------------------------------- default styles
DEFAULT_STYLES: dict[str, dict[str, Any]] = {
    "neutral": {
        "description": "Normal cleanup: fillers out, capitals and punctuation fixed, wording untouched.",
        "prompt": "",
        "paragraphs": True,      # long streamed dictations: a long pause after 3+ sentences starts a new paragraph
    },
    "casual": {
        "description": "Texting / chat: lowercase, slang abbreviations, almost no punctuation, no final period.",
        "prompt": ("Casual texting style, the way a person types fast in a chat app. All lowercase (including \"i\"). "
                   "No period at the end and none between short sentences; keep ? or ! only when they matter. "
                   "Minimal punctuation: a comma only where a sentence would be confusing without it. "
                   "Drop apostrophes in common contractions (dont, cant, im, thats, lets, ive). "
                   "Use texting abbreviations ONLY for phrases the speaker actually said: idk, nvm, tbh, rn, omw, ngl, "
                   "imo, btw, omg, brb, idc, ttyl, lmk, fr. Still remove filler sounds (um, uh, er, hmm, \"you know\"). "
                   "Do not polish the grammar, and do not shorten single words (write \"you\", not \"u\"). "
                   "Never add slang, emoji or words that were not said."),
        "lowercase": True,
        "strip_trailing_period": True,
        "abbreviations": True,
        "drop_apostrophes": True,
        # the model sometimes shortens single words on its own; nobody SAYS "u" - put the real word back
        "word_map": {"u": "you", "ur": "your", "cuz": "because", "pls": "please", "plz": "please", "thx": "thanks"},
        "capitalize": False,
    },
    "professional": {
        "description": "Email / documents: proper capitalization, full sentences, correct grammar, no slang.",
        "prompt": ("Professional writing (email, documents). Proper capitalization and punctuation, complete "
                   "sentences, correct grammar, no slang or texting abbreviations. Reword awkward phrasing slightly "
                   "(\"can you\" -> \"could you\", \"gonna\" -> \"going to\", \"aint\" -> \"is not\"). "
                   "KEEP every greeting and thanks the speaker actually said (\"hey\" -> \"Hi\", \"thanks\" -> \"Thanks.\"). "
                   "Never ADD a greeting, sign-off, name, subject line or fact that was not said. "
                   "Keep contractions as the speaker used them."),
        "lowercase": False,
        "strip_trailing_period": False,
        "abbreviations": False,
        "capitalize": True,
        "paragraphs": True,
    },
    "code": {
        "description": "Editors and terminals: literal wording, no capitalised first letter, no final period.",
        "prompt": ("Code/terminal context. Keep the wording literal, do not capitalize the first letter of "
                   "the text, and put no final period."),
        "lowercase": False,
        "strip_trailing_period": "single",
        "abbreviations": False,
        "capitalize": False,
    },
}

# built-in few-shot examples per style (config styles.<name>.examples overrides)
DEFAULT_EXAMPLES: dict[str, list[list[str]]] = {
    "casual": [
        ["I don't know, never mind, let's just do it tomorrow.", "idk nvm lets just do it tomorrow"],
        ["um yeah that sounds good to me, I'm on my way right now",
         "yeah that sounds good to me im omw rn"],
        ["Honestly I thought the movie was fine, but to be honest the ending was kind of boring.",
         "honestly i thought the movie was fine but tbh the ending was kinda boring"],
        ["Can you send me the link? Let me know when you're free.", "can you send me the link? lmk when youre free"],
    ],
    "professional": [
        ["hey can you send me that report by friday thanks", "Hi, could you send me the report by Friday? Thanks."],
        ["hey sarah quick question about the invoice", "Hi Sarah, quick question about the invoice."],
        ["um so we gonna need to push the launch back a week cause the testing aint done",
         "We are going to need to push the launch back a week because the testing is not done."],
        ["i looked at the numbers and they dont match what finance sent us yesterday",
         "I looked at the numbers and they don't match what finance sent us yesterday."],
        ["can you let me know if the 3 pm call works for you",
         "Could you let me know if the 3 PM call works for you?"],
    ],
    "code": [
        ["uh add a null check before the loop and then return early",
         "add a null check before the loop and then return early"],
    ],
}

# phrase -> abbreviation (word-boundary safe, case-insensitive; apostrophes in the phrase are optional when matching)
DEFAULT_ABBREVIATIONS: dict[str, str] = {
    "i don't know": "idk",
    "i do not know": "idk",
    "never mind": "nvm",
    "nevermind": "nvm",
    "to be honest": "tbh",
    "right now": "rn",
    "on my way": "omw",
    "not gonna lie": "ngl",
    "not going to lie": "ngl",
    "in my opinion": "imo",
    "by the way": "btw",
    "oh my god": "omg",
    "oh my gosh": "omg",
    "be right back": "brb",
    "i don't care": "idc",
    "talk to you later": "ttyl",
    "let me know": "lmk",
    "for real": "fr",
    "as far as i know": "afaik",
    "in real life": "irl",
    "just kidding": "jk",
    "laughing out loud": "lol",
    "as soon as possible": "asap",
}

# contractions where dropping the apostrophe is unambiguous (its/well/ill/were/id/hell are NOT in here)
APOSTROPHE_DROPS: dict[str, str] = {
    "don't": "dont", "can't": "cant", "won't": "wont", "didn't": "didnt", "doesn't": "doesnt",
    "isn't": "isnt", "wasn't": "wasnt", "aren't": "arent", "weren't": "werent", "wouldn't": "wouldnt",
    "couldn't": "couldnt", "shouldn't": "shouldnt", "haven't": "havent", "hasn't": "hasnt",
    "hadn't": "hadnt", "i'm": "im", "i've": "ive", "that's": "thats", "let's": "lets", "what's": "whats",
    "you're": "youre", "they're": "theyre", "you've": "youve", "we've": "weve", "they've": "theyve",
    "there's": "theres", "here's": "heres", "ain't": "aint", "y'all": "yall"
}

# --------------------------------------------------------------------------- default profiles
def _b(**kw) -> dict:
    return kw


NOTE_WORK = "work chat: neutral on purpose - change style to casual for texting-style messages"


DEFAULT_PROFILES: list[dict[str, Any]] = [
    # --- work chat: neutral on purpose. Change "style" to "casual" if you want Slack/Teams texting-style.
    _b(name="Slack", note=NOTE_WORK, process=["slack.exe"], style="neutral"),
    _b(name="Slack (web)", note=NOTE_WORK, process=["@browsers"], title=r"\bSlack\b", style="neutral"),
    _b(name="Teams", note=NOTE_WORK, process=["teams.exe", "ms-teams.exe", "msteams.exe"], style="neutral"),
    _b(name="Teams (web)", note=NOTE_WORK, process=["@browsers"], title=r"\b(?:Microsoft )?Teams\b",
       style="neutral"),
    # --- casual chat
    _b(name="Discord", process=["discord.exe", "discordptb.exe", "discordcanary.exe", "discorddevelopment.exe"],
       style="casual"),
    _b(name="Discord (web)", process=["@browsers"],
       title=r"^\W*(?:\(\d+\)\s*)?discord\s*(?:[|\-–—•]|$)", style="casual"),
    _b(name="WhatsApp", process=["whatsapp.exe", "whatsapp.root.exe", "whatsappdesktop.exe"], style="casual"),
    _b(name="WhatsApp (web)", process=["@browsers"], title=r"^\W*(?:\(\d+\)\s*)?WhatsApp\b", style="casual"),
    _b(name="Telegram", process=["telegram.exe", "telegramdesktop.exe", "ayugram.exe", "kotatogram.exe"],
       style="casual"),
    _b(name="Telegram (web)", process=["@browsers"], title=r"^\W*(?:\(\d+\)\s*)?Telegram\b", style="casual"),
    _b(name="Messenger", process=["messenger.exe"], style="casual"),
    _b(name="Messenger (web)", process=["@browsers"],
       title=r"^\W*(?:\(\d+\)\s*)?(?:Messenger|Chats?)\s*(?:[|\-–—]|$)|\bMessenger\s*\|\s*Facebook",
       style="casual"),
    _b(name="Signal", process=["signal.exe"], style="casual"),
    _b(name="Instagram DMs", process=["@browsers"],
       title=r"Instagram\s*[•·|\-]\s*(?:Direct|Chats?|Inbox|Messages)|(?:Direct|Inbox)\s*[•·]\s*Instagram",
       style="casual"),
    _b(name="X DMs", process=["@browsers"],
       title=r"^\W*(?:\(\d+\)\s*)?(?:Messages|Chat)\s*/\s*(?:X|Twitter)\b", style="casual"),
    _b(name="Phone-style messengers", process=["beeper.exe", "texts.exe", "imessage.exe", "element.exe",
                                               "guilded.exe", "skype.exe"], style="casual"),
    _b(name="Messages (web)", process=["@browsers"],
       title=r"\bMessages for web\b|\bGoogle Messages\b|\biMessage\b|\bBeeper\b|\bTexts\.com\b", style="casual"),
    _b(name="Snapchat (web)", process=["@browsers"], title=r"\bSnapchat\b", style="casual"),
    _b(name="Reddit chat", process=["@browsers"],
       title=r"\bReddit\s+Chat\b|^\W*(?:\(\d+\)\s*)?Chat\s*[|\-–—]\s*Reddit|\bchat\.reddit\.com\b",
       style="casual"),
    # --- professional
    _b(name="Gmail", process=["@browsers"], title=r"(?:^|[-–—|]\s*)Gmail(?:\s*[-–—|]|$)",
       style="professional"),
    _b(name="Outlook", process=["outlook.exe", "olk.exe", "hxoutlook.exe", "hxmail.exe", "thunderbird.exe",
                                "mailspring.exe", "em client.exe", "emclient.exe"], style="professional"),
    _b(name="Outlook / Office web", process=["@browsers"],
       title=r"\bOutlook\b|outlook\.(?:live|office)|\bOffice 365 Mail\b", style="professional"),
    _b(name="Google Docs", process=["@browsers"], title=r"[-–—]\s*Google Docs\b", style="professional"),
    _b(name="Word", process=["winword.exe"], style="professional"),
    _b(name="LinkedIn", process=["@browsers"], title=r"\bLinkedIn\b", style="professional"),
    _b(name="Proton / Yahoo mail", process=["@browsers"], title=r"\bProton\s*Mail\b|\bYahoo\s*Mail\b",
       style="professional"),
    # --- code
    _b(name="Code editors", process=[
        "code.exe", "code - insiders.exe", "cursor.exe", "windsurf.exe", "zed.exe", "devenv.exe", "idea64.exe",
        "idea.exe", "pycharm64.exe", "pycharm.exe", "webstorm64.exe", "clion64.exe", "rider64.exe",
        "goland64.exe", "phpstorm64.exe", "rubymine64.exe", "datagrip64.exe", "androidstudio64.exe",
        "studio64.exe", "sublime_text.exe", "notepad++.exe", "nvim-qt.exe"], style="code"),
    _b(name="Terminals", process=[
        "windowsterminal.exe", "wt.exe", "powershell.exe", "pwsh.exe", "cmd.exe", "conhost.exe",
        "openconsole.exe", "alacritty.exe", "wezterm-gui.exe", "mintty.exe", "bash.exe", "kitty.exe",
        "tabby.exe", "hyper.exe"], style="code"),
]

DEFAULT_PROCESS_GROUPS: dict[str, list[str]] = {"browsers": BROWSERS}


# --------------------------------------------------------------------------- resolution
@dataclass
class StyleContext:
    exe: str = ""
    title: str = ""
    style: str = NEUTRAL
    rule: str = ""            # matched rule name ("" = fallback)
    app: str = ""             # friendly app label for UI ("Discord", "chrome")
    auto: bool = True

    @property
    def label(self) -> str:
        """Overlay text, e.g. 'casual · Discord'."""
        return f"{self.style} · {self.app}" if self.app else self.style


def canon_style(name: Optional[str]) -> str:
    n = (name or NEUTRAL).strip().lower()
    return _ALIASES.get(n, n) or NEUTRAL


def _norm_exe(e: str) -> str:
    e = (e or "").strip().lower()
    return e


_re_cache: dict[str, Optional[re.Pattern]] = {}


def _compile(pat: str) -> Optional[re.Pattern]:
    if pat not in _re_cache:
        try:
            _re_cache[pat] = re.compile(pat, re.IGNORECASE)
        except re.error as e:
            log.warning("invalid title regex in profiles %r: %s", pat, e)
            _re_cache[pat] = None
    return _re_cache[pat]


def expand_processes(procs: Any, groups: dict[str, list[str]]) -> set[str]:
    if isinstance(procs, str):
        procs = [procs]
    out: set[str] = set()
    for p in procs or []:
        p = _norm_exe(str(p))
        if p.startswith("@"):
            out.update(_norm_exe(x) for x in groups.get(p[1:], []))
        elif p:
            out.add(p)
    return out


def rule_matches(rule: dict, exe: str, title: str, groups: dict[str, list[str]]) -> bool:
    procs = rule.get("process")
    pat = rule.get("title")
    if not procs and not pat:
        return False
    if procs and exe not in expand_processes(procs, groups):
        return False
    if pat:
        rx = _compile(str(pat))
        if rx is None or not title or not rx.search(title):
            return False
    return True


def pretty_app(exe: str) -> str:
    e = (exe or "").lower()
    if e.endswith(".exe"):
        e = e[:-4]
    return e


def resolve(exe: str, title: str, cfg) -> StyleContext:
    """Pick the style for the window ``(exe, title)`` from ``cfg`` (first matching profile wins)."""
    exe = _norm_exe(exe)
    title = title or ""
    ctx = StyleContext(exe=exe, title=title, app=pretty_app(exe))
    auto = bool(cfg.get("auto_style", True))
    ctx.auto = auto
    if not auto:
        ctx.style = NEUTRAL
        ctx.app = ""
        return ctx
    groups = cfg.get("process_groups", DEFAULT_PROCESS_GROUPS) or {}
    for rule in cfg.get("profiles", DEFAULT_PROFILES) or []:
        try:
            if isinstance(rule, dict) and rule_matches(rule, exe, title, groups):
                ctx.style = canon_style(rule.get("style"))
                ctx.rule = str(rule.get("name") or "")
                ctx.app = ctx.rule or ctx.app
                break
        except Exception:
            log.exception("bad profile rule %r", rule)
    # legacy config (apps.code / apps.chat lists from older versions) still honoured as a last resort
    if not ctx.rule and exe:
        for key, st in (("code", "code"), ("chat", "casual")):
            if exe in {_norm_exe(x) for x in (cfg.get(f"apps.{key}", []) or [])} and \
                    exe not in {_norm_exe(x) for x in LEGACY_DEFAULT_APPS.get(key, [])}:
                ctx.style, ctx.rule = st, f"legacy apps.{key}"
                break
    if ctx.style not in (cfg.get("styles", {}) or {}) and ctx.style not in DEFAULT_STYLES:
        log.warning("rule %r references unknown style %r - using neutral", ctx.rule, ctx.style)
        ctx.style = NEUTRAL
    return ctx


# user entries added to the old `apps` lists are honoured; the old shipped defaults are not (profiles cover them)
LEGACY_DEFAULT_APPS = {
    "code": ["code.exe", "code - insiders.exe", "cursor.exe", "windsurf.exe", "devenv.exe", "idea64.exe",
             "pycharm64.exe", "webstorm64.exe", "clion64.exe", "rider64.exe", "sublime_text.exe",
             "notepad++.exe", "windowsterminal.exe", "wt.exe", "powershell.exe", "pwsh.exe",
             "cmd.exe", "conhost.exe", "alacritty.exe", "wezterm-gui.exe", "mintty.exe", "bash.exe"],
    "chat": ["discord.exe", "slack.exe", "telegram.exe", "whatsapp.exe", "signal.exe", "teams.exe",
             "ms-teams.exe", "element.exe", "messenger.exe", "guilded.exe"],
}


def get_style(cfg, name: str) -> dict:
    """Merged definition of a style (config overrides built-ins; unknown -> neutral)."""
    name = canon_style(name)
    styles = cfg.get("styles", {}) or {}
    d = styles.get(name)
    if d is None:
        d = DEFAULT_STYLES.get(name) or styles.get(NEUTRAL) or DEFAULT_STYLES[NEUTRAL]
        name = name if name in DEFAULT_STYLES else NEUTRAL
    d = dict(d)
    d["name"] = name
    if "examples" not in d:
        d["examples"] = copy.deepcopy(DEFAULT_EXAMPLES.get(name, []))
    return d


# --------------------------------------------------------------------------- deterministic post-pass
_abbr_cache: dict[tuple, list] = {}


def _compile_abbr(abbr: dict[str, str]) -> list[tuple[re.Pattern, str]]:
    key = tuple(sorted(abbr.items()))
    if key not in _abbr_cache:
        pats = []
        for ph in sorted((p for p in abbr if p.strip()), key=lambda s: -len(s)):
            toks = [re.escape(t).replace("'", "'?") for t in ph.strip().replace("’", "'").split()]
            rx = r"(?<![\w'])" + r"\s+".join(toks) + r"(?![\w'])"
            pats.append((re.compile(rx, re.IGNORECASE), abbr[ph]))
        _abbr_cache[key] = pats
    return _abbr_cache[key]


def apply_abbreviations(text: str, abbr: dict[str, str]) -> str:
    text = text.replace("’", "'")
    for rx, rep in _compile_abbr(abbr or {}):
        text = rx.sub(lambda m, r=rep: r, text)
    return text


def expand_abbreviations(text: str, abbr: dict[str, str]) -> str:
    """abbreviation -> original phrase (used by the LLM drift guard so shortening isn't called drift)."""
    for ph, ab in sorted(((p, a) for p, a in (abbr or {}).items() if p.strip() and a.strip()),
                         key=lambda x: -len(x[1])):
        text = re.sub(r"(?<![\w'])" + re.escape(ab) + r"(?![\w'])", lambda m, p=ph: p, text, flags=re.IGNORECASE)
    return text


_drop_rx = re.compile(r"(?<![\w'])(" + "|".join(re.escape(k) for k in sorted(APOSTROPHE_DROPS, key=len, reverse=True))
                      + r")(?![\w'])", re.IGNORECASE)


def drop_apostrophes(text: str) -> str:
    text = text.replace("’", "'")
    return _drop_rx.sub(lambda m: APOSTROPHE_DROPS[m.group(1).lower()], text)


def apply_word_map(text: str, wm: dict[str, str]) -> str:
    """Whole-word, case-insensitive single word replacements ({"u": "you"})."""
    for src, dst in wm.items():
        if src.strip():
            text = re.sub(r"(?<![\w'.-])" + re.escape(src.strip()) + r"(?![\w'-]|\.\w)", lambda m, d=dst: d, text,
                          flags=re.IGNORECASE)
    return text


def strip_final_period(text: str, mode: Any) -> str:
    t = text.rstrip()
    if not mode or not t.endswith(".") or t.endswith(".."):
        return t
    if mode == "single":
        # only a lone sentence loses its period (multi-sentence / multi-line text keeps its punctuation)
        if len(re.findall(r"[.!?](?:\s|$)", t)) > 1 or "\n" in t:
            return t
    return t[:-1].rstrip()


def apply_style_def(text: str, sd: dict, abbr: Optional[dict[str, str]] = None,
                    preserve: Optional[list[str]] = None) -> str:
    """Deterministic post-processing of a style definition. Idempotent; safe on LLM output or regex output."""
    t = text
    if not t.strip():
        return t
    if sd.get("abbreviations"):
        t = apply_abbreviations(t, abbr if abbr is not None else DEFAULT_ABBREVIATIONS)
    if sd.get("word_map"):
        t = apply_word_map(t, sd["word_map"])
    if sd.get("drop_apostrophes"):
        t = drop_apostrophes(t)
    if sd.get("lowercase"):
        t = t.lower()
        for w in preserve or []:   # keep dictionary spellings (names, acronyms) intact
            w = w.strip()
            if w:
                t = re.sub(r"(?<![\w])" + re.escape(w) + r"(?![\w])", lambda m, w=w: w, t, flags=re.IGNORECASE)
    return strip_final_period(t, sd.get("strip_trailing_period"))
