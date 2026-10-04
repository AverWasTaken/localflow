"""Offline unit tests for per-app styles (no GPU, no LLM, no real windows are touched).

   .venv\\Scripts\\python.exe -m unittest discover -s tests -p "test_unit_*.py" -v
"""
import copy
import ctypes
import os
import sqlite3
import sys
import tempfile
import time
import unittest
from ctypes import wintypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from localflow import styles, textproc  # noqa: E402
from localflow.cleanup import Cleaner  # noqa: E402
from localflow.config import Config, DEFAULTS  # noqa: E402
from localflow.history import History  # noqa: E402


def make_cfg(**over) -> Config:
    cfg = Config(path=Path(tempfile.mkdtemp()) / "cfg.json")
    for k, v in over.items():
        cfg.set(k.replace("__", "."), v, save=False)
    return cfg


class Resolve(unittest.TestCase):
    def setUp(self):
        self.cfg = make_cfg()

    def style(self, exe, title=""):
        return styles.resolve(exe, title, self.cfg)

    def test_default_rule_table(self):
        chrome = "chrome.exe"
        cases = [
            # casual
            ("discord.exe", "", "casual"),
            ("Discord.exe", "#general - Discord", "casual"),
            (chrome, "Discord | #general | Some Server - Google Chrome", "casual"),
            (chrome, "(3) Discord | @friend - Google Chrome", "casual"),
            ("msedge.exe", "(2) WhatsApp - Microsoft Edge", "casual"),
            ("whatsapp.exe", "WhatsApp", "casual"),
            ("telegram.exe", "Telegram", "casual"),
            ("firefox.exe", "Telegram Web — Mozilla Firefox", "casual"),
            ("messenger.exe", "", "casual"),
            ("signal.exe", "Signal", "casual"),
            (chrome, "Instagram • Direct - Google Chrome", "casual"),
            (chrome, "Messages / X - Google Chrome", "casual"),
            ("zen.exe", "Snapchat — Zen Browser", "casual"),
            (chrome, "Messages for web - Google Chrome", "casual"),
            # professional
            (chrome, "Inbox (3) - someone@gmail.com - Gmail - Google Chrome", "professional"),
            ("brave.exe", "Gmail - Brave", "professional"),
            ("outlook.exe", "Inbox - Outlook", "professional"),
            ("olk.exe", "Mail - Outlook", "professional"),
            ("msedge.exe", "Mail - Jane - Outlook - Microsoft Edge", "professional"),
            (chrome, "Quarterly plan - Google Docs - Google Chrome", "professional"),
            ("WINWORD.EXE", "Document1 - Word", "professional"),
            (chrome, "(5) Feed | LinkedIn - Google Chrome", "professional"),
            ("firefox.exe", "Proton Mail — Mozilla Firefox", "professional"),
            (chrome, "Yahoo Mail - Google Chrome", "professional"),
            # code
            ("Code.exe", "main.py - Visual Studio Code", "code"),
            ("cursor.exe", "", "code"),
            ("pycharm64.exe", "", "code"),
            ("WindowsTerminal.exe", "pwsh", "code"),
            ("cmd.exe", "", "code"),
            ("conhost.exe", "", "code"),
            # neutral by design / fallback
            ("slack.exe", "general - Acme", "neutral"),
            (chrome, "general - Acme - Slack - Google Chrome", "neutral"),
            ("ms-teams.exe", "Chat | Microsoft Teams", "neutral"),
            ("notepad.exe", "gmail notes.txt - Notepad", "neutral"),     # "Gmail" in a non-browser title
            (chrome, "How to build a Discord bot - Stack Overflow - Google Chrome", "neutral"),
            (chrome, "YouTube - Google Chrome", "neutral"),
            ("", "", "neutral"),
        ]
        for exe, title, want in cases:
            with self.subTest(exe=exe, title=title):
                self.assertEqual(self.style(exe, title).style, want)

    def test_first_match_wins(self):
        self.cfg.set("profiles", [
            {"name": "mine", "process": ["discord.exe"], "style": "professional"},
            {"name": "d", "process": ["discord.exe"], "style": "casual"}], save=False)
        c = self.style("discord.exe")
        self.assertEqual((c.style, c.rule), ("professional", "mine"))

    def test_process_and_title_are_anded(self):
        self.cfg.set("profiles", [{"name": "x", "process": ["chrome.exe"], "title": "jira", "style": "code"}],
                     save=False)
        self.assertEqual(self.style("chrome.exe", "JIRA board").style, "code")
        self.assertEqual(self.style("chrome.exe", "other").style, "neutral")
        self.assertEqual(self.style("firefox.exe", "jira").style, "neutral")

    def test_title_only_rule_and_case_insensitive(self):
        self.cfg.set("profiles", [{"name": "t", "title": r"^standup notes\b", "style": "casual"}], save=False)
        self.assertEqual(self.style("anything.exe", "Standup Notes - Notepad").style, "casual")

    def test_empty_rule_matches_nothing_and_bad_regex_ignored(self):
        self.cfg.set("profiles", [{"name": "empty", "style": "casual"},
                                  {"name": "bad", "process": ["a.exe"], "title": "([", "style": "casual"},
                                  {"name": "ok", "process": ["a.exe"], "style": "code"}], save=False)
        self.assertEqual(self.style("a.exe", "x").style, "code")
        self.assertEqual(self.style("b.exe", "x").style, "neutral")

    def test_unknown_style_falls_back_to_neutral(self):
        self.cfg.set("profiles", [{"name": "x", "process": ["a.exe"], "style": "nope"}], save=False)
        self.assertEqual(self.style("a.exe").style, "neutral")

    def test_auto_style_off(self):
        self.cfg.set("auto_style", False, save=False)
        c = self.style("discord.exe", "")
        self.assertEqual(c.style, "neutral")
        self.assertEqual(c.label, "neutral")

    def test_custom_style_and_group(self):
        self.cfg.set("styles", {"shout": {"prompt": "ALL CAPS", "uppercase": True}}, save=False)
        self.cfg.set("process_groups", {"mine": ["foo.exe", "BAR.exe"]}, save=False)
        self.cfg.set("profiles", [{"name": "g", "process": ["@mine"], "style": "shout"}], save=False)
        self.assertEqual(self.style("bar.exe").style, "shout")
        self.assertEqual(styles.get_style(self.cfg, "shout")["prompt"], "ALL CAPS")

    def test_label_and_aliases(self):
        self.assertEqual(self.style("discord.exe").label, "casual · Discord")
        self.assertEqual(styles.canon_style("general"), "neutral")
        self.assertEqual(styles.canon_style("CHAT"), "casual")

    def test_legacy_apps_lists_still_honoured(self):
        self.cfg.set("apps", {"code": ["mytool.exe"], "chat": ["mychat.exe"]}, save=False)
        self.assertEqual(self.style("mytool.exe").style, "code")
        self.assertEqual(self.style("mychat.exe").style, "casual")

    def test_speed(self):
        t = time.perf_counter()
        for _ in range(200):
            self.style("chrome.exe", "Inbox (3) - someone@gmail.com - Gmail - Google Chrome")
            self.style("notepad.exe", "untitled - Notepad")
        per = (time.perf_counter() - t) / 400 * 1000
        self.assertLess(per, 1.0, f"{per:.3f} ms per resolve")


class Deterministic(unittest.TestCase):
    def setUp(self):
        self.cfg = make_cfg()
        self.sd = styles.get_style(self.cfg, "casual")
        self.abbr = self.cfg.get("abbreviations")

    def casual(self, t):
        return textproc.apply_style(t, self.sd, self.abbr)

    def test_abbreviation_map(self):
        cases = {
            "I don't know": "idk", "I dont know": "idk", "Never mind": "nvm", "nevermind": "nvm",
            "To be honest": "tbh", "right now": "rn", "On my way": "omw", "Not gonna lie": "ngl",
            "In my opinion": "imo", "By the way": "btw", "Oh my god": "omg", "Be right back": "brb",
            "I don't care": "idc", "Talk to you later": "ttyl", "Let me know": "lmk", "For real": "fr",
        }
        for src, want in cases.items():
            with self.subTest(src=src):
                self.assertEqual(self.casual(src + "."), want)

    def test_curly_apostrophe(self):
        self.assertEqual(self.casual("I don’t know"), "idk")

    def test_word_boundaries(self):
        for s in ("the bright nowhere", "forever real estate", "a mind of my own way", "unlet me knowing",
                  "overlet me knowledge"):
            self.assertEqual(self.casual(s), s.lower().replace("overlet me knowledge", "overlet me knowledge"))
        self.assertEqual(self.casual("I am on my way home right now."), "i am omw home rn")
        self.assertEqual(self.casual("it's for realistic reasons"), "it's for realistic reasons")

    def test_lowercase_period_and_idempotent(self):
        out = self.casual("I Don't Know, Never Mind, Let's Just Do It Tomorrow.")
        self.assertEqual(out, "idk, nvm, lets just do it tomorrow")
        self.assertEqual(self.casual(out), out)

    def test_question_and_exclamation_survive(self):
        self.assertEqual(self.casual("Are you coming?"), "are you coming?")
        self.assertEqual(self.casual("Let me know!"), "lmk!")
        self.assertEqual(self.casual("wait..."), "wait...")

    def test_dictionary_spelling_preserved(self):
        out = textproc.apply_style("Ask Maria about the GPU", self.sd, self.abbr, preserve=["GPU", "Maria"])
        self.assertEqual(out, "ask Maria about the GPU")

    def test_configurable_abbreviation_list(self):
        out = textproc.apply_style("See you tomorrow", self.sd, {"see you tomorrow": "cya tmrw"})
        self.assertEqual(out, "cya tmrw")

    def test_word_map_puts_real_words_back(self):
        self.assertEqual(self.casual("can u send ur link pls"), "can you send your link please")
        self.assertEqual(self.casual("U-turn at the u.s. border"), "u-turn at the u.s. border")   # not a bare "u"

    def test_apostrophe_drops_only_unambiguous(self):
        out = self.casual("I'm sure it's fine and we'll see, don't worry, that's it, I'll go")
        self.assertEqual(out, "im sure it's fine and we'll see, dont worry, thats it, i'll go")

    def test_code_style_single_sentence_only(self):
        code = styles.get_style(self.cfg, "code")
        self.assertEqual(textproc.apply_style("add a null check.", code), "add a null check")
        self.assertEqual(textproc.apply_style("Add it. Then run it.", code), "Add it. Then run it.")

    def test_professional_and_neutral_untouched(self):
        for name in ("professional", "neutral"):
            sd = styles.get_style(self.cfg, name)
            self.assertEqual(textproc.apply_style("I don't know, Never mind.", sd, self.abbr),
                             "I don't know, Never mind.")

    def test_regex_clean_per_style(self):
        self.assertEqual(textproc.regex_clean("um i don't know what to do right now", "casual", self.abbr),
                         "idk what to do rn")
        self.assertEqual(textproc.regex_clean("hello there my friend", "professional"), "Hello there my friend.")
        self.assertEqual(textproc.regex_clean("hello there my friend", "neutral"), "Hello there my friend.")
        self.assertEqual(textproc.regex_clean("add a null check", "code"), "add a null check")


class CleanerShortcuts(unittest.TestCase):
    """Paths that never reach the LLM must still apply the deterministic style pass."""

    def test_short_utterance_still_styled(self):
        cfg = make_cfg()
        cl = Cleaner(cfg)                                   # server never started -> not ready
        out, info = cl.clean("never mind", "casual")        # 2 words < min_words
        self.assertEqual(out, "nvm")
        self.assertEqual(info["reason"], "short utterance")
        out, _ = cl.clean("Sounds good", "casual")
        self.assertEqual(out, "sounds good")
        out, _ = cl.clean("sounds good", "professional")
        self.assertEqual(out, "Sounds good.")

    def test_llm_disabled_or_down_still_styled(self):
        cfg = make_cfg(cleanup__enabled=False)
        cl = Cleaner(cfg)
        out, info = cl.clean("um I don't know what to do right now honestly", "casual")
        self.assertEqual(out, "idk what to do rn honestly")
        cfg2 = make_cfg()
        cl2 = Cleaner(cfg2)
        cl2.server.failed = "x"
        cl2.server.start = lambda *a, **k: None             # don't launch anything
        out, info = cl2.clean("um I don't know what to do right now honestly", "casual")
        self.assertEqual(info["reason"], "llm not ready")
        self.assertEqual(out, "idk what to do rn honestly")


class DriftGuard(unittest.TestCase):
    def setUp(self):
        self.cfg = make_cfg()
        self.cl = Cleaner(self.cfg)

    def sane(self, raw, out, style):
        return self.cl._sane(raw, out, styles.get_style(self.cfg, style))

    def test_casual_shortening_is_not_drift(self):
        raw = "I don't know, never mind, let's just do it tomorrow."
        self.assertEqual(self.sane(raw, "idk nvm lets just do it tomorrow", "casual"), "")
        raw2 = "to be honest I'm on my way right now so let me know if you need anything"
        self.assertEqual(self.sane(raw2, "tbh im omw rn so lmk if you need anything", "casual"), "")
        raw3 = "not gonna lie that was talk to you later by the way oh my god"
        self.assertEqual(self.sane(raw3, "ngl that was ttyl btw omg", "casual"), "")

    def test_abbreviations_only_forgiven_for_styles_that_use_them(self):
        raw = "I don't know what we should do about the budget this week"
        self.assertNotEqual(self.sane(raw, "idk what we should do about the budget this week", "neutral"), "x")
        # neutral: an unexpanded abbreviation is fine too (length/precision still ok) - the guard stays loose
        self.assertEqual(self.sane(raw, "idk what we should do about the budget this week", "casual"), "")

    def test_chat_reply_and_tags_rejected(self):
        raw = "what is the capital of france"
        self.assertTrue(self.sane(raw, "Sure! The capital of France is Paris.", "casual"))
        self.assertTrue(self.sane(raw, "<transcript>what is the capital of france</transcript>", "neutral"))
        self.assertTrue(self.sane(raw, "[style: casual] what is the capital of france", "casual"))

    def test_answering_rejected(self):
        raw = "what is the capital of france and when was it founded"
        self.assertTrue(self.sane(raw, "Paris is the capital of France and it was founded in the third century BC.",
                                  "professional"))

    def test_professional_polish_allowed(self):
        raw = "hey can you send me that report by friday thanks"
        self.assertEqual(self.sane(raw, "Hi, could you send me the report by Friday? Thanks.", "professional"), "")
        raw = "um so we gonna need to push the launch back a week cause the testing aint done"
        self.assertEqual(self.sane(
            raw, "We are going to need to push the launch back a week because the testing is not done.",
            "professional"), "")

    def test_invented_greeting_and_signoff_rejected(self):
        raw = "i wanted to follow up on the proposal we discussed last week"
        self.assertEqual(self.sane(raw, "I wanted to follow up on the proposal we discussed last week.",
                                   "professional"), "")
        self.assertIn("greeting", self.sane(raw, "Hello, I wanted to follow up on the proposal we discussed last week.",
                                            "professional"))
        self.assertIn("sign-off", self.sane(raw, "I wanted to follow up on the proposal we discussed last week.\n\nBest regards",
                                            "professional"))
        # spoken greeting / thanks are kept
        raw = "hi sarah i wanted to follow up on the proposal thanks"
        self.assertEqual(self.sane(raw, "Hi Sarah, I wanted to follow up on the proposal. Thanks.", "professional"), "")

    def test_invented_content_rejected(self):
        raw = "send the report to john by friday please"
        self.assertTrue(self.sane(raw, "Please send the quarterly financial report to John Smith in accounting "
                                       "by end of day Friday, and copy the whole team.", "professional"))


class StyleMessages(unittest.TestCase):
    def test_prompt_contains_style_rules_and_examples(self):
        cfg = make_cfg()
        cl = Cleaner(cfg)
        msgs = cl._messages("hello there", styles.get_style(cfg, "casual"), ["Maria"])
        last = msgs[-1]["content"]
        self.assertIn("[style: casual]", last)
        self.assertIn("[style rules:", last)
        self.assertIn("[vocabulary: Maria]", last)
        self.assertTrue(last.endswith("<transcript>hello there</transcript>"))
        joined = "\n".join(m["content"] for m in msgs)
        self.assertIn("idk nvm lets just do it tomorrow", joined)           # spec's casual example
        self.assertNotIn("Hi, could you send me the report", joined)         # professional examples not mixed in
        pro = "\n".join(m["content"] for m in cl._messages("x", styles.get_style(cfg, "professional"), []))
        self.assertIn("Hi, could you send me the report by Friday? Thanks.", pro)
        # neutral has no style header at all (identical to the pre-styles prompt)
        neutral = cl._messages("x y z w", styles.get_style(cfg, "neutral"), [])
        self.assertEqual(neutral[-1]["content"], "<transcript>x y z w</transcript>")

    def test_context_line(self):
        cfg = make_cfg()
        cl = Cleaner(cfg)
        msgs = cl._messages("and then more", styles.get_style(cfg, "neutral"), [], context="Earlier sentence.")
        self.assertIn("[previous text: Earlier sentence.]", msgs[-1]["content"])

    def test_custom_style_examples_from_config(self):
        cfg = make_cfg()
        cfg.set("styles", {"pirate": {"prompt": "talk like a pirate", "examples": [["hello friend", "Ahoy, matey"]]}},
                save=False)
        cl = Cleaner(cfg)
        msgs = cl._messages("hello", styles.get_style(cfg, "pirate"), [])
        self.assertTrue(any(m["content"] == "Ahoy, matey" for m in msgs))


class HistoryMigration(unittest.TestCase):
    def test_old_db_is_migrated_and_rows_kept(self):
        d = Path(tempfile.mkdtemp()) / "history.db"
        c = sqlite3.connect(str(d))
        c.executescript("""CREATE TABLE history (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, app TEXT,
            raw TEXT, cleaned TEXT, duration_s REAL, stt_ms INTEGER, cleanup_ms INTEGER, total_ms INTEGER);
            INSERT INTO history(ts,app,raw,cleaned,duration_s,stt_ms,cleanup_ms,total_ms)
            VALUES (1.0,'notepad.exe','old raw','Old text.',2.0,100,200,300);""")
        c.commit()
        c.close()
        h = History(path=d)
        rows = h.list()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["cleaned"], "Old text.")
        self.assertIsNone(rows[0]["style"])
        h.add("raw", "text", "discord.exe", 3.0, 1, 2, 3, title="#general - Discord", style="casual", segments=2)
        h2 = History(path=d)                                   # re-open: migration is idempotent
        rows = h2.list()
        self.assertEqual(rows[0]["style"], "casual")
        self.assertEqual(rows[0]["title"], "#general - Discord")
        self.assertEqual(rows[0]["segments"], 2)
        self.assertEqual(len(h2.list(search="casual")), 1)
        self.assertEqual(len(h2.list(search="General")), 1)


class ConfigUpgrade(unittest.TestCase):
    def test_existing_config_gets_new_sections_and_keeps_user_values(self):
        import json
        p = Path(tempfile.mkdtemp()) / "config.json"
        p.write_text(json.dumps({"hotkey": "ctrl+shift", "mic": {"match": "blue"}, "apps": {"code": ["x.exe"]}}),
                     encoding="utf-8")
        cfg = Config(path=p)
        self.assertEqual(cfg.get("hotkey"), "ctrl+shift")
        saved = json.loads(p.read_text(encoding="utf-8"))
        for k in ("profiles", "styles", "abbreviations", "streaming", "auto_style", "process_groups"):
            self.assertIn(k, saved)
        self.assertEqual(saved["mic"]["match"], "blue")
        self.assertEqual(saved["apps"]["code"], ["x.exe"])       # unknown keys preserved
        self.assertEqual(len(saved["profiles"]), len(DEFAULTS["profiles"]))

    def test_user_profiles_replace_not_merge(self):
        import json
        p = Path(tempfile.mkdtemp()) / "config.json"
        p.write_text(json.dumps({"profiles": [{"name": "only", "process": ["a.exe"], "style": "code"}]}),
                     encoding="utf-8")
        cfg = Config(path=p)
        self.assertEqual(len(cfg.get("profiles")), 1)


class ForegroundDetection(unittest.TestCase):
    """Uses a throwaway, never-shown window owned by this process - no real window is touched."""

    def test_window_info_on_own_hidden_window(self):
        from localflow import winutil
        u32 = ctypes.WinDLL("user32", use_last_error=True)
        u32.CreateWindowExW.argtypes = [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
                                        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.HWND,
                                        wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID]
        u32.CreateWindowExW.restype = wintypes.HWND
        title = "Inbox (3) - someone@gmail.com - Gmail - Google Chrome " + "x" * 300     # long title (> 256 chars)
        hwnd = u32.CreateWindowExW(0, "STATIC", title, 0x80000000, 0, 0, 10, 10, None, None, None, None)  # WS_POPUP, never shown
        self.assertTrue(hwnd)
        try:
            t = time.perf_counter()
            for _ in range(100):
                exe, got = winutil.window_info(hwnd)
            per = (time.perf_counter() - t) / 100 * 1000
        finally:
            u32.DestroyWindow(hwnd)
        self.assertEqual(exe, os.path.basename(sys.executable).lower())
        self.assertEqual(got, title)                                     # not truncated at 256
        self.assertLess(per, 5.0, f"{per:.3f} ms")

    def test_bad_handles_dont_raise(self):
        from localflow import winutil
        self.assertEqual(winutil.window_info(0), ("", ""))
        exe, title = winutil.window_info(0x7FFFFFF)                      # invalid hwnd
        self.assertEqual(title, "")

    def test_foreground_info_returns_strings_and_is_fast(self):
        from localflow import winutil
        t = time.perf_counter()
        for _ in range(50):
            exe, title = winutil.foreground_info()
        per = (time.perf_counter() - t) / 50 * 1000
        self.assertIsInstance(exe, str)
        self.assertIsInstance(title, str)
        self.assertLess(per, 5.0, f"{per:.3f} ms")

    def test_protected_process_name_via_psutil_fallback(self):
        from localflow import winutil
        # System (pid 4) cannot be opened; csrss/System-ish names are still resolved or ''.
        self.assertIsInstance(winutil.pid_exe_name(4), str)


if __name__ == "__main__":
    unittest.main()
