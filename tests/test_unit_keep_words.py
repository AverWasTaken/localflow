"""keep_words guard (localflow/keepwords.py + Cleaner._keep_words) and the LocalFlow fine-tune prompt routing fallback.
No llama-server is started.

   .venv\\Scripts\\python.exe -m unittest tests.test_unit_keep_words -v
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from localflow import keepwords, styles  # noqa: E402
from localflow.cleanup import Cleaner  # noqa: E402
from localflow.config import Config  # noqa: E402

VOCAB = ["fifine", "reyna", "claude", "skye", "qwen", "aj"]
ABBR = styles.DEFAULT_ABBREVIATIONS
CASUAL_WM = styles.DEFAULT_STYLES["casual"]["word_map"]


def fix(raw, out, vocab=VOCAB, settings=(), word_map=None):
    return keepwords.repair(raw, out, vocab, ABBR, word_map or {}, settings)[0]


class Reverts(unittest.TestCase):
    """Swaps from real history rows: the spoken words come back."""

    def test_helpers(self):
        self.assertEqual(fix("I meant unhelped technicians", "I meant helpers."), "I meant unhelped technicians.")

    def test_undelivered(self):
        self.assertEqual(fix("what i meant to say was unhelped technicians",
                             "What I meant to say was undelivered technicians."),
                         "What I meant to say was unhelped technicians.")

    def test_unhelpful(self):
        self.assertEqual(
            fix("Can you change the thing that checks for unhelped technicians in the configuration chat from five "
                "minutes to two minutes?",
                "Can you change the thing that checks for unhelpful technicians in the configuration chat from 5 "
                "minutes to 2 minutes?"),
            "Can you change the thing that checks for unhelped technicians in the configuration chat from 5 "
            "minutes to 2 minutes?")

    def test_region_all_or_nothing(self):
        # "yaw is" -> "voice are": the agreement change is part of the swap, so both words go back
        self.assertEqual(
            fix("Awesome your pitch and your yaw is all good So let me just configure a few things",
                "Awesome, your pitch and your voice are all good. So let me just configure a few things."),
            "Awesome, your pitch and your yaw is all good. So let me just configure a few things.")

    def test_hyphenated_jargon(self):
        self.assertEqual(fix("So the whole pitch-yaw thing.", "So the whole YAPAC thing."),
                         "So the whole pitch-yaw thing.")

    def test_dictionary_word_inserted_for_symbol(self):
        self.assertEqual(fix("inside my document slash coding and then", "Inside my document/fifine/coding, and then"),
                         "Inside my document/coding, and then")

    def test_dictionary_words_not_similar(self):
        self.assertEqual(
            fix("somebody didn't have Freelook on, and they did a hashtag follow players and then the bot snapped",
                "Somebody didn't have Fifine on, and they did a hashtag follow Reyna players and then the bot snapped."),
            "Somebody didn't have Freelook on, and they did a hashtag follow players and then the bot snapped.")

    def test_profanity_swap(self):
        self.assertEqual(
            fix("she's a cashier and she serves orange chicken and shit and helps people in the drive-thru",
                "She's a cashier, she serves orange chicken, and she helps people in the drive-thru."),
            "She's a cashier, she serves orange chicken, and shit and helps people in the drive-thru.")

    def test_replacement_target_kept_unrelated_word_reverted(self):
        # production cleans the raw with dictionary replacements applied (quen -> qwen)
        self.assertEqual(fix("So even on the qwen model, this mistake was made.",
                             "So even on the quine model, this mistake was made."),
                         "So even on the qwen model, this mistake was made.")

    def test_inserted_words_removed(self):
        self.assertEqual(fix("I need what do I need I need the install height",
                             "I need to know what do I need, I need the install height."),
                         "I need what do I need, I need the install height.")
        # "to" took the place of a spoken "like": the spoken word comes back (it may be filler, but it was said)
        self.assertEqual(fix("just like make it look, you know, half decent", "just to make it look half decent"),
                         "just like make it look half decent")

    def test_filler_swapped_for_word(self):
        self.assertEqual(fix("Travel time is paid, you get like per mile and per hour.",
                             "Travel time is paid; you get paid per mile and per hour."),
                         "Travel time is paid; you get like per mile and per hour.")

    def test_number_change_never_allowed(self):
        self.assertEqual(fix("from five minutes to two minutes", "From 5 minutes to 3 minutes."),
                         "From 5 minutes to two minutes.")
        self.assertEqual(fix("remind me tonight at about like maybe 520 to ask Javier",
                             "Remind me tonight at about maybe 5 PM to ask Javier."),
                         "Remind me tonight at about maybe 520 to ask Javier.")

    def test_sentence_split_undone_on_revert(self):
        self.assertEqual(fix("you can put anything useful on there if you want me to look at it that would be helpful",
                             "You can put anything useful on there if you want. I can look at it; that would be helpful."),
                         "You can put anything useful on there if you want me to look at it; that would be helpful.")

    def test_spoken_comma_back_when_insertion_removed(self):
        self.assertEqual(fix("Is it deep sea, clawed code, what is it?", "Is it Deep Sea or Clawed Code? What is it?"),
                         "Is it Deep Sea, Clawed Code? What is it?")

    def test_swap_of_a_dropped_word_restores_it(self):
        # the aligner leaves out "like" / "i mean" / spoken symbol names; a swap of one must not delete it
        self.assertEqual(fix("i like pizza a lot", "I love pizza a lot."), "I like pizza a lot.")
        self.assertEqual(fix("i mean it this time", "I meant it this time."), "I mean it this time.")
        self.assertEqual(fix("the quote was too high", "The price was too high."), "The quote was too high.")

    def test_one_word_for_one_word_not_fuzzy(self):
        self.assertEqual(fix("ping the technicians about the configuration",
                             "Ping the technicals about the confirmation."),
                         "Ping the technicians about the configuration.")

    def test_revert_capitalized_after_digit_start(self):
        self.assertEqual(fix("three billion people use it", "3 billion humans use it."), "3 billion people use it.")
        self.assertEqual(fix("yaw is fine. ok", "Voice is fine. Ok"), "Yaw is fine. Ok")

    def test_bracketed_and_quoted_insertions(self):
        self.assertEqual(fix("meet me at the store", "Meet me at the store (the big one)."), "Meet me at the store.")
        self.assertEqual(fix("meet me at the store tonight", 'Meet me at the "grocery" store tonight.'),
                         "Meet me at the store tonight.")
        self.assertEqual(fix("we should go now", '"Honestly, we should go now."'), '"We should go now."')

    def test_neighbour_quotes_and_brackets_kept(self):
        self.assertEqual(fix("pick deep sea clawed code", 'Pick "Deep Sea" or "Clawed Code".'),
                         'Pick "Deep Sea" "Clawed Code".')
        self.assertEqual(fix("say yes no to him", "Say 'yes' and 'no' to him."), "Say 'yes' 'no' to him.")
        self.assertEqual(fix("use foo bar", "Use `foo` and `bar`."), "Use `foo` `bar`.")
        self.assertEqual(fix("use foo bar", "Use (foo) plus (bar)."), "Use (foo) (bar).")
        self.assertEqual(fix("go to the store", '"Really" go to the store.'), "Go to the store.")

    def test_restored_words_keep_their_joiners(self):
        self.assertEqual(fix("we got 3.5 stars overall", "We got four stars overall."), "We got 3.5 stars overall.")
        self.assertEqual(fix("open release.js and run it", "Open the release notes and run it."),
                         "Open release.js and run it.")

    def test_symbol_name_restored_when_its_symbol_was_not_written(self):
        self.assertEqual(fix("the quote was high", "The client's price was high."), "The quote was high.")
        self.assertEqual(fix("press the dash button twice", "Press the 'minus' button twice."),
                         "Press the 'dash' button twice.")
        self.assertEqual(fix("the period was long", "The well-known era was long."), "The period was long.")

    def test_raw_symbols_count_as_words(self):
        self.assertEqual(fix("it costs $5 a month", "It costs five dollars a month."), "It costs five dollars a month.")
        self.assertEqual(fix("50% of users left", "Fifty percent of users left."), "Fifty percent of users left.")

    def test_model_sentence_break_kept_for_unpunctuated_raw(self):
        self.assertEqual(fix("i went home technicians are bad", "I went home. Technicals are bad."),
                         "I went home. Technicians are bad.")

    def test_reverts_reported(self):
        _, rv = keepwords.repair("I meant unhelped technicians", "I meant helpers.", VOCAB, ABBR, {}, ())
        self.assertEqual([(r.out_words, r.raw_words) for r in rv], [("helpers", "unhelped technicians")])


class Kept(unittest.TestCase):
    """Legitimate cleanup must come through unchanged."""

    def same(self, raw, out, **kw):
        self.assertEqual(fix(raw, out, **kw), out, f"{raw!r} -> {out!r} wrongly repaired")

    def test_homophones(self):
        self.same("you're more Then welcome to express any creative freedom",
                  "You're more than welcome to express any creative freedom.")
        self.same("their going to the store later", "They're going to the store later.")
        self.same("its a good idea i think", "It's a good idea, I think.")
        self.same("i want to come to", "I want to come too.")
        self.same("your welcome to join us", "You're welcome to join us.")

    def test_agreement(self):
        self.same("when i spawn a subagent it will correctly show but then", "When I spawn a subagent, it correctly shows, but then")
        self.same("the pictures he sent in DMs was the pictures we were waiting for",
                  "The pictures he sent in DMs were the pictures we were waiting for.")
        self.same("so there's three files in the folder", "So there are three files in the folder.")
        self.same("the files is ready to go", "The files are ready to go.")
        self.same("she have a lot of work today", "She has a lot of work today.")
        self.same("i need a apple and a orange", "I need an apple and an orange.")

    def test_dictionary_spelling(self):
        self.same("I mean, their sky had to have gone like 23 and 9.", "i mean their skye had to have gone like 23 and 9")
        self.same("tell a j i said hi", "Tell AJ I said hi.")

    def test_contractions_and_casual(self):
        self.same("I don't know man I'm gonna go right now", "idk man im gonna go rn", word_map=CASUAL_WM)
        self.same("it's kind of weird that you do not see it", "its kinda weird that you dont see it", word_map=CASUAL_WM)
        self.same("I am going to be there", "I'm going to be there.")
        self.same("we are going to need more time", "We're gonna need more time.")

    def test_compounds(self):
        self.same("launch a few more sub agents to work on it", "Launch a few more subagents to work on it.")
        self.same("I'll check out the new build", "I'll checkout the new build.")

    def test_numbers(self):
        self.same("meet at four pm on the twenty first", "Meet at 4 PM on the 21st.")
        self.same("meet at four pm", "Meet at 4:00 PM.")
        self.same("meet at three thirty", "Meet at 3:30.")
        self.same("twenty twenty six was wild", "2026 was wild.")
        self.same("two thousand twenty six was wild", "2026 was wild.")
        self.same("it costs fifteen hundred dollars", "It costs $1,500.")
        self.same("if you had like 12,000 users on your website", "If you had like 12k users on your website.")
        self.same("it went up fifty percent", "It went up 50%.")
        self.same("it costs five million dollars", "It costs $5 million.")
        self.same("it raised two point five million", "It raised 2.5 million.")
        # formats the parser does not model are left alone instead of half-replaced inside the number
        self.same("it takes two and a half hours", "It takes 2.5 hours.")
        self.same("version one point oh", "Version 1.0")
        self.same("add one half cup", "Add 1/2 cup.")
        self.same("it was a dollar fifty", "It was $1.50.")
        self.same("meet me at half past four", "Meet me at 4:30.")
        self.same("my shopping list is number one milk number two eggs number three bread",
                  "My shopping list is:\n1. Milk\n2. Eggs\n3. Bread")

    def test_formats(self):
        self.same("email me at sarah dot li at example dot com", "Email me at sarah.li@example.com.")
        self.same("open main dot py and test underscore user dot py", "Open main.py and test_user.py.")
        self.same("hey john comma thanks for the update period new paragraph I'll review it tonight",
                  "Hey John, thanks for the update.\n\nI'll review it tonight.")
        self.same("that was a long period of time", "That was a long period of time.")

    def test_fillers(self):
        self.same("um so like I was thinking we should uh meet tomorrow", "I was thinking we should meet tomorrow.")
        self.same("I like pizza a lot you know", "I like pizza a lot.")

    def test_self_corrections(self):
        self.same("Let's get lunch at noon. Actually, no, make it one o'clock, at the usual place.",
                  "Let's get lunch at 1 o'clock at the usual place.")
        self.same("monday at 3 i mean tuesday", "Tuesday at 3")
        self.same("send it to john on friday actually no send it to jane", "Send it to Jane on Friday")
        self.same("um so like I was thinking we should uh meet at 3 actually no make it 4 tomorrow",
                  "I was thinking we should meet at 4 tomorrow.")
        self.same("email me at sarah dot lee at example dot com i mean sarah dot li at example dot com",
                  "Email me at sarah.li@example.com.")

    def test_stt_split_long_word(self):
        self.same("anything in the code that's bad, inconsistent, underperform it, or anything",
                  "Anything in the code that's bad, inconsistent, underperforming, or anything")

    def test_deletions_untouched(self):
        self.same("so basically the the build is broken right now", "The build is broken right now.")

    def test_reordered_spoken_word(self):
        # a counted spoken word may move
        self.same("call me tomorrow or today", "Call me today or tomorrow.")


class SettingsWords(unittest.TestCase):
    """Swaps the user's own settings ask for keep working (the "aj" -> "aiden morales" prank)."""

    def test_replacement_target(self):
        self.assertEqual(fix("yo aj are you on today", "Yo Aiden Morales, are you on today?", settings=["Aiden Morales"]),
                         "Yo Aiden Morales, are you on today?")

    def test_without_settings_reverted(self):
        self.assertEqual(fix("yo aj are you on today", "Yo Aiden Morales, are you on today?"),
                         "Yo aj, are you on today?")

    def test_cleaner_settings_text(self):
        cfg = Config(path=Path(tempfile.mkdtemp()) / "cfg.json")
        cfg.set("dictionary.replacements", {"aj": "Aiden Morales"}, save=False)
        cfg.set("snippets", {"my site": "https://example.org/jusjcb"}, save=False)
        cl = Cleaner(cfg)
        txt = cl._settings_text(styles.get_style(cfg, "neutral"))
        self.assertIn("Aiden Morales", txt)
        self.assertIn("https://example.org/jusjcb", txt)

    def test_custom_style_prompt_counts_builtin_does_not(self):
        cfg = Config(path=Path(tempfile.mkdtemp()) / "cfg.json")
        cl = Cleaner(cfg)
        self.assertEqual(cl._settings_text(styles.get_style(cfg, "casual")), [])     # built-in prompt text
        # the live config's casual prompt: the shipped text + one sentence the user appended
        cfg.set("styles.casual.prompt", styles.DEFAULT_STYLES["casual"]["prompt"] +
                " make sure to format coding correctly", save=False)
        sd = styles.get_style(cfg, "casual")
        txt = " ".join(cl._settings_text(sd)).lower()
        for w in ("code", "people", "confused", "problem", "report"):
            self.assertNotIn(w, txt.split())
        self.assertIn("unhelped", fix("the technicians were unhelped all day", "the people were confused all day",
                                      settings=cl._settings_text(sd), word_map=CASUAL_WM))
        cfg.set("styles.neutral.prompt", "Always write AJ as Aiden Morales.", save=False)
        sd = styles.get_style(cfg, "neutral")
        self.assertEqual(fix("yo aj are you on today", "Yo Aiden Morales, are you on today?",
                             settings=cl._settings_text(sd)), "Yo Aiden Morales, are you on today?")

    def test_appended_rule_words_count_even_if_in_a_shipped_prompt(self):
        cfg = Config(path=Path(tempfile.mkdtemp()) / "cfg.json")
        cl = Cleaner(cfg)
        cfg.set("styles.casual.prompt", styles.DEFAULT_STYLES["casual"]["prompt"] +
                ' Always say "hey" instead of "hello".', save=False)
        txt = " ".join(cl._settings_text(styles.get_style(cfg, "casual")))
        self.assertIn("hey", txt)
        self.assertNotIn("backticks", txt)
        self.assertEqual(fix("hello there buddy how are you", "hey there buddy how are you",
                             settings=cl._settings_text(styles.get_style(cfg, "casual")), word_map=CASUAL_WM),
                         "hey there buddy how are you")

    def test_custom_examples_count(self):
        cfg = Config(path=Path(tempfile.mkdtemp()) / "cfg.json")
        cfg.set("styles.neutral.examples", [["yo aj", "Yo Aiden Morales"]], save=False)
        cl = Cleaner(cfg)
        self.assertIn("Yo Aiden Morales", cl._settings_text(styles.get_style(cfg, "neutral")))

    def test_common_prompt_words_do_not_count(self):
        # "the"/"to" in a custom prompt must not make every inserted "to" acceptable
        self.assertEqual(fix("I need what do I need", "I need to know what do I need.",
                             settings=["Write the words to the user, keep it short."]),
                         "I need what do I need.")


class Robustness(unittest.TestCase):
    def test_never_raises(self):
        with mock.patch.object(keepwords, "_repair", side_effect=RuntimeError("boom")):
            self.assertEqual(keepwords.repair("a b c d", "A b c d.", [], {}, {}, ()), ("A b c d.", []))

    def test_empty_inputs(self):
        self.assertEqual(keepwords.repair("", "x", [], {}, {}, ()), ("x", []))
        self.assertEqual(keepwords.repair("x", "", [], {}, {}, ()), ("", []))
        self.assertEqual(keepwords.repair("hello there", "...", [], {}, {}, ()), ("...", []))

    def test_unicode_and_odd_text(self):
        for raw, out in [("café au lait please", "Café au lait, please."), ("´´ weird ’ quotes ’", "Weird quotes."),
                         ("one two three four five", "1, 2, 3, 4, 5."), ("a\nb\nc d", "A\nB\nC d.")]:
            keepwords.repair(raw, out, VOCAB, ABBR, CASUAL_WM, ["x"])      # must not raise


class CleanerIntegration(unittest.TestCase):
    def make(self, llm_out, keep=True):
        cfg = Config(path=Path(tempfile.mkdtemp()) / "cfg.json")
        cfg.set("cleanup.keep_words", keep, save=False)
        cl = Cleaner(cfg)
        cl.server.ready = True
        cl._llm = lambda *a, **k: llm_out
        return cl

    RAW = "Awesome your pitch and your yaw is all good so let me configure a few things"
    BAD = "Awesome, your pitch and your voice are all good, so let me configure a few things."
    GOOD = "Awesome, your pitch and your yaw is all good, so let me configure a few things."

    def test_neutral_repaired(self):
        out, info = self.make(self.BAD).clean(self.RAW, "neutral")
        self.assertEqual(out, self.GOOD)
        self.assertTrue(info["llm"])

    def test_streaming_unit_repaired(self):
        out, _ = self.make(self.BAD).clean(self.RAW, "neutral", context="Previous text.", final=False)
        self.assertEqual(out, self.GOOD)

    def test_casual_and_code_repaired(self):
        out, _ = self.make(self.BAD.lower()).clean(self.RAW, "casual")
        self.assertIn("yaw is", out)
        out, _ = self.make(self.BAD).clean(self.RAW, "code")
        self.assertIn("yaw is", out)

    def test_professional_and_email_not_repaired(self):
        for style in ("professional", "email"):
            out, _ = self.make(self.BAD).clean(self.RAW, style)
            self.assertIn("voice are", out, style)

    def test_setting_off(self):
        out, _ = self.make(self.BAD, keep=False).clean(self.RAW, "neutral")
        self.assertEqual(out, self.BAD)

    def test_repair_failing_guard_falls_back_to_regex(self):
        cl = self.make(self.BAD)
        with mock.patch.object(keepwords, "repair", return_value=("", [keepwords.Revert("x", "y", 0)])):
            out, info = cl.clean(self.RAW, "neutral")
        self.assertFalse(info["llm"])
        self.assertTrue(info["reason"].startswith("guard: keep_words"), info["reason"])
        self.assertIn("yaw is", out)                 # regex cleanup of the raw

    def test_exception_keeps_llm_output(self):
        cl = self.make(self.BAD)
        with mock.patch.object(keepwords, "repair", side_effect=RuntimeError("boom")):
            out, info = cl.clean(self.RAW, "neutral")
        self.assertEqual(out, self.BAD)
        self.assertTrue(info["llm"])

    def test_vocab_passed_through(self):
        out, _ = self.make("They did a hashtag follow Reyna players and then the bot snapped.").clean(
            "they did a hashtag follow players and then the bot snapped", "neutral", vocab=["Reyna"])
        self.assertEqual(out, "They did a hashtag follow players and then the bot snapped.")


class LocalFlowFamilyRouting(unittest.TestCase):
    R1E3 = "models/LFM3-1.7B-LocalFlow-r1e3-Q8_0.gguf"

    def make(self, model, **sets):
        cfg = Config(path=Path(tempfile.mkdtemp()) / "cfg.json")
        cfg.set("cleanup.model_path", model, save=False)
        for k, v in sets.items():
            cfg.set(f"cleanup.{k}", v, save=False)
        cl = Cleaner(cfg)
        cl.server.launched = cl.server.launch()[1]
        return cl

    def test_new_finetune_gets_compact_and_think_stops(self):
        cl = self.make(self.R1E3)
        self.assertEqual(cl._prompt_mode(), "compact")
        self.assertEqual(cl._request_overrides(), {"stop": ["<think>", "</think>"]})
        msgs = cl.cleanup_request("so um this is a test of the thing", styles.get_style(cl.cfg, "neutral"), [])[0]
        self.assertEqual(len(msgs["messages"]), 2)
        self.assertEqual(msgs["stop"], ["<think>", "</think>"])

    def test_exact_entry_wins(self):
        cl = self.make(self.R1E3, model_prompt={"LFM3-1.7B-LocalFlow-r1e3-Q8_0.gguf": "full"},
                       model_request={"lfm3-1.7b-localflow-r1e3-q8_0.gguf": {"top_k": 3}})
        self.assertEqual(cl._prompt_mode(), "full")
        self.assertEqual(cl._request_overrides(), {"top_k": 3})

    def test_other_models_unaffected(self):
        cl = self.make("models/Qwen3.5-4B-Q8_0.gguf")
        self.assertEqual(cl._prompt_mode(), "full")
        self.assertEqual(cl._request_overrides(), {})

    def test_fallback_not_shared_between_calls(self):
        cl = self.make(self.R1E3)
        cl._request_overrides()["stop"].append("X")
        self.assertEqual(cl._request_overrides(), {"stop": ["<think>", "</think>"]})

    def test_model_args_and_reasoning_untouched(self):
        cl = self.make(self.R1E3)
        self.assertEqual(cl.server.model_args(Path(self.R1E3)), [])
        self.assertEqual(cl.reasoning_budget(), 0)
        self.assertNotIn("--reasoning-budget", cl.server.launch()[0])


if __name__ == "__main__":
    unittest.main()
