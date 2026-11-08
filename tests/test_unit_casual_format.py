"""Offline tests for casual formatting: code-span protection, prompt migration, streamed paragraphs."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from localflow import styles  # noqa: E402
from localflow.config import CASUAL_PROMPT_V1, OLD_CASUAL_PROMPT, Config, apply_migrations  # noqa: E402
from localflow.streaming import join_pieces  # noqa: E402

CASUAL = styles.DEFAULT_STYLES["casual"]
NEW_PROMPT = CASUAL["prompt"]


def casual(text, preserve=None):
    return styles.apply_style_def(text, CASUAL, preserve=preserve)


class CodeSpans(unittest.TestCase):
    def test_apostrophe_inside_backticks_kept(self):
        self.assertEqual(casual("I don't think `don't` works"), "i dont think `don't` works")

    def test_case_inside_backticks_kept(self):
        self.assertEqual(casual("Open the `README.md` file"), "open the `README.md` file")

    def test_word_map_skips_code(self):
        self.assertEqual(casual("u should run `ls -u`"), "you should run `ls -u`")

    def test_abbreviation_skips_code(self):
        self.assertEqual(casual("never mind, run `echo never mind`"), "nvm, run `echo never mind`")

    def test_lone_backtick_is_normal_text(self):
        self.assertEqual(casual("A lone ` Backtick U"), "a lone ` backtick you")

    def test_fenced_block_kept(self):
        self.assertEqual(casual("Look:\n```\nRun It\n```"), "look:\n```\nRun It\n```")

    def test_quotes_are_not_protected(self):
        self.assertEqual(casual('I Said "Don\'t Do It"'), 'i said "dont do it"')

    def test_preserve_vocab_outside_code(self):
        self.assertEqual(casual("Ask Alex about `Alex.py`", preserve=["Alex"]), "ask Alex about `Alex.py`")

    def test_final_period_after_code(self):
        self.assertEqual(casual("Run `npm install`."), "run `npm install`")

    def test_idempotent(self):
        once = casual("I don't know, open `Config.JSON` u know")
        self.assertEqual(casual(once), once)


class PromptMigration(unittest.TestCase):
    def cfg_path(self, prompt, migrations=None):
        p = Path(tempfile.mkdtemp()) / "cfg.json"
        casual_def = dict(CASUAL)
        casual_def["prompt"] = prompt
        data = {"styles": {"casual": casual_def}, "migrations": migrations or ["email_style_v1"]}
        p.write_text(json.dumps(data), encoding="utf-8")
        return p

    def test_new_prompt_has_format_rules(self):
        self.assertNotEqual(NEW_PROMPT, OLD_CASUAL_PROMPT)
        self.assertTrue(NEW_PROMPT.startswith(OLD_CASUAL_PROMPT[:-1]))
        self.assertIn("backticks", NEW_PROMPT)

    def test_old_prompt_upgraded_and_saved(self):
        p = self.cfg_path(OLD_CASUAL_PROMPT)
        cfg = Config(path=p)
        self.assertEqual(cfg.get("styles")["casual"]["prompt"], NEW_PROMPT)
        on_disk = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(on_disk["styles"]["casual"]["prompt"], NEW_PROMPT)
        self.assertIn("casual_format_v1", on_disk["migrations"])
        self.assertTrue(cfg.get("styles")["casual"].get("paragraphs"))   # new default key merges in

    def test_custom_prompt_untouched(self):
        p = self.cfg_path("my own casual rules")
        cfg = Config(path=p)
        self.assertEqual(cfg.get("styles")["casual"]["prompt"], "my own casual rules")
        self.assertIn("casual_format_v1", cfg.get("migrations"))

    def test_runs_once(self):
        p = self.cfg_path(OLD_CASUAL_PROMPT)
        Config(path=p)
        d = json.loads(p.read_text(encoding="utf-8"))
        d["styles"]["casual"]["prompt"] = OLD_CASUAL_PROMPT      # user deliberately puts the old prompt back
        p.write_text(json.dumps(d), encoding="utf-8")
        cfg = Config(path=p)
        self.assertEqual(cfg.get("styles")["casual"]["prompt"], OLD_CASUAL_PROMPT)
        self.assertEqual(cfg.get("migrations").count("casual_format_v1"), 1)

    def test_missing_styles_is_safe(self):
        d = {"migrations": ["email_style_v1"]}
        self.assertTrue(apply_migrations(d))
        self.assertIn("casual_format_v1", d["migrations"])
        self.assertIn("casual_format_v2", d["migrations"])


class PromptMigrationV2(unittest.TestCase):
    """casual_format_v2: the v1 prompt (quotes/backticks) gets the spoken-code + self-correction rules."""

    def cfg_path(self, prompt, migrations):
        p = Path(tempfile.mkdtemp()) / "cfg.json"
        casual_def = dict(CASUAL)
        casual_def["prompt"] = prompt
        p.write_text(json.dumps({"styles": {"casual": casual_def}, "migrations": migrations}), encoding="utf-8")
        return p

    def test_new_prompt_has_code_and_correction_rules(self):
        self.assertNotEqual(NEW_PROMPT, CASUAL_PROMPT_V1)
        self.assertIn("reads out loud", NEW_PROMPT)
        self.assertIn("Self-corrections still apply", NEW_PROMPT)

    def test_v1_prompt_upgraded_and_saved(self):
        p = self.cfg_path(CASUAL_PROMPT_V1, ["email_style_v1", "casual_format_v1"])
        cfg = Config(path=p)
        self.assertEqual(cfg.get("styles")["casual"]["prompt"], NEW_PROMPT)
        on_disk = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(on_disk["styles"]["casual"]["prompt"], NEW_PROMPT)
        self.assertEqual(on_disk["migrations"], ["email_style_v1", "casual_format_v1", "casual_format_v2"])

    def test_old_prompt_chains_to_new(self):
        p = self.cfg_path(OLD_CASUAL_PROMPT, ["email_style_v1"])
        cfg = Config(path=p)
        self.assertEqual(cfg.get("styles")["casual"]["prompt"], NEW_PROMPT)
        self.assertEqual(cfg.get("migrations"), ["email_style_v1", "casual_format_v1", "casual_format_v2"])

    def test_edited_prompt_untouched(self):
        edited = CASUAL_PROMPT_V1 + " Also never use the word lol."
        p = self.cfg_path(edited, ["email_style_v1", "casual_format_v1"])
        cfg = Config(path=p)
        self.assertEqual(cfg.get("styles")["casual"]["prompt"], edited)
        self.assertIn("casual_format_v2", cfg.get("migrations"))

    def test_runs_once(self):
        p = self.cfg_path(CASUAL_PROMPT_V1, ["email_style_v1", "casual_format_v1"])
        Config(path=p)
        d = json.loads(p.read_text(encoding="utf-8"))
        d["styles"]["casual"]["prompt"] = CASUAL_PROMPT_V1      # user deliberately puts the v1 prompt back
        p.write_text(json.dumps(d), encoding="utf-8")
        cfg = Config(path=p)
        self.assertEqual(cfg.get("styles")["casual"]["prompt"], CASUAL_PROMPT_V1)
        self.assertEqual(cfg.get("migrations").count("casual_format_v2"), 1)


class CasualExamples(unittest.TestCase):
    def test_spoken_code_and_correction_examples(self):
        ex = dict((u, a) for u, a in styles.DEFAULT_EXAMPLES["casual"])
        self.assertEqual(ex["Like if you type print hi there it just shows hi there on the screen"],
                         'like if you type `print("hi there")` it just shows `hi there` on the screen')
        self.assertEqual(ex["were meeting at the park friday, or wait no sorry saturday at noon"],
                         "were meeting at the park saturday at noon")
        q = "can you print the report for me before the meeting"
        self.assertEqual(ex[q], q)

    def test_code_example_survives_post_pass(self):
        out = 'like if you type `print("hi there")` it just shows `hi there` on the screen'
        self.assertEqual(casual(out), out)


class StreamedParagraphs(unittest.TestCase):
    # casual-cleaned pieces have no periods; the raw STT text does
    RAW = ["So the first thing is setup. You install it. Then you run it once. It takes a minute or two to finish.",
           "Second thing, the bots. They run on an old laptop. Linux keeps them online.",
           "And that's basically it."]
    CLEAN = ["so the first thing is setup you install it then you run it once it takes a minute or two to finish "
             "and then you are pretty much done with that part of it honestly",
             "second thing the bots they run on an old laptop linux keeps them online",
             "and thats basically it"]

    def test_long_pause_breaks_with_raws(self):
        out = join_pieces(self.CLEAN, [0, 1500, 300], 900, raws=self.RAW)
        self.assertEqual(out.count("\n\n"), 1)
        self.assertTrue(out.split("\n\n")[1].startswith("second thing"))

    def test_without_raws_casual_never_breaks(self):
        self.assertNotIn("\n", join_pieces(self.CLEAN, [0, 1500, 1500], 900))

    def test_no_pauses_never_breaks(self):
        self.assertNotIn("\n", join_pieces(self.CLEAN, None, 900, raws=self.RAW))

    def test_short_pause_no_break(self):
        self.assertNotIn("\n", join_pieces(self.CLEAN, [0, 400, 400], 900, raws=self.RAW))

    def test_too_few_words_no_break(self):
        out = join_pieces(["one. two. three.", "four"], [0, 2000], 900, raws=["One. Two. Three.", "Four."])
        self.assertNotIn("\n", out)

    def test_inpiece_break_restarts_count(self):
        clean = ["intro bit\n\n" + self.CLEAN[0].split(" and then")[0], self.CLEAN[1]]
        out = join_pieces(clean, [0, 1500], 900, raws=self.RAW[:2])
        self.assertEqual(out.count("\n\n"), 1)        # only the LLM's own break; the counter restarted after it


if __name__ == "__main__":
    unittest.main()
