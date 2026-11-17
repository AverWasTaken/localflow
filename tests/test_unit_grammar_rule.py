"""Offline tests for the light grammar rule (SYSTEM_PROMPT rule 3): prompt text, style opt-outs, config migration."""
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from localflow import styles  # noqa: E402
from localflow.cleanup import FEWSHOT, SYSTEM_PROMPT, build_messages  # noqa: E402
from localflow.config import CASUAL_PROMPT_V2, CODE_PROMPT_V1, Config, apply_migrations  # noqa: E402

RULE = ('Also fix these clear slips, and no other grammar: a verb that disagrees in number with its subject '
        '("the files is" -> "the files are"; "there\'s"/"here\'s" before a plural -> "there are"/"here are"), a '
        'doubled word ("the the"), and "a"/"an". Two sentences joined by a comma become two sentences.')
CASUAL_OPT_OUT = "Do not polish the grammar (no agreement fixes)"
CODE_PROSE_ONLY = "rule 3 grammar fixes apply to prose sentences only, never to commands, code or identifiers"


def tmp_cfg():
    return Config(path=Path(tempfile.mkdtemp()) / "config.json")


class PromptText(unittest.TestCase):
    def test_rule_in_rule_3(self):
        rule3 = next(l for l in SYSTEM_PROMPT.splitlines() if l.startswith("3. "))
        self.assertIn(RULE, rule3)

    def test_neutral_and_code_get_rule_in_both_modes(self):
        cfg = tmp_cfg()
        for style in ("neutral", "code"):
            sd = styles.get_style(cfg, style)
            for mode in ("full", "compact"):
                msgs = build_messages("there's two files", sd, [], mode=mode)
                self.assertIn(RULE, msgs[0]["content"], (style, mode))
                if style == "code":
                    self.assertIn(CODE_PROSE_ONLY, msgs[-1]["content"], mode)

    def test_casual_opts_out_in_both_modes(self):
        sd = styles.get_style(tmp_cfg(), "casual")
        for mode in ("full", "compact"):
            last = build_messages("there's two files", sd, [], mode=mode)[-1]["content"]
            self.assertIn(CASUAL_OPT_OUT, last, mode)

    def test_neutral_compact_has_no_opt_out(self):
        sd = styles.get_style(tmp_cfg(), "neutral")
        last = build_messages("there's two files", sd, [], mode="compact")[-1]["content"]
        self.assertNotIn("no agreement fixes", last)


class ExamplesAgreeWithRule(unittest.TestCase):
    """Built-in examples outside casual must not keep a slip the rule fixes."""

    def outputs(self):
        outs = [a for _, a in FEWSHOT]
        for name, ex in styles.DEFAULT_EXAMPLES.items():
            if name != "casual":
                outs += [a for _, a in ex]
        return outs

    def test_no_doubled_words(self):
        for a in self.outputs():
            self.assertIsNone(re.search(r"\b(\w+) \1\b", a, re.I), a)

    def test_no_theres_before_plural(self):
        for a in self.outputs():
            self.assertIsNone(re.search(r"\b(?:there|here|where)'s (?:any |some |two |three |many |a lot of )?\w+s\b",
                                        a, re.I), a)


class GrammarMigration(unittest.TestCase):
    def cfg_path(self, casual=None, code=None, migrations=None):
        p = Path(tempfile.mkdtemp()) / "cfg.json"
        st = {}
        if casual is not None:
            st["casual"] = dict(styles.DEFAULT_STYLES["casual"], prompt=casual)
        if code is not None:
            st["code"] = dict(styles.DEFAULT_STYLES["code"], prompt=code)
        mig = migrations if migrations is not None else ["email_style_v1", "casual_format_v1", "casual_format_v2",
                                                         "model_defaults_v1"]
        p.write_text(json.dumps({"styles": st, "migrations": mig}), encoding="utf-8")
        return p

    def test_old_defaults_upgraded_and_saved(self):
        p = self.cfg_path(CASUAL_PROMPT_V2, CODE_PROMPT_V1)
        cfg = Config(path=p)
        self.assertEqual(cfg.get("styles")["casual"]["prompt"], styles.DEFAULT_STYLES["casual"]["prompt"])
        self.assertEqual(cfg.get("styles")["code"]["prompt"], styles.DEFAULT_STYLES["code"]["prompt"])
        on_disk = json.loads(p.read_text(encoding="utf-8"))
        self.assertIn(CASUAL_OPT_OUT, on_disk["styles"]["casual"]["prompt"])
        self.assertIn(CODE_PROSE_ONLY, on_disk["styles"]["code"]["prompt"])
        self.assertIn("grammar_v1", on_disk["migrations"])

    def test_casual_with_user_suffix_keeps_suffix(self):
        p = self.cfg_path(CASUAL_PROMPT_V2 + " make sure to format coding correctly")
        cfg = Config(path=p)
        self.assertEqual(cfg.get("styles")["casual"]["prompt"],
                         styles.DEFAULT_STYLES["casual"]["prompt"] + " make sure to format coding correctly")

    def test_other_edits_untouched(self):
        casual = CASUAL_PROMPT_V2.replace("All lowercase", "Mostly lowercase")
        code = CODE_PROMPT_V1 + " Always use snake_case."
        cfg = Config(path=self.cfg_path(casual, code))
        self.assertEqual(cfg.get("styles")["casual"]["prompt"], casual)
        self.assertEqual(cfg.get("styles")["code"]["prompt"], code)
        self.assertIn("grammar_v1", cfg.get("migrations"))

    def test_casual_suffix_without_space_untouched(self):
        casual = CASUAL_PROMPT_V2 + "!"
        cfg = Config(path=self.cfg_path(casual))
        self.assertEqual(cfg.get("styles")["casual"]["prompt"], casual)

    def test_runs_once(self):
        p = self.cfg_path(code=CODE_PROMPT_V1)
        Config(path=p)
        d = json.loads(p.read_text(encoding="utf-8"))
        d["styles"]["code"]["prompt"] = CODE_PROMPT_V1      # user deliberately puts the old prompt back
        p.write_text(json.dumps(d), encoding="utf-8")
        cfg = Config(path=p)
        self.assertEqual(cfg.get("styles")["code"]["prompt"], CODE_PROMPT_V1)
        self.assertEqual(cfg.get("migrations").count("grammar_v1"), 1)

    def test_bad_shapes_are_safe(self):
        for data in ({}, {"styles": []}, {"styles": {"casual": "x", "code": {"prompt": 3}}}):
            d = dict(data, migrations=[])
            apply_migrations(d)
            self.assertIn("grammar_v1", d["migrations"])

    def test_current_defaults_unchanged(self):
        d = {"styles": {"casual": dict(styles.DEFAULT_STYLES["casual"]), "code": dict(styles.DEFAULT_STYLES["code"])},
             "migrations": ["email_style_v1", "casual_format_v1", "casual_format_v2", "model_defaults_v1"]}
        snap = json.dumps(d["styles"], sort_keys=True)
        apply_migrations(d)
        self.assertEqual(json.dumps(d["styles"], sort_keys=True), snap)


if __name__ == "__main__":
    unittest.main()
