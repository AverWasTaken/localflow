"""Offline tests for the `email` style: layout post-pass, rule migration, whole-text cleanup."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from localflow import styles, textproc  # noqa: E402
from localflow.config import Config, DEFAULTS, apply_migrations  # noqa: E402
from localflow.streaming import StreamSession  # noqa: E402

SD = styles.DEFAULT_STYLES["email"]
lay = styles.layout_email


class Layout(unittest.TestCase):
    def test_greeting_body_closing(self):
        t = "Hi Sarah, I looked at the invoice. It is overdue by a week. Thanks, talk soon."
        self.assertEqual(lay(t), "Hi Sarah,\n\nI looked at the invoice. It is overdue by a week.\n\nThanks, talk soon.")

    def test_idempotent(self):
        for t in ["Hi Sarah, I looked at the invoice. It is overdue by a week. Thanks, talk soon.",
                  "Hi John, sounds good, see you then.", "Sounds good, see you then.",
                  "Hello,\n\nThe offer is great. I must decline.\n\nThank you anyway, and have a good day.",
                  "Hi Sarah,\n\nThe report is done. I sent it.\n\nThanks again. Have a good day.",
                  "Dear Dr. Lee, the results are in. They look fine. Best,"]:
            once = lay(t)
            self.assertEqual(lay(once), once)

    def test_short_replies_stay_one_line(self):
        for t in ["Hi John, sounds good, see you then.", "Sounds good, see you then.", "Thanks for the update.",
                  "Hey, thanks."]:
            self.assertEqual(lay(t), t)

    def test_no_greeting_body_only(self):
        t = "The draft needs more detail. The table does not match the summary."
        self.assertEqual(lay(t), t)

    def test_already_laid_out_untouched(self):
        t = "Hi Tom,\n\nThe plan works for us. We need a week.\n\nBest,"
        self.assertEqual(lay(t), t)

    def test_closing_needs_two_sentences_before(self):
        t = "I will send it today. Thanks."
        self.assertEqual(lay(t), t)

    def test_best_case_is_not_a_closing(self):
        t = "We checked the plan. It is fine. Best case, we ship on Friday."
        self.assertEqual(lay(t), t)

    def test_long_last_sentence_not_a_closing(self):
        t = ("The plan is fine. We checked it. Thanks to the whole team for the effort over the last quarter "
             "we are ahead of schedule on every milestone.")
        self.assertEqual(lay(t), t)

    def test_two_sentence_closing(self):
        t = "The plan is fine. We checked it twice. Thanks for your time. Have a great day."
        self.assertEqual(lay(t), "The plan is fine. We checked it twice.\n\nThanks for your time. Have a great day.")

    def test_false_positive_closings(self):
        for last in ("Take care of the invoice by Friday.", "Have a good look at the attached file.",
                     "Thanks to Bob for the quick fix."):
            t = "The plan is fine. We checked it. " + last
            self.assertEqual(lay(t), t)

    def test_best_with_punctuation_and_titles(self):
        self.assertEqual(lay("The plan is fine. We checked it. Best,"), "The plan is fine. We checked it.\n\nBest,")
        self.assertEqual(lay("Dear Dr. Lee, the results are in. They look fine."),
                         "Dear Dr. Lee,\n\nThe results are in. They look fine.")
        self.assertTrue(lay("Hi Tom, " + "word " * 16).startswith("Hi Tom,\n\nWord"))
        self.assertIn("\n\niPhone", lay("Hi Tom, iPhone " + "word " * 16))

    def test_glued_closing_after_comma(self):
        t = ("Hey there, I appreciate the offer for the job. I think I have to decline because it does not fit my "
             "current timeline but thank you anyways, have a good day.")
        self.assertEqual(lay(t), "Hey there,\n\nI appreciate the offer for the job. I think I have to decline because "
                                 "it does not fit my current timeline but thank you anyways.\n\nHave a good day.")
        self.assertEqual(lay(lay(t)), lay(t))
        short = "It works, have a good day."
        self.assertEqual(lay(short), short)

    def test_glued_closing_false_positives(self):
        body = "We reviewed the whole plan and the budget, and the team agreed to meet again tomorrow morning. "
        for last in ("Also, take care of the invoices.", "If it rains, have a great time indoors.",
                     "I need them, have a good look at them first.", "We had a party, cheers everyone said.",
                     "Bring the slides, take care of the printouts."):
            t = body + last
            self.assertEqual(lay(t), t)

    def test_glued_closing_not_in_bullets(self):
        t = "- call Bob about the contract and the schedule\n- fix the sink and the door and then walk the dog today, have a good day"
        self.assertEqual(lay(t), t)

    def test_greeting_longer_body_split_by_words(self):
        t = "Hello there, " + "word " * 16 + "end"
        self.assertTrue(lay(t).startswith("Hello there,\n\nWord "))

    def test_apply_style_def_and_regex_fallback(self):
        out = textproc.regex_clean("hi sarah, I looked at the invoice. it is overdue. thanks", SD)
        self.assertEqual(out, "Hi sarah,\n\nI looked at the invoice. It is overdue.\n\nThanks.")

    def test_other_styles_unchanged(self):
        t = "Hi Sarah, I looked at the invoice. It is overdue by a week. Thanks, talk soon."
        for name in ("neutral", "professional", "casual", "code"):
            self.assertNotIn("\n", styles.apply_style_def(t, styles.DEFAULT_STYLES[name]))


class Migration(unittest.TestCase):
    def cfg_path(self, profiles=None, migrations=None):
        p = Path(tempfile.mkdtemp()) / "cfg.json"
        data = {"profiles": profiles if profiles is not None else [dict(r) for r in styles.DEFAULT_PROFILES]}
        if migrations is not None:
            data["migrations"] = migrations
        p.write_text(json.dumps(data), encoding="utf-8")
        return p

    @staticmethod
    def old_profiles():
        out = []
        for r in styles.DEFAULT_PROFILES:
            r = dict(r)
            if r["name"] in ("Gmail", "Outlook", "Outlook / Office web", "Proton / Yahoo mail"):
                r["style"] = "professional"
            out.append(r)
        return out

    def test_defaults_use_email(self):
        by = {r["name"]: r["style"] for r in styles.DEFAULT_PROFILES}
        for n in ("Gmail", "Outlook", "Outlook / Office web", "Proton / Yahoo mail"):
            self.assertEqual(by[n], "email")
        self.assertEqual(by["Word"], "professional")

    def test_migrates_old_rules_and_saves(self):
        p = self.cfg_path(self.old_profiles())
        cfg = Config(path=p)
        by = {r["name"]: r["style"] for r in cfg.get("profiles")}
        self.assertEqual([by[n] for n in ("Gmail", "Outlook", "Outlook / Office web", "Proton / Yahoo mail")],
                         ["email"] * 4)
        self.assertEqual(by["Word"], "professional")
        on_disk = json.loads(p.read_text(encoding="utf-8"))
        self.assertIn("email_style_v1", on_disk["migrations"])
        self.assertEqual({r["name"]: r["style"] for r in on_disk["profiles"]}["Gmail"], "email")
        self.assertIn("email", cfg.get("styles"))

    def test_runs_once_user_choice_sticks(self):
        p = self.cfg_path(self.old_profiles())
        Config(path=p)
        d = json.loads(p.read_text(encoding="utf-8"))
        for r in d["profiles"]:
            if r["name"] == "Gmail":
                r["style"] = "professional"      # user deliberately switches back
        p.write_text(json.dumps(d), encoding="utf-8")
        cfg = Config(path=p)
        self.assertEqual({r["name"]: r["style"] for r in cfg.get("profiles")}["Gmail"], "professional")
        self.assertEqual(cfg.get("migrations").count("email_style_v1"), 1)

    def test_user_edited_rules_left_alone(self):
        profs = self.old_profiles()
        for r in profs:
            if r["name"] == "Gmail":
                r["style"] = "casual"
            if r["name"] == "Outlook":
                r["style"] = "neutral"
        cfg = Config(path=self.cfg_path(profs))
        by = {r["name"]: r["style"] for r in cfg.get("profiles")}
        self.assertEqual(by["Gmail"], "casual")
        self.assertEqual(by["Outlook"], "neutral")
        self.assertEqual(by["Proton / Yahoo mail"], "email")

    def test_apply_migrations_idempotent(self):
        d = {"profiles": self.old_profiles()}
        self.assertTrue(apply_migrations(d))
        snap = json.dumps(d, sort_keys=True)
        self.assertFalse(apply_migrations(d))
        self.assertEqual(json.dumps(d, sort_keys=True), snap)

    def test_resolve_gmail_picks_email(self):
        cfg = Config(path=self.cfg_path(self.old_profiles()))
        ctx = styles.resolve("zen.exe", "Inbox (3) - me@gmail.com - Gmail", cfg)
        self.assertEqual(ctx.style, "email")

    def test_fresh_install_defaults(self):
        cfg = Config(path=Path(tempfile.mkdtemp()) / "cfg.json")
        self.assertIn("email_style_v1", cfg.get("migrations"))
        self.assertEqual(styles.resolve("zen.exe", "Inbox - Gmail", cfg).style, "email")


class WholeCleanup(unittest.TestCase):
    def test_email_style_disables_incremental(self):
        cfg = Config(path=Path(tempfile.mkdtemp()) / "cfg.json")
        for st, expect in (("email", False), ("professional", True)):
            s = StreamSession(cfg, object(), object(), styles.StyleContext(style=st), 16000)
            self.assertEqual(s.incremental, expect, st)

    def test_global_flag_still_wins_off(self):
        cfg = Config(path=Path(tempfile.mkdtemp()) / "cfg.json")
        cfg.set("streaming.incremental_cleanup", False, save=False)
        s = StreamSession(cfg, object(), object(), styles.StyleContext(style="professional"), 16000)
        self.assertFalse(s.incremental)


if __name__ == "__main__":
    unittest.main()
