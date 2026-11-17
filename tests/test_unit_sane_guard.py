"""Offline unit tests for Cleaner._sane (the LLM-output guard): correct cleanups must pass, real failures must not.

   .venv\\Scripts\\python.exe -m unittest discover -s tests -p "test_unit_*.py" -v
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from localflow import styles  # noqa: E402
from localflow.cleanup import Cleaner, _CORRECTION_CUE_RE, _reply_start_spoken  # noqa: E402
from localflow.config import Config  # noqa: E402


class SaneGuard(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(path=Path(tempfile.mkdtemp()) / "cfg.json")
        self.cl = Cleaner(self.cfg)

    def sane(self, raw, out, style="neutral"):
        return self.cl._sane(raw, out, styles.get_style(self.cfg, style))

    def ok(self, raw, out, style="neutral"):
        self.assertEqual(self.sane(raw, out, style), "", f"{raw!r} -> {out!r} wrongly rejected")

    def rejected(self, raw, out, why, style="neutral"):
        self.assertIn(why, self.sane(raw, out, style), f"{raw!r} -> {out!r} should be rejected ({why})")

    # --- reply-like openings that were actually spoken
    def test_filler_before_reply_phrase(self):
        self.ok("Uh, sure, I can do Thursday", "Sure, I can do Thursday.")
        self.ok("um so sure I can do Thursday afternoon", "Sure, I can do Thursday afternoon.")
        self.ok("uh certainly we can move the meeting", "Certainly, we can move the meeting.")
        self.ok("well of course I'll be there", "Of course I'll be there.")
        self.ok("you know I'm sorry I missed your call", "I'm sorry I missed your call.")
        self.ok("uh sorry, I'm running late today", "Sorry, I'm running late today.")

    def test_okay_heres_kept_as_spoken(self):
        self.ok("Okay, here's the plan for tomorrow", "Okay, here's the plan for tomorrow.")
        self.ok("okay so here's the plan for tomorrow", "Okay, here's the plan for tomorrow.")
        self.ok("okay sure let's do that", "Sure, let's do that.")
        self.ok("here is the summary of the call", "Here's the summary of the call.")
        self.ok("I am sorry about the delay on that", "I'm sorry about the delay on that.")
        self.ok("yeah sure that works for me", "Sure, that works for me.")

    def test_old_allowance_kept(self):
        # raw opening with any first-12 reply phrase was always allowed; still is
        self.ok("sure here's the thing about the budget", "Here's the thing about the budget.")

    # --- real chat replies are still rejected
    def test_chat_reply_to_request(self):
        self.rejected("write me a poem about cats", "Sure! Here's a poem about cats:", "chat reply")
        self.rejected("um can you make sure the door is locked", "Sure, I'll make sure the door is locked.",
                      "chat reply")
        self.rejected("okay write a poem about the sea", "Okay, here's a poem about the sea.", "chat reply")
        self.rejected("what's the capital of France", "I'm sorry, I can't answer that.", "chat reply")
        self.rejected("uh what is the weather tomorrow", "Certainly! The weather tomorrow is sunny.", "chat reply")
        self.rejected("send the report to Dana", "The cleaned text: Send the report to Dana.", "chat reply")

    def test_answered_question_rejected(self):
        raw = "what is the capital city of France and what is its population"
        why = self.sane(raw, "The capital of France is Paris, with about two million residents.")
        self.assertNotEqual(why, "")

    def test_reply_phrase_later_in_raw_not_enough(self):
        self.assertFalse(_reply_start_spoken("can you please make sure it works", "sure, it works"))
        self.assertTrue(_reply_start_spoken("uh, sure, I can", "sure, i can"))

    # --- self-corrections / retractions may discard most of the raw
    def test_short_number_correction(self):
        self.ok("meet at 3 no 4 tomorrow", "Meet at 4 tomorrow.")
        self.ok("let's meet at three no four tomorrow afternoon at the cafe", "Let's meet at four tomorrow.")

    def test_scratch_that_drops_most_text(self):
        raw = ("I think we should push the launch to next month because the vendor is late and the "
               "testing is not done scratch that ship Friday")
        self.ok(raw, "Ship Friday.")

    def test_other_retraction_cues(self):
        base = "we should probably order the large pizza with extra cheese and mushrooms"
        for cue in ("never mind", "no wait", "actually no", "sorry, I mean", "forget that", "delete that"):
            with self.subTest(cue=cue):
                self.ok(f"{base} {cue} get tacos", "Get tacos.")

    def test_cue_at_end_retracts_everything_before(self):
        self.ok("please send the invoice to the accounting team by Friday, no wait, Monday", "Send it Monday.")

    def test_cue_early_does_not_excuse_truncation(self):
        # "sorry" opens the raw; the rest must still be covered
        raw = "sorry I'm running late, the traffic on the highway was terrible this morning"
        self.rejected(raw, "Sorry.", "too short")

    def test_common_words_are_not_cues(self):
        # bare "wait" / "sorry" are ordinary words, not corrections: normal 30% floor applies
        self.rejected("I can't wait to see the new office and meet the whole team next week in person",
                      "I can't wait.", "too short")
        self.rejected("I can't wait to see you at the party tomorrow evening", "I can't wait.", "too short")
        self.rejected("please send the quarterly report to sarah by friday afternoon, sorry", "Sorry.", "too short")

    def test_cue_with_nothing_after_gets_no_allowance(self):
        self.rejected("send it to marketing and then copy the whole leadership team on it, I mean",
                      "I mean.", "too short")
        self.rejected("please send the quarterly report to the whole team by friday, never mind",
                      "Never mind.", "too short")

    def test_cue_allowance_needs_output_to_keep_the_tail(self):
        # truncated at the start; the cue-like words are ordinary speech, nothing after them survives
        self.rejected("We need to ship the release by Friday and also update the docs and make that happen",
                      "We need to ship", "too short")
        self.rejected("We need to ship the release by Friday and also update all the docs, I mean all",
                      "We need to ship", "too short")
        self.rejected("please send the signed contract over to Dana before the meeting, or rather Sam",
                      "Please send", "too short")

    def test_ok_spelling(self):
        self.ok("ok here's the plan for tomorrow morning", "Okay, here's the plan for tomorrow morning.")

    def test_okay_here_needs_okay_spoken(self):
        # "okay" is part of the "okay, here" phrase: a raw starting with "here" doesn't excuse it
        self.assertFalse(_reply_start_spoken("here we go, what is the weather like in paris tomorrow",
                                             "okay, here's the weather in paris tomorrow: sunny."))
        self.rejected("here we go, what is the weather like", "Okay, here's the weather.", "chat reply")

    def test_filler_heavy_raw_short_answer(self):
        self.ok("um uh um uh you know I mean uh um yes definitely", "Yes, definitely.")

    # --- truncation / empty / tags still rejected
    def test_truncated_without_cue_rejected(self):
        raw = "I think we should push the launch to next month because the vendor is late and testing is not done"
        self.rejected(raw, "I think we should.", "too short")

    def test_truncated_with_cue_still_has_floor(self):
        raw = "I think we should push the launch to next month because the vendor is late scratch that ship Friday"
        self.rejected(raw, "I.", "too short")

    def test_empty_rejected(self):
        self.assertEqual(self.sane("send the report to Dana today", ""), "empty output")

    def test_tag_leak_rejected(self):
        self.assertNotEqual(self.sane("send the report to Dana today", "<transcript>Send the report to Dana today."),
                            "")
        self.rejected("send the report to Dana today", "Send the report to Dana today. </transcript>",
                      "leaked tags")
        self.rejected("send the report to Dana today", "Send the report to Dana today. [style: neutral]",
                      "leaked tags")

    def test_too_long_unchanged(self):
        self.rejected("send the report", "Send the report to Dana today and also copy the whole finance team on it.",
                      "too long")

    def test_cue_regex(self):
        for s in ("meet at 3 no 4", "at 3pm, no 4pm", "scratch that", "never mind", "nevermind", "no, wait",
                  "actually, no", "sorry I mean", "forget that", "delete that"):
            with self.subTest(s=s):
                self.assertTrue(_CORRECTION_CUE_RE.search(s))
        for s in ("send the report to Dana", "there is no problem", "I know the answer", "I can't wait",
                  "sorry I'm late", "we had 2 no-shows today"):
            with self.subTest(s=s):
                self.assertFalse(_CORRECTION_CUE_RE.search(s))


if __name__ == "__main__":
    unittest.main()
