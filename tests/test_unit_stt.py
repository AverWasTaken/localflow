"""Whisper decoding and repetition filtering without loading a model."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from localflow.stt import Transcriber


class RepeatCollapseTests(unittest.TestCase):
    def test_reported_loop_with_truncated_ending(self):
        intro = "So I think for the box or like the like factory machines ... "
        phrase = "So I think like in the build"
        text = intro + (phrase + " ") * 25 + "So"
        self.assertEqual(Transcriber._collapse_repeat(text), intro + phrase)

    def test_preserves_continuation_starting_with_the_same_word(self):
        intro = "I was looking at the machines. "
        phrase = "So I think like in the build"
        continuation = " So we should have some help"
        text = intro + " ".join([phrase] * 5) + continuation
        self.assertEqual(Transcriber._collapse_repeat(text), intro + phrase + continuation)

    def test_whole_transcript_repeats(self):
        for count in (2, 3, 4, 6):
            with self.subTest(count=count):
                text = "Check the build. " + "CHECK the build! " * (count - 1)
                self.assertEqual(Transcriber._collapse_repeat(text), "Check the build.")
                text = " ".join(["check the build"] * count)
                self.assertEqual(Transcriber._collapse_repeat(text), "check the build")
        long_phrase = " ".join(f"word{i}" for i in range(30))
        self.assertEqual(Transcriber._collapse_repeat(" ".join([long_phrase] * 5)), long_phrase)
        self.assertEqual(Transcriber._collapse_repeat("go back " * 6), "go back")

    def test_word_loop_collapses_in_place(self):
        self.assertEqual(Transcriber._collapse_repeat("the the the the the"), "the")
        text = "Look at the, THE the the factory"
        self.assertEqual(Transcriber._collapse_repeat(text), "Look at the, factory")

    def test_preserves_natural_repetition(self):
        for text in ("that that", "very very", "no no no", "wait wait wait", "I think I think",
                     "on and on and on", "over and over and over", "It went on and on and on again."):
            with self.subTest(text=text):
                self.assertEqual(Transcriber._collapse_repeat(text), text)

    def test_preserves_punctuation_only_tokens(self):
        for text in ("hello - - - - world", "a ... ... ... ... b", "- - - -", "... ... ... ...",
                     "a ... a ... a ... a ..."):
            with self.subTest(text=text):
                self.assertEqual(Transcriber._collapse_repeat(text), text)

    def test_preserves_empty_whitespace_and_single_word_inputs(self):
        for text in ("", " \t\n ", "Hello.", " Hello. "):
            with self.subTest(text=text):
                self.assertEqual(Transcriber._collapse_repeat(text), text)

    def test_adjacent_and_nested_loops_collapse(self):
        phrase = "keep Keep keep keep going"
        text = "Please " + " ".join([phrase] * 3) + " now now now now"
        self.assertEqual(Transcriber._collapse_repeat(text), "Please keep going now")


class WhisperDecodeTests(unittest.TestCase):
    def setUp(self):
        self.transcriber = Transcriber({"stt": {"language": "en"}})
        self.transcriber.model_name = "large-v3-turbo"
        self.transcriber.model = Mock()
        self.audio = np.zeros(16000, dtype=np.float32)

    def set_segments(self, *texts, avg_logprob=-0.1, no_speech_prob=0.01, duration_after_vad=1.0):
        segments = [
            SimpleNamespace(text=text, start=float(i), end=float(i + 1), avg_logprob=avg_logprob,
                            no_speech_prob=no_speech_prob)
            for i, text in enumerate(texts)
        ]
        self.transcriber.model.transcribe.return_value = (
            iter(segments), SimpleNamespace(language="en", duration_after_vad=duration_after_vad),
        )

    def assert_fallback_options(self):
        kwargs = self.transcriber.model.transcribe.call_args.kwargs
        self.assertEqual(kwargs["temperature"], (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
        self.assertEqual(kwargs["compression_ratio_threshold"], 2.4)

    def test_dictation_retries_and_removes_loop_suffix_at_a_segment_boundary(self):
        phrase = "So I think like in the build"
        self.set_segments("Check the machines. " + (phrase + " ") * 4 + "So", "We should have some help.")
        result = self.transcriber.transcribe(self.audio)
        self.assert_fallback_options()
        self.assertEqual(result.text, "Check the machines. " + phrase + " We should have some help.")

    def test_dictation_still_drops_uncertain_one_or_two_word_noise_loops(self):
        for text in (" the" * 13, " the" * 24, " Thank you." * 8, " you" * 7):
            with self.subTest(text=text):
                self.set_segments(text, avg_logprob=-1.0, no_speech_prob=0.5, duration_after_vad=3.0)
                result = self.transcriber.transcribe(self.audio)
                self.assertEqual(result.text, "")
                self.assertEqual(result.dropped, "repetition loop")

    def test_dictation_still_drops_confident_noise_loops(self):
        for text in (" the" * 13, " the" * 24, " Thank you." * 8, " you" * 7):
            with self.subTest(text=text):
                self.set_segments(text, duration_after_vad=3.0)
                result = self.transcriber.transcribe(self.audio)
                self.assertEqual(result.text, "")
                self.assertEqual(result.dropped, "repetition loop")

    def test_dictation_retains_a_confident_two_word_loop_after_collapse(self):
        self.set_segments(" Go now." * 6)
        self.assertEqual(self.transcriber.transcribe(self.audio).text, "Go now.")

    def test_timestamped_transcription_retries_and_collapses_longer_loops(self):
        self.set_segments("Check the build. check the build! Check the build. Check the",
                          "Check the notes. Check the notes.")
        segments = self.transcriber.transcribe_segments(self.audio, offset_s=10.0)
        self.assert_fallback_options()
        self.assertEqual(segments, [
            (10.0, 11.0, "Check the build."),
            (11.0, 12.0, "Check the notes. Check the notes."),
        ])


if __name__ == "__main__":
    unittest.main()
