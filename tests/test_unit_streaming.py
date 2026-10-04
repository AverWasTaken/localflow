"""Offline unit tests for chunked ("streaming") transcription: fake STT + fake cleaner, real Silero VAD on the
SAPI clips in tests\\audio. No GPU, no LLM, nothing typed anywhere.

   .venv\\Scripts\\python.exe -m unittest discover -s tests -p "test_unit_*.py" -v
"""
import sys
import tempfile
import threading
import time
import unittest
import wave
from math import gcd
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from localflow import streaming, styles  # noqa: E402
from localflow.config import Config  # noqa: E402
from localflow.pipeline import Pipeline  # noqa: E402
from localflow.streaming import (FRAME, StreamResampler, StreamSession, VadStream, correction_cue,  # noqa: E402
                                 ends_sentence, join_pieces)

AUDIO = Path(__file__).parent / "audio"
SR = 16000


def setUpModule():
    """The SAPI clips are not in the repo: generate them (silently, to files) on first run."""
    if (AUDIO / "08_punct.wav").exists():
        return
    import subprocess
    try:
        subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                        str(Path(__file__).parent / "gen_speech.ps1")], check=True, capture_output=True, timeout=180)
    except Exception as e:
        print(f"could not generate test audio ({e}); tests that need it will be skipped")


def wav(name):
    if not (AUDIO / f"{name}.wav").exists():
        raise unittest.SkipTest(f"{name}.wav missing - run tests/gen_speech.ps1 (needs Windows SAPI voices)")
    with wave.open(str(AUDIO / f"{name}.wav"), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def silence(sec, noise=0.002, seed=0):
    return (np.random.default_rng(seed).standard_normal(int(sec * SR)) * noise).astype(np.float32)


def compose(parts):
    """parts: list of np arrays or float (silence seconds). Returns audio and the list of gap (start, end) seconds."""
    out, gaps, t = [], [], 0.0
    for p in parts:
        if isinstance(p, float):
            p = silence(p)
            gaps.append((t, t + len(p) / SR))
        out.append(p)
        t += len(p) / SR
    return np.concatenate(out), gaps


def make_cfg(**over):
    cfg = Config(path=Path(tempfile.mkdtemp()) / "cfg.json")
    for k, v in over.items():
        cfg.set(k.replace("__", "."), v, save=False)
    return cfg


class FakeSTT:
    def __init__(self, texts=None, delay=0.0, fail=False):
        self.calls = []
        self.texts = list(texts) if texts else None
        self.delay, self.fail = delay, fail
        self.lock = threading.Lock()

    def transcribe(self, audio, hotwords=None, initial_prompt=None):
        with self.lock:
            self.calls.append(SimpleNamespace(n=len(audio), prompt=initial_prompt, hotwords=hotwords))
            n = len(self.calls)
        if self.fail:
            raise RuntimeError("boom")
        time.sleep(self.delay)
        text = self.texts.pop(0) if self.texts else f"This is piece number {n}."
        return SimpleNamespace(text=text, ms=int(self.delay * 1000), dropped="")


class FakeCleaner:
    def __init__(self):
        self.calls = []

    def clean(self, raw, style="neutral", vocab=None, context="", final=True):
        self.calls.append(SimpleNamespace(raw=raw, style=style, context=context, final=final))
        return raw.strip().upper(), {"llm": True, "ms": 3, "reason": "", "style": style}


def feed_fast(sess, audio, chunk=SR // 2, nap=0.012):
    for k in range(0, len(audio), chunk):
        sess.feed(audio[k:k + chunk])
        time.sleep(nap)


def whole_audio_flags(audio):
    from faster_whisper.vad import get_vad_model
    n = len(audio) // FRAME * FRAME
    p = np.asarray(get_vad_model()(audio[:n])).reshape(-1)
    return p >= 0.5


class Helpers(unittest.TestCase):
    def test_correction_cue(self):
        for t in ("Actually no, make it 4 PM.", "no wait, it's Tuesday", "Sorry, I mean 12,000 dollars.",
                  "um actually let's skip it", "scratch that", "I mean the other one", "No, make it Wednesday.",
                  "or rather Friday", "Wait, that's wrong"):
            self.assertTrue(correction_cue(t), t)
        for t in ("The actually good news is", "We should wait for the results.", "Note that I mean business",
                  "Nothing changed.", "Not now."):
            self.assertFalse(correction_cue(t), t)

    def test_ends_sentence(self):
        self.assertTrue(ends_sentence("It works."))
        self.assertTrue(ends_sentence('He said "stop?"'))
        self.assertFalse(ends_sentence("so I think we should"))
        self.assertFalse(ends_sentence("we should,"))

    def test_split_complete(self):
        sc = streaming.split_complete
        self.assertEqual(sc("Good morning everyone, here is the update. The new search feature"),
                         ("Good morning everyone, here is the update.", "The new search feature"))
        # last boundary wins; abbreviations and lowercase continuations are not boundaries
        self.assertEqual(sc("We ship at 3 p.m. on Tuesday and then we celebrate. It was good. And then we"),
                         ("We ship at 3 p.m. on Tuesday and then we celebrate. It was good.", "And then we"))
        self.assertEqual(sc("Ask Dr. Smith about the plan for next quarter and the budget"),
                         ("", "Ask Dr. Smith about the plan for next quarter and the budget"))
        self.assertEqual(sc("Is this ready? Maybe not yet"), ("", "Is this ready? Maybe not yet"))   # head < 6 words
        self.assertEqual(sc("so I think we should go to the store and then"), ("", "so I think we should go to the store and then"))
        self.assertEqual(sc("The meeting starts at 2 a.m. Please avoid deploying anything before then okay"),
                         ("", "The meeting starts at 2 a.m. Please avoid deploying anything before then okay"))

    def test_join_pieces_paragraphs(self):
        long3 = ("Alpha beta gamma delta epsilon zeta eta theta iota kappa. Lambda mu nu xi omicron pi rho sigma tau upsilon. "
                 "Phi chi psi omega alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu.")
        short = "Next point."
        # long pause after 3+ sentences / 35+ words -> paragraph break
        self.assertEqual(join_pieces([long3, short], [0, 1200]), long3 + "\n\n" + short)
        # short pause, or too little text before it -> plain space
        self.assertEqual(join_pieces([long3, short], [0, 400]), long3 + " " + short)
        self.assertEqual(join_pieces([short, short], [0, 1500]), short + " " + short)
        # no pauses given (casual / code) -> never
        self.assertEqual(join_pieces([long3, short]), long3 + " " + short)
        # the second paragraph starts counting afresh
        self.assertEqual(join_pieces([long3, short, short], [0, 1500, 1500]), long3 + "\n\n" + short + " " + short)

    def test_join_pieces(self):
        self.assertEqual(join_pieces(["One.", "Two."]), "One. Two.")
        self.assertEqual(join_pieces(["One.", "", "Two."]), "One. Two.")
        self.assertEqual(join_pieces(["List:\n1. a", "\n\nNext"]), "List:\n1. a\n\nNext")

    def test_resampler_matches_oneshot(self):
        from scipy.signal import resample_poly
        rng = np.random.default_rng(1)
        for sr in (48000, 44100, 32000, 22050, 16000):
            x = (rng.standard_normal(sr * 4) * 0.1).astype(np.float32)
            rs, outs, i = StreamResampler(sr), [], 0
            while i < len(x):
                n = int(rng.integers(300, 5000))
                outs.append(rs.push(x[i:i + n]))
                i += n
            y = np.concatenate(outs)
            ref = x if sr == SR else resample_poly(x, SR // gcd(SR, sr), sr // gcd(SR, sr)).astype(np.float32)
            self.assertGreater(len(y), len(ref) - 1000)              # look-ahead holds back <= ~40 ms
            self.assertLess(float(np.abs(y - ref[:len(y)]).max()), 1e-5, sr)

    def test_vad_stream_equals_whole_run(self):
        from faster_whisper.vad import get_vad_model
        a = wav("06_long")
        n = len(a) // FRAME * FRAME
        whole = np.asarray(get_vad_model()(a[:n])).reshape(-1)
        for chunk in (777, 1600, 8192):
            v = VadStream()
            inc = np.concatenate([v.push(a[i:min(n, i + chunk)]) for i in range(0, n, chunk)])
            self.assertEqual(len(inc), len(whole))
            self.assertLess(float(np.abs(inc - whole).max()), 1e-5)

    def test_runs(self):
        f = np.array([0, 0, 1, 1, 1, 0, 1], bool)
        self.assertEqual(streaming._runs(f), [(False, 0, 2), (True, 2, 3), (False, 5, 1), (True, 6, 1)])
        self.assertEqual(streaming._runs(np.zeros(0, bool)), [])


class Cutting(unittest.TestCase):
    def session(self, stt=None, cleaner=None, **streaming_over):
        cfg = make_cfg(**{f"streaming__{k}": v for k, v in streaming_over.items()})
        sess = StreamSession(cfg, stt or FakeSTT(), cleaner or FakeCleaner(), styles.StyleContext(style="neutral"), SR)
        sess.start()
        return sess, cfg

    def test_cuts_only_at_pauses_and_tail_only_after_release(self):
        audio, gaps = compose([silence(0.2), wav("06_long"), 1.0, wav("02_fillers"), 1.0, wav("03_correction"), 1.0,
                               wav("06_long"), silence(0.2)])
        stt = FakeSTT()
        sess, _ = self.session(stt=stt)
        feed_fast(sess, audio)
        time.sleep(0.4)
        res = sess.finish(audio)
        self.assertGreaterEqual(sess.n_segments, 2, sess._seg_log)
        flags = whole_audio_flags(audio)
        for a, b, _t, _ms in sess._seg_log[:-1]:                 # every cut is inside non-speech
            f = int(b * SR) // FRAME
            self.assertFalse(flags[max(0, f - 3):f + 3].any(), f"cut at {b:.2f}s lands in speech")
        # one transcription per segment + one for the tail; the tail is only what came after the last cut
        # (a tail that is only silence is not sent to the model at all)
        self.assertIn(len(stt.calls), (sess.n_segments, sess.n_segments + 1))
        self.assertLess(stt.calls[-1].n, len(audio) * 0.6)
        covered = sum(c.n for c in stt.calls)
        self.assertLessEqual(covered, len(audio))                    # nothing transcribed twice
        self.assertGreater(covered, len(audio) - 1 * SR)              # nothing lost except silent edges
        # continuity: later pieces get the earlier text as initial_prompt
        self.assertIsNone(stt.calls[0].prompt)
        self.assertIn("piece number 1", stt.calls[1].prompt)
        self.assertEqual(res.n_segments, sess.n_segments)
        pauses = [u.pause_ms for u in sess._units]
        self.assertEqual(pauses[0], 0)
        for ms in pauses[1:]:
            self.assertTrue(600 <= ms <= 2500, pauses)               # planted 1.0 s gaps + SAPI's own padding
        self.assertIn("PIECE NUMBER 1.", res.text)
        self.assertEqual(res.text.count("PIECE NUMBER"), len(stt.calls))

    def test_segments_never_below_min_speech(self):
        audio, _ = compose([silence(0.2), wav("06_long"), 1.0, wav("02_fillers"), 1.0, wav("06_long")])
        sess, _ = self.session(min_segment_s=4.0)
        feed_fast(sess, audio)
        sess.finish(audio)
        for a, b, _t, _ms in sess._seg_log[:-1]:
            self.assertGreaterEqual(b - a, 4.0)

    def test_short_recording_is_single_pass(self):
        audio, _ = compose([silence(0.2), wav("03_correction"), 0.9, wav("01_plain")])     # ~10 s but min_record 12
        sess, _ = self.session(min_record_s=12.0)
        feed_fast(sess, audio)
        time.sleep(0.3)
        self.assertEqual(sess.n_segments, 0)
        sess.cancel()

    def test_forced_cut_at_lowest_energy(self):
        # make pauses unusable (needs 5 s of silence) so the 8 s cap forces cuts
        audio, _ = compose([wav("06_long"), 0.3, wav("06_long"), 0.3, wav("02_fillers")])
        sess, _ = self.session(pause_ms=5000, max_segment_s=8.0, min_record_s=4.0)
        feed_fast(sess, audio)
        time.sleep(0.4)
        sess.finish(audio)
        self.assertGreaterEqual(sess.n_segments, 2)
        hop = 160
        for a, b, _t, _ms in sess._seg_log[:-1]:
            self.assertLessEqual(b - a, 8.0 + 0.7)
            lo = int(b * SR) - int(2.5 * SR)
            win = audio[lo:lo + int(2.5 * SR) + int(0.8 * SR)]
            e = np.sqrt((win[:len(win) // hop * hop].reshape(-1, hop) ** 2).mean(axis=1))
            cut_e = np.sqrt((audio[int(b * SR) - 400:int(b * SR) + 400] ** 2).mean())
            self.assertLessEqual(cut_e, np.percentile(e, 15) + 1e-4, f"cut at {b:.2f}s is not a quiet point")

    def test_cancel_stops_threads(self):
        sess, _ = self.session()
        sess.feed(silence(1.0))
        sess.cancel()
        time.sleep(0.4)
        self.assertFalse(any(t.is_alive() for t in sess._threads))

    def test_stt_failure_makes_finish_raise(self):
        audio, _ = compose([silence(0.2), wav("06_long"), 1.0, wav("02_fillers"), 1.0, wav("06_long")])
        sess, _ = self.session(stt=FakeSTT(fail=True))
        feed_fast(sess, audio)
        time.sleep(0.4)
        with self.assertRaises(RuntimeError):
            sess.finish(audio)

    def test_real_audio_input_poller(self):
        """attach() pulls native-rate chunks from an AudioInput-like object and resamples to 16 kHz."""
        audio, _ = compose([silence(0.2), wav("06_long"), 1.0, wav("02_fillers"), 1.0, wav("06_long")])
        from scipy.signal import resample_poly
        a48 = resample_poly(audio, 3, 1).astype(np.float32)

        class FakeAudio:
            sample_rate = 48000

            def __init__(self):
                self.chunks, self.live = [], True

            def pull_chunks(self, since):
                return (self.chunks[since:], len(self.chunks)) if self.live else (None, since)

        fa = FakeAudio()
        stt = FakeSTT()
        sess, _ = self.session(stt=stt)
        sess.attach(fa, poll_s=0.02)
        for k in range(0, len(a48), 960 * 25):                      # 0.5 s per step
            fa.chunks.append(a48[k:k + 960 * 25])
            time.sleep(0.012)
        time.sleep(0.3)
        res = sess.finish(audio)
        fa.live = False
        self.assertGreaterEqual(sess.n_segments, 1)
        self.assertGreater(sum(c.n for c in stt.calls), len(audio) - 1 * SR)
        self.assertTrue(res.text)


class RealAudioInput(unittest.TestCase):
    def test_session_on_a_real_audioinput_object(self):
        """AudioInput.begin()/pull_chunks()/end() with synthetic 48 kHz callbacks (no microphone, no stream)."""
        from scipy.signal import resample_poly
        from localflow.audio import AudioInput
        cfg = make_cfg()
        ai = AudioInput(cfg)
        ai._sr = 48000
        audio, _ = compose([silence(0.2), wav("06_long"), 1.0, wav("02_fillers"), 1.0, wav("06_long")])
        a48 = resample_poly(audio, 3, 1).astype(np.float32)
        stt = FakeSTT()
        sess = StreamSession(cfg, stt, FakeCleaner(), styles.StyleContext(style="neutral"), ai.sample_rate)
        sess.start()
        ai.begin()
        self.assertTrue(ai.recording)
        sess.attach(ai, poll_s=0.02)
        blk = 960                                                  # 20 ms PortAudio blocks
        for k in range(0, len(a48), blk):
            ai._callback(a48[k:k + blk, None], blk, None, None)
            if k % (blk * 25) == 0:
                time.sleep(0.012)
        time.sleep(0.3)
        full = ai.end()                                            # what the pipeline gets on release
        self.assertFalse(ai.recording)
        self.assertAlmostEqual(len(full) / SR, len(audio) / SR, delta=0.05)
        res = sess.finish(full)
        self.assertGreaterEqual(sess.n_segments, 1)
        self.assertGreater(sum(c.n for c in stt.calls), len(full) - 1 * SR)
        self.assertTrue(res.text)
        self.assertEqual(ai.pull_chunks(0), (None, 0))             # recording ended -> poller exits


class CleanupUnits(unittest.TestCase):
    def make(self, texts, **over):
        cfg = make_cfg(**{f"streaming__{k}": v for k, v in over.items()})
        cl = FakeCleaner()
        sess = StreamSession(cfg, FakeSTT(texts), cl, styles.StyleContext(style="casual"), SR)
        sess.start()
        return sess, cl

    def tail(self, sess, text_audio_sec=1.0):
        a = (np.random.default_rng(3).standard_normal(int(text_audio_sec * SR)) * 0.05).astype(np.float32)
        return a

    def test_sentences_are_cleaned_in_background_with_context(self):
        sess, cl = self.make(["Ship it on Friday.", "Then we party.", "The tail words"])
        sess._cut = 0
        sess._transcribe_piece("seg", 0, SR * 2, self.tail(sess))
        sess._transcribe_piece("seg", SR * 2, SR * 4, self.tail(sess))
        sess.n_segments = 2
        sess._clean_q.join()
        self.assertEqual([c.raw for c in cl.calls], ["Ship it on Friday.", "Then we party."])
        self.assertEqual(cl.calls[0].context, "")
        self.assertEqual(cl.calls[1].context, "SHIP IT ON FRIDAY.")
        self.assertTrue(all(c.final is False for c in cl.calls))
        res = sess.finish(np.concatenate([np.zeros(SR * 4, np.float32), self.tail(sess)]))
        self.assertEqual([c.raw for c in cl.calls][-1], "The tail words")
        self.assertEqual(cl.calls[-1].context, "SHIP IT ON FRIDAY. THEN WE PARTY.")
        self.assertEqual(len(cl.calls), 3)                              # earlier units were NOT cleaned again
        # style pass (casual: lowercase, no final period) runs once on the joined text
        self.assertEqual(res.text, "ship it on friday. then we party. the tail words")

    def test_unfinished_sentence_waits_for_the_next_segment(self):
        sess, cl = self.make(["so I think we should", "go to the store.", "and buy milk"])
        sess._transcribe_piece("seg", 0, SR * 2, self.tail(sess))
        sess._clean_q.join()
        self.assertEqual(cl.calls, [])                                   # no terminal punctuation -> held
        sess._transcribe_piece("seg", SR * 2, SR * 4, self.tail(sess))
        sess._clean_q.join()
        self.assertEqual([c.raw for c in cl.calls], ["so I think we should go to the store."])
        sess.n_segments = 2
        res = sess.finish(np.concatenate([np.zeros(SR * 4, np.float32), self.tail(sess)]))
        self.assertEqual(res.text, "so i think we should go to the store. and buy milk")

    def test_correction_across_boundary_is_merged_and_reclean(self):
        sess, cl = self.make(["Let's meet at 3 PM on Tuesday.", "Actually no, make it 4 PM on Wednesday."])
        sess._transcribe_piece("seg", 0, SR * 2, self.tail(sess))
        sess._clean_q.join()
        sess.n_segments = 1
        res = sess.finish(np.concatenate([np.zeros(SR * 2, np.float32), self.tail(sess)]))
        self.assertEqual(res.merged, 1)
        self.assertEqual(cl.calls[-1].raw, "Let's meet at 3 PM on Tuesday. Actually no, make it 4 PM on Wednesday.")
        self.assertEqual(res.units, 1)
        self.assertNotIn("TUESDAY.\n", res.text)
        self.assertEqual(res.text.lower().count("lets meet"), 1)

    def test_correction_between_background_segments(self):
        sess, cl = self.make(["The budget is 15,000 dollars.", "Sorry, I mean 12,000 dollars.", "Then more."])
        sess._transcribe_piece("seg", 0, SR * 2, self.tail(sess))
        sess._clean_q.join()
        sess._transcribe_piece("seg", SR * 2, SR * 4, self.tail(sess))
        sess._clean_q.join()
        self.assertEqual(cl.calls[-1].raw, "The budget is 15,000 dollars. Sorry, I mean 12,000 dollars.")
        self.assertEqual(sess._units[0].cleaned, "THE BUDGET IS 15,000 DOLLARS. SORRY, I MEAN 12,000 DOLLARS.")
        self.assertEqual(len(sess._units), 1)

    def test_finished_sentences_cleaned_while_fragment_waits(self):
        sess, cl = self.make(["Good morning everyone, here is the weekly update. The new search feature",
                              "is live for all users."])
        sess._transcribe_piece("seg", 0, SR * 2, self.tail(sess))
        sess._clean_q.join()
        self.assertEqual([c.raw for c in cl.calls], ["Good morning everyone, here is the weekly update."])
        sess._transcribe_piece("seg", SR * 2, SR * 4, self.tail(sess))
        sess._clean_q.join()
        self.assertEqual(cl.calls[-1].raw, "The new search feature is live for all users.")
        self.assertEqual(cl.calls[-1].context, "GOOD MORNING EVERYONE, HERE IS THE WEEKLY UPDATE.")

    def test_non_incremental_cleans_everything_once_at_release(self):
        sess, cl = self.make(["One sentence here.", "Two sentence here.", "tail"], incremental_cleanup=False)
        sess._transcribe_piece("seg", 0, SR * 2, self.tail(sess))
        sess._transcribe_piece("seg", SR * 2, SR * 4, self.tail(sess))
        sess.n_segments = 2
        self.assertEqual(cl.calls, [])
        res = sess.finish(np.concatenate([np.zeros(SR * 4, np.float32), self.tail(sess)]))
        self.assertEqual(len(cl.calls), 1)
        self.assertEqual(cl.calls[0].raw, "One sentence here. Two sentence here. tail")
        self.assertEqual(res.units, 1)

    def test_context_is_trimmed_to_word_boundary(self):
        sess, cl = self.make(["word " * 100 + "end.", "second one."], context_chars=40)
        sess._transcribe_piece("seg", 0, SR * 2, self.tail(sess))
        sess._clean_q.join()
        sess._transcribe_piece("seg", SR * 2, SR * 4, self.tail(sess))
        sess._clean_q.join()
        ctx = cl.calls[1].context
        self.assertLessEqual(len(ctx), 40)
        self.assertTrue(ctx.startswith("WORD") or ctx.startswith("END"))


class PipelineIntegration(unittest.TestCase):
    def test_streamed_path_and_fallback(self):
        cfg = make_cfg()
        cfg.set("snippets", {"my email": "me@example.com"}, save=False)
        cfg.set("dictionary", {"words": [], "replacements": {"kubernetes": "Kubernetes"}}, save=False)
        inserted, hist = [], []

        class H:
            def add(self, *a, **k):
                hist.append((a, k))

        class S:  # whole-clip STT used by the fallback path
            def transcribe(self, audio, hotwords=None, initial_prompt=None):
                return SimpleNamespace(text="fallback text from the whole clip", ms=7, dropped="")

        class C(FakeCleaner):
            pass

        pipe = Pipeline(cfg, S(), C(), H(), lambda t, **k: inserted.append(t))
        audio = (np.random.default_rng(5).standard_normal(SR * 3) * 0.05).astype(np.float32)

        good = SimpleNamespace(n_segments=2, failed=None, cancel=lambda: None, finish=lambda a: streaming.StreamResult(
            raw="raw joined", text="Deploy kubernetes now. Send my email later", n_segments=2, tail_stt_ms=90,
            stt_total_ms=300, cleanup_ms=120, llm=True, units=2))
        ctx = styles.StyleContext(exe="discord.exe", title="#general", style="casual", rule="Discord", app="Discord")
        r = pipe.process(audio, style_ctx=ctx, session=good)
        self.assertTrue(r["streamed"])
        self.assertEqual(r["segments"], 3)
        self.assertEqual(r["stt_ms"], 90)
        self.assertEqual(r["text"], "Deploy Kubernetes now. Send me@example.com later")
        self.assertEqual(inserted, [r["text"]])
        self.assertEqual(hist[-1][1]["style"], "casual")
        self.assertEqual(hist[-1][1]["title"], "#general")
        self.assertEqual(hist[-1][1]["segments"], 3)
        self.assertEqual(r["style"], "casual")

        def boom(a):
            raise RuntimeError("stream broke")

        bad = SimpleNamespace(n_segments=2, failed=None, cancel=lambda: None, finish=boom)
        r = pipe.process(audio, style_ctx=ctx, session=bad)
        self.assertFalse(r["streamed"])
        self.assertEqual(r["status"], "ok")
        self.assertIn("FALLBACK TEXT", r["text"].upper())

        zero = SimpleNamespace(n_segments=0, failed=None, cancel=lambda: None, finish=boom)
        r = pipe.process(audio, style_ctx=ctx, session=zero)         # nothing was cut -> normal single pass
        self.assertFalse(r["streamed"])

    def test_style_resolved_from_app_when_no_ctx(self):
        cfg = make_cfg()
        pipe = Pipeline(cfg, SimpleNamespace(transcribe=lambda *a, **k: SimpleNamespace(text="hello there my friend",
                                                                                         ms=1, dropped="")),
                        FakeCleaner(), None, lambda *a, **k: {})
        audio = (np.random.default_rng(5).standard_normal(SR * 2) * 0.05).astype(np.float32)
        r = pipe.process(audio, app_exe="discord.exe", app_title="x", insert=False)
        self.assertEqual(r["style"], "casual")


if __name__ == "__main__":
    unittest.main()
