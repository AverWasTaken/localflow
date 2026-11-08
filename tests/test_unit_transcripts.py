"""Offline tests for the transcripts engine: store, merge + echo marking, chunk planning, recorder writer +
recovery from a killed writer, assistant chunk budgeting (fake HTTP), job runner priority / pause / cancel /
resume, and the service façade.

No audio device is opened and nothing is played: recorder tests feed synthetic PCM through the callback path.
Run: .venv\\Scripts\\python.exe tests\\test_unit_transcripts.py
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from localflow import assistant as asst  # noqa: E402
from localflow import jobs  # noqa: E402
from localflow import recorder as rec  # noqa: E402
from localflow.gate import ForegroundGate  # noqa: E402
from localflow.stt import ModelUnloaded  # noqa: E402
from localflow.transcripts import TranscriptStore  # noqa: E402


class Cfg(dict):
    """Config stand-in with dotted get()."""

    def get(self, dotted, default=None):
        cur = self
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur


def wait_for(pred, timeout=10.0, step=0.02):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(step)
    return pred()


class TmpCase(unittest.TestCase):
    def setUp(self):
        # cleanups run last-in-first-out: services/runners registered by tests stop before the dir is removed
        self._td = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._td.cleanup)
        self.tmp = Path(self._td.name)
        self.root = self.tmp / "recordings"
        self.root.mkdir()
        self.store = TranscriptStore(self.tmp / "t.db", self.root)
        self.addCleanup(self.store.close)


# ============================================================================ store
class StoreTests(TmpCase):
    def test_crud_search_rename(self):
        a = self.store.create("file", "Lecture one", source_path="C:/x/lecture.mp3", style="neutral")
        b = self.store.create("meeting", "Meeting A")
        self.store.update(a, status="done", text="the mitochondria is the powerhouse", raw_text="raw")
        self.assertEqual(self.store.get(a)["status"], "done")
        self.assertEqual([r["id"] for r in self.store.list()], [b, a])
        self.assertEqual([r["id"] for r in self.store.list("powerhouse")], [a])
        self.assertEqual([r["id"] for r in self.store.list("meeting a")], [b])
        self.assertEqual(self.store.list("100%"), [])           # LIKE wildcards are escaped
        self.store.rename(b, "  Standup ")
        self.assertEqual(self.store.get(b)["title"], "Standup")
        with self.assertRaises(ValueError):
            self.store.update(a, status="bogus")
        with self.assertRaises(ValueError):
            self.store.update(a, nope=1)

    def test_info_and_segments(self):
        a = self.store.create("file", "x")
        self.store.update(a, segments_json={"segments": [{"start": 0, "end": 1, "text": "hi"}]})
        self.assertEqual(self.store.segments(a)["segments"][0]["text"], "hi")
        self.store.update_info(a, language="en", fallbacks=2)
        self.store.update_info(a, fallbacks=None)
        self.assertEqual(self.store.get(a)["info"], {"language": "en"})

    def test_delete_only_removes_audio_inside_root(self):
        inside = self.root / "20260101-000000"
        inside.mkdir()
        (inside / "mic.flac").write_bytes(b"x")
        outside = self.tmp / "user_audio"
        outside.mkdir()
        a = self.store.create("meeting", "m", audio_dir=str(inside))
        b = self.store.create("file", "f", audio_dir=str(outside))
        self.store.add_ai_result(a, "summary", "", "sum")
        self.store.save_progress(a, "stt:mic:0", [[0, 1, "x"]])
        self.assertTrue(self.store.delete(a))
        self.assertFalse(inside.exists())
        self.assertEqual(self.store.list_ai_results(a), [])
        self.assertEqual(self.store.load_progress(a), {})
        self.assertTrue(self.store.delete(b))
        self.assertTrue(outside.exists())                        # never deleted
        self.assertFalse(self.store.delete(999))

    def test_ai_results_and_export(self):
        a = self.store.create("meeting", "Planning")
        self.store.update(a, text="[00:00:01] You: hello there\n\n[00:00:03] Others: hi", duration_s=65)
        self.store.add_ai_result(a, "summary", "", "A greeting.")
        self.store.add_ai_result(a, "custom", "who spoke?", "Both.")
        res = self.store.list_ai_results(a)
        self.assertEqual([r["preset"] for r in res], ["custom", "summary"])
        txt = self.store.export_text(a, "txt")
        self.assertTrue(txt.startswith("[00:00:01] You: hello there"))
        md = self.store.export_text(a, "md")
        self.assertIn("# Planning", md)
        self.assertIn("**[00:00:01] You:** hello there", md)
        self.assertIn("### Summary", md)
        self.assertIn("### who spoke?", md)
        self.assertIn("00:01:05", md)
        with self.assertRaises(ValueError):
            self.store.export_text(a, "pdf")

    def test_progress_roundtrip(self):
        a = self.store.create("file", "x")
        self.store.save_progress(a, "plan:file", [[0, 16000]])
        self.store.save_progress(a, "stt:file:0", [[0.0, 1.0, "hi"]])
        self.assertEqual(self.store.load_progress(a)["stt:file:0"], [[0.0, 1.0, "hi"]])
        self.store.clear_progress(a)
        self.assertEqual(self.store.load_progress(a), {})

    def test_thread_safety(self):
        errors = []

        def worker(n):
            try:
                for i in range(25):
                    t = self.store.create("file", f"t{n}-{i}")
                    self.store.update(t, progress=i / 25.0, text="x" * i)
                    self.store.save_progress(t, f"k{i}", {"i": i})
                    self.store.list("t")
            except Exception as e:      # pragma: no cover - failure path
                errors.append(e)
        ths = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(self.store.list(limit=10000)), 150)


# ============================================================================ merge / echo / turns
class MergeEchoTests(unittest.TestCase):
    def test_echo_marked_not_deleted(self):
        you = [(10.1, 13.0, "So the budget is approved for next quarter."), (20.0, 22.0, "Great, I agree with that.")]
        others = [(10.0, 13.2, "So the budget is approved for next quarter, right?"), (30.0, 31.0, "Okay.")]
        segs = jobs.merge_tracks(you, others)
        self.assertEqual(len(segs), 4)
        self.assertEqual(jobs.mark_echo(segs, 0.75), 1)
        echo = [s for s in segs if s.get("echo")]
        self.assertEqual(len(echo), 1)
        self.assertEqual(echo[0]["speaker"], "You")
        self.assertTrue(echo[0]["text"].startswith("So the budget"))
        text = jobs.render_turns(jobs.build_turns(segs))
        self.assertNotIn("You: So the budget", text)
        self.assertIn("Others: So the budget", text)
        self.assertIn("You: Great, I agree", text)
        with_echo = jobs.render_turns(jobs.build_turns(segs, include_echo=True))
        self.assertIn("You: So the budget", with_echo)

    def test_same_words_at_other_time_is_not_echo(self):
        segs = jobs.merge_tracks([(50.0, 52.0, "So the budget is approved.")], [(10.0, 12.0, "So the budget is approved.")])
        self.assertEqual(jobs.mark_echo(segs), 0)

    def test_overlapping_but_different_is_not_echo(self):
        segs = jobs.merge_tracks([(10.0, 12.0, "I need to leave early today, sorry.")],
                                 [(10.5, 12.5, "The quarterly numbers look strong overall.")])
        self.assertEqual(jobs.mark_echo(segs), 0)

    def test_threshold(self):
        segs = jobs.merge_tracks([(1.0, 3.0, "alpha beta gamma delta")], [(1.0, 3.0, "alpha beta gamma omega")])
        self.assertEqual(jobs.mark_echo(segs, 0.75), 1)
        self.assertEqual(jobs.mark_echo(segs, 0.8), 0)            # re-marking clears old flags

    def test_turns_join_same_speaker(self):
        segs = jobs.merge_tracks([(0.0, 1.0, "Hi."), (1.2, 2.0, "How are you?"), (40.0, 41.0, "Still there?")],
                                 [(2.5, 3.5, "Good, thanks.")])
        turns = jobs.build_turns(segs)
        self.assertEqual([t["speaker"] for t in turns], ["You", "Others", "You"])
        self.assertEqual(turns[0]["raw"], "Hi. How are you?")
        self.assertEqual(jobs.render_turns(turns).split("\n\n")[0], "[00:00:00] You: Hi. How are you?")

    def test_render_meeting_uses_cleaned_turns_only_without_echo(self):
        segs = jobs.merge_tracks([(1.0, 2.0, "uh hello")], [(1.0, 2.0, "uh hello")])
        jobs.mark_echo(segs)
        turns = jobs.build_turns(segs)
        turns[0]["text"] = "Hello."
        sj = {"segments": segs, "turns": turns}
        self.assertEqual(jobs.render_meeting(sj, False), "[00:00:01] Others: Hello.")
        self.assertIn("You: uh hello", jobs.render_meeting(sj, True))

    def test_paragraphs_and_cleanup_split(self):
        segs = [(0.0, 2.0, "First sentence here."), (2.2, 4.0, "Second one."), (7.0, 9.0, "After a pause.")]
        paras = jobs.group_paragraphs(segs)
        self.assertEqual([p["raw"] for p in paras], ["First sentence here. Second one.", "After a pause."])
        long_segs = [(i * 1.0, i * 1.0 + 0.9, "word " * 9 + "end.") for i in range(20)]
        paras = jobs.group_paragraphs(long_segs)
        self.assertGreater(len(paras), 1)
        self.assertTrue(all(len(p["raw"].split()) <= 80 * 1.6 + 10 for p in paras))
        text = " ".join(["This is sentence number %d with some words." % i for i in range(60)])
        pieces = jobs.split_for_cleanup(text, 110)
        self.assertGreater(len(pieces), 2)
        self.assertEqual(" ".join(pieces).split(), text.split())
        self.assertTrue(all(len(p.split()) <= 110 + 12 for p in pieces))
        blob = "word " * 400
        self.assertTrue(all(len(p.split()) <= 176 for p in jobs.split_for_cleanup(blob, 110)))


# ============================================================================ chunk planning
class ChunkPlanTests(unittest.TestCase):
    SR = 16000

    def test_merges_regions_up_to_max(self):
        regions = [(0, 5), (6, 12), (13, 25), (26, 40), (41, 44)]
        fake = lambda x: [{"start": s * self.SR, "end": e * self.SR} for s, e in regions]  # noqa: E731
        audio = np.zeros(50 * self.SR, dtype=np.int16)
        chunks = jobs.plan_chunks(audio, 30.0, vad_fn=fake)
        self.assertEqual([[s // self.SR, e // self.SR] for s, e in chunks], [[0, 25], [26, 44]])
        self.assertTrue(all(e - s <= 30 * self.SR for s, e in chunks))

    def test_long_region_hard_split_and_overlap_trim(self):
        fake = lambda x: [{"start": 0, "end": 70 * self.SR}, {"start": 69 * self.SR, "end": 75 * self.SR}]  # noqa
        chunks = jobs.plan_chunks(np.zeros(80 * self.SR, dtype=np.int16), 30.0, vad_fn=fake)
        self.assertTrue(all(e - s <= 30 * self.SR for s, e in chunks))
        for (s1, e1), (s2, e2) in zip(chunks, chunks[1:]):
            self.assertLessEqual(e1, s2)                         # no audio transcribed twice
        self.assertEqual(chunks[0][0], 0)
        self.assertEqual(chunks[-1][1], 75 * self.SR)

    def test_vad_blocks_get_absolute_offsets(self):
        calls = []

        def fake(x):
            calls.append(x.size)
            return [{"start": 10 * self.SR, "end": 12 * self.SR}]
        n = int((jobs.VAD_BLOCK_S * 2 + 30) * self.SR)
        chunks = jobs.plan_chunks(np.zeros(n, dtype=np.int16), 30.0, vad_fn=fake)
        self.assertEqual(len(calls), 3)
        starts = [s // self.SR for s, _ in chunks]
        self.assertEqual(starts, [10, jobs.VAD_BLOCK_S + 10, 2 * jobs.VAD_BLOCK_S + 10])

    def test_real_vad_on_speech_file(self):
        wav = ROOT / "tests" / "bench_audio" / "bench_90s.wav"
        if not wav.exists():
            self.skipTest("bench audio missing")
        audio = jobs.decode_file_16k(str(wav), 3600)
        self.assertGreater(audio.size, 60 * self.SR)
        chunks = jobs.plan_chunks(audio)
        self.assertGreaterEqual(len(chunks), 3)
        self.assertTrue(all(0 < e - s <= 30 * self.SR for s, e in chunks))
        covered = sum(e - s for s, e in chunks) / audio.size
        self.assertGreater(covered, 0.5)

    def test_decode_refuses_too_long(self):
        wav = ROOT / "tests" / "bench_audio" / "bench_20s.wav"
        if not wav.exists():
            self.skipTest("bench audio missing")
        with self.assertRaises(jobs.JobError):
            jobs.decode_file_16k(str(wav), 5)

    def test_load_track_resample_no_seams(self):
        import soundfile as sf
        with tempfile.TemporaryDirectory() as td:
            sr = 48000
            t = np.arange(int(sr * 130.5)) / sr                   # crosses two 60 s block boundaries
            x = (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16)
            p = Path(td) / "mic.flac"
            sf.write(str(p), x, sr, subtype="PCM_16")
            y = jobs.load_track_16k(p, sr)
            self.assertEqual(abs(y.size - int(130.5 * 16000)), 0)
            ref = (np.sin(2 * np.pi * 440 * np.arange(y.size) / 16000) * 8000)
            err = np.abs(y[100:-100].astype(np.float64) - ref[100:-100])
            self.assertLess(err.max(), 200)                       # no clicks at block seams
            pcm = Path(td) / "sys.pcm"
            x.tofile(pcm)
            y2 = jobs.load_track_16k(pcm, sr)
            self.assertTrue(np.array_equal(y, y2))


# ============================================================================ recorder (synthetic)
class FakeRecorder(rec.MeetingRecorder):
    """Feeds synthetic 20 ms blocks through the real callback path instead of opening devices."""
    RATE = 48000

    def __init__(self, *a, gap_at=None, gap_s=0.0, **kw):
        super().__init__(*a, **kw)
        self.gap_at = gap_at
        self.gap_s = gap_s
        self._feed_stop = threading.Event()
        self._feeders = []

    def _open_streams(self):
        self._feed_stop.clear()
        for key, val in (("mic", 1000), ("system", 2000)):
            tr = self._tracks[key]
            tr.rate, tr.channels, tr.device = self.RATE, 2 if key == "system" else 1, f"fake {key}"
            th = threading.Thread(target=self._feed, args=(tr, val), daemon=True)
            self._feeders.append(th)
            th.start()

    def _feed(self, tr, val):
        n = self.RATE // 50
        block = np.full(n * tr.channels, val, dtype=np.int16).tobytes()
        t_start = time.monotonic()
        k = 0
        gapped = False
        while not self._feed_stop.is_set():
            if (self.gap_at is not None and tr.key == "system" and not gapped
                    and time.monotonic() - t_start >= self.gap_at):
                gapped = True
                time.sleep(self.gap_s)
                k = int((time.monotonic() - t_start) * 50)      # device delivered nothing during the gap
            self._on_input(tr, block, n)
            k += 1
            nxt = t_start + k * 0.02
            time.sleep(max(0.0, nxt - time.monotonic()))

    def _close_streams(self):
        self._feed_stop.set()
        for th in self._feeders:
            th.join(2)
        self._feeders = []


class RecorderTests(TmpCase):
    def test_start_stop_writes_verified_flac(self):
        import soundfile as sf
        levels = []
        r = FakeRecorder(Cfg(meetings={"min_free_gb": 0.001}), on_level=lambda m, s: levels.append((m, s)))
        d = self.root / "a"
        r.start(d)
        self.assertEqual(rec.read_meta(d)["state"], "recording")
        time.sleep(1.5)
        self.assertTrue(r.recording)
        self.assertGreater(r.elapsed, 1.0)
        meta = r.stop()
        self.assertFalse(r.recording)
        self.assertEqual(meta["state"], "stopped")
        for key, val in (("mic", 1000), ("system", 2000)):
            tr = meta["tracks"][key]
            self.assertEqual(tr["file"], f"{key}.flac")
            self.assertFalse((d / f"{key}.pcm").exists())
            data, sr = sf.read(str(d / tr["file"]), dtype="int16")
            self.assertEqual(sr, 48000)
            self.assertEqual(len(data), tr["samples"])
            self.assertAlmostEqual(len(data) / sr, meta["duration_s"] - tr["offset_s"], delta=0.15)
            self.assertTrue(np.all(data == val))                  # stereo loopback downmixed to mono
        self.assertTrue(levels and levels[-1][0] > 0 and levels[-1][1] > 0)

    def test_gap_is_zero_filled_to_wall_clock(self):
        r = FakeRecorder(Cfg(meetings={"min_free_gb": 0.001}), gap_at=0.5, gap_s=1.2)
        d = self.root / "gap"
        r.start(d)
        time.sleep(2.5)
        meta = r.stop()
        sys_tr, mic_tr = meta["tracks"]["system"], meta["tracks"]["mic"]
        self.assertGreater(sys_tr["zero_filled"], 48000 * 1.0)
        # both tracks still cover the same wall-clock span
        self.assertAlmostEqual(sys_tr["samples"] + sys_tr["offset_s"] * 48000,
                               mic_tr["samples"] + mic_tr["offset_s"] * 48000, delta=48000 * 0.1)

    def test_refuses_when_disk_full(self):
        r = FakeRecorder(Cfg(meetings={"min_free_gb": 10 ** 9}))
        with self.assertRaises(rec.RecorderError):
            r.start(self.root / "full")

    def test_max_length_auto_stops(self):
        stopped = []
        r = FakeRecorder(Cfg(meetings={"min_free_gb": 0.001, "max_hours": 1.0 / 3600}),
                         on_auto_stop=lambda meta, why: stopped.append((meta, why)))
        r.start(self.root / "max")
        self.assertTrue(wait_for(lambda: stopped, 5))
        self.assertIn("maximum", stopped[0][1])
        self.assertFalse(r.recording)
        self.assertEqual(stopped[0][0]["state"], "stopped")

    def test_pcm_to_flac_keeps_pcm_on_failure(self):
        pcm = self.tmp / "x.pcm"
        np.arange(1000, dtype=np.int16).tofile(pcm)
        with mock.patch("soundfile.SoundFile", side_effect=RuntimeError("disk")):
            with self.assertRaises(RuntimeError):
                rec.pcm_to_flac(pcm, self.tmp / "x.flac", 16000)
        self.assertTrue(pcm.exists())
        self.assertEqual(rec.pcm_to_flac(pcm, self.tmp / "x.flac", 16000), 1000)
        self.assertFalse(pcm.exists())

    def test_recovery_from_killed_writer(self):
        """A subprocess records (synthetic feed) and is force-killed; recovery converts and marks it."""
        d = self.root / "killed"
        code = f"""
import sys, time
sys.path.insert(0, {str(ROOT)!r}); sys.path.insert(0, {str(ROOT / 'tests')!r})
from test_unit_transcripts import FakeRecorder, Cfg
r = FakeRecorder(Cfg(meetings={{"min_free_gb": 0.001}}))
r.start({str(d)!r})
print("started", flush=True)
time.sleep(60)
"""
        p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            line = p.stdout.readline().decode().strip()
            self.assertEqual(line, "started", p.stderr.read().decode() if line != "started" else "")
            time.sleep(2.5)                                        # > 1 s flush interval
        finally:
            p.kill()                                               # TerminateProcess: no cleanup runs
            p.wait(10)
        self.assertEqual(rec.read_meta(d)["state"], "recording")
        self.assertTrue((d / "mic.pcm").exists())
        out = rec.recover_interrupted(self.root)
        self.assertEqual([x[0].name for x in out], ["killed"])
        meta = rec.read_meta(d)
        self.assertEqual(meta["state"], "interrupted")
        for key in rec.TRACKS:
            self.assertFalse((d / f"{key}.pcm").exists())
            self.assertTrue((d / f"{key}.flac").exists())
            self.assertGreater(meta["tracks"][key]["samples"], 48000 * 1.0)
        self.assertGreater(meta["duration_s"], 1.0)
        self.assertEqual(rec.recover_interrupted(self.root), [])   # idempotent


# ============================================================================ assistant
class FakeResp:
    def __init__(self, chunks, gate=None, busy_after=None):
        self.chunks = chunks
        self.gate = gate
        self.busy_after = busy_after
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.closed = True

    def raise_for_status(self):
        pass

    def iter_lines(self):
        for i, c in enumerate(self.chunks):
            if self.busy_after is not None and i == self.busy_after:
                self.gate.begin()
                threading.Timer(0.2, self.gate.end).start()
            yield b"data: " + json.dumps({"choices": [{"delta": {"content": c}}]}).encode()
        yield b"data: [DONE]"


class FakeSession:
    def __init__(self, responder):
        self.responder = responder
        self.bodies = []

    def post(self, url, json=None, stream=False, timeout=None):
        self.bodies.append(json)
        return self.responder(len(self.bodies), json)


class AssistantTests(TmpCase):
    def make(self, responder, ready=lambda: True, ctx=8192, gate=None):
        gate = gate or ForegroundGate()
        sess = FakeSession(responder)
        a = asst.Assistant(Cfg(cleanup={"ctx": ctx, "enabled": True}).get, gate, self.store,
                           lambda: "http://x", ready, session=sess)
        a.retry_sleep_s = 0.01
        a.ready_poll_s = 0.02
        return a, sess, gate

    def test_budget_and_chunking(self):
        b = asst.budget_chars(8192, 700, 1024)
        self.assertEqual(b, int((8192 - int(700 / 3.5) - 1024 - 256) * 3.5))
        text = "\n\n".join(("Sentence %d is here. " % i) * 30 for i in range(40))
        chunks = asst.chunk_text(text, 2000)
        self.assertTrue(all(len(c) <= 2000 for c in chunks))
        self.assertEqual("".join(chunks).replace("\n", "").replace(" ", ""), text.replace("\n", "").replace(" ", ""))
        self.assertTrue(all(len(c) <= 500 for c in asst.chunk_text("x" * 3000, 500)))

    def test_short_transcript_single_request_saved(self):
        tid = self.store.create("file", "x")
        self.store.update(tid, text="We agreed to ship on Friday. Bob will write the notes.")
        partials = []
        a, sess, _ = self.make(lambda n, body: FakeResp(["- Ship ", "Friday", "\n- Bob: notes"]))
        out = a.ask(tid, "action_items", on_partial=partials.append)
        self.assertEqual(out, "- Ship Friday\n- Bob: notes")
        self.assertEqual(len(sess.bodies), 1)
        body = sess.bodies[0]
        self.assertTrue(body["stream"])
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertIn("action item", body["messages"][1]["content"])
        self.assertEqual(partials[-1], out)
        res = self.store.list_ai_results(tid)
        self.assertEqual((res[0]["preset"], res[0]["output"]), ("action_items", out))

    def test_custom_prompt_and_think_stripped(self):
        tid = self.store.create("file", "x")
        self.store.update(tid, text="hello world")
        partials = []
        a, sess, _ = self.make(lambda n, body: FakeResp(["<think>hmm", "</think>", "Answer."]))
        self.assertEqual(a.ask(tid, "What was said?", on_partial=partials.append), "Answer.")
        self.assertEqual(partials[0], "")                          # thinking never shown
        self.assertEqual(self.store.list_ai_results(tid)[0]["prompt"], "What was said?")
        with self.assertRaises(asst.AssistantError):
            a.ask(tid, "   ")

    def test_map_reduce_fits_context(self):
        ctx = 2048
        tid = self.store.create("meeting", "long")
        text = "\n\n".join(f"[00:{i // 60:02d}:{i % 60:02d}] Others: " + ("We discussed item %d at length. " % i) * 12
                           for i in range(60))
        self.store.update(tid, text=text)
        a, sess, _ = self.make(lambda n, body: FakeResp([f"note {n}"]), ctx=ctx)
        out = a.ask(tid, "summary")
        self.assertTrue(out.startswith("note"))
        self.assertGreater(len(sess.bodies), 3)
        for body in sess.bodies:
            chars = sum(len(m["content"]) for m in body["messages"])
            self.assertLessEqual(chars / 3.5 + body["max_tokens"], ctx, "request would overflow the context")
        self.assertIn("notes", sess.bodies[-1]["messages"][1]["content"])

    def test_preempted_by_dictation_retries(self):
        tid = self.store.create("file", "x")
        self.store.update(tid, text="some words here")
        gate = ForegroundGate()
        first = {}

        def responder(n, body):
            if n == 1:
                first["r"] = FakeResp(["part", "ial", "more"], gate=gate, busy_after=1)
                return first["r"]
            return FakeResp(["Full answer."])
        a, sess, _ = self.make(responder, gate=gate)
        self.assertEqual(a.ask(tid, "summary"), "Full answer.")
        self.assertEqual(len(sess.bodies), 2)
        self.assertTrue(first["r"].closed)

    def test_waits_until_ready_and_cancel(self):
        tid = self.store.create("file", "x")
        self.store.update(tid, text="some words here")
        state = {"ready": False}
        a, sess, _ = self.make(lambda n, body: FakeResp(["ok"]), ready=lambda: state["ready"])
        threading.Timer(0.3, lambda: state.update(ready=True)).start()
        t0 = time.monotonic()
        self.assertEqual(a.ask(tid, "summary"), "ok")
        self.assertGreater(time.monotonic() - t0, 0.25)
        state["ready"] = False
        stop = threading.Event()
        threading.Timer(0.2, stop.set).start()
        with self.assertRaises(asst.AssistantCancelled):
            a.ask(tid, "summary", stop=stop)

    def test_connection_errors_give_up(self):
        import requests
        tid = self.store.create("file", "x")
        self.store.update(tid, text="some words here")

        def responder(n, body):
            raise requests.ConnectionError("refused")
        a, sess, _ = self.make(responder)
        with self.assertRaises(asst.AssistantError):
            a.ask(tid, "summary")
        self.assertEqual(len(sess.bodies), 4)


# ============================================================================ job runner
class FakeSTT:
    def __init__(self, unloaded_first=0, delay=0.0):
        self.calls = []
        self.unloaded = unloaded_first
        self.delay = delay

    def transcribe_segments(self, audio, offset_s, language=None):
        if self.unloaded > 0:
            self.unloaded -= 1
            raise ModelUnloaded("unloaded")
        self.calls.append((time.monotonic(), offset_s, float(audio[0]) if audio.size else 0.0))
        time.sleep(self.delay)
        val = float(audio[0]) if audio.size else 0.0
        who = "mic" if abs(val - 1000 / 32768) < 1e-4 else ("sys" if abs(val - 2000 / 32768) < 1e-4 else "file")
        if who == "sys" or (who == "mic" and offset_s < 1):
            text = "The project deadline moves to Friday afternoon."        # mic copy at t~0 is an echo
        elif who == "mic":
            text = "Sounds good to me."
        else:
            text = f"Chunk at {offset_s:.0f} seconds."
        return [(offset_s + 0.5, offset_s + 2.5, text)]


class FakeCleaner:
    def __init__(self, not_ready_first=0):
        self.calls = []
        self.not_ready = not_ready_first

    def clean(self, raw, style="neutral", vocab=None, context="", final=True):
        if self.not_ready > 0:
            self.not_ready -= 1
            return raw, {"llm": False, "reason": "llm not ready"}
        self.calls.append((raw, style, context))
        return raw.upper(), {"llm": True, "reason": ""}


def chunks_plan(n, step=30):
    return lambda audio, *a, **k: [[i * step * 16000, (i * step + 20) * 16000] for i in range(n)]


class RunnerTests(TmpCase):
    def make(self, stt=None, cleaner=None, paused=lambda: False, gate=None):
        self.stt = stt or FakeSTT()
        self.cleaner = cleaner or FakeCleaner()
        self.gate = gate or ForegroundGate()
        self.events = []
        r = jobs.JobRunner(self.stt, self.cleaner, self.gate, self.store, paused,
                           on_progress=lambda *a: self.events.append(a), on_done=lambda *a: self.events.append(a))
        r.retry_sleep_s = 0.02
        self.addCleanup(r.shutdown, 5)
        return r

    def file_job(self, style="raw"):
        f = self.tmp / "a.wav"
        f.write_bytes(b"RIFF")
        return self.store.create("file", "a", source_path=str(f), style=style)

    def patches(self, n_chunks=4, seconds=120):
        p1 = mock.patch.object(jobs, "decode_file_16k", lambda path, max_s: np.zeros(seconds * 16000, np.int16))
        p2 = mock.patch.object(jobs, "plan_chunks", chunks_plan(n_chunks))
        p1.start()
        p2.start()
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)

    def test_file_job_raw(self):
        self.patches(4)
        r = self.make()
        tid = self.file_job()
        r.submit(tid)
        self.assertTrue(wait_for(lambda: self.store.get(tid)["status"] == "done"))
        row = self.store.get(tid)
        self.assertEqual(len(self.stt.calls), 4)
        self.assertEqual([c[1] for c in self.stt.calls], [0, 30, 60, 90])        # absolute offsets
        self.assertIn("Chunk at 60 seconds.", row["text"])
        self.assertEqual(row["text"], row["raw_text"])
        self.assertEqual(self.cleaner.calls, [])
        self.assertEqual(len(self.store.segments(tid)["segments"]), 4)
        self.assertEqual(self.store.load_progress(tid), {})                       # cleared when done
        self.assertIn((tid, "done"), self.events)

    def test_file_job_style_cleans_per_paragraph_and_waits_for_llm(self):
        self.patches(3)
        r = self.make(cleaner=FakeCleaner(not_ready_first=1))
        with mock.patch.object(jobs.JobRunner, "_sleep", lambda self, job, s: time.sleep(0.01)):
            tid = self.file_job("formal")
            r.submit(tid)
            self.assertTrue(wait_for(lambda: self.store.get(tid)["status"] == "done"))
        row = self.store.get(tid)
        self.assertEqual(len(self.cleaner.calls), 3)                               # one per paragraph (gaps > 1.5 s)
        self.assertTrue(all(c[1] == "formal" for c in self.cleaner.calls))
        self.assertEqual(self.cleaner.calls[1][2], self.cleaner.calls[0][0].upper())  # previous text as context
        self.assertIn("CHUNK AT 30 SECONDS.", row["text"])
        self.assertIn("Chunk at 30 seconds.", row["raw_text"])

    def make_cfg(self, cfg, cleaner):
        self.stt, self.cleaner, self.gate = FakeSTT(), cleaner, ForegroundGate()
        r = jobs.JobRunner(self.stt, cleaner, self.gate, self.store, lambda: False, cfg=cfg)
        r.retry_sleep_s = 0.02
        self.addCleanup(r.shutdown, 5)
        return r

    def test_llm_wait_is_capped_then_regex_fallback_with_note(self):
        self.patches(3)
        cleaner = FakeCleaner(not_ready_first=10 ** 6)                            # the LLM never comes up
        r = self.make_cfg(Cfg(assistant={"wait_ready_s": 0.3}), cleaner)
        tid = self.file_job("formal")
        t0 = time.monotonic()
        r.submit(tid)
        self.assertTrue(wait_for(lambda: self.store.get(tid)["status"] in ("done", "failed")))
        row = self.store.get(tid)
        self.assertEqual(row["status"], "done")
        self.assertEqual(row["error"], jobs.CLEANUP_UNAVAILABLE_NOTE)
        self.assertIn("Chunk at 30 seconds.", row["text"])                        # regex-fallback text kept
        self.assertEqual(cleaner.calls, [])
        # waited once (~0.3 s) for the whole job, not once per paragraph
        self.assertLess(time.monotonic() - t0, 3.0)

    def test_connection_errors_share_the_wait_budget(self):
        self.patches(3)

        class DeadCleaner(FakeCleaner):
            def clean(self, raw, style="neutral", vocab=None, context="", final=True):
                self.calls.append(raw)
                return raw, {"llm": False, "reason": "error: ConnectionError"}
        cleaner = DeadCleaner()
        r = self.make_cfg(Cfg(assistant={"wait_ready_s": 0.3}), cleaner)
        tid = self.file_job("formal")
        t0 = time.monotonic()
        r.submit(tid)
        self.assertTrue(wait_for(lambda: self.store.get(tid)["status"] in ("done", "failed")))
        self.assertLess(time.monotonic() - t0, 3.0)                               # not 20 x 3 s per piece
        row = self.store.get(tid)
        self.assertEqual((row["status"], row["error"]), ("done", jobs.CLEANUP_UNAVAILABLE_NOTE))
        # zero budget: one attempt per piece, no retry sleeps
        cleaner.calls.clear()
        r2 = self.make_cfg(Cfg(assistant={"wait_ready_s": 0}), cleaner)
        tid2 = self.file_job("formal")
        r2.submit(tid2)
        self.assertTrue(wait_for(lambda: self.store.get(tid2)["status"] == "done", timeout=3))
        self.assertEqual(len(cleaner.calls), 3)

    def test_llm_ready_within_cap_has_no_note(self):
        self.patches(2)
        cleaner = FakeCleaner(not_ready_first=1)
        r = self.make_cfg(Cfg(assistant={"wait_ready_s": 60}), cleaner)
        tid = self.file_job("formal")
        r.submit(tid)
        self.assertTrue(wait_for(lambda: self.store.get(tid)["status"] == "done", timeout=15))
        row = self.store.get(tid)
        self.assertEqual(row["error"], "")
        self.assertIn("CHUNK AT 30 SECONDS.", row["text"])

    def test_foreground_blocks_next_chunk(self):
        self.patches(3)
        gate = ForegroundGate()
        stt = FakeSTT(delay=0.2)
        r = self.make(stt=stt, gate=gate)
        tid = self.file_job()
        r.submit(tid)
        self.assertTrue(wait_for(lambda: len(stt.calls) >= 1))
        gate.begin()                       # dictation starts while chunk 1 runs
        t_begin = time.monotonic()
        time.sleep(1.0)
        self.assertEqual(len(stt.calls), 1, "a chunk started during foreground work")
        gate.end()
        t_end = time.monotonic()
        self.assertTrue(wait_for(lambda: self.store.get(tid)["status"] == "done"))
        self.assertGreaterEqual(stt.calls[1][0] - t_end, 0.45)                     # settle time honoured
        self.assertGreater(stt.calls[1][0], t_begin)

    def test_paused_blocks_and_unloaded_retries(self):
        self.patches(2)
        state = {"paused": True}
        stt = FakeSTT(unloaded_first=2)
        r = self.make(stt=stt, paused=lambda: state["paused"])
        tid = self.file_job()
        r.submit(tid)
        time.sleep(1.0)
        self.assertEqual(stt.calls, [])
        self.assertEqual(stt.unloaded, 2)
        state["paused"] = False
        self.assertTrue(wait_for(lambda: self.store.get(tid)["status"] == "done"))
        self.assertEqual(len(stt.calls), 2)

    def test_cancel_running_and_queued(self):
        self.patches(50)
        stt = FakeSTT(delay=0.05)
        r = self.make(stt=stt)
        a, b = self.file_job(), self.file_job()
        r.submit(a)
        r.submit(b)
        self.assertTrue(wait_for(lambda: len(stt.calls) >= 2))
        self.assertTrue(r.cancel(b))
        self.assertEqual(self.store.get(b)["status"], "cancelled")
        self.assertTrue(r.cancel(a))
        self.assertTrue(wait_for(lambda: self.store.get(a)["status"] == "cancelled"))
        self.assertEqual(self.store.load_progress(a), {})
        n = len(stt.calls)
        time.sleep(0.3)
        self.assertEqual(len(stt.calls), n)

    def test_shutdown_interrupts_and_resume_continues(self):
        self.patches(6)
        stt = FakeSTT(delay=0.1)
        r = self.make(stt=stt)
        tid = self.file_job()
        r.submit(tid)
        self.assertTrue(wait_for(lambda: len(stt.calls) >= 3))
        r.shutdown(5)
        self.assertEqual(self.store.get(tid)["status"], "interrupted")
        done_before = sum(1 for k in self.store.load_progress(tid) if k.startswith("stt:"))
        self.assertGreaterEqual(done_before, 3)
        stt2 = FakeSTT()
        r2 = self.make(stt=stt2)
        r2.submit(tid)
        self.assertTrue(wait_for(lambda: self.store.get(tid)["status"] == "done"))
        self.assertEqual(len(stt2.calls), 6 - done_before)                        # resumed, not restarted
        self.assertEqual(len(self.store.segments(tid)["segments"]), 6)

    def test_missing_file_fails(self):
        r = self.make()
        tid = self.store.create("file", "gone", source_path=str(self.tmp / "nope.mp3"))
        r.submit(tid)
        self.assertTrue(wait_for(lambda: self.store.get(tid)["status"] == "failed"))
        self.assertIn("not found", self.store.get(tid)["error"])

    def make_meeting_dir(self, name="m1", seconds=5):
        import soundfile as sf
        d = self.root / name
        d.mkdir()
        for key, val, off in (("mic", 1000, 0.02), ("system", 2000, 0.05)):
            sf.write(str(d / f"{key}.flac"), np.full(seconds * 48000, val, np.int16), 48000, subtype="PCM_16")
        meta = {"state": "stopped", "t0": time.time(), "duration_s": seconds,
                "tracks": {"mic": {"file": "mic.flac", "rate": 48000, "offset_s": 0.02, "samples": seconds * 48000},
                           "system": {"file": "system.flac", "rate": 48000, "offset_s": 0.05,
                                      "samples": seconds * 48000}}}
        rec.write_meta(d, meta)
        return d

    def test_meeting_job_merges_and_marks_echo(self):
        d = self.make_meeting_dir()
        with mock.patch.object(jobs, "plan_chunks", lambda audio, *a, **k: [[0, 16000], [3 * 16000, 4 * 16000]]):
            r = self.make()
            tid = self.store.create("meeting", "m", audio_dir=str(d))
            r.submit(tid)
            self.assertTrue(wait_for(lambda: self.store.get(tid)["status"] == "done"))
        row = self.store.get(tid)
        sj = self.store.segments(tid)
        self.assertEqual(len(sj["segments"]), 4)
        echo = [s for s in sj["segments"] if s.get("echo")]
        self.assertEqual(len(echo), 1)
        self.assertEqual(echo[0]["speaker"], "You")
        self.assertAlmostEqual(echo[0]["start"], 0.52, places=2)                   # offset_s applied
        self.assertIn("Others: The project deadline", row["text"])
        self.assertNotIn("You: The project deadline", row["text"])
        self.assertIn("You: Sounds good to me.", row["text"])
        self.assertEqual(row["info"]["echo_marked"], 1)
        self.assertEqual(rec.read_meta(d)["state"], "done")

    def test_meeting_job_finalizes_leftover_pcm(self):
        d = self.root / "pcm"
        d.mkdir()
        np.full(48000 * 2, 1000, np.int16).tofile(d / "mic.pcm")
        np.full(48000 * 2, 2000, np.int16).tofile(d / "system.pcm")
        rec.write_meta(d, {"state": "stopped", "t0": time.time(),
                           "tracks": {"mic": {"file": "mic.pcm", "rate": 48000, "offset_s": 0.0},
                                      "system": {"file": "system.pcm", "rate": 48000, "offset_s": 0.0}}})
        with mock.patch.object(jobs, "plan_chunks", lambda audio, *a, **k: [[0, 16000]]):
            r = self.make()
            tid = self.store.create("meeting", "m", audio_dir=str(d))
            r.submit(tid)
            self.assertTrue(wait_for(lambda: self.store.get(tid)["status"] == "done"))
        self.assertTrue((d / "mic.flac").exists())
        self.assertFalse((d / "mic.pcm").exists())


# ============================================================================ service façade
class ServiceTests(TmpCase):
    def make(self, cfg=None):
        cleaner = mock.MagicMock()
        cleaner.server.base_url = "http://127.0.0.1:1"
        cleaner.server.ready = False
        svc = jobs.TranscriptService(cfg or Cfg(meetings={"min_free_gb": 0.001}), FakeSTT(), FakeCleaner(),
                                     ForegroundGate(), db_path=self.tmp / "svc.db", recordings_root=self.root)
        svc.runner.retry_sleep_s = 0.02
        self.addCleanup(svc.shutdown, 5)
        return svc

    def test_import_file_and_export(self):
        wav = ROOT / "tests" / "audio" / "01_plain.wav"
        svc = self.make()
        with mock.patch.object(jobs, "plan_chunks", chunks_plan(1)):
            tid = svc.import_file(str(wav), style="raw")
            self.assertTrue(wait_for(lambda: svc.get(tid)["status"] == "done"))
        self.assertEqual(svc.get(tid)["title"], "01_plain")
        out = self.tmp / "o.md"
        svc.export(tid, "md", str(out))
        self.assertIn("# 01_plain", out.read_text(encoding="utf-8"))
        with self.assertRaises(FileNotFoundError):
            svc.import_file(str(self.tmp / "missing.wav"))

    def test_recover_interrupted_recording_and_jobs(self):
        svc = self.make()
        d = self.root / "20260101-101010"
        d.mkdir()
        np.full(48000, 1000, np.int16).tofile(d / "mic.pcm")
        np.full(48000, 2000, np.int16).tofile(d / "system.pcm")
        rec.write_meta(d, {"state": "recording", "pid": 0, "t0": time.time() - 100,
                           "tracks": {"mic": {"file": "mic.pcm", "rate": 48000, "offset_s": 0.0},
                                      "system": {"file": "system.pcm", "rate": 48000, "offset_s": 0.0}}})
        stale = svc.store.create("file", "stale", source_path=str(self.tmp / "gone.mp3"), status="transcribing")
        ids = svc.recover(auto_resume=False)
        self.assertEqual(len(ids), 2)
        row = svc.store.find_by_audio_dir(str(d))
        self.assertEqual(row["status"], "interrupted")
        self.assertEqual(svc.get(stale)["status"], "interrupted")
        self.assertEqual(rec.read_meta(d)["state"], "interrupted")
        self.assertTrue((d / "mic.flac").exists())
        with mock.patch.object(jobs, "plan_chunks", lambda audio, *a, **k: [[0, 8000]]):
            self.assertTrue(svc.resume(row["id"]))
            self.assertTrue(wait_for(lambda: svc.get(row["id"])["status"] == "done"))

    def test_meeting_start_stop_with_fake_recorder(self):
        svc = self.make()
        svc.recorder = FakeRecorder(svc.cfg, on_level=svc._levels, on_auto_stop=svc._auto_stopped)
        states = []
        svc.on_meeting = states.append
        with mock.patch.object(jobs, "plan_chunks", lambda audio, *a, **k: [[0, 8000]]):
            st = svc.start_meeting()
            self.assertTrue(st["recording"])
            time.sleep(0.6)
            tid = svc.stop_meeting()
            self.assertIsNotNone(tid)
            self.assertTrue(wait_for(lambda: svc.get(tid)["status"] == "done"))
        self.assertFalse(svc.meeting_state()["recording"])
        self.assertTrue(states[0]["recording"] and not states[-1]["recording"])
        row = svc.get(tid)
        self.assertEqual(row["kind"], "meeting")
        self.assertTrue(row["title"].startswith("Meeting "))
        # echo toggle re-renders
        txt_on = svc.set_include_echo(tid, True)
        self.assertEqual(svc.get(tid)["include_echo"], 1)
        self.assertIn("You:", txt_on)
        svc.set_include_echo(tid, False)
        self.assertNotIn("You: The project", svc.get(tid)["text"])
        d = Path(row["audio_dir"])
        self.assertTrue(d.exists())
        self.assertTrue(svc.delete(tid))
        self.assertFalse(d.exists())

    def test_shutdown_saves_running_recording(self):
        svc = self.make()
        svc.recorder = FakeRecorder(svc.cfg, on_level=svc._levels, on_auto_stop=svc._auto_stopped)
        svc.start_meeting()
        time.sleep(0.4)
        d = svc.recorder.rec_dir
        svc.shutdown(5)
        store = TranscriptStore(self.tmp / "svc.db", self.root)
        self.addCleanup(store.close)
        row = store.find_by_audio_dir(str(d))
        self.assertIsNotNone(row)
        self.assertEqual(row["status"], "queued")                 # recover() resumes it on next start
        self.assertEqual(rec.read_meta(d)["state"], "stopped")

    def test_ask_async(self):
        svc = self.make()
        tid = svc.store.create("file", "x")
        svc.store.update(tid, text="hello there", status="done")
        svc.assistant.http = FakeSession(lambda n, body: FakeResp(["Hi ", "summary."]))
        svc.assistant.is_ready_fn = lambda: True
        svc.assistant.base_url_fn = lambda: "http://x"
        got = []
        h = svc.ask(tid, "summary", on_done=lambda res, err: got.append((res, err)))
        self.assertTrue(h.wait(5))
        self.assertEqual(got[0][1], "")
        self.assertEqual(got[0][0]["output"], "Hi summary.")
        self.assertEqual(svc.list_ai_results(tid)[0]["output"], "Hi summary.")


# ============================================================================ regressions (verifier round 1)
class RegressionTests(TmpCase):
    def test_auto_stop_never_hits_a_newer_recording(self):
        r = FakeRecorder(Cfg(meetings={"min_free_gb": 0.001}))
        first = self.root / "first"
        r.start(first)
        time.sleep(0.3)
        r.stop(finalize=False)
        second = self.root / "second"
        r.start(second)
        meta = r.stop(finalize=False, reason="late auto-stop of the first", expect_dir=first)
        self.assertFalse(meta["stopped_now"])
        self.assertTrue(r.recording)                                # the new recording keeps going
        meta = r.stop(finalize=False)
        self.assertTrue(meta["stopped_now"])
        self.assertEqual(meta["dir"], str(second))
        self.assertFalse(r.stop()["stopped_now"])                   # second stop is a no-op

    def test_stop_does_not_hang_after_writer_failure(self):
        r = FakeRecorder(Cfg(meetings={"min_free_gb": 0.001}))
        r.start(self.root / "wfail")
        orig = r._handle

        def boom(*a):
            raise OSError("disk gone")
        r._handle = boom
        self.assertTrue(wait_for(lambda: not r.recording, 5))      # writer error -> auto-stop
        r._handle = orig
        t0 = time.monotonic()
        meta = r.stop()
        self.assertLess(time.monotonic() - t0, 2.0)
        self.assertIn("write error", r.error)
        self.assertEqual(meta["state"], "stopped")

    def test_cancel_during_last_chunk_wins(self):
        stt = FakeSTT(delay=0.5)
        r = jobs.JobRunner(stt, FakeCleaner(), ForegroundGate(), self.store, lambda: False)
        self.addCleanup(r.shutdown, 5)
        f = self.tmp / "a.wav"
        f.write_bytes(b"RIFF")
        with mock.patch.object(jobs, "decode_file_16k", lambda p, m: np.zeros(16000 * 5, np.int16)), \
                mock.patch.object(jobs, "plan_chunks", chunks_plan(1)):
            tid = self.store.create("file", "a", source_path=str(f))
            r.submit(tid)
            self.assertTrue(wait_for(lambda: len(stt.calls) == 1))
            self.assertTrue(r.cancel(tid))
            self.assertTrue(wait_for(lambda: self.store.get(tid)["status"] in ("cancelled", "done")))
        self.assertEqual(self.store.get(tid)["status"], "cancelled")
        self.assertEqual(self.store.get(tid)["text"], "")

    def test_resume_restarts_when_source_changed(self):
        stt = FakeSTT(delay=0.1)
        f = self.tmp / "a.wav"
        f.write_bytes(b"RIFF")
        with mock.patch.object(jobs, "decode_file_16k", lambda p, m: np.zeros(16000 * 200, np.int16)), \
                mock.patch.object(jobs, "plan_chunks", chunks_plan(5)):
            r = jobs.JobRunner(stt, FakeCleaner(), ForegroundGate(), self.store, lambda: False)
            tid = self.store.create("file", "a", source_path=str(f))
            r.submit(tid)
            self.assertTrue(wait_for(lambda: len(stt.calls) >= 2))
            r.shutdown(5)
            f.write_bytes(b"RIFF-different-and-longer")
            stt2 = FakeSTT()
            r2 = jobs.JobRunner(stt2, FakeCleaner(), ForegroundGate(), self.store, lambda: False)
            self.addCleanup(r2.shutdown, 5)
            r2.submit(tid)
            self.assertTrue(wait_for(lambda: self.store.get(tid)["status"] == "done"))
        self.assertEqual(len(stt2.calls), 5)

    def test_short_reply_after_their_words_is_not_echo(self):
        segs = jobs.merge_tracks([(1.5, 2.0, "Yes.")], [(0.0, 1.0, "Yes.")])
        self.assertEqual(jobs.mark_echo(segs), 0)
        segs = jobs.merge_tracks([(0.1, 0.9, "Yes.")], [(0.0, 1.0, "Yes.")])
        self.assertEqual(jobs.mark_echo(segs), 1)

    def test_map_reduce_keeps_every_note(self):
        tid = self.store.create("meeting", "long")
        text = "\n\n".join("Others: " + ("We discussed item %d at length. " % i) * 12 for i in range(60))
        self.store.update(tid, text=text)

        def responder(n, body):
            cap = body["max_tokens"]
            return FakeResp([(f"NOTE{n} " + "x" * 2000)[: cap * 4]])     # notes as long as allowed
        sess = FakeSession(responder)
        a = asst.Assistant(Cfg(cleanup={"ctx": 2048}).get, ForegroundGate(), self.store, lambda: "http://x",
                           lambda: True, session=sess)
        a.ask(tid, "summary")
        for body in sess.bodies:
            chars = sum(len(m["content"]) for m in body["messages"])
            self.assertLessEqual(chars / 3.5 + body["max_tokens"], 2048)
        # every map note made it into some combine/final request (nothing silently cut off)
        n_map = sum(1 for b in sess.bodies if "This is part" in b["messages"][1]["content"])
        later = " ".join(b["messages"][1]["content"] for b in sess.bodies
                         if "This is part" not in b["messages"][1]["content"])
        for i in range(1, n_map + 1):
            self.assertIn(f"NOTE{i} ", later)

    def test_prompt_too_long_is_rejected(self):
        tid = self.store.create("file", "x")
        self.store.update(tid, text="hello " * 50)
        a = asst.Assistant(Cfg(cleanup={"ctx": 8192}).get, ForegroundGate(), self.store, lambda: "http://x",
                           lambda: True, session=FakeSession(lambda n, b: FakeResp(["x"])))
        with self.assertRaises(asst.AssistantError):
            a.ask(tid, "q " * 14000)

    def test_stt_segment_filter_needs_uncertainty(self):
        from types import SimpleNamespace
        from localflow.stt import Transcriber
        segs = [SimpleNamespace(text=" Visit www.example.com for the notes.", start=0.0, end=4.0,
                                no_speech_prob=0.01, avg_logprob=-0.1),
                SimpleNamespace(text=" Thanks for watching!", start=5.0, end=6.0, no_speech_prob=0.6,
                                avg_logprob=-1.2),
                SimpleNamespace(text=" Thank you.", start=7.0, end=8.0, no_speech_prob=0.05, avg_logprob=-0.2)]
        t = Transcriber(Cfg(stt={"language": "en"}))
        t.model_name = "large-v3-turbo"
        t.model = SimpleNamespace(transcribe=lambda audio, **kw: (iter(segs), SimpleNamespace(language="en")))
        out = t.transcribe_segments(np.zeros(16000 * 10, np.float32), 100.0)
        self.assertEqual([x[2] for x in out], ["Visit www.example.com for the notes.", "Thank you."])
        self.assertEqual(out[0][:2], (100.0, 104.0))
        t.model = None
        with self.assertRaises(ModelUnloaded):
            t.transcribe_segments(np.zeros(16000, np.float32), 0)


class ServiceRegressionTests(TmpCase):
    def make(self):
        svc = jobs.TranscriptService(Cfg(meetings={"min_free_gb": 0.001}), FakeSTT(delay=0.3), FakeCleaner(),
                                     ForegroundGate(), db_path=self.tmp / "svc.db", recordings_root=self.root)
        svc.runner.retry_sleep_s = 0.02
        self.addCleanup(svc.shutdown, 5)
        return svc

    def test_recover_creates_row_for_stopped_recording_without_one(self):
        svc = self.make()
        d = self.root / "20260102-090000"
        d.mkdir()
        np.full(48000, 1000, np.int16).tofile(d / "mic.pcm")
        rec.write_meta(d, {"state": "stopped", "t0": time.time(), "dir": str(d),
                           "tracks": {"mic": {"file": "mic.pcm", "rate": 48000, "offset_s": 0.0}}})
        gone = self.root / "20260102-100000"
        gone.mkdir()
        rec.write_meta(gone, {"state": "stopped", "t0": time.time(), "tracks": {}})
        (gone / TranscriptStore.DELETED_MARKER).write_text("x")
        svc.recover(auto_resume=False)
        row = svc.store.find_by_audio_dir(str(d))
        self.assertIsNotNone(row)
        self.assertEqual(row["status"], "interrupted")
        self.assertTrue((d / "mic.flac").exists())
        self.assertFalse(gone.exists())                             # pending delete finished, not resurrected
        svc.recover(auto_resume=False)
        self.assertEqual(len([r for r in svc.list() if r["audio_dir"] == str(d)]), 1)

    def test_delete_while_job_runs_defers_audio_removal(self):
        svc = self.make()
        d = self.root / "20260103-090000"
        d.mkdir()
        import soundfile as sf
        sf.write(str(d / "mic.flac"), np.full(48000 * 3, 1000, np.int16), 48000, subtype="PCM_16")
        rec.write_meta(d, {"state": "stopped", "t0": time.time(), "dir": str(d),
                           "tracks": {"mic": {"file": "mic.flac", "rate": 48000, "offset_s": 0.0,
                                              "samples": 48000 * 3}}})
        tid = svc.store.create("meeting", "m", audio_dir=str(d))
        release = threading.Event()
        orig = svc.runner._stt_chunk

        def slow(*a, **k):
            release.wait(10)                                        # ignores cancel for a while (long unit)
            return orig(*a, **k)
        svc.runner._stt_chunk = slow
        with mock.patch.object(jobs, "plan_chunks", lambda audio, *a, **k: [[0, 16000]]), \
                mock.patch.object(time, "sleep", lambda s: None):
            svc.runner.submit(tid)
            self.assertTrue(wait_for(lambda: svc.runner.current() == tid))
            self.assertTrue(svc.delete(tid))
        self.assertIsNone(svc.get(tid))
        self.assertTrue(d.exists())                                 # job still owns it
        release.set()
        self.assertTrue(wait_for(lambda: not d.exists(), 10))

    def test_shutdown_cancels_asks(self):
        svc = self.make()
        tid = svc.store.create("file", "x")
        svc.store.update(tid, text="hello there", status="done")
        svc.assistant.is_ready_fn = lambda: False                  # waits for the model forever
        svc.assistant.ready_poll_s = 0.05
        h = svc.ask(tid, "summary")
        time.sleep(0.2)
        svc.shutdown(5)
        self.assertFalse(h.thread.is_alive())
        self.assertTrue(h.stop.is_set())


# ============================================================================ regressions (verifier round 2)
class Regression2Tests(TmpCase):
    def test_stuck_writer_blocks_reuse_and_queueing(self):
        r = FakeRecorder(Cfg(meetings={"min_free_gb": 0.001}))
        d = self.root / "stuck"
        r.start(d)
        time.sleep(0.2)
        release = threading.Event()
        orig = r._handle
        r._handle = lambda *a: (release.wait(30), orig(*a))
        time.sleep(0.2)
        real_join = threading.Thread.join
        with mock.patch.object(threading.Thread, "join", lambda self, timeout=None: real_join(self, 0.3)):
            meta = r.stop()
        self.assertTrue(meta["stopped_now"] and meta["writer_busy"])
        self.assertEqual(rec.read_meta(d)["state"], "recording")    # left for startup recovery
        with self.assertRaises(rec.RecorderError):
            r.start(self.root / "next")
        release.set()
        self.assertTrue(wait_for(lambda: not r._stuck_writer.is_alive(), 5))
        r.start(self.root / "next")
        r.stop()

    def test_delete_marks_dir_before_row_goes(self):
        d = self.root / "20260104-090000"
        d.mkdir()
        rec.write_meta(d, {"state": "stopped", "t0": time.time(), "tracks": {}})
        tid = self.store.create("meeting", "m", audio_dir=str(d))
        self.store.delete(tid, delete_audio=False)                  # e.g. crash before the dir was removed
        self.assertTrue((d / TranscriptStore.DELETED_MARKER).exists())
        svc = jobs.TranscriptService(Cfg(), FakeSTT(), FakeCleaner(), ForegroundGate(),
                                     db_path=self.tmp / "t.db", recordings_root=self.root)
        self.addCleanup(svc.shutdown, 5)
        svc.recover(auto_resume=False)
        self.assertFalse(d.exists())
        self.assertEqual(svc.list(), [])

    def test_ai_result_not_saved_for_deleted_transcript(self):
        tid = self.store.create("file", "x")
        self.store.delete(tid)
        self.assertEqual(self.store.add_ai_result(tid, "summary", "", "late"), 0)
        self.assertEqual(self.store.list_ai_results(tid), [])

    def test_delete_cancels_running_ask(self):
        svc = jobs.TranscriptService(Cfg(), FakeSTT(), FakeCleaner(), ForegroundGate(),
                                     db_path=self.tmp / "s.db", recordings_root=self.root)
        self.addCleanup(svc.shutdown, 5)
        tid = svc.store.create("file", "x")
        svc.store.update(tid, text="hello there", status="done")
        svc.assistant.is_ready_fn = lambda: False
        svc.assistant.ready_poll_s = 0.05
        got = []
        h = svc.ask(tid, "summary", on_done=lambda res, err: got.append(err))
        time.sleep(0.2)
        svc.delete(tid)
        self.assertTrue(h.wait(5))
        self.assertTrue(h.stop.is_set())
        self.assertEqual(svc.store.list_ai_results(tid), [])

    def test_confident_repetition_is_kept(self):
        from types import SimpleNamespace
        from localflow.stt import Transcriber
        segs = [SimpleNamespace(text=" One two, one two, one two.", start=0.0, end=3.0, no_speech_prob=0.01,
                                avg_logprob=-0.1),
                SimpleNamespace(text=" the the the the the the", start=4.0, end=6.0, no_speech_prob=0.5,
                                avg_logprob=-1.0)]
        t = Transcriber(Cfg(stt={}))
        t.model_name = "large-v3-turbo"
        t.model = SimpleNamespace(transcribe=lambda audio, **kw: (iter(segs), None))
        self.assertEqual([x[2] for x in t.transcribe_segments(np.zeros(16000 * 8, np.float32))],
                         ["One two, one two, one two."])

    def test_recover_skips_dir_of_running_job(self):
        svc = jobs.TranscriptService(Cfg(), FakeSTT(), FakeCleaner(), ForegroundGate(),
                                     db_path=self.tmp / "r.db", recordings_root=self.root)
        self.addCleanup(svc.shutdown, 5)
        d = self.root / "20260105-090000"
        d.mkdir()
        np.full(48000, 1000, np.int16).tofile(d / "mic.pcm")
        rec.write_meta(d, {"state": "transcribing", "t0": time.time(),
                           "tracks": {"mic": {"file": "mic.pcm", "rate": 48000, "offset_s": 0.0}}})
        tid = svc.store.create("meeting", "m", audio_dir=str(d), status="transcribing")
        with mock.patch.object(svc.runner, "pending", lambda: [tid]):
            svc.recover(auto_resume=False)
        self.assertEqual(rec.read_meta(d)["state"], "transcribing")
        self.assertTrue((d / "mic.pcm").exists())


if __name__ == "__main__":
    unittest.main()
