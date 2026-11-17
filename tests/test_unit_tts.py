"""Offline tests for spoken command-mode answers: text clean-up, the TTS server protocol / playback (fake model,
fake sounddevice), the app-side process manager (fake process running the real Server class) and the app wiring.
No GPU, no model, no speakers.

   .venv\\Scripts\\python.exe tests\\test_unit_tts.py
"""
import json
import os
import queue
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from localflow import tts  # noqa: E402
from localflow import tts_server as srv  # noqa: E402


def wait_for(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


# ============================================================================ text
class SpeechText(unittest.TestCase):
    def s(self, t):
        return tts.speech_text(t)

    def test_markdown_stripped(self):
        self.assertEqual(self.s("**Bold** and *italic* with `code` and [a link](http://x.com)."),
                         "Bold and italic with code and a link.")
        self.assertEqual(self.s("## Steps\n- Preheat oven\n- Mix flour & sugar\n1. Bake"),
                         "Steps. Preheat oven. Mix flour and sugar. Bake.")
        self.assertEqual(self.s("Here:\n```python\nprint(1)\n```\nDone"), "Here: Done.")
        self.assertEqual(self.s("| a | b |\n|---|---|\n| 1 | 2 |"), "a, b. 1, 2.")
        self.assertEqual(self.s("See https://example.com/x?y=1 now"), "See a link now.")

    def test_symbols_and_units(self):
        self.assertEqual(self.s("It is 72°F, about 22 °C."), "It is 72 degrees Fahrenheit, about 22 degrees Celsius.")
        self.assertEqual(self.s("A 15% tip on $84 is $12.60."), "A 15 percent tip on 84 dollars is 12.60 dollars.")
        self.assertEqual(self.s("It costs $3 million."), "It costs 3 million dollars.")
        self.assertEqual(self.s("Drive 60 mph or 100 km/h for 5 km."),
                         "Drive 60 miles per hour or 100 kilometers per hour for 5 kilometers.")
        self.assertEqual(self.s("Weight: 150 lbs, 2 kg"), "Weight: 150 pounds, 2 kilograms.")
        self.assertEqual(self.s("Room is 10ft by 12 ft, 120 sq ft."), "Room is 10 feet by 12 feet, 120 square feet.")

    def test_feet_inches_only_when_unambiguous(self):
        self.assertEqual(self.s("He is 5'11\" tall."), "He is 5 feet 11 inches tall.")
        self.assertEqual(self.s("The board is 12 in. wide."), "The board is 12 inches wide.")
        self.assertEqual(self.s("It measures 12 in."), "It measures 12 inches.")
        self.assertEqual(self.s("A 24in monitor."), "A 24 inches monitor.")
        self.assertEqual(self.s("We got 12 in total."), "We got 12 in total.")         # not "12 inches total"
        self.assertEqual(self.s("I'll be there in 5 minutes."), "I'll be there in 5 minutes.")
        self.assertEqual(self.s("The '90s were great."), "The '90s were great.")
        self.assertEqual(self.s("Page 4 in the book."), "Page 4 in the book.")

    def test_plain_words_untouched(self):
        self.assertEqual(self.s("Jane Austen"), "Jane Austen.")
        self.assertEqual(self.s("I am 5 km away and 3 in a row"), "I am 5 kilometers away and 3 in a row.")
        self.assertEqual(self.s("Call me at 5 and bring 2 cm of tape"), "Call me at 5 and bring 2 centimeters of tape.")

    def test_trimmed_to_about_30_seconds_on_a_sentence(self):
        long = "Sentence number one is right here. " * 40
        out = self.s(long)
        self.assertLessEqual(len(out.split()), tts.MAX_WORDS)
        self.assertTrue(out.endswith("here."))
        words = self.s("word " * 300)
        self.assertEqual(len(words.split()), tts.MAX_WORDS)
        self.assertTrue(words.endswith("."))
        self.assertLessEqual(tts.estimate_seconds(out), tts.MAX_SPEECH_S)

    def test_answers_that_start_with_a_number_keep_it(self):
        self.assertEqual(self.s("42. That is the answer."), "42. That is the answer.")
        self.assertEqual(self.s("50. The US has 50 states."), "50. The US has 50 states.")
        self.assertEqual(self.s("1. one\n2. two"), "one. two.")             # a real list loses its markers

    def test_more_symbol_cases(self):
        self.assertEqual(self.s("It costs $1,299, plus tax."), "It costs 1,299 dollars, plus tax.")
        self.assertEqual(self.s("The answer is '42'."), "The answer is '42'.")
        self.assertEqual(self.s("12 * 4 = 48"), "12 times 4 equals 48.")
        self.assertEqual(self.s("5 > 3 and x < 10"), "5 is greater than 3 and x is less than 10.")
        self.assertEqual(self.s("-3"), "minus 3.")
        self.assertEqual(self.s("pages 3-5"), "pages 3-5.")
        self.assertEqual(self.s("2 * 3 * 4 = 24"), "2 times 3 times 4 equals 24.")
        self.assertEqual(self.s("2 × 3 × 4"), "2 times 3 times 4.")
        self.assertEqual(self.s("x >= 5 and y <= 2"), "x is at least 5 and y is at most 2.")

    def test_menu_paths_are_not_comparisons(self):
        self.assertEqual(self.s("Open Settings > Privacy > Microphone and turn it on."),
                         "Open Settings, then Privacy, then Microphone and turn it on.")
        self.assertEqual(self.s("Go to File > Save As"), "Go to File, then Save As.")

    def test_closing_quote_after_number_is_not_feet(self):
        self.assertEqual(self.s("Press 'F5' to refresh"), "Press 'F5' to refresh.")
        self.assertEqual(self.s("'Windows 11' and 'Python 3'"), "'Windows 11' and 'Python 3'.")
        self.assertEqual(self.s("She said 'I have 42' and left."), "She said 'I have 42' and left.")
        self.assertEqual(self.s("It's 6' tall. Then 'ok'."), "It's 6 feet tall. Then 'ok'.")

    def test_number_first_line_kept_next_to_bullets(self):
        self.assertEqual(self.s("42. That is the answer.\n- Douglas Adams wrote it.\n- Published in 1979."),
                         "42. That is the answer. Douglas Adams wrote it. Published in 1979.")

    def test_empty(self):
        self.assertEqual(self.s(""), "")
        self.assertEqual(self.s("```\ncode only\n```"), "")
        self.assertEqual(tts.estimate_seconds(""), 0.0)


# ============================================================================ protocol
class Protocol(unittest.TestCase):
    def test_parse_command(self):
        p = srv.parse_command
        self.assertIsNone(p(""))
        self.assertIsNone(p("not json"))
        self.assertIsNone(p('{"cmd": "dance"}'))
        self.assertIsNone(p('["speak"]'))
        self.assertIsNone(p('{"cmd": "speak", "text": "  "}'))
        self.assertEqual(p('{"cmd": "stop"}'), {"cmd": "stop"})
        d = p('{"cmd": "speak", "id": "7", "text": "hi", "voice": "Uncle Fu", "volume": 3}')
        self.assertEqual((d["id"], d["voice"], d["volume"], d["device"]), (7, "uncle_fu", 1.0, ""))
        d = p('{"cmd": "speak", "id": null, "text": "hi", "volume": "x"}')
        self.assertEqual((d["id"], d["voice"], d["volume"]), (0, "ryan", 1.0))

    def test_events_are_json_lines(self):
        out = []
        e = srv.Events(SimpleNamespace(write=out.append, flush=lambda: None))
        e.emit("done", id=3, audio_s=1.5, reason="end")
        self.assertEqual(json.loads(out[0]), {"ev": "done", "id": 3, "audio_s": 1.5, "reason": "end"})
        self.assertTrue(out[0].endswith("\n"))

    def test_events_survive_closed_pipe(self):
        def boom(_):
            raise OSError("pipe closed")
        srv.Events(SimpleNamespace(write=boom, flush=lambda: None)).emit("ready")

    def test_max_new_tokens_is_30s(self):
        self.assertEqual(srv.max_new_tokens(), 375)


class FakePlayer:
    def __init__(self, req):
        self.req = req
        self.opened = self.closed = False
        self.abort = None
        self.chunks = []

    def open(self):
        self.opened = True

    def write(self, a):
        self.chunks.append(np.asarray(a))

    def drain(self, cancel, timeout):
        return not cancel.wait(0.01)

    def close(self, abort=False):
        self.closed, self.abort = True, abort


class ServerCore(unittest.TestCase):
    def make(self, gen):
        self.events = []
        self.players = []
        ev = srv.Events(SimpleNamespace(write=lambda s: self.events.append(json.loads(s)), flush=lambda: None))

        def mk(req):
            p = FakePlayer(req)
            self.players.append(p)
            return p
        return srv.Server(ev, gen, mk, voices=["ryan", "aiden"])

    def kinds(self, rid=None):
        return [e["ev"] for e in self.events if rid is None or e.get("id") == rid]

    def test_speak_streams_chunks(self):
        def gen(text, voice, cancel):
            self.voice = voice
            for _ in range(3):
                yield np.full(2400, 0.1, np.float32)
        s = self.make(gen)
        s.speak({"id": 1, "text": "hi", "voice": "aiden", "volume": 1.0, "device": ""})
        self.assertEqual(self.kinds(1), ["started", "playing", "done"])
        self.assertEqual(self.events[-1]["reason"], "end")
        self.assertAlmostEqual(self.events[-1]["audio_s"], 0.3, places=2)
        self.assertEqual(self.voice, "aiden")
        self.assertTrue(self.players[0].opened and self.players[0].closed)
        self.assertFalse(self.players[0].abort)

    def test_unknown_voice_falls_back(self):
        def gen(text, voice, cancel):
            self.voice = voice
            yield np.zeros(10, np.float32)
        s = self.make(gen)
        s.speak({"id": 1, "text": "hi", "voice": "nobody", "volume": 1.0, "device": ""})
        self.assertEqual(self.voice, "ryan")

    def test_stop_mid_generation(self):
        def gen(text, voice, cancel):
            for i in range(100):
                if i == 2:
                    s.handle({"cmd": "stop"})
                yield np.zeros(2400, np.float32)
        s = self.make(gen)
        s.speak({"id": 4, "text": "hi", "voice": "ryan", "volume": 1.0, "device": ""})
        self.assertEqual(self.events[-1]["reason"], "stopped")
        self.assertLessEqual(len(self.players[0].chunks), 3)
        self.assertTrue(self.players[0].abort)                 # playback cut right away

    def test_stop_silences_player_immediately(self):
        """The generator sees cancel only between chunks; the stdin thread must mute the output at once."""
        in_chunk = threading.Event()
        go_on = threading.Event()

        def gen(text, voice, cancel):
            yield np.zeros(2400, np.float32)
            in_chunk.set()
            go_on.wait(2)                       # a slow chunk is being generated
            yield np.zeros(2400, np.float32)
        s = self.make(gen)
        silenced = []
        FakePlayer.silence = lambda p: silenced.append(time.monotonic())
        try:
            t = threading.Thread(target=s.speak, args=({"id": 5, "text": "hi", "voice": "ryan", "volume": 1.0,
                                                        "device": ""},))
            t.start()
            self.assertTrue(in_chunk.wait(2))
            t0 = time.monotonic()
            s.handle({"cmd": "stop"})
            self.assertTrue(silenced and silenced[0] - t0 < 0.05)    # before the chunk finished
            go_on.set()
            t.join(2)
            self.assertEqual(self.events[-1]["reason"], "stopped")
        finally:
            del FakePlayer.silence

    def test_cap_at_30s(self):
        def gen(text, voice, cancel):
            while True:
                yield np.zeros(srv.SR, np.float32)              # 1 s per chunk, forever
        s = self.make(gen)
        s.speak({"id": 2, "text": "hi", "voice": "ryan", "volume": 1.0, "device": ""})
        self.assertEqual(self.events[-1]["reason"], "cap")
        self.assertAlmostEqual(self.events[-1]["audio_s"], srv.MAX_AUDIO_S, places=1)

    def test_generation_error_reported(self):
        def gen(text, voice, cancel):
            yield np.zeros(100, np.float32)
            raise RuntimeError("cuda oops")
        s = self.make(gen)
        s.speak({"id": 3, "text": "hi", "voice": "ryan", "volume": 1.0, "device": ""})
        self.assertIn("error", self.kinds(3))
        self.assertEqual(self.events[-1], {"ev": "done", "id": 3, "audio_s": 0.0, "reason": "error"})

    def test_run_loop_preempts_and_skips_stale(self):
        started = threading.Event()
        release = threading.Event()

        def gen(text, voice, cancel):
            if text == "long":
                started.set()
                while not cancel.is_set():
                    yield np.zeros(240, np.float32)
                    time.sleep(0.005)
            else:
                yield np.zeros(240, np.float32)
        s = self.make(gen)
        lines = queue.Queue()

        def stdin():
            while True:
                ln = lines.get()
                if ln is None:
                    return
                yield ln
        threading.Thread(target=s.read_commands, args=(stdin(),), daemon=True).start()
        loop = threading.Thread(target=s.run, daemon=True)
        loop.start()
        lines.put(json.dumps({"cmd": "speak", "id": 1, "text": "long"}) + "\n")
        self.assertTrue(started.wait(2))
        lines.put(json.dumps({"cmd": "speak", "id": 2, "text": "short"}) + "\n")
        self.assertTrue(wait_for(lambda: ("done", 2) in [(e["ev"], e.get("id")) for e in self.events]))
        done1 = next(e for e in self.events if e["ev"] == "done" and e["id"] == 1)
        self.assertEqual(done1["reason"], "stopped")
        # a stop drops a queued request that has not started yet
        s.handle({"cmd": "stop"})
        s.jobs.put({"cmd": "speak", "id": 3, "text": "short", "voice": "ryan", "volume": 1.0, "device": ""})
        time.sleep(0.2)
        self.assertNotIn(3, [e.get("id") for e in self.events])
        lines.put(None)                                          # EOF -> quit
        loop.join(2)
        self.assertFalse(loop.is_alive())
        release.set()


# ============================================================================ playback (fake sounddevice)
class FakeStream:
    def __init__(self, **kw):
        self.kw = kw
        self.latency = 0.01
        self.calls = []

    def start(self):
        self.calls.append("start")

    def stop(self):
        self.calls.append("stop")

    def abort(self):
        self.calls.append("abort")

    def close(self):
        self.calls.append("close")


class FakeSd:
    def __init__(self, devices=None, apis=None):
        self.devices = devices or [
            {"name": "Microphone (USB)", "max_output_channels": 0, "hostapi": 1},
            {"name": "Speakers (Realtek(R) Audio)", "max_output_channels": 2, "hostapi": 1},
            {"name": "Headphones (Arctis Nova)", "max_output_channels": 2, "hostapi": 1},
            {"name": "Speakers (Realtek(R) Audi", "max_output_channels": 2, "hostapi": 0},
        ]
        self.apis = apis or [{"name": "MME", "default_output_device": 3},
                             {"name": "Windows WASAPI", "default_output_device": 1}]
        self.default = SimpleNamespace(device=(0, 3))
        self.inits = 0
        self.streams = []

    def query_hostapis(self):
        return self.apis

    def query_devices(self):
        return self.devices

    def _terminate(self):
        pass

    def _initialize(self):
        self.inits += 1

    def WasapiSettings(self, auto_convert=False):
        return ("wasapi", auto_convert)

    def OutputStream(self, **kw):
        s = FakeStream(**kw)
        self.streams.append(s)
        return s


class Playback(unittest.TestCase):
    def test_pick_output(self):
        sd = FakeSd()
        self.assertEqual(srv.pick_output(sd, ""), (1, True))                     # WASAPI default
        self.assertEqual(srv.pick_output(sd, "Headphones (Arctis Nova)"), (2, True))
        self.assertEqual(srv.pick_output(sd, "arctis nova"), (2, True))          # contains
        self.assertEqual(srv.pick_output(sd, "Gone device"), (1, True))          # unplugged -> default
        self.assertEqual(srv.pick_output(sd, "Microphone (USB)"), (1, True))     # inputs never picked

    def test_player_feeds_callback_and_cuts(self):
        sd = FakeSd()
        p = srv.SdPlayer(sd, "Headphones (Arctis Nova)", volume=0.5)
        p.open()
        st = sd.streams[0]
        self.assertEqual(sd.inits, 1)                                             # device list refreshed
        self.assertEqual(st.kw["device"], 2)
        self.assertEqual(st.kw["samplerate"], srv.SR)
        self.assertEqual(st.kw["extra_settings"], ("wasapi", True))               # auto_convert for 24 kHz
        p.write(np.full(300, 0.8, np.float32))
        p.write(np.full(300, -0.4, np.float32))
        out = np.zeros((400, 2), np.float32)
        st.kw["callback"](out, 400, None, None)
        self.assertTrue(np.allclose(out[:300, 0], 0.4))                           # volume applied
        self.assertTrue(np.allclose(out[300:400, 0], -0.2))
        self.assertTrue(np.allclose(out[:, 1], out[:, 0]))                        # both channels
        self.assertEqual(p.pending(), 200)
        out2 = np.ones((400, 1), np.float32)
        st.kw["callback"](out2, 400, None, None)
        self.assertTrue(np.allclose(out2[200:], 0.0))                             # underrun -> silence
        self.assertTrue(p.drain(threading.Event(), timeout=1))
        p.write(np.ones(1000, np.float32))
        p.close(abort=True)
        self.assertEqual(st.calls, ["start", "abort", "close"])
        self.assertEqual(p.pending(), 1000 - 0)        # nothing more is played after the cut
        self.assertIsNone(p.stream)

    def test_silence_mutes_callback_and_drops_writes(self):
        sd = FakeSd()
        p = srv.SdPlayer(sd)
        p.open()
        p.write(np.ones(1000, np.float32))
        p.silence()
        p.write(np.ones(1000, np.float32))
        out = np.ones((400, 1), np.float32)
        sd.streams[0].kw["callback"](out, 400, None, None)
        self.assertTrue(np.allclose(out, 0.0))
        self.assertEqual(p.pending(), 0)

    def test_drain_cancelled(self):
        sd = FakeSd()
        p = srv.SdPlayer(sd)
        p.open()
        p.write(np.ones(48000, np.float32))
        c = threading.Event()
        c.set()
        self.assertFalse(p.drain(c, timeout=1))


# ============================================================================ manager (fake process, real Server)
class _Pipe:
    """stdout of the fake process: an iterator of event lines."""

    def __init__(self):
        self.q = queue.Queue()

    def write(self, s):
        self.q.put(s.encode() if isinstance(s, str) else s)

    def flush(self):
        pass

    def __iter__(self):
        while True:
            x = self.q.get()
            if x is None:
                return
            yield x

    def close(self):
        self.q.put(None)


class _Stdin:
    def __init__(self, proc):
        self.proc = proc
        self.closed = False

    def write(self, b):
        if self.closed or self.proc.returncode is not None:
            raise OSError("broken pipe")
        for ln in b.decode().splitlines():
            self.proc.lines.append(json.loads(ln))
            self.proc.inq.put(ln + "\n")

    def flush(self):
        pass

    def close(self):
        self.closed = True
        self.proc.inq.put(None)


class FakeProc:
    """Runs the real tts_server.Server (fake model, fake player) in threads behind a Popen-like object."""
    pid = None
    count = 0

    def __init__(self, ready=True, gen=None, crash_after_ready=False):
        FakeProc.count += 1
        self.lines = []
        self.inq = queue.Queue()
        self.stdin = _Stdin(self)
        self.stdout = _Pipe()
        self.returncode = None
        self.players = []
        self.killed = False
        ev = srv.Events(self.stdout)

        def default_gen(text, voice, cancel):
            for _ in range(200):
                if cancel.is_set():
                    return
                time.sleep(0.01)
                yield np.zeros(240, np.float32)

        def mk(req):
            p = FakePlayer(req)
            self.players.append(p)
            return p
        self.server = srv.Server(ev, gen or default_gen, mk, voices=["ryan"])

        def stdin_iter():
            while True:
                ln = self.inq.get()
                if ln is None:
                    return
                yield ln

        def main():
            if ready:
                ev.emit("ready", voices=["ryan"], sr=24000, load_s=0.1, warmup_s=0.1)
                if crash_after_ready:
                    self._exit(1)
                    return
                threading.Thread(target=self.server.read_commands, args=(stdin_iter(),), daemon=True).start()
                self.server.run()
                self._exit(0)
            else:
                self._exit(2)
        threading.Thread(target=main, daemon=True).start()

    def _exit(self, code):
        if self.returncode is None:
            self.returncode = code
        self.stdout.close()

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        end = time.monotonic() + (timeout or 5)
        while self.returncode is None and time.monotonic() < end:
            time.sleep(0.01)
        if self.returncode is None:
            raise TimeoutError
        return self.returncode

    def kill(self):
        self.killed = True
        self.server.handle({"cmd": "quit"})
        self._exit(-9)


class Cfg:
    def __init__(self, **over):
        self.d = {"command_mode.speak_answers": True, "command_mode.enabled": True,
                  "command_mode.answer_questions": True, "command_mode.tts_voice": "Ryan",
                  "command_mode.tts_device": "", "command_mode.tts_volume": 0.7, "enabled": True}
        self.d.update(over)

    def get(self, k, default=None):
        return self.d.get(k, default)


class Manager(unittest.TestCase):
    def make(self, cfg=None, procs=None, blocked=lambda: False, installed=True, **kw):
        self.spawned = []
        procs = list(procs or [])

        def popen(cmd, **pkw):
            self.cmd, self.pkw = cmd, pkw
            p = procs.pop(0)() if procs else FakeProc()
            self.spawned.append(p)
            return p
        self.cfg = cfg or Cfg()
        kw.setdefault("log_path", Path(tempfile.mkdtemp()) / "tts-server.log")     # never the real log
        return tts.TtsManager(self.cfg, blocked=blocked, popen=popen, is_installed=lambda: installed,
                              restart_delays=kw.pop("restart_delays", (0.01, 0.01, 0.01)), **kw)

    def ready(self, m):
        m.start()
        self.assertTrue(wait_for(lambda: m.ready), m.state)

    def tearDown(self):
        m = getattr(self, "m", None)
        if m is not None:
            m.shutdown("test", wait=False)

    def test_speak_waits_for_ready(self):
        self.m = m = self.make(procs=[lambda: FakeProc(ready=False)], restart_delays=())
        self.assertEqual(m.speak("hello", rid=1), 0.0)                 # not started
        m.start()
        self.assertEqual(m.speak("hello", rid=1), 0.0)                 # starting, not READY
        self.assertTrue(wait_for(lambda: m.state == "failed"))         # load failed, no restarts left

    def test_speak_sends_request_and_tracks_audible(self):
        self.m = m = self.make()
        self.ready(m)
        est = m.speak("It's **72°F**.", rid=1)
        self.assertGreater(est, 0)
        p = self.spawned[0]
        req = [x for x in p.lines if x["cmd"] == "speak"][0]
        self.assertEqual(req["text"], "It's 72 degrees Fahrenheit.")   # normalised before sending
        self.assertEqual((req["voice"], req["device"], req["volume"]), ("Ryan", "", 0.7))
        self.assertTrue(wait_for(lambda: m._audible))
        self.assertTrue(m.speaking)
        self.assertTrue(m.recording_started(2))                         # audible -> skip pre-roll
        self.assertEqual(p.lines[-1], {"cmd": "stop"})
        self.assertFalse(m.speaking)
        self.assertTrue(wait_for(lambda: p.players and p.players[0].abort is True))

    def test_command_line_and_window_flags(self):
        self.m = m = self.make(wav_out="C:/x")
        self.ready(m)
        self.assertEqual(self.cmd[0], str(tts.TTS_VENV_PY))
        self.assertIn(str(tts.TTS_SERVER), self.cmd)
        self.assertEqual(self.cmd[self.cmd.index("--wav-out") + 1], "C:/x")
        self.assertEqual(self.pkw["creationflags"] & tts.CREATE_NO_WINDOW, tts.CREATE_NO_WINDOW)
        self.assertNotIn("PYTHONPATH", self.pkw["env"])
        self.assertEqual(self.pkw["env"]["HF_HUB_OFFLINE"], "1")

    def test_preroll_window(self):
        self.m = m = self.make()
        self.ready(m)
        self.assertFalse(m.recording_started(1))                        # nothing was playing
        m.speak("hello there", rid=1)
        self.assertTrue(wait_for(lambda: m._audible))
        self.assertTrue(wait_for(lambda: not m.speaking, timeout=5))    # finished by itself
        self.assertTrue(m.recording_started(2))                         # within 0.5 s of the end
        m._audible_end = time.monotonic() - 1.0
        self.assertFalse(m.recording_started(3))

    def test_stale_rid_dropped(self):
        self.m = m = self.make()
        self.ready(m)
        m.recording_started(5)
        self.assertEqual(m.speak("old answer", rid=4), 0.0)
        self.assertFalse([x for x in self.spawned[0].lines if x["cmd"] == "speak"])
        self.assertGreater(m.speak("new answer", rid=5), 0)

    def test_newer_answer_preempts(self):
        self.m = m = self.make()
        self.ready(m)
        m.speak("first answer here", rid=1)
        self.assertTrue(wait_for(lambda: m._audible))
        m.speak("second answer", rid=1)
        p = self.spawned[0]
        self.assertTrue(wait_for(lambda: len(p.players) == 2))
        self.assertTrue(wait_for(lambda: p.players[0].abort is True))   # first one cut
        self.assertEqual(m._speaking, 2)

    def test_disabled_inert_missing_blocked_never_spawn(self):
        for kw in ({"cfg": Cfg(**{"command_mode.speak_answers": False})},
                   {"cfg": Cfg(**{"command_mode.answer_questions": False})},
                   {"inert": True}, {"installed": False}, {"blocked": lambda: True}):
            m = self.make(**kw)
            with self.assertLogs("localflow.tts", "INFO") if kw.get("installed") is False else _nullctx():
                m.start()
            time.sleep(0.05)
            self.assertEqual(self.spawned, [], kw)
            self.assertEqual(m.state, "off")
            self.assertEqual(m.speak("hi", rid=1), 0.0)

    def test_missing_install_logged_once(self):
        m = self.make(installed=False)
        with self.assertLogs("localflow.tts", "INFO") as lg:
            m.start()
            m.start()
            tts.log.info("marker")
        self.assertEqual(sum("not installed" in x for x in lg.output), 1)

    def test_shutdown_kills_and_frees(self):
        self.m = m = self.make()
        self.ready(m)
        p = self.spawned[0]
        m.shutdown("game mode")
        self.assertEqual(m.state, "off")
        self.assertIsNotNone(p.returncode)
        self.assertEqual(p.lines[-1], {"cmd": "quit"})
        time.sleep(0.1)
        self.assertEqual(len(self.spawned), 1)                          # no restart after an intended stop
        self.assertEqual(m.speak("hi", rid=9), 0.0)
        m.start()                                                       # game over / turned back on
        self.assertTrue(wait_for(lambda: m.ready))
        self.assertEqual(len(self.spawned), 2)

    def test_shutdown_while_spawning_leaves_nothing(self):
        gate = threading.Event()

        def slow():
            gate.wait(2)
            return FakeProc()
        self.m = m = self.make(procs=[slow])
        m.start()
        time.sleep(0.05)
        m.shutdown("game mode")
        gate.set()
        self.assertTrue(wait_for(lambda: self.spawned and self.spawned[0].returncode is not None))
        self.assertEqual(m.state, "off")
        self.assertIsNone(m._proc)

    def test_crash_restarts_capped_at_three(self):
        self.m = m = self.make(procs=[lambda: FakeProc(crash_after_ready=True)] * 5)
        m.start()
        self.assertTrue(wait_for(lambda: m.state == "failed", timeout=5))
        self.assertEqual(len(self.spawned), 4)                          # first run + 3 restarts
        m.start()                                                       # turning it on again retries
        self.assertTrue(wait_for(lambda: len(self.spawned) >= 5))

    def test_crash_then_recovers(self):
        self.m = m = self.make(procs=[lambda: FakeProc(crash_after_ready=True), FakeProc])
        m.start()
        self.assertTrue(wait_for(lambda: m.ready and len(self.spawned) == 2, timeout=5))

    def test_no_restart_while_blocked(self):
        state = {"game": False}
        self.m = m = self.make(procs=[lambda: FakeProc(crash_after_ready=True)], blocked=lambda: state["game"],
                               restart_delays=(0.2, 0.2, 0.2))
        m.start()
        self.assertTrue(wait_for(lambda: len(self.spawned) == 1 and m.state == "starting"))
        state["game"] = True
        time.sleep(0.4)
        self.assertEqual(len(self.spawned), 1)
        self.assertEqual(m.state, "off")


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ============================================================================ app wiring
class FakeTts:
    def __init__(self, est=6.0, recent=False):
        self.calls = []
        self.est = est
        self.recent = recent
        self.on = True

    def speak(self, text, rid=None):
        self.calls.append(("speak", text, rid))
        return self.est

    def recording_started(self, rid):
        self.calls.append(("recording_started", rid))
        return self.recent

    def stop(self, why=""):
        self.calls.append(("stop", why))

    def start(self):
        self.calls.append(("start",))

    def shutdown(self, why="", wait=True):
        self.calls.append(("shutdown", why))

    def enabled(self):
        return self.on

    def names(self):
        return [c[0] for c in self.calls]


try:
    from localflow import app as app_mod  # noqa: E402
    from tests import test_unit_command as tc  # noqa: E402
    _APP_OK = True
except Exception:       # pragma: no cover
    _APP_OK = False


@unittest.skipUnless(_APP_OK, "app modules unavailable")
class AppWiring(tc.AppBase if _APP_OK else unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.app.cleaner.answer.return_value = ("Jane Austen", {"llm": True, "ms": 9, "reason": ""})
        self.app.tts = self.fake = FakeTts()

    def say(self, text):
        self.app.stt.transcribe.return_value = SimpleNamespace(text=text, ms=5, dropped="")

    def test_default_app_tts_is_inert(self):
        m = app_mod.LocalFlowApp.__init__.__defaults__
        self.assertIn(False, m)                 # real_tts defaults to False

    def test_answer_spoken_and_card_extended(self):
        a = self.app
        self.say("Who wrote Pride and Prejudice?")
        self.speak("command")
        rid = a._rec_id
        self.assertIn(("speak", "Jane Austen", rid), self.fake.calls)
        self.assertEqual(a.overlay.answers[0][2], 6000 + 2000)          # card stays up while it is read
        self.assertGreater(a._answer_until - time.monotonic(), 7)

    def test_short_speech_keeps_normal_card_length(self):
        self.fake.est = 1.0
        self.say("Who wrote Pride and Prejudice?")
        self.speak("command")
        self.assertEqual(self.app.overlay.answers[0][2], 4000 + 50 * len("Jane Austen"))

    def test_not_spoken_during_meeting(self):
        a = self.app
        a._meeting_on = True
        self.say("Who wrote Pride and Prejudice?")
        self.speak("command")
        self.assertNotIn("speak", self.fake.names())
        self.assertEqual(a.overlay.answers[0][2], 4000 + 50 * len("Jane Austen"))

    def test_not_spoken_when_newer_recording_started(self):
        a = self.app

        def answer(*x):
            a._rec_id += 1
            return "Jane Austen", {"llm": True, "ms": 9, "reason": ""}
        a.cleaner.answer.side_effect = answer
        self.say("Who wrote Pride and Prejudice?")
        self.speak("command")
        self.assertNotIn("speak", self.fake.names())

    def test_quick_answer_spoken(self):
        self.say("How many hours is it from 7:30am to 5pm?")
        self.speak("command")
        spoken = [c for c in self.fake.calls if c[0] == "speak"]
        self.assertEqual(len(spoken), 1)
        self.assertIn("9", spoken[0][1])

    def test_commands_and_errors_not_spoken(self):
        self.say("make it formal")
        self.speak("command")                                           # inserted command output
        self.app.cleaner.answer.return_value = ("", {"llm": True, "ms": 9, "reason": "timeout"})
        self.say("Who wrote Pride and Prejudice?")
        self.speak("command")                                           # failed answer -> error toast
        self.assertNotIn("speak", self.fake.names())

    def test_recording_start_stops_speech_first_and_skips_preroll(self):
        a = self.app
        order = []
        self.fake.recent = True
        a.audio.drop_preroll = lambda: order.append("drop")
        real_begin = a.audio.begin
        a.audio.begin = lambda: (order.append("begin"), real_begin())
        real_fg = a._fg_begin
        a._fg_begin = lambda rid: (order.append("fg"), real_fg(rid))
        rec = self.fake.recording_started
        self.fake.recording_started = lambda rid: (order.append("tts"), rec(rid))[1]
        a._do_start("command")
        self.assertEqual(order, ["tts", "fg", "drop", "begin"])
        self.assertIn(("recording_started", a._rec_id), self.fake.calls)
        a._do_cancel("esc")
        order.clear()
        self.fake.recent = False
        a._do_start("dictate")
        self.assertEqual(order, ["tts", "fg", "begin"])                 # pre-roll kept
        a._do_cancel("esc")

    def test_meeting_start_stops_speech(self):
        self.app._on_meeting_state({"recording": True})
        self.assertIn(("stop", "meeting recording"), self.fake.calls)

    def test_game_enter_exit(self):
        a = self.app
        a.stt = mock.Mock(ready=True, device="cuda", model_name="m")
        a.watcher = mock.Mock()
        a.watcher.game_running.return_value = False
        a._game_enter()
        self.assertIn(("shutdown", "game mode"), self.fake.calls)
        self.fake.calls.clear()
        with mock.patch("faster_whisper.vad.get_vad_model", create=True):
            a._game_exit()
        self.assertFalse(a.game_active)
        self.assertEqual(self.fake.names()[-1], "start")

    def _apply(self, *changed):
        self.app._apply_config(lambda *p: p in changed)

    def test_config_toggle(self):
        a = self.app
        a._ready = True
        self.fake.on = False
        self._apply(("command_mode", "speak_answers"))
        self.assertIn(("shutdown", "spoken answers off"), self.fake.calls)
        self.fake.on = True
        self.fake.calls.clear()
        self._apply(("command_mode", "speak_answers"))
        self.assertEqual(self.fake.calls, [("start",)])
        a.game_active = True                                            # never under a game
        self.fake.calls.clear()
        self._apply(("command_mode", "speak_answers"))
        self.assertEqual(self.fake.calls, [])
        a.game_active = False

    def test_pause_and_resume(self):
        a = self.app
        a._ready = True
        a.cfg.set("enabled", False, save=False)
        self._apply(("enabled",))
        self.assertIn(("shutdown", "paused"), self.fake.calls)
        self.assertNotIn(("start",), self.fake.calls)
        self.fake.calls.clear()
        a.cfg.set("enabled", True, save=False)
        self._apply(("enabled",))
        self.assertIn(("start",), self.fake.calls)


if __name__ == "__main__":
    unittest.main()
