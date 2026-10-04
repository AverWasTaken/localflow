"""Offline tests for microphone selection and default-device following.

sounddevice and the MMDevice lookup are faked: no real microphone is opened, nothing is recorded or played.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from localflow import audio, devwatch  # noqa: E402
from localflow.config import DEFAULTS, Config  # noqa: E402

# host api indexes, in the order fake_sd() lists them
WASAPI, MME, DS = 0, 1, 2


def dev(name, hostapi, ins=1, sr=48000.0):
    return {"name": name, "hostapi": hostapi, "max_input_channels": ins, "max_output_channels": 0,
            "default_samplerate": sr}


def fake_sd(devices, wasapi_default=-1, global_default=-1, wasapi_present=True):
    """A stand-in for the sounddevice module; ``devices`` is the PortAudio device list."""
    apis = [{"name": "Windows WASAPI", "default_input_device": wasapi_default},
            {"name": "MME", "default_input_device": -1},
            {"name": "Windows DirectSound", "default_input_device": -1}]
    if not wasapi_present:
        apis = apis[1:]

    def query_devices(device=None, kind=None):
        if device is None and kind is None:
            return list(devices)
        if kind == "input":
            if global_default < 0:
                raise ValueError("No input device found")
            return dict(devices[global_default], index=global_default)
        return devices[device]

    def query_hostapis(index=None):
        return apis if index is None else apis[index]

    return SimpleNamespace(query_devices=query_devices, query_hostapis=query_hostapis,
                           default=SimpleNamespace(device=(global_default, -1)))


# the same two mics seen through each host API, like PortAudio on Windows reports them
DEVICES = [
    dev("Microphone (Realtek Audio)", MME),            # 0
    dev("Microphone (USB Audio Device)", MME),         # 1
    dev("Microphone (Realtek Audio)", WASAPI),         # 2
    dev("Microphone (USB Audio Device)", WASAPI),      # 3
    dev("Speakers (Realtek Audio)", WASAPI, ins=0),    # 4 output only
]


def patched(sd_fake):
    return mock.patch.object(audio, "sd", sd_fake)


class DefaultDeviceTests(unittest.TestCase):
    def test_wasapi_default_wins(self):
        with patched(fake_sd(DEVICES, wasapi_default=3, global_default=0)):
            d, why = audio.default_input_device()
        self.assertEqual(d["name"], "Microphone (USB Audio Device)")
        self.assertEqual(d["index"], 3)
        self.assertEqual(d["hostapi"], "Windows WASAPI")
        self.assertIn("WASAPI", why)

    def test_falls_back_to_portaudio_default_when_wasapi_has_none(self):
        with patched(fake_sd(DEVICES, wasapi_default=-1, global_default=1)):
            d, why = audio.default_input_device()
        self.assertEqual(d["index"], 1)
        self.assertIn("PortAudio", why)

    def test_falls_back_when_wasapi_missing(self):
        with patched(fake_sd(DEVICES, global_default=0, wasapi_present=False)):
            d, _ = audio.default_input_device()
        self.assertEqual(d["index"], 0)

    def test_wasapi_default_that_is_not_an_input_is_skipped(self):
        with patched(fake_sd(DEVICES, wasapi_default=4, global_default=1)):
            d, _ = audio.default_input_device()
        self.assertEqual(d["index"], 1)

    def test_no_input_devices_at_all(self):
        with patched(fake_sd([], wasapi_default=-1, global_default=-1)):
            d, why = audio.default_input_device()
            self.assertEqual(audio.default_input_name(), "")
        self.assertIsNone(d)
        self.assertEqual(why, "system default")

    def test_default_input_name(self):
        with patched(fake_sd(DEVICES, wasapi_default=2)):
            self.assertEqual(audio.default_input_name(), "Microphone (Realtek Audio)")


class PickDeviceTests(unittest.TestCase):
    def setUp(self):
        self.sd = fake_sd(DEVICES, wasapi_default=3, global_default=1)

    def pick(self, match="", exact=""):
        with patched(self.sd):
            return audio.pick_device(match, exact)

    def test_empty_match_and_device_use_windows_default(self):
        d, why = self.pick("", "")
        self.assertEqual(d["index"], 3)
        self.assertIn("system default", why)

    def test_match_picks_substring_and_prefers_wasapi(self):
        d, why = self.pick("realtek")
        self.assertEqual(d["index"], 2)      # WASAPI copy, not the MME one
        self.assertEqual(why, "matched 'realtek'")

    def test_match_with_no_hit_falls_back_to_default(self):
        d, why = self.pick("nothing-like-this")
        self.assertEqual(d["index"], 3)
        self.assertIn("system default", why)

    def test_exact_device_pick(self):
        d, why = self.pick("", "Microphone (Realtek Audio)")
        self.assertEqual(d["index"], 2)
        self.assertTrue(why.startswith("selected"))

    def test_exact_beats_match(self):
        d, _ = self.pick("usb", "Microphone (Realtek Audio)")
        self.assertEqual(d["index"], 2)

    def test_missing_exact_falls_back_to_default_not_match(self):
        d, why = self.pick("realtek", "Unplugged Mic")
        self.assertEqual(d["index"], 3)
        self.assertIn("system default", why)

    def test_output_only_devices_are_never_picked(self):
        d, _ = self.pick("speakers")
        self.assertEqual(d["index"], 3)      # no input called Speakers -> default


class ConfigDefaultsTests(unittest.TestCase):
    def test_defaults_follow_windows_default(self):
        self.assertEqual(DEFAULTS["mic"], {"match": "", "device": ""})

    def test_saved_user_match_survives_the_merge(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "config.json"
            p.write_text(json.dumps({"mic": {"match": "usb", "device": ""}}), encoding="utf-8")
            cfg = Config(path=p)
            self.assertEqual(cfg.get("mic.match"), "usb")
            self.assertEqual(cfg.get("mic.device"), "")
            # saving from Settings only sends mic.device; the match must stay
            cfg.update({"mic": {"device": "Some Mic"}})
            self.assertEqual(cfg.get("mic.match"), "usb")
            self.assertEqual(json.loads(p.read_text(encoding="utf-8"))["mic"],
                             {"match": "usb", "device": "Some Mic"})

    def test_fresh_config_has_empty_match(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = Config(path=Path(td) / "config.json")
            self.assertEqual(cfg.get("mic.match"), "")


class FakeStream:
    active = True


def make_input(preferred_found=False, recording=False, opened_for="A"):
    with tempfile.TemporaryDirectory() as td:
        cfg = Config(path=Path(td) / "c.json")
    a = audio.AudioInput(cfg)
    a._stream = FakeStream()
    a._last_cb = 1e12                      # looks alive
    a._default_id = opened_for
    a.preferred_found = preferred_found
    a._rec = [] if recording else None
    return a


class DefaultFollowingTests(unittest.TestCase):
    """The watcher re-opens the stream when the Windows default changes, but never mid-dictation."""

    def make(self, **kw):
        a = make_input(**kw)
        a.reopen = mock.Mock()
        return a

    @staticmethod
    def run_watch(a, current_id):
        # run exactly one iteration of the watcher loop
        a._stop_monitor = SimpleNamespace(wait=mock.Mock(side_effect=[False, True]), set=lambda: None)
        with mock.patch.object(devwatch, "default_capture_id", return_value=current_id):
            a._watch()

    def test_change_triggers_reopen(self):
        a = self.make()
        seen = []
        a.on_device_change = seen.append
        self.run_watch(a, "B")
        a.reopen.assert_called_once()
        self.assertEqual(len(seen), 1)

    def test_same_id_does_nothing(self):
        a = self.make()
        self.run_watch(a, "A")
        a.reopen.assert_not_called()

    def test_failed_lookup_is_not_a_change(self):
        a = self.make()
        self.run_watch(a, None)
        a.reopen.assert_not_called()

    def test_deferred_while_recording(self):
        a = self.make(recording=True)
        self.run_watch(a, "B")
        a.reopen.assert_not_called()
        # once the dictation is over the next poll switches
        a._rec = None
        a._last_default_check = 0.0
        self.run_watch(a, "B")
        a.reopen.assert_called_once()

    def test_explicit_device_is_not_overridden(self):
        a = self.make(preferred_found=True)
        self.run_watch(a, "B")
        a.reopen.assert_not_called()

    def test_reopen_snapshots_default_id_before_reinit(self):
        a = make_input()
        a._rec = None
        order = []
        with mock.patch.object(devwatch, "default_capture_id", side_effect=lambda: order.append("id") or "X"), \
                mock.patch.object(audio.AudioInput, "_reinit_portaudio",
                                  staticmethod(lambda: order.append("reinit"))), \
                mock.patch.object(audio.AudioInput, "_open", lambda self: order.append("open") or True):
            a.reopen()
        self.assertEqual(order, ["id", "reinit", "open"])
        self.assertEqual(a._default_id, "X")

    def test_reopen_is_skipped_while_recording(self):
        a = make_input(recording=True)
        with mock.patch.object(audio.AudioInput, "_open") as op:
            a.reopen()
        op.assert_not_called()


class DevwatchSmokeTests(unittest.TestCase):
    def test_returns_str_or_none(self):
        # read-only MMDevice query: no audio is opened. None is fine on machines without a recording device.
        r = devwatch.default_capture_id()
        self.assertTrue(r is None or (isinstance(r, str) and r))


if __name__ == "__main__":
    unittest.main()
