"""Real-model smoke test for the transcripts engine (GPU + the RUNNING app's llama-server; plays no audio).

- times transcribe_segments on dense 30 s speech chunks (worst-case unit of background GPU work)
- file job (raw + "neutral" style through the live LLM) on tests/bench_audio/bench_90s.wav
- meeting job on a synthetic two-track recording built from speech WAV files: "Others" = bench_45s, "You" =
  bench_20s later on + an attenuated, delayed copy of Others (speaker echo) -> echo must be marked
- assistant summary of the meeting via the live llama-server

Read-only towards the app: the config file is read as JSON (never saved), the live llama-server is only queried
(never started/stopped), everything else lives in a temp dir.
Run: .venv\\Scripts\\python.exe tests\\smoke_transcripts.py [--no-llm]
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from localflow import jobs, recorder as rec  # noqa: E402
from localflow.gate import ForegroundGate  # noqa: E402
from localflow.stt import Transcriber  # noqa: E402
from localflow.transcripts import TranscriptStore  # noqa: E402

BENCH = ROOT / "tests" / "bench_audio"


class Cfg(dict):
    def get(self, dotted, default=None):
        cur = self
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur


def load_cfg() -> Cfg:
    import os
    p = Path(os.environ["APPDATA"]) / "LocalFlow" / "config.json"
    data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    data.setdefault("stt", {}).setdefault("model", "large-v3-turbo")
    return Cfg(data)


def wait(store, tid, timeout=900):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        st = store.get(tid)["status"]
        if st in ("done", "failed", "cancelled"):
            return st
        time.sleep(0.25)
    return "timeout"


def main():
    use_llm = "--no-llm" not in sys.argv
    cfg = load_cfg()
    stt = Transcriber(cfg)
    t = time.perf_counter()
    stt.load(strict=True)
    print(f"STT loaded: {stt.model_name} on {stt.device}/{stt.compute} in {time.perf_counter() - t:.1f} s")

    # ---- 1) worst-case chunk timing
    audio = jobs.decode_file_16k(str(BENCH / "bench_90s.wav"), 3600).astype(np.float32) / 32768.0
    times = []
    for start in range(0, int(audio.size / 16000) - 29, 10):
        ch = audio[start * 16000:(start + 30) * 16000]
        t = time.perf_counter()
        segs = stt.transcribe_segments(ch, start)
        times.append(time.perf_counter() - t)
    print(f"30 s chunks: n={len(times)} mean={np.mean(times):.2f}s max={max(times):.2f}s "
          f"(last: {len(segs)} segs, first='{segs[0][2][:60] if segs else ''}')")
    silent = stt.transcribe_segments(np.zeros(16000 * 20, np.float32), 0)
    noise = stt.transcribe_segments((np.random.default_rng(1).standard_normal(16000 * 20) * 0.003).astype(np.float32), 0)
    print("silence ->", silent, "| low noise ->", noise)

    td = Path(tempfile.mkdtemp(prefix="lf_smoke_"))
    root = td / "recordings"
    root.mkdir()
    store = TranscriptStore(td / "t.db", root)
    gate = ForegroundGate()
    cleaner = None
    if use_llm:
        from localflow.cleanup import Cleaner

        class LiveCleaner(Cleaner):
            """Talks to the app's running llama-server; never starts, stops or restarts it."""

            def __init__(self, c):
                super().__init__(c)
                self.server.ready = True

            def _recover(self, proc):
                pass
        cleaner = LiveCleaner(cfg)
    runner = jobs.JobRunner(stt, cleaner, gate, store, lambda: False, cfg=cfg,
                            on_progress=lambda tid, st, p, m: None)

    # ---- 2) file jobs
    f = str(BENCH / "bench_90s.wav")
    ref = (BENCH / "bench_90s.txt").read_text(encoding="utf-8").strip()
    tid = store.create("file", "bench90 raw", source_path=f, style="raw")
    t = time.perf_counter()
    runner.submit(tid)
    print("file raw:", wait(store, tid), f"{time.perf_counter() - t:.1f} s")
    row = store.get(tid)
    print("  paragraphs:", row["text"].count("\n\n") + 1, "| words:", len(row["text"].split()), "/ ref", len(ref.split()))
    print("  text:", row["text"][:300].replace("\n", " / "))
    if use_llm:
        tid2 = store.create("file", "bench90 neutral", source_path=f, style="neutral")
        t = time.perf_counter()
        runner.submit(tid2)
        print("file neutral:", wait(store, tid2), f"{time.perf_counter() - t:.1f} s", store.get(tid2)["info"])
        print("  text:", store.get(tid2)["text"][:300].replace("\n", " / "))

    # ---- 3) meeting job on a synthetic two-track recording (no audio played)
    import soundfile as sf
    from scipy.signal import resample_poly
    others = resample_poly(jobs.decode_file_16k(str(BENCH / "bench_45s.wav"), 3600).astype(np.float32), 3, 1)
    you = resample_poly(jobs.decode_file_16k(str(BENCH / "bench_20s.wav"), 3600).astype(np.float32), 3, 1)
    sr = 48000
    total = int(sr * (50 + you.size / sr + 3))
    sys_tr = np.zeros(total, np.float32)
    sys_tr[:others.size] = others
    mic_tr = np.zeros(total, np.float32)
    d = int(0.08 * sr)                                     # speaker -> mic echo: 80 ms later, quieter
    mic_tr[d:d + others.size] += others * 0.35
    mic_tr[50 * sr:50 * sr + you.size] += you
    md = root / "20260101-120000"
    md.mkdir()
    for key, x, off in (("mic", mic_tr, 0.30), ("system", sys_tr, 0.28)):
        sf.write(str(md / f"{key}.flac"), np.clip(x, -32768, 32767).astype(np.int16), sr, subtype="PCM_16")
    rec.write_meta(md, {"state": "stopped", "t0": time.time(), "duration_s": total / sr,
                        "tracks": {"mic": {"file": "mic.flac", "rate": sr, "offset_s": 0.30, "samples": total},
                                   "system": {"file": "system.flac", "rate": sr, "offset_s": 0.28, "samples": total}}})
    mid = store.create("meeting", "synthetic meeting", audio_dir=str(md), style="raw")
    t = time.perf_counter()
    runner.submit(mid)
    print("meeting:", wait(store, mid), f"{time.perf_counter() - t:.1f} s", store.get(mid)["info"])
    sj = store.segments(mid)
    n_you = sum(1 for s in sj["segments"] if s["speaker"] == "You")
    n_echo = sum(1 for s in sj["segments"] if s.get("echo"))
    n_oth = sum(1 for s in sj["segments"] if s["speaker"] == "Others")
    print(f"  segments: Others={n_oth} You={n_you} (echo-marked {n_echo})")
    for s in sj["segments"]:
        if s["speaker"] == "You" and not s.get("echo"):
            print("  unmarked You:", round(s["start"], 1), s["text"][:80])
    print("  transcript:\n   " + store.get(mid)["text"][:900].replace("\n\n", "\n   "))
    print("slowest STT chunk in jobs: %.2f s" % runner.max_chunk_s)

    # ---- 4) assistant against the live server
    if use_llm:
        from localflow.assistant import Assistant
        a = Assistant(cfg.get, gate, store, lambda: cleaner.server.base_url, lambda: True)
        t = time.perf_counter()
        out = a.ask(mid, "action_items")
        print(f"assistant action_items ({time.perf_counter() - t:.1f} s):\n{out[:600]}")
    runner.shutdown(10)
    store.close()
    print("temp dir:", td)


if __name__ == "__main__":
    main()
