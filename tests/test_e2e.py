"""End-to-end pipeline test on SAPI-synthesised speech (no insertion).  Run:
   .venv\\Scripts\\python.exe tests\\test_e2e.py
Reports STT accuracy (WER vs the reference sentence) and per-stage latency."""
import re
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from localflow.cleanup import Cleaner  # noqa: E402
from localflow.config import Config  # noqa: E402
from localflow.pipeline import Pipeline  # noqa: E402
from localflow.stt import Transcriber  # noqa: E402

AUDIO = Path(__file__).parent / "audio"


def load_wav(p):
    with wave.open(str(p), "rb") as w:
        assert w.getframerate() == 16000
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    return x


def words(s):
    return re.findall(r"[a-z0-9']+", s.lower())


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / max(1, len(r))


def main():
    cfg = Config(path=Path(tempfile.mkdtemp()) / "cfg.json")
    cfg.set("cleanup.port", 8799, save=False)    # never touch the running app's llama-server on :8765
    stt = Transcriber(cfg)
    t = time.perf_counter()
    stt.load()
    print(f"STT load+warmup: {time.perf_counter() - t:.1f}s  device={stt.device} compute={stt.compute} model={stt.model_name}")
    cl = Cleaner(cfg)
    cl.server.start(wait=True)
    cl._warm()
    pipe = Pipeline(cfg, stt, cl, None, lambda *a, **k: {})
    rows = []
    for wav in sorted(AUDIO.glob("*.wav")):
        ref = wav.with_suffix(".txt").read_text(encoding="utf-8-sig").strip()
        audio = load_wav(wav)
        # add 150 ms leading / 200 ms trailing silence like a real recording
        audio = np.concatenate([np.zeros(2400, np.float32), audio, np.zeros(3200, np.float32)])
        t0 = time.perf_counter()
        r = pipe.process(audio, app_exe="notepad.exe", t_release=t0, insert=False)
        wall = (time.perf_counter() - t0) * 1000
        w = wer(ref, r["raw"]) if r["raw"] else 1.0
        rows.append((wav.stem, r, w, wall))
        print(f"\n{wav.stem}  ({r['duration_s']:.1f}s)  status={r['status']} {r['reason']}")
        print(f"  ref  : {ref}\n  raw  : {r['raw']}\n  final: {r['text']!r}")
        print(f"  WER {w:.2f} | stt {r['stt_ms']} ms | cleanup {r['cleanup_ms']} ms (llm={r['llm']}) | total {wall:.0f} ms")
    ok = [x for x in rows if x[1]["status"] == "ok"]
    print("\n==== summary ====")
    print(f"mean WER (raw STT): {np.mean([x[2] for x in ok]):.3f}")
    print(f"STT ms  median {np.median([x[1]['stt_ms'] for x in ok]):.0f}  max {max(x[1]['stt_ms'] for x in ok)}")
    print(f"clean ms median {np.median([x[1]['cleanup_ms'] for x in ok]):.0f}  max {max(x[1]['cleanup_ms'] for x in ok)}")
    print(f"total ms median {np.median([x[3] for x in ok]):.0f}  max {max(x[3] for x in ok):.0f}")
    # silence / noise must be discarded with no text
    rng = np.random.default_rng(1)
    for name, sig in (("digital silence", np.zeros(32000, np.float32)),
                      ("low noise", (rng.standard_normal(48000) * 0.004).astype(np.float32)),
                      ("louder noise", (rng.standard_normal(48000) * 0.03).astype(np.float32))):
        r = pipe.process(sig, insert=False)
        print(f"{name}: status={r['status']} reason={r['reason']!r} text={r['text']!r}")
        assert r["text"] == "", f"hallucinated text on {name}: {r['text']!r}"
    cl.shutdown()


if __name__ == "__main__":
    main()
