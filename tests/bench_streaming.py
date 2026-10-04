"""Single-pass vs streaming benchmark on SAPI speech (run tests\\gen_bench_corpus.py first).

The audio is fed to the StreamSession in REAL TIME (20 ms chunks on a wall-clock schedule) so the background
worker competes for the GPU exactly as it would while someone is talking. Release-to-text is measured from the
moment the last sample was fed until Pipeline.process() returns (insert=False, so no clipboard/paste time).

   .venv\\Scripts\\python.exe tests\\bench_streaming.py [--reps 2] [--clips bench_05s bench_20s ...]
                                                       [--modes single stream_stt stream_full] [--style neutral]

Uses its own llama-server on port 8799 so it never touches the running app's server on 8765.
Nothing is typed or pasted anywhere.
"""
import argparse
import json
import re
import statistics
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from localflow import styles, textproc  # noqa: E402
from localflow.cleanup import Cleaner  # noqa: E402
from localflow.config import Config  # noqa: E402
from localflow.pipeline import Pipeline  # noqa: E402
from localflow.streaming import StreamSession  # noqa: E402
from localflow.stt import Transcriber  # noqa: E402

BENCH = Path(__file__).parent / "bench_audio"


def load_wav(p):
    with wave.open(str(p), "rb") as w:
        assert w.getframerate() == 16000
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768


def wer(ref: str, hyp: str) -> float:
    r, h = textproc.simple_words(ref), textproc.simple_words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / max(1, len(r))


def squeeze(audio: np.ndarray, max_gap_s: float = 0.12) -> np.ndarray:
    """Remove pauses (leaves max_gap_s of silence) -> continuous speech with no natural cut point."""
    from localflow.streaming import FRAME, VadStream
    n = len(audio) // FRAME * FRAME
    p = VadStream().push(audio[:n])
    sp = np.convolve(p >= 0.5, np.ones(5), mode="same") > 0          # +-2 frames (65 ms) around speech
    out, gap = [], 0
    for i in range(0, n, FRAME):
        fr = audio[i:i + FRAME]
        if sp[i // FRAME]:
            gap = 0
            out.append(fr)
        elif gap * FRAME < max_gap_s * 16000:
            gap += 1
            out.append(fr)
    return np.concatenate(out)


def feed_real_time(sess: StreamSession, audio: np.ndarray, speed: float = 1.0) -> None:
    chunk = 320
    t0 = time.perf_counter()
    for k in range(0, len(audio), chunk):
        due = t0 + (k / 16000.0) / speed
        now = time.perf_counter()
        if due > now:
            time.sleep(due - now)
        sess.feed(audio[k:k + chunk])


def run_once(mode, pipe, cfg, stt, cleaner, audio, ctx, speed):
    if mode == "single":
        cfg.set("streaming.enabled", False, save=False)
        t_rel = time.perf_counter()
        r = pipe.process(audio, t_release=t_rel, insert=False, style_ctx=ctx)
        return r
    cfg.set("streaming.enabled", True, save=False)
    cfg.set("streaming.incremental_cleanup", mode == "stream_full", save=False)
    sess = StreamSession(cfg, stt, cleaner, ctx, 16000)
    sess.start()
    feed_real_time(sess, audio, speed)
    t_rel = time.perf_counter()
    return pipe.process(audio, t_release=t_rel, insert=False, style_ctx=ctx, session=sess)


def main():
    import logging
    logging.basicConfig(level=logging.WARNING, format="    %(message)s")
    logging.getLogger("localflow.streaming").setLevel(logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--clips", nargs="*", default=["bench_05s", "bench_20s", "bench_45s", "bench_90s", "bench_corr"])
    ap.add_argument("--modes", nargs="*", default=["single", "stream_stt", "stream_full"])
    ap.add_argument("--style", default="neutral")
    ap.add_argument("--speed", type=float, default=1.0, help="feed speed multiplier (1.0 = real time)")
    ap.add_argument("--squeeze", type=float, nargs="?", const=0.0, default=None,
                    help="shrink every pause to this many seconds first (0 = continuous speech: forces the "
                         "lowest-energy cut)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    cfg = Config(path=Path(tempfile.mkdtemp()) / "cfg.json")
    cfg.set("cleanup.port", 8799, save=False)
    stt = Transcriber(cfg)
    stt.load()
    cleaner = Cleaner(cfg)
    cleaner.server.start(wait=True)
    assert cleaner.server.ready, cleaner.server.failed
    cleaner._warm()
    pipe = Pipeline(cfg, stt, cleaner, None, lambda *x, **k: {})
    ctx = styles.StyleContext(style=a.style, app="bench")
    results = []
    for clip in a.clips:
        audio = load_wav(BENCH / f"{clip}.wav")
        if a.squeeze is not None:
            audio = squeeze(audio, a.squeeze)
        ref = (BENCH / f"{clip}.txt").read_text(encoding="utf-8-sig").strip()
        print(f"\n=== {clip}  ({len(audio) / 16000:.1f}s, {len(ref.split())} words) ===", flush=True)
        for mode in a.modes:
            rows = []
            for rep in range(a.reps):
                r = run_once(mode, pipe, cfg, stt, cleaner, audio, ctx, a.speed)
                rows.append(r)
                print(f"  {mode:11s} rep{rep}: total {r['total_ms']:5d} ms | stt(after release) {r['stt_ms']:4d} "
                      f"cleanup(after release) {r['cleanup_ms']:5d} | segs {r['segments']} llm={r['llm']} "
                      f"{r['reason']} | WERraw {wer(ref, r['raw']):.3f} WERfinal {wer(ref, r['text']):.3f}", flush=True)
            med = lambda k: int(statistics.median(x[k] for x in rows))      # noqa: E731
            res = dict(clip=clip, mode=mode, dur=len(audio) / 16000, total_ms=med("total_ms"), stt_ms=med("stt_ms"),
                       stt_total_ms=med("stt_total_ms"), cleanup_ms=med("cleanup_ms"),
                       segments=rows[0]["segments"], llm=all(x["llm"] for x in rows),
                       wer_raw=statistics.median(wer(ref, x["raw"]) for x in rows),
                       wer_final=statistics.median(wer(ref, x["text"]) for x in rows),
                       text=rows[-1]["text"], raw=rows[-1]["raw"], reason=rows[-1]["reason"])
            results.append(res)
    print("\n| clip | dur | mode | release->text ms | STT ms (after release) | cleanup ms (after release) | segs | WER raw | WER final |")
    print("|---|---|---|---|---|---|---|---|---|")
    for x in results:
        print(f"| {x['clip']} | {x['dur']:.0f}s | {x['mode']} | {x['total_ms']} | {x['stt_ms']} | {x['cleanup_ms']} | "
              f"{x['segments']} | {x['wer_raw']:.3f} | {x['wer_final']:.3f} |")
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    cleaner.shutdown()


if __name__ == "__main__":
    main()
