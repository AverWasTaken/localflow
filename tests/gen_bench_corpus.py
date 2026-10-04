"""Regenerate the LocalFlow speech benchmark corpus into tests/bench_audio/.

Each sentence (or clause, for hesitation splits) is synthesized with Windows
SAPI (System.Speech, 'Microsoft David Desktop', 16 kHz / 16-bit / mono), then
the pieces are concatenated with generated digital-silence gaps that emulate a
person dictating with natural pauses. A very quiet gaussian noise floor is
added over the whole clip. Deterministic (seeded RNG), apart from SAPI itself.

Outputs per clip <name>: .wav, .txt (reference), .json (timings + pauses);
bench_corr additionally gets bench_corr.raw.txt (literal spoken words).

Usage:  .venv\\Scripts\\python.exe tests\\gen_bench_corpus.py [clip ...]
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
import wave
import zlib
from pathlib import Path

import numpy as np

SR = 16000
HERE = Path(__file__).resolve().parent
OUT = HERE / "bench_audio"
SEED = 20261003
VOICE = "Microsoft David Desktop"
LEAD_MS, TRAIL_MS = 150, 200
GAP_MS = (450, 1100)       # between sentences
HESITATE_MS = (550, 800)   # mid-sentence, at a comma
HESITATE_P = 0.2           # ~1 in 5 sentences
NOISE_RMS = 0.002

# --------------------------------------------------------------------------
# Content. Plain clips: a list of sentences (reference = sentences joined).
# --------------------------------------------------------------------------
CLIPS: dict[str, list[str]] = {
    "bench_05s": [
        "Hi Sarah, the Q3 report is ready for review.",
        "Please take a look today.",
    ],
    "bench_20s": [
        "Hi Marcus, quick update on the Kubernetes migration.",
        "We moved the staging cluster over on Monday, and everything is running smoothly so far.",
        "The GPU nodes still need new drivers, so that will take until Thursday.",
        "I will send the full report by 5 p.m. on Friday.",
        "Thanks again for your patience.",
    ],
    "bench_45s": [
        "Good morning everyone, here is the weekly project update.",
        "The new search feature shipped on Tuesday, and early feedback has been very positive.",
        "Latency dropped from 450 milliseconds to about 120 milliseconds after we added the cache.",
        "We still have two open bugs in the export tool, both related to large files.",
        "Priya is investigating the first one, and I will take the second.",
        "On the infrastructure side, the Kubernetes upgrade is scheduled for Saturday at 2 a.m.",
        "Please avoid deploying anything between midnight and 6 a.m. that night.",
        "Next week we begin planning for Q4, so send me your top three priorities by Wednesday.",
        "Finally, the team lunch is on Thursday at 12:30 p.m., and everyone is welcome.",
        "Thanks, and have a great week.",
    ],
    "bench_90s": [
        "Hello team, this is my end of month summary for the platform group.",
        "Let me start with the numbers, because they are mostly good news.",
        "Revenue for Q3 came in at 2.4 million dollars, which is 8 percent above our forecast.",
        "Customer churn fell to 3.1 percent, the lowest figure we have recorded in two years.",
        "The credit goes largely to the support team, who answered 94 percent of tickets within one hour.",
        "Turning to engineering, we finished migrating the last three services to Kubernetes.",
        "That means the old virtual machines can be shut down by the end of October.",
        "We also added four new GPU servers to the training cluster, which cut model training time nearly in half.",
        "Daniel and Aisha did most of that work, and I want to thank them publicly.",
        "On the less cheerful side, we had two outages this month.",
        "The first was a misconfigured load balancer on the 7th, and it lasted about 40 minutes.",
        "The second was a database failover that took 15 minutes longer than it should have.",
        "We have written up both incidents, and the action items are in the shared tracker.",
        "In terms of hiring, we have offers out to two backend engineers and one designer.",
        "If all three accept, the new people will start on the first Monday of November.",
        "Looking ahead, our main goal for Q4 is to reduce page load time below one second.",
        "That will require a rewrite of the image pipeline, so I would like to schedule a design review.",
        "How does next Tuesday at 10 a.m. look for everyone?",
        "Please reply to this message with any conflicts, and I will find another time.",
        "Thank you all for the hard work this quarter.",
    ],
}

# Self-correction clip. Each entry: (spoken text, reference text or None to
# drop it, forced pause-after in ms or None for a random gap). A correction
# sentence follows a fixed 650-750 ms pause, i.e. it straddles a pause; the
# reference carries the INTENDED final text with the correction applied.
CORR_SEGMENTS: list[tuple[str, str | None, int | None]] = [
    ("Hi Elena, thanks for sending over the draft.",
     "Hi Elena, thanks for sending over the draft.", None),
    ("I think we should review it together, so let's meet at 3 p.m. on Tuesday.",
     None, 700),
    ("Actually no, make it 4 p.m. on Wednesday.",
     "I think we should review it together, so let's meet at 4 p.m. on Wednesday.", None),
    ("The budget for the project is 15,000 dollars.", None, 650),
    ("Sorry, I mean 12,000 dollars.",
     "The budget for the project is 12,000 dollars.", None),
    ("We will present it to the steering committee in Q3.",
     "We will present it to the steering committee in Q3.", None),
    ("Bring the GPU benchmark results, along with the Kubernetes cost estimate.",
     None, 750),
    ("Never mind, ask Marcus to bring the cost estimate himself.",
     "Bring the GPU benchmark results, and ask Marcus to bring the Kubernetes cost estimate himself.", None),
    ("Thanks, talk soon.", "Thanks, talk soon.", None),
]

# --------------------------------------------------------------------------
PS_SCRIPT = r"""
param([string]$Manifest, [string]$Voice)
Add-Type -AssemblyName System.Speech
$items = Get-Content -Raw -Encoding UTF8 $Manifest | ConvertFrom-Json
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
foreach ($it in $items) {
  $s = New-Object System.Speech.Synthesis.SpeechSynthesizer
  try { $s.SelectVoice($Voice) } catch {}
  $s.Rate = 1
  $s.SetOutputToWaveFile($it.path, $fmt)
  $s.Speak($it.text)
  $s.Dispose()
}
"""


def synth(texts: list[str], tmp: Path) -> list[np.ndarray]:
    """Synthesize every text in one PowerShell run; return trimmed float32 arrays."""
    items = [{"path": str(tmp / f"p{i:04d}.wav"), "text": t} for i, t in enumerate(texts)]
    (tmp / "manifest.json").write_text(json.dumps(items), encoding="utf-8")
    (tmp / "synth.ps1").write_text(PS_SCRIPT, encoding="utf-8-sig")
    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
         str(tmp / "synth.ps1"), "-Manifest", str(tmp / "manifest.json"), "-Voice", VOICE],
        check=True,
    )
    return [trim(read_wav(Path(it["path"]))) for it in items]


def read_wav(p: Path) -> np.ndarray:
    with wave.open(str(p), "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2, p
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def trim(x: np.ndarray, thr: float = 0.004, margin_ms: int = 25) -> np.ndarray:
    """Strip SAPI's own leading/trailing silence so our gaps are exact."""
    idx = np.flatnonzero(np.abs(x) > thr)
    if idx.size == 0:
        return x
    m = SR * margin_ms // 1000
    return x[max(0, idx[0] - m): idx[-1] + 1 + m]


def silence(ms: float) -> np.ndarray:
    return np.zeros(int(round(SR * ms / 1000)), dtype=np.float32)


def write_wav(p: Path, x: np.ndarray) -> None:
    pcm = np.clip(np.round(x * 32767), -32768, 32767).astype("<i2")
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def split_at_comma(s: str) -> tuple[str, str] | None:
    """Split at the comma nearest the middle (>=2 words on each side)."""
    pos = [i for i, c in enumerate(s) if c == "," and s[i + 1:i + 2] == " " and len(s[:i].split()) >= 2 and len(s[i + 1:].split()) >= 2]
    if not pos:
        return None
    i = min(pos, key=lambda k: abs(k - len(s) / 2))
    return s[: i + 1].strip(), s[i + 1:].strip()


def plan(sentences: list[tuple[str, int | None]], rng: random.Random):
    """Return (pieces [(text, sentence_idx)], pause-after-each-piece [(ms, kind)])."""
    cand = [i for i, (s, _) in enumerate(sentences) if split_at_comma(s)]
    chosen = {i for i in cand if rng.random() < HESITATE_P}
    if not chosen and cand and len(sentences) >= 4:   # make sure longer clips have one
        chosen = {cand[len(cand) // 2]}
    pieces: list[tuple[str, int]] = []
    pauses: list[tuple[int, str]] = []
    for i, (s, forced) in enumerate(sentences):
        parts = split_at_comma(s) if i in chosen else None
        for part in (parts or (s,)):
            pieces.append((part, i))
            pauses.append((rng.randint(*HESITATE_MS), "hesitation"))
        pauses[-1] = (forced if forced is not None else rng.randint(*GAP_MS), "sentence")
    return pieces, pauses


def build(name: str, sentences: list[tuple[str, int | None]], reference: str, raw: str | None, tmp: Path):
    rng = random.Random(f"{SEED}:{name}")
    pieces, pauses = plan(sentences, rng)
    audio = synth([p[0] for p in pieces], tmp)
    parts = [silence(LEAD_MS)]
    t = LEAD_MS / 1000
    timings = []
    pause_list = [{"start_s": 0.0, "end_s": LEAD_MS / 1000, "ms": LEAD_MS, "kind": "lead"}]
    for k, (a, (text, sidx)) in enumerate(zip(audio, pieces)):
        d = len(a) / SR
        timings.append({"start_s": round(t, 3), "end_s": round(t + d, 3), "text": text, "sentence": sidx})
        parts.append(a)
        t += d
        ms, kind = (TRAIL_MS, "trail") if k == len(audio) - 1 else pauses[k]
        parts.append(silence(ms))
        pause_list.append({"start_s": round(t, 3), "end_s": round(t + ms / 1000, 3), "ms": ms, "kind": kind})
        t += ms / 1000
    x = np.concatenate(parts)
    nrng = np.random.default_rng(zlib.crc32(f"{SEED}:{name}".encode()))
    x = x + nrng.normal(0.0, NOISE_RMS, x.shape).astype(np.float32)
    write_wav(OUT / f"{name}.wav", x)
    (OUT / f"{name}.txt").write_text(reference + "\n", encoding="utf-8")
    if raw is not None:
        (OUT / f"{name}.raw.txt").write_text(raw + "\n", encoding="utf-8")
    meta = {"duration_s": round(len(x) / SR, 3), "pieces": timings, "pauses": pause_list}
    (OUT / f"{name}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    want = set(sys.argv[1:])
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for name, sents in CLIPS.items():
            if want and name not in want:
                continue
            m = build(name, [(s, None) for s in sents], " ".join(sents), None, tmp)
            print(f"{name}: {m['duration_s']:.2f}s, {len(m['pieces'])} pieces")
        if not want or "bench_corr" in want:
            sents = [(s, p) for s, _, p in CORR_SEGMENTS]
            ref = " ".join(r for _, r, _ in CORR_SEGMENTS if r)
            raw = " ".join(s for s, _, _ in CORR_SEGMENTS)
            m = build("bench_corr", sents, ref, raw, tmp)
            print(f"bench_corr: {m['duration_s']:.2f}s, {len(m['pieces'])} pieces")


if __name__ == "__main__":
    main()
