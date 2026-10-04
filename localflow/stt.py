"""Speech-to-text: faster-whisper (CTranslate2) kept warm on the GPU, with hallucination filtering."""
from __future__ import annotations

import glob
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# Blackwell (sm_120) has no precompiled CTranslate2 kernels in some wheels; the driver JIT-compiles PTX
# the first time (~30 s) and caches it. Make sure that cache is big enough to never be evicted.
os.environ.setdefault("CUDA_CACHE_MAXSIZE", str(4 * 1024 ** 3))
os.environ.setdefault("CUDA_CACHE_DISABLE", "0")

MODEL_REPOS = {
    "large-v3-turbo": "h2oai/faster-whisper-large-v3-turbo",
    "turbo": "h2oai/faster-whisper-large-v3-turbo",
    "small.en": "Systran/faster-whisper-small.en",
    "base.en": "Systran/faster-whisper-base.en",
    "small": "Systran/faster-whisper-small",
    "base": "Systran/faster-whisper-base",
    "medium.en": "Systran/faster-whisper-medium.en",
    "large-v3": "Systran/faster-whisper-large-v3",
}

_dll_dirs_added = False


def add_cuda_dll_dirs() -> None:
    """Make the pip-installed nvidia-cublas/cudnn DLLs findable by ctranslate2."""
    global _dll_dirs_added
    if _dll_dirs_added:
        return
    _dll_dirs_added = True
    try:
        import nvidia
        for base in nvidia.__path__:
            for b in glob.glob(os.path.join(base, "*", "bin")):
                os.add_dll_directory(b)
                os.environ["PATH"] = b + os.pathsep + os.environ.get("PATH", "")
                log.debug("cuda dll dir: %s", b)
        # IMPORTANT: ctranslate2 otherwise resolves cublas64_12.dll from an old system CUDA toolkit on PATH
        # (12.4: no native sm_120 kernels -> encoder ~2.4x slower). Preload the pip 12.9 build by full path
        # so the loader reuses it by name.
        import ctypes
        for base in nvidia.__path__:
            for name in ("cublasLt64_12.dll", "cublas64_12.dll"):
                p = os.path.join(base, "cublas", "bin", name)
                if os.path.exists(p):
                    ctypes.CDLL(p)
                    log.info("preloaded %s", p)
    except Exception as e:
        log.warning("nvidia pip packages not found (%s); relying on system CUDA", e)


def _cuda_device_count() -> int:
    try:
        import ctranslate2
        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 1     # unknown: let the real load decide


# Classic Whisper hallucinations on silence / noise.
_HALLUCINATIONS = {
    "thank you", "thank you.", "thanks", "thanks.", "thank you very much", "thank you so much",
    "thanks for watching", "thank you for watching", "thanks for watching!", "thank you for watching!",
    "please subscribe", "like and subscribe", "bye", "bye.", "bye bye", "goodbye", "you", "you.",
    "i'm sorry", "so", "okay", "ok", "oh", "uh", "um", "hmm", "mm-hmm", "the end", "...", ".", "?", "!",
    "subtitles by the amara.org community", "transcribed by", "www.mooji.org", "ご視聴ありがとうございました",
}
_ALWAYS_DROP_SUBSTR = ("amara.org", "thanks for watching", "thank you for watching", "subscribe to",
                       "subtitles by", "captions by", "transcription by", "www.", ".com/")


def _norm(t: str) -> str:
    return re.sub(r"[^\w' ]+", "", t.lower()).strip()


@dataclass
class STTResult:
    text: str = ""
    duration_s: float = 0.0
    speech_s: float = 0.0
    ms: int = 0
    language: str = ""
    dropped: str = ""            # reason if the transcript was discarded
    avg_logprob: float = 0.0
    segments: list = field(default_factory=list)


class Transcriber:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self.device = "?"
        self.compute = "?"
        self.model_name = "?"
        self.ready = False
        self.error: Optional[str] = None
        self.load_ms = 0
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ loading
    @staticmethod
    def _patch_dynamic_padding(min_frames: int = 1000) -> None:
        """Whisper pads every window to 30 s (3000 frames), so a 3 s dictation costs as much as 30 s.
        Pad to the next 2.5 s bucket instead, with a floor of 10 s (shorter floors cause repeated text)."""
        import faster_whisper.transcribe as ft
        if getattr(ft, "_lf_patched", False):
            return
        orig = ft.pad_or_trim

        def dyn(arr, length=3000, *, axis=-1):
            n = arr.shape[axis]
            if length != 3000:
                return orig(arr, length, axis=axis)
            tgt = min(3000, max(min_frames, int(np.ceil(n / 250.0)) * 250))
            return orig(arr, tgt, axis=axis)

        ft.pad_or_trim = dyn
        ft._lf_patched = True

    def _resolve(self, name: str) -> tuple[str, bool]:
        if os.path.isdir(name):
            return name, True
        repo = MODEL_REPOS.get(name, name)
        try:
            from huggingface_hub import snapshot_download
            return snapshot_download(repo, local_files_only=True), True
        except Exception:
            return repo, False   # let faster-whisper download it

    def load(self) -> None:
        add_cuda_dll_dirs()
        from faster_whisper import WhisperModel
        self._patch_dynamic_padding()
        st = self.cfg.get("stt", {})
        t0 = time.perf_counter()
        name = st.get("model", "large-v3-turbo")
        device = st.get("device", "cuda")
        compute = st.get("compute_type", "float16")
        try:
            if device == "cuda" and _cuda_device_count() == 0:
                # skip straight to the CPU model instead of downloading the big GPU model just to fail on it
                raise RuntimeError("no CUDA device available")
            path, _ = self._resolve(name)
            self.model = WhisperModel(path, device=device, compute_type=compute)
            self.device, self.compute, self.model_name = device, compute, name
            self._warmup()
        except Exception as e:
            log.error("!!! GPU/primary STT load FAILED (%s: %s) - falling back to CPU int8 %s !!!",
                      type(e).__name__, e, st.get("cpu_fallback_model", "small.en"))
            self.error = f"{type(e).__name__}: {e}"
            fb = st.get("cpu_fallback_model", "small.en")
            path, _ = self._resolve(fb)
            self.model = WhisperModel(path, device="cpu", compute_type="int8", cpu_threads=8)
            self.device, self.compute, self.model_name = "cpu", "int8", fb
            self._warmup()
        self.load_ms = int((time.perf_counter() - t0) * 1000)
        self.ready = True
        log.info("STT ready: model=%s device=%s compute=%s load=%d ms", self.model_name, self.device,
                 self.compute, self.load_ms)

    def _warmup(self) -> None:
        t = time.perf_counter()
        rng = np.random.default_rng(0)
        audio = (rng.standard_normal(16000 * 2) * 0.01).astype(np.float32)
        for _ in range(2):
            segs, _ = self.model.transcribe(audio, beam_size=1, vad_filter=False, language="en",
                                            without_timestamps=True)
            list(segs)
        log.info("STT warm-up took %d ms", (time.perf_counter() - t) * 1000)

    # ------------------------------------------------------------------ transcription
    def transcribe(self, audio: np.ndarray, hotwords: list[str] | None = None,
                   initial_prompt: str | None = None) -> STTResult:
        """Transcribe 16 kHz mono float32. ``initial_prompt`` (the tail of the previous text) gives continuity
        when a long dictation is transcribed in chunks. Serialized: safe to call from several threads."""
        with self._lock:
            return self._transcribe(audio, hotwords, initial_prompt)

    def _transcribe(self, audio: np.ndarray, hotwords: list[str] | None, initial_prompt: str | None) -> STTResult:
        st = self.cfg.get("stt", {})
        res = STTResult(duration_s=len(audio) / 16000.0)
        t0 = time.perf_counter()
        lang = st.get("language", "en")
        lang = None if lang in ("auto", "", None) else lang
        if self.model_name.endswith(".en"):
            lang = "en"
        kwargs = dict(
            beam_size=int(st.get("beam_size", 1)), language=lang, vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=350, speech_pad_ms=200, threshold=0.45),
            condition_on_previous_text=False, temperature=0.0,
            no_speech_threshold=float(st.get("no_speech_threshold", 0.6)),
            log_prob_threshold=float(st.get("log_prob_threshold", -1.0)),
            compression_ratio_threshold=2.4, without_timestamps=True,
        )
        if hotwords:
            hw = ", ".join(hotwords)
            kwargs["hotwords"] = hw
        if initial_prompt:
            kwargs["initial_prompt"] = initial_prompt
        segs_iter, info = self.model.transcribe(audio, **kwargs)
        segs = list(segs_iter)
        res.language = info.language
        res.speech_s = float(getattr(info, "duration_after_vad", res.duration_s))
        res.ms = int((time.perf_counter() - t0) * 1000)
        if not segs:
            res.dropped = "no speech (VAD)"
            return res
        res.segments = [(s.text, s.avg_logprob, s.no_speech_prob) for s in segs]
        text = " ".join(s.text.strip() for s in segs).strip()
        res.avg_logprob = float(np.mean([s.avg_logprob for s in segs]))
        max_nsp = max(s.no_speech_prob for s in segs)
        text = self._collapse_repeat(text)
        res.text = text
        reason = self._hallucination_reason(text, res, max_nsp)
        if reason:
            res.dropped = reason
            res.text = ""
        return res

    @staticmethod
    def _collapse_repeat(text: str) -> str:
        """'A. A.' / 'A A A' (whole transcript is one phrase repeated) -> 'A'."""
        words = text.split()
        n = len(words)
        for k in range(2, 5):
            if n % k == 0 and n >= 2:
                size = n // k
                chunk = [w.lower().strip(".,!?") for w in words[:size]]
                if all([w.lower().strip(".,!?") for w in words[i * size:(i + 1) * size]] == chunk for i in range(k)):
                    return " ".join(words[:size])
        return text

    @staticmethod
    def _hallucination_reason(text: str, res: STTResult, max_nsp: float) -> str:
        n = _norm(text)
        if not n:
            return "empty"
        low = text.lower()
        if any(s in low for s in _ALWAYS_DROP_SUBSTR) and res.speech_s < 6:
            return f"hallucination substring ({text[:40]!r})"
        if n in _HALLUCINATIONS or low.strip() in _HALLUCINATIONS:
            # a genuine one-word "thank you" is plausible; only drop when the audio looks like noise
            if res.speech_s < 1.5 and (max_nsp > 0.2 or res.avg_logprob < -0.7):
                return f"hallucination blacklist ({text[:40]!r})"
            if res.speech_s < 0.7:
                return f"hallucination blacklist/very short ({text[:40]!r})"
        words = n.split()
        if len(words) >= 6 and len(set(words)) <= 2:
            return "repetition loop"
        return ""
