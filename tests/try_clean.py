"""Poke the cleaner interactively. stdin lines look like  "[casual]some raw text"  (style prefix optional).
Uses its own llama-server on :8799 so the running app's server on :8765 is never touched."""
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from localflow import styles
from localflow.cleanup import Cleaner
from localflow.config import Config
cfg = Config(path=Path(tempfile.mkdtemp()) / "c.json")
cfg.set("cleanup.port", 8799, save=False)
cl = Cleaner(cfg)
cl.server.start(wait=True); cl._warm()
for line in sys.stdin.read().splitlines():
    style = "neutral"
    if line.startswith("["):
        style, line = line[1:].split("]", 1)
    raw = line.strip()
    sd = styles.get_style(cfg, style)
    out = cl._llm(raw, sd, [])
    print("RAW :", raw, "\nLLM :", repr(out), "\nGUARD:", cl._sane(raw, out, sd) or "ok", "\n")
cl.shutdown()
