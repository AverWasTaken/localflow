"""Manual hardware check (opens the real mic for ~3 s each): reports device + level for the Windows default
mic, for a name that does not exist (must fall back to the default), and for any mic.match given on the command
line, e.g. ``python tests/test_mic.py usb``. Not part of the offline unit tests."""
import sys, tempfile, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from localflow.audio import AudioInput
from localflow.config import Config

for match in ("", "no-such-mic-xyz", *sys.argv[1:]):
    cfg = Config(path=Path(tempfile.mkdtemp()) / "c.json")
    cfg.data["mic"]["match"] = match
    levels = []
    a = AudioInput(cfg, on_level=levels.append)
    a.start()
    time.sleep(0.8)
    a.begin()
    time.sleep(2.0)
    x = a.end()
    print(f"match={match!r}: device={a.device_name!r} preferred_found={a.preferred_found} "
          f"captured {len(x)/16000:.2f}s @16k, rms={np.sqrt(np.mean(x**2)):.4f}, peak={np.abs(x).max():.3f}, callbacks={len(levels)}")
    a.stop()
