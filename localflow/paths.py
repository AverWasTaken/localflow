"""Filesystem locations used by LocalFlow."""
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("APPDATA", str(Path.home()))) / "LocalFlow"
CONFIG_PATH = DATA_DIR / "config.json"
HISTORY_DB = DATA_DIR / "history.db"
LOG_PATH = DATA_DIR / "localflow.log"
BIN_DIR = PROJECT_DIR / "bin"
MODELS_DIR = PROJECT_DIR / "models"
ASSETS_DIR = PROJECT_DIR / "localflow" / "assets"

DATA_DIR.mkdir(parents=True, exist_ok=True)
