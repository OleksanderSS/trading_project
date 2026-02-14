# config/config_manager.py

import os
from pathlib import Path
import logging

logger = logging.getLogger("config_manager")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

def detect_environment() -> str:
    """Detects execution environment: Colab, PyCharm, VSCode, CI or local."""
    # Check for actual Colab environment (not just /content directory)
    colab_indicators = [
        'COLAB_GPU' in os.environ,
        'COLAB_TPU_ADDR' in os.environ,
        'DATALAB_SETTINGS' in os.environ,
        Path('/usr/local/bin/jupyter').exists() and Path('/content/drive').exists()
    ]
    
    if any(colab_indicators):
        env = "colab"
    elif "PYCHARM_HOSTED" in os.environ:
        env = "pycharm"
    elif "VSCODE_PID" in os.environ:
        env = "vscode"
    elif "GITHUB_ACTIONS" in os.environ or "CI" in os.environ:
        env = "ci"
    else:
        env = "local"
    logger.info(f"[ConfigManager] Environment detected: {env}")
    return env

def resolve_paths(env: str) -> dict:
    """Returns dictionary with all necessary paths according to environment."""
    if env == "colab":
        base = Path("/content/drive/MyDrive/trading_project")
    else:
        base = Path(os.environ.get("PROJECT_BASE", Path(__file__).resolve().parent.parent))

    return {
        "data": str(base / "data"),
        "models": str(base / "models"),
        "config": str(base / "config"),
        "output": str(base / "output"),
        "logs": str(base / "logs"),
        "news_config": str(base / "config" / "news_sources.yaml"),
        "news_data": str(base / "data" / "news"),
        "price_data": str(base / "data" / "prices"),
        "macro_data": str(base / "data" / "macro"),
        "db": str(base / "data" / "news.db"),
    }
