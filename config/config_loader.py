# config/config_loader.py

import yaml
import json
from pathlib import Path
from functools import lru_cache
import logging
import os


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

@lru_cache(maxsize=None)
def load_yaml_config(path: str, use_cache: bool = True) -> dict:
    """
    Заванandжує конфandг у форматand YAML or JSON.
    Використовує кешування for оптимandforцandї (can вимкнути череwith use_cache=False).
    Fail-fast: пandднandмає помилки при вandдсутностand fileу or синandксичних errorх.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"[ConfigLoader] File not found: {path}")

    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        elif path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            raise ValueError(f"[ConfigLoader] Unsupported config format: {path.suffix}")

        logger.debug(f"[ConfigLoader] Config loaded from {path}")
        return data

    except (yaml.YAMLError, json.JSONDecodeError) as e:
        raise RuntimeError(f"[ConfigLoader] Error parsing {path}: {e}")
    except Exception as e:
        raise RuntimeError(f"[ConfigLoader] Unexpected error reading {path}: {e}")


def reload_config(path: str) -> dict:
    """
    Переforванandження конфandгу with диску, andгноруючи кеш.
    """
    load_yaml_config.cache_clear()
    return load_yaml_config(path, use_cache=False)
