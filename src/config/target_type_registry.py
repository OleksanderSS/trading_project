"""Target type taxonomy and registry loader.

Single source of truth for src/config/targets.yaml's `type:` field
(classification_binary / classification_multiclass / regression /
indicator_prediction) -- shared by the Colab training cell
(scripts/colab/colab_clean_cell.py's ConfigLoader), the live pipeline's
champion selector (src/pipeline/hybrid/champion_selector.py), and the
standalone CLI (scripts/colab/select_champions.py) so none of them can
drift out of sync on what a given target actually is.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CLASSIFICATION_BINARY_TYPE = "classification_binary"
CLASSIFICATION_MULTICLASS_TYPE = "classification_multiclass"
CLASSIFICATION_TARGET_TYPES = {CLASSIFICATION_BINARY_TYPE, CLASSIFICATION_MULTICLASS_TYPE}

DEFAULT_TARGETS_YAML = Path(__file__).resolve().parent / "targets.yaml"


def load_target_types(targets_yaml_path: Path | str = DEFAULT_TARGETS_YAML) -> dict[str, str]:
    """Read {target_name: type} for every target declared in targets.yaml.

    Returns {} (never raises) on a missing/unreadable file -- callers are
    expected to treat an unregistered target as "regression" via
    `.get(name, "regression")`, matching this project's pre-registry
    default behavior rather than crashing on an unrecognized target name.
    """
    path = Path(targets_yaml_path)
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
    except Exception:
        return {}
    return {
        name: str(cfg.get("type", "regression"))
        for name, cfg in (raw.get("targets") or {}).items()
        if isinstance(cfg, dict)
    }
