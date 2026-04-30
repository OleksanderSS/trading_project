"""
Model ensemble manager for combining light and heavy model predictions.

The previous file was truncated mid-import, which prevented the repository from
passing even a syntax audit. This implementation keeps the public surface small
and safe while the richer ensemble logic can evolve around it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger


logger = ProjectLogger.get_logger(__name__)


@dataclass
class EnsembleMember:
    """A loaded model and its voting weight."""

    name: str
    model: Any
    weight: float = 1.0


class ModelEnsembleManager:
    """Loads models and produces weighted-average predictions."""

    def __init__(self, models_dir: str | Path = "data/trained_models"):
        self.models_dir = Path(models_dir)
        self.members: List[EnsembleMember] = []

    def load_model(self, name: str, path: str | Path, weight: float = 1.0) -> bool:
        """Load a serialized model from disk into the ensemble."""
        model_path = Path(path)
        if not model_path.exists():
            logger.warning("Model file not found: %s", model_path)
            return False

        model = joblib.load(model_path)
        self.members.append(EnsembleMember(name=name, model=model, weight=float(weight)))
        logger.info("Loaded ensemble member '%s' from %s", name, model_path)
        return True

    def load_directory(self, pattern: str = "*.joblib") -> int:
        """Load every matching model artifact from the configured directory."""
        if not self.models_dir.exists():
            logger.warning("Models directory not found: %s", self.models_dir)
            return 0

        loaded = 0
        for model_path in self.models_dir.glob(pattern):
            loaded += int(self.load_model(model_path.stem, model_path))
        return loaded

    def predict(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return weighted ensemble predictions."""
        if not self.members:
            raise ValueError("No ensemble members loaded")

        predictions = []
        weights = []
        for member in self.members:
            pred = member.model.predict(features)
            predictions.append(np.asarray(pred, dtype=float))
            weights.append(member.weight)

        return np.average(np.vstack(predictions), axis=0, weights=np.asarray(weights))

    def get_metadata(self) -> Dict[str, Any]:
        """Return a compact summary of the loaded ensemble."""
        return {
            "models_dir": str(self.models_dir),
            "members": [{"name": m.name, "weight": m.weight} for m in self.members],
            "member_count": len(self.members),
        }
