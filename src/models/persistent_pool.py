"""
PersistentModelPool: Extended ModelPool with persistence and quality tracking

Extends ModelPool with:
- Cache index persistence (survives restarts)
- Metadata tracking per model
- Quality scores
- Warm-up mechanism
- Enhanced statistics

Usage:
    pool = PersistentModelPool(max_models=50, cache_dir=".model_cache")

    # Add with metadata
    pool.add_model_with_metadata(
        "BTC_LSTM_v2", model,
        metadata={"ticker": "BTC", "version": "2.0"},
        quality_score=0.85
    )

    # Get with quality check
    model = pool.get_model_with_quality_check(
        "BTC_LSTM_v2", loader_fn, min_quality=0.7
    )

    # Warm-up
    pool.warm_up(["model1", "model2"], loader_fns)
"""
import json
import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.core.logging.logger import ProjectLogger
from src.models.model_pool import ModelPool

logger = ProjectLogger.get_logger(__name__)


class PersistentModelPool(ModelPool):
    """
    Extended model pool with persistence and quality tracking.

    New features:
    - Metadata tracking per model
    - Quality scores
    - Warm-up mechanism
    - Cache index persistence
    - Export/import

    Attributes:
        cache_dir: Directory for cache index
        model_metadata: Dict[model_id, metadata]
        quality_scores: Dict[model_id, quality_score]
    """

    def __init__(self, max_models: int = 50, cache_dir: str = ".model_cache"):
        """
        Initialize persistent model pool.

        Args:
            max_models: Maximum number of models to keep in memory
            cache_dir: Directory for cache index persistence
        """
        super().__init__(max_models)

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.model_metadata: dict[str, dict[str, Any]] = {}
        self.quality_scores: dict[str, float] = {}

        # Load existing cache index
        self._load_cache_index()

        logger.info(f"PersistentModelPool initialized: cache_dir={self.cache_dir}, max_models={max_models}")

    def add_model_with_metadata(
        self,
        model_id: str,
        model: Any,
        metadata: dict[str, Any],
        quality_score: float = 0.0
    ) -> None:
        """
        Add model with metadata and quality score.

        Args:
            model_id: Unique model identifier
            model: Model instance
            metadata: Model metadata (ticker, version, etc.)
            quality_score: Quality score (0.0-1.0)

        Example:
            pool.add_model_with_metadata(
                "BTC_LSTM_v2", model,
                metadata={"ticker": "BTC", "version": "2.0", "trained_at": "2026-05-01"},
                quality_score=0.85
            )
        """
        # Add to base pool
        self.add_model(model_id, model)

        # Store metadata
        self.model_metadata[model_id] = {
            **metadata,
            'added_at': datetime.now().isoformat(),
            'version': metadata.get('version', '1.0.0')
        }

        # Store quality score
        self.quality_scores[model_id] = quality_score

        # Persist to disk
        self._save_cache_index()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Added {model_id} with quality {quality_score:.2f}")

    def get_model_with_quality_check(
        self,
        model_id: str,
        loader_fn: Callable,
        min_quality: float = 0.5
    ) -> Any | None:
        """
        Get model with quality validation.

        Args:
            model_id: Model identifier
            loader_fn: Function to load model if not in pool
            min_quality: Minimum acceptable quality score

        Returns:
            Model instance or None if quality too low

        Example:
            model = pool.get_model_with_quality_check(
                "BTC_LSTM_v2", loader_fn, min_quality=0.7
            )
            if model:
                predictions = model.predict(features)
        """
        # Get model from base pool
        model = self.get_model(model_id, loader_fn)
        if not model:
            return None

        # Check quality
        quality = self.quality_scores.get(model_id, 0.0)
        if quality < min_quality:
            logger.warning(
                f"Model {model_id} quality {quality:.2f} below threshold {min_quality}"
            )
            return None

        return model

    def warm_up(self, model_ids: list[str], loader_fns: dict[str, Callable]) -> None:
        """
        Pre-load models into pool.

        Args:
            model_ids: List of model IDs to load
            loader_fns: Dict mapping model_id to loader function

        Example:
            critical_models = ["catboost_v1", "lightgbm_v1", "xgboost_v1"]
            loader_fns = {
                model: lambda m=model: registry.clone(m)
                for model in critical_models
            }
            pool.warm_up(critical_models, loader_fns)
        """
        logger.info(f"Warming up {len(model_ids)} models...")
        loaded = 0

        for model_id in model_ids:
            if model_id not in loader_fns:
                logger.warning(f"No loader function for {model_id}")
                continue

            model = self.get_model(model_id, loader_fns[model_id])
            if model:
                loaded += 1

        logger.info(f"Warm-up complete: {loaded}/{len(model_ids)} loaded")

    def update_quality_score(self, model_id: str, new_score: float) -> None:
        """
        Update quality score for model.

        Args:
            model_id: Model identifier
            new_score: New quality score (0.0-1.0)
        """
        if model_id in self.quality_scores:
            old_score = self.quality_scores[model_id]
            self.quality_scores[model_id] = new_score
            self._save_cache_index()
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Updated {model_id} quality: {old_score:.2f} → {new_score:.2f}")
        else:
            logger.warning(f"Model {model_id} not found in quality scores")

    def get_metadata(self, model_id: str) -> dict[str, Any] | None:
        """
        Get metadata for model.

        Args:
            model_id: Model identifier

        Returns:
            Metadata dict or None if not found
        """
        return self.model_metadata.get(model_id)

    def _load_cache_index(self) -> None:
        """Load cache index from disk."""
        index_path = self.cache_dir / "cache_index.json"
        if not index_path.exists():
            logger.info("No existing cache index found. Starting fresh.")
            return

        try:
            with open(index_path) as f:
                data = json.load(f)
                self.model_metadata = data.get('metadata', {})
                self.quality_scores = data.get('quality', {})
            logger.info(f"Loaded cache index: {len(self.model_metadata)} models")
        except Exception as e:
            logger.error(f"Failed to load cache index: {e}")

    def _save_cache_index(self) -> None:
        """Save cache index to disk."""
        index_path = self.cache_dir / "cache_index.json"

        try:
            with open(index_path, 'w') as f:
                json.dump({
                    'metadata': self.model_metadata,
                    'quality': self.quality_scores,
                    'updated_at': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")

    def get_enhanced_stats(self) -> dict[str, Any]:
        """
        Get enhanced statistics.

        Returns:
            Dict with base stats + quality metrics
        """
        base_stats = self.get_stats()

        # Calculate average quality
        if self.quality_scores:
            avg_quality = np.mean(list(self.quality_scores.values()))
        else:
            avg_quality = 0.0

        return {
            **base_stats,
            'avg_quality': avg_quality,
            'models_with_metadata': len(self.model_metadata),
            'cache_dir': str(self.cache_dir)
        }

    def export_to_disk(self, export_dir: str) -> None:
        """
        Export all models to disk.

        Args:
            export_dir: Directory to export models
        """
        import joblib

        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)

        with self._lock:
            for model_id, model in self.models.items():
                model_path = export_path / f"{model_id}.joblib"
                try:
                    joblib.dump(model, model_path)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Exported {model_id} to {model_path}")
                except Exception as e:
                    logger.error(f"Failed to export {model_id}: {e}")

        # Save cache index
        self._save_cache_index()
        logger.info(f"Exported {len(self.models)} models to {export_dir}")


# Global singleton
_persistent_pool: PersistentModelPool | None = None


def get_persistent_pool(max_models: int = 50, cache_dir: str = ".model_cache") -> PersistentModelPool:
    """
    Get or create global persistent pool (singleton).

    Args:
        max_models: Maximum models in pool
        cache_dir: Cache directory

    Returns:
        Global PersistentModelPool instance
    """
    global _persistent_pool

    if _persistent_pool is None:
        _persistent_pool = PersistentModelPool(max_models=max_models, cache_dir=cache_dir)

    return _persistent_pool
