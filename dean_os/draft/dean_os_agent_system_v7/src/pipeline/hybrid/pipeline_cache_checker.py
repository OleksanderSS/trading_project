"""
Pipeline Cache Checker - Checks cache before running pipeline stages.
This module provides cache validation to avoid running pipeline stages 0-3 when data is already cached.
"""

import pandas as pd

from src.core.logging.logger import ProjectLogger


class PipelineCacheChecker:
    """Checks if valid cache exists before running pipeline."""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.logger = ProjectLogger.get_logger(__name__)

    def check_cache_before_run(self, force_training: bool = False) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        """
        Check if valid cache exists before running pipeline.
        Returns (features_df, targets_df) if cache is valid, None otherwise.
        """
        if force_training:
            self.logger.info("Force training mode - skipping cache check")
            return None

        # Get cache manager from orchestrator
        cache_manager = getattr(self.orchestrator, "data_cache_manager", None)
        if cache_manager is None:
            self.logger.warning("No cache manager found in orchestrator")
            return None

        # Get paths
        output_dir = self.orchestrator.config.output_dir
        batch_dir = output_dir / self.orchestrator.batch_name
        features_path = batch_dir / 'features.parquet'
        targets_path = batch_dir / 'targets.parquet'

        # Check if files exist
        if not features_path.exists() or not targets_path.exists():
            self.logger.info(f"Cache files not found: features={features_path.exists()}, targets={targets_path.exists()}")
            return None

        # Try to read cache
        try:
            features_df = pd.read_parquet(features_path)
            targets_df = pd.read_parquet(targets_path)

            # Validate data
            if features_df.empty or targets_df.empty:
                self.logger.warning("Cache files are empty")
                return None

            self.logger.info(f"✅ Found valid cache: features={features_df.shape}, targets={targets_df.shape}")
            return features_df, targets_df

        except Exception as e:
            self.logger.warning(f"Error reading cache: {e}")
            return None

    def should_use_cache(self, force_training: bool = False) -> bool:
        """
        Quick check if cache should be used.
        Returns True if cache exists and is valid.
        """
        return self.check_cache_before_run(force_training) is not None
