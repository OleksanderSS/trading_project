"""
Orchestrator Interface - Public API methods.
Contains all public methods that the main orchestrator delegates to.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger

from .pipeline_config import FinalStagesParams, PipelineParams

logger = ProjectLogger.get_logger(__name__)


class OrchestratorInterface:
    """Public interface methods for Hybrid Orchestrator."""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.logger = ProjectLogger.get_logger(__name__)

    def check_if_feature_selection_needed(self, batch_dir: Path, new_rows_count: int, force: bool = False) -> dict[str, Any]:
        """Checks if new feature selection is required."""
        # Check for initial run or forced selection
        initial_check = self._check_initial_or_forced_selection(batch_dir, force)
        if initial_check['needed']:
            return initial_check

        # Check if we have enough new data
        threshold = self.orchestrator.config.models_config.get('feature_selection', {}).get('new_data_threshold', 1000)
        if new_rows_count >= threshold:
            return {
                'needed': True,
                'reason': f'New data threshold reached: {new_rows_count} >= {threshold}',
                'threshold': threshold,
                'new_rows': new_rows_count
            }

        return {
            'needed': False,
            'reason': f'Insufficient new data: {new_rows_count} < {threshold}',
            'threshold': threshold,
            'new_rows': new_rows_count
        }

    def _check_initial_or_forced_selection(self, batch_dir: Path, force: bool = False) -> dict[str, Any]:
        """Check for initial run or forced selection."""
        if force:
            return {
                'needed': True,
                'reason': 'Feature selection forced by user',
                'forced': True
            }

        # Check if this is the first run
        feature_files = list(batch_dir.glob("selected_features_*.json"))
        if not feature_files:
            return {
                'needed': True,
                'reason': 'Initial feature selection required',
                'initial': True
            }

        return {
            'needed': False,
            'reason': 'Feature selection already exists',
            'existing_files': len(feature_files)
        }

    async def prepare_colab_data(self, batch_dir: Path, batch_name: str,
                               features_df: pd.DataFrame, targets_df: pd.DataFrame,
                               prices_dict: dict[str, pd.DataFrame] | None = None,
                               tickers: list[str] | None = None, timeframes: list[str] | None = None,
                               test_ticker: str | None = None, test_target: str | None = None,
                               test_model: str | None = None, epochs: int | None = None,
                               max_iterations: int | None = None, **kwargs) -> dict[str, Any]:
        """
        Prepare data for Colab training.

        Args:
            batch_dir: Directory path for the batch
            batch_name: Name of the batch
            features_df: Features DataFrame
            targets_df: Targets DataFrame
            prices_dict: Dictionary of price DataFrames by timeframe
            tickers: List of ticker symbols
            timeframes: List of timeframes
            test_ticker: Optional test ticker for test mode
            test_target: Optional test target for test mode
            test_model: Optional test model for test mode
            epochs: Optional epochs for test mode
            max_iterations: Optional max iterations for test mode
            **kwargs: Additional parameters for batch preparation

        Returns:
            Dictionary with batch preparation results including batch directory and metadata
        """
        self.logger.info("   Delegating to colab_manager to save batch data...")

        # Create BatchPreparationConfig
        from src.pipeline.hybrid.colab_manager import BatchPreparationConfig

        config = BatchPreparationConfig(
            tickers=tickers if tickers is not None else [],
            timeframes=timeframes if timeframes is not None else [],
            batch_name=batch_name,
            accumulate=kwargs.get('accumulate', True),
            force_feature_selection=kwargs.get('force_feature_selection', False),
            # Test mode parameters
            test_ticker=test_ticker,
            test_target=test_target,
            test_model=test_model,
            epochs=epochs,
            max_iterations=max_iterations
        )

        # Delegate to colab_manager to package and save the data
        if (features_df is None or features_df.empty) and (targets_df is None or targets_df.empty):
            self.logger.warning("Empty features/targets passed to prepare_colab_data. Loading from Parquet fallback.")
            try:
                features_df = pd.read_parquet("data/processed/features/enriched_features.parquet")
                targets_df = pd.read_parquet("data/processed/features/targets.parquet")
            except Exception as e:
                self.logger.error(f"❌ Failed to load fallback data from Parquet: {e}")
                return {}

        result = self.orchestrator.colab_manager.prepare_colab_batch(
            features_df=features_df,
            targets_df=targets_df,
            prices_dict=prices_dict or {}, # Pass prices_dict to colab_manager
            config=config,
            news_df=kwargs.get('news_df'),
            economic_df=kwargs.get('economic_df')
        )
        return result if result is not None else {}

    def load_colab_results(self, batch_name: str) -> dict[str, Any]:
        """Loads training results from Colab."""
        result = self.orchestrator.colab_manager.load_colab_results(batch_name)
        return result if result is not None else {}

    def extract_batch_name_from_path(self, path_str: str) -> str | None:
        """Extract batch name from path."""
        parts = Path(path_str.replace('/', '\\')).parts
        if 'accumulated' in parts:
            idx = parts.index('accumulated')
            if len(parts) > idx + 1:
                return parts[idx + 1]
        return None

    async def run_full_hybrid_pipeline(self, tickers: list[str] | None = None,
                                      timeframes: list[str] | None = None,
                                      accumulate: bool = True, force_training: bool = False,
                                      skip_colab: bool = False,
                                      force_feature_selection: bool = False) -> dict[str, Any]:
        """Run full hybrid pipeline with all parameters."""
        params = PipelineParams(
            tickers=tickers,
            timeframes=timeframes,
            accumulate=accumulate,
            force_training=force_training,
            skip_colab=skip_colab,
            force_feature_selection=force_feature_selection
        )

        result = await self.orchestrator.pipeline_manager.run_full_hybrid_pipeline(params)
        return result if result is not None else {}

    async def run_final_stages(self, features_df: pd.DataFrame | None, targets_df: pd.DataFrame | None,
                              colab_results: dict[str, Any] | None = None,
                              light_results: dict[str, Any] | None = None,
                              tickers: list[str] | None = None,
                              timeframes: list[str] | None = None,
                              batch_name: str | None = None,
                              news_data: pd.DataFrame | None = None,
                              economic_data: pd.DataFrame | None = None,
                              market_indicators: pd.DataFrame | None = None,
                              stages_to_run: list[int] | None = None) -> dict[str, Any]:
        """Run final stages of the pipeline."""
        params = FinalStagesParams(
            features_df=features_df,
            targets_df=targets_df,
            colab_results=colab_results,
            light_results=light_results,
            tickers=tickers,
            timeframes=timeframes,
            batch_name=batch_name,
            stages_to_run=stages_to_run,
            news_data=news_data,
            economic_data=economic_data,
            market_indicators=market_indicators
        )

        result = await self.orchestrator.pipeline_manager.run_final_stages(params)
        return result if result is not None else {}
