"""
Pipeline Executor for Hybrid Orchestrator.
Handles pipeline stage execution and result management.
"""

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)


class PipelineExecutor:
    """Executes pipeline stages and manages results."""

    def __init__(self, config, data_manager, feature_processor):
        self.config = config
        self.data_manager = data_manager
        self.feature_processor = feature_processor
        from .metadata_manager import MetadataManager
        self.metadata_manager = MetadataManager(config)
        self.logger = ProjectLogger.get_logger(__name__)

    async def run_pipeline_stages(self, tickers: list[str] | None,
                                   timeframes: list[str] | None,
                                   stages: list[int]) -> dict[str, Any]:
        """Runs pipeline stages."""
        results = {}
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        for stage in stages:
            self.logger.info(f"Running stage {stage}")

            if stage == 1:
                results['stage_1'] = await self._run_stage_1(tickers, timeframes)
            elif stage == 2:
                results['stage_2'] = await self._run_stage_2(tickers, timeframes)
            elif stage == 3:
                results['stage_3'] = await self._run_stage_3(tickers, timeframes)
            # Add more stages as needed

        # Save results
        saved_files = self._save_pipeline_results(results, timestamp)

        return {
            'results': results,
            'saved_files': saved_files,
            'timestamp': timestamp,
            'stages_completed': stages
        }

    async def _run_stage_1(self, tickers: list[str] | None,
                           timeframes: list[str] | None) -> dict[str, Any]:
        """Run stage 1 - Data collection."""
        # Implementation would go here
        return {'status': 'completed', 'data_collected': True}

    async def _run_stage_2(self, tickers: list[str] | None,
                           timeframes: list[str] | None) -> dict[str, Any]:
        """Run stage 2 - Data cleaning."""
        # Implementation would go here
        return {'status': 'completed', 'data_cleaned': True}

    async def _run_stage_3(self, tickers: list[str] | None,
                           timeframes: list[str] | None) -> dict[str, Any]:
        """Run stage 3 - Data enrichment."""
        # Implementation would go here
        return {'status': 'completed', 'data_enriched': True}

    def _save_pipeline_results(self, results: dict[str, Any], timestamp: str) -> dict[str, str]:
        """Saves results from pipeline stages."""
        saved_files = {}

        for stage_name, stage_result in results.items():
            filename = f"{stage_name}_{timestamp}.pkl"
            saved_files[stage_name] = str(self._save_stage_result(stage_result, filename))

        return saved_files

    def _save_stage_result(self, data: Any, filename: str) -> Path:
        """Save stage result to file."""
        path = self.config.output_dir / filename

        if isinstance(data, pd.DataFrame):
            data.to_parquet(path)
        elif isinstance(data, dict):
            with open(path.with_suffix('.json'), 'w') as f:
                json.dump(data, f, indent=2, default=str)
        else:
            with open(path, 'wb') as f:
                pickle.dump(data, f)

        return path

    def create_pipeline_metadata(self, timestamp: str, tickers: list[str] | None,
                                  timeframes: list[str] | None, stages: list[int],
                                  saved_files: dict[str, str], batch_name: str) -> dict[str, Any]:
        """Create pipeline metadata."""
        from .metadata_manager import MetadataParams
        metadata_params = MetadataParams(
            timestamp=timestamp,
            tickers=tickers,
            timeframes=timeframes,
            stages=stages,
            saved_files=saved_files,
            batch_name=batch_name
        )
        return self.metadata_manager.create_pipeline_metadata(metadata_params)

    def save_metadata(self, metadata: dict[str, Any], timestamp: str) -> None:
        """Save metadata to files."""
        from .metadata_manager import SaveMetadataParams
        save_params = SaveMetadataParams(
            metadata=metadata,
            timestamp=timestamp,
            batch_name="pipeline",
            output_dir=self.config.output_dir
        )
        self.metadata_manager.save_metadata(save_params)

    def save_data(self, data: dict[str, pd.DataFrame], path: Path) -> None:
        """Saves a dictionary of DataFrames to parquet."""
        if not data:
            return

        try:
            flat_data = self._flatten_data_dict(data)
            if not flat_data:
                return

            path.mkdir(parents=True, exist_ok=True)

            for key, df in flat_data.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    file_path = path / f"{key}.parquet"
                    df.to_parquet(file_path)
                    self.logger.info(f"Saved {key} data to {file_path}")

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error saving data: {e}")

    def _flatten_data_dict(self, data: dict[str, Any], parent_key: str = "") -> dict[str, pd.DataFrame]:
        """Flatten nested dictionary of DataFrames."""
        flat = {}

        for key, value in data.items():
            new_key = f"{parent_key}_{key}" if parent_key else key

            if isinstance(value, dict):
                nested = self._flatten_data_dict(value, new_key)
                flat.update(nested)
            elif isinstance(value, pd.DataFrame):
                flat[new_key] = value

        return flat
