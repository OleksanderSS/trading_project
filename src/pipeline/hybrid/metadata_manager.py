"""
Metadata Manager for Hybrid Orchestrator.
Handles all metadata creation and management operations.
"""

import json
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.core.logging.logger import ProjectLogger


@dataclass
class MetadataParams:
    """Parameters for metadata creation."""
    timestamp: str
    tickers: list[str] | None
    timeframes: list[str] | None
    stages: list[int]
    saved_files: dict[str, str]
    batch_name: str


@dataclass
class SaveMetadataParams:
    """Parameters for metadata saving."""
    metadata: dict[str, Any]
    timestamp: str
    batch_name: str
    output_dir: Path


@dataclass
class BatchMetadataParams:
    """Parameters for batch metadata creation."""
    metadata: dict[str, Any]
    timestamp: str
    batch_name: str

logger = ProjectLogger.get_logger(__name__)


class MetadataManager:
    """Manages metadata operations for hybrid pipeline."""

    def __init__(self, config):
        self.config = config
        self.logger = ProjectLogger.get_logger(__name__)

    def create_pipeline_metadata(self, params: MetadataParams) -> dict[str, Any]:
        """Create pipeline metadata."""
        return {
            'timestamp': params.timestamp,
            'batch_name': params.batch_name,
            'tickers': params.tickers or [],
            'timeframes': params.timeframes or [],
            'stages_completed': params.stages,
            'saved_files': params.saved_files,
            'light_models': self.config.get('models.pipeline.light_models', []),
            'heavy_models': self.config.get('models.pipeline.heavy_models', [])
        }

    def save_metadata(self, params: SaveMetadataParams) -> None:
        """Save metadata to files."""
        metadata_path = params.output_dir / f"{params.batch_name}_metadata_{params.timestamp}.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(params.metadata, f, indent=2, ensure_ascii=False)

        # Also save as pickle for faster loading
        pickle_path = params.output_dir / f"{params.batch_name}_metadata_{params.timestamp}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(params.metadata, f)

        self.logger.info(f"Metadata saved: {metadata_path}")

    def create_batch_metadata_dict(self, params: BatchMetadataParams) -> dict[str, Any]:
        """Create batch metadata dictionary."""
        return {
            'batch_name': params.batch_name,
            'timestamp': params.timestamp,
            'pipeline_metadata': params.metadata,
            'files': params.metadata['saved_files']
        }

    def create_final_summary(self, results: dict[str, Any], tickers: list[str] | None) -> dict[str, Any]:
        """Create final execution summary."""
        return {
            'status': 'completed',
            'results': results,
            'tickers': tickers or [],
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': results.get('duration_seconds', 0)
        }

    def save_final_results(self, final_summary: dict[str, Any], output_dir: Path) -> Path:
        """Save final results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"final_results_{timestamp}.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_summary, f, indent=2, ensure_ascii=False)

        return output_path

    def log_pipeline_completion(self, total_duration: float) -> None:
        """Log pipeline completion."""
        self.logger.info(f"Total time: {total_duration:.1f}s ({total_duration/60:.1f}m)")

    def extract_batch_name_from_path(self, path_str: str) -> str | None:
        """Extract batch name from path."""
        parts = Path(path_str.replace('/', '\\')).parts
        if 'accumulated' in parts:
            idx = parts.index('accumulated')
            if len(parts) > idx + 1:
                return parts[idx + 1]
        return None
