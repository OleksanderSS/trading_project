from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.targets.target_orchestrator import TargetOrchestrator

logger = ProjectLogger.get_logger('TargetGenerator')

class TargetGenerator:
    """Generates machine learning targets/labels."""

    def __init__(self, config_manager: Any):
        self.logger = logger
        targets_list = config_manager.get('targets').as_dict() if hasattr(config_manager.get('targets'), 'as_dict') else config_manager.get('targets')
        self.target_orchestrator = TargetOrchestrator(targets_list=targets_list)

    def generate_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all configured targets."""
        self.logger.info("Generating machine learning targets...")
        return self.target_orchestrator.generate_targets(df)

    def append_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the original features with aligned target columns appended."""
        targets_df = self.generate_targets(df)
        target_cols = [col for col in targets_df.columns if col.startswith('target_')]
        if not target_cols:
            return df

        result = df.copy()
        for col in target_cols:
            result[col] = targets_df[col].reindex(result.index)
        return result
