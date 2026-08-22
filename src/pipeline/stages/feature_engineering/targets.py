from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.pipeline.target_column_utils import is_direct_target_column
from src.targets.target_orchestrator import TargetOrchestrator

logger = ProjectLogger.get_logger('TargetGenerator')

class TargetGenerator:
    """Generates machine learning targets/labels."""

    def __init__(self, config_manager: Any):
        self.logger = logger
        targets_list = config_manager.get('targets').as_dict() if hasattr(config_manager.get('targets'), 'as_dict') else config_manager.get('targets')
        self.targets_list = targets_list
        self.target_orchestrator = TargetOrchestrator(targets_list=targets_list)

    def generate_targets(
        self,
        df: pd.DataFrame,
        timeframe: str | None = None,
    ) -> pd.DataFrame:
        """Generate all configured targets."""
        self.logger.info("Generating machine learning targets...")
        resolved_timeframe = timeframe or self._infer_timeframe(df)
        orchestrator = self.target_orchestrator
        if resolved_timeframe:
            orchestrator = TargetOrchestrator(
                targets_list=self.targets_list,
                timeframe=resolved_timeframe,
            )
        return orchestrator.generate_targets(df)

    def append_targets(
        self,
        df: pd.DataFrame,
        timeframe: str | None = None,
    ) -> pd.DataFrame:
        """Return the original features with aligned target columns appended."""
        targets_df = self.generate_targets(df, timeframe=timeframe)
        target_cols = [col for col in targets_df.columns if is_direct_target_column(col)]
        if not target_cols:
            return df

        # Whole-column assignment only, so a shallow copy is safe; see
        # `_restore_service_columns` for the measurement.
        result = df.copy(deep=False)
        for col in target_cols:
            result[col] = targets_df[col].reindex(result.index)
        return result

    def _infer_timeframe(self, df: pd.DataFrame) -> str | None:
        if "interval" not in df.columns:
            return None
        values = df["interval"].dropna().astype(str).unique()
        return str(values[0]) if len(values) == 1 else None
