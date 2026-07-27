from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.validation.validators import UnifiedValidator

logger = ProjectLogger.get_logger('ProcessingValidator')

class ProcessingValidator:
    """Handles data validation for the processing stage."""

    def __init__(self):
        self.logger = logger
        self.validator = UnifiedValidator()

    def run_system_validation(self, filtered_results: dict[str, Any]) -> dict[str, Any]:
        """Run comprehensive system validation on processed data."""
        self.logger.info("Running system-wide data validation...")
        result = self.validator.validate_cleaned_data(filtered_results)
        if result.get('is_valid'):
            self.logger.info("✅ System-wide data validation passed.")
        else:
            self.logger.warning(
                f"⚠️ System-wide data validation found {len(result.get('issues', []))} "
                f"issue(s): {result.get('issues')}"
            )
        return result

    def create_quality_metrics(self, cleaned_data: dict[str, Any]) -> dict[str, Any]:
        """Generate quality metrics for the processed dataset."""
        totals = {'rows': 0, 'cells': 0, 'missing': 0}
        for value in cleaned_data.values():
            self._accumulate_quality_totals(value, totals)

        data_consistency_score = (
            1.0 - (totals['missing'] / totals['cells']) if totals['cells'] > 0 else 1.0
        )

        return {
            'total_rows': totals['rows'],
            'missing_values_count': totals['missing'],
            'data_consistency_score': round(data_consistency_score, 4),
        }

    def _accumulate_quality_totals(self, value: Any, totals: dict[str, int]) -> None:
        """Recursively accumulate row/cell/missing-value counts (cleaned_data
        can nest DataFrames inside dicts, e.g. {'prices': {'1d': df, '1h': df}})."""
        if isinstance(value, pd.DataFrame):
            totals['rows'] += len(value)
            totals['cells'] += value.size
            totals['missing'] += int(value.isna().sum().sum())
        elif isinstance(value, dict):
            for nested in value.values():
                self._accumulate_quality_totals(nested, totals)
