"""
DriftAnalyzer using Evidently AI.
Detects data drift between reference (train) and current (test/recent) data.
"""
from pathlib import Path
from typing import Any

import pandas as pd

from src.analytics.interfaces import IAnalyzer
from src.core.logging.logger import ProjectLogger

try:
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
    from evidently.report import Report
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False

logger = ProjectLogger.get_logger(__name__)

class DriftAnalyzer(IAnalyzer):
    """
    Analyzes data and target drift using Evidently AI.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.report_dir = Path(self.config.get('report_dir', 'reports/drift'))
        self.report_dir.mkdir(parents=True, exist_ok=True)
        logger.info("DriftAnalyzer initialized.")

    def analyze(self, data: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """
        Executes drift analysis.

        Args:
            data: Dictionary containing:
                - 'reference_data': DataFrame used for training
                - 'current_data': DataFrame with recent data
            **kwargs:
                - 'target_col': Name of the target column (optional)
        """
        if not EVIDENTLY_AVAILABLE:
            logger.warning("Evidently AI is not available. Skipping drift analysis.")
            return {"status": "SKIPPED", "reason": "Evidently not installed"}

        ref_df = data.get('reference_data')
        curr_df = data.get('current_data')

        # Fallback: if only one dataset provided, split it
        if ref_df is None and curr_df is not None:
            split_idx = int(len(curr_df) * 0.7)
            ref_df = curr_df.iloc[:split_idx]
            curr_df = curr_df.iloc[split_idx:]
            logger.info(f"Split data into reference ({len(ref_df)}) and current ({len(curr_df)})")

        if ref_df is None or curr_df is None or ref_df.empty or curr_df.empty:
            logger.warning("Insufficient data for drift analysis.")
            return {"status": "SKIPPED", "reason": "Insufficient data"}

        target_col = kwargs.get('target_col')

        try:
            # Create report
            presets = [DataDriftPreset()]
            if target_col and target_col in ref_df.columns and target_col in curr_df.columns:
                presets.append(TargetDriftPreset())

            report = Report(metrics=presets)
            report.run(reference_data=ref_df, current_data=curr_df)

            # Save HTML report
            report_path = self.report_dir / f"drift_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html"
            report.save_html(str(report_path))

            # Extract summary metrics
            result_json = report.as_dict()
            drift_share = result_json.get('metrics', [{}])[0].get('result', {}).get('drift_share', 0)
            number_of_drifted_columns = result_json.get('metrics', [{}])[0].get('result', {}).get('number_of_drifted_columns', 0)

            logger.info(f"✅ Drift analysis complete. Drift share: {drift_share:.2%}. Report: {report_path.name}")

            return {
                "status": "OK",
                "drift_share": drift_share,
                "drifted_columns_count": number_of_drifted_columns,
                "report_path": str(report_path),
                "is_drift_detected": drift_share > self.config.get('drift_threshold', 0.3)
            }

        except Exception as e:
            logger.error(f"Drift analysis failed: {e}", exc_info=True)
            return {"status": "ERROR", "reason": str(e)}
