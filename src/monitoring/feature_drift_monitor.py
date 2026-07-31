"""
Feature Drift Monitor using Evidently AI
Detects feature distribution changes over time.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.exceptions import DataProcessingError
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("FeatureDriftMonitor")

# Evidently AI. The 0.7 line moved the classic Report/preset API under
# `evidently.legacy`; importing only the pre-0.7 paths made an INSTALLED
# Evidently look absent and logged "not installed", which is why this monitor
# was dead on a machine that had the package (0.7.21).
try:  # Evidently >= 0.7
    from evidently.legacy.metric_preset import DataDriftPreset
    from evidently.legacy.metrics import DatasetDriftMetric
    from evidently.legacy.report import Report
    EVIDENTLY_AVAILABLE = True
except ImportError:
    try:  # Evidently < 0.7
        from evidently.metric_preset import DataDriftPreset
        from evidently.metrics import DatasetDriftMetric
        from evidently.report import Report
        EVIDENTLY_AVAILABLE = True
    except ImportError:
        EVIDENTLY_AVAILABLE = False
        logger.warning(
            "⚠️ Evidently AI unavailable (neither the >=0.7 `evidently.legacy` "
            "paths nor the pre-0.7 ones import). Install with: pip install evidently"
        )


class FeatureDriftMonitor:
    """
    Monitors feature drift using Evidently AI.

    Audit Point: FEATURE LAYER → Feature Drift Detection
    """

    def __init__(
        self,
        reference_data: pd.DataFrame | None = None,
        drift_threshold: float = 0.5,
        report_dir: str = "reports/drift"
    ):
        """
        Initialize drift monitor.

        Args:
            reference_data: Reference dataset (training data)
            drift_threshold: Threshold for drift detection (0-1)
            report_dir: Directory to save drift reports
        """
        self.reference_data = reference_data
        self.drift_threshold = drift_threshold
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        self.drift_history: list[dict[str, Any]] = []
        self.metrics = {
            'checks_performed': 0,
            'drifts_detected': 0,
            'last_check_time': None,
            'last_drift_score': None
        }

        if not EVIDENTLY_AVAILABLE:
            logger.warning("⚠️ Evidently AI not available. Drift monitoring disabled.")

    def set_reference_data(self, reference_data: pd.DataFrame):
        """Set or update reference data."""
        self.reference_data = reference_data.copy()
        logger.info(f"✅ Reference data set: {len(reference_data)} rows, {len(reference_data.columns)} columns")

    def check_drift(
        self,
        current_data: pd.DataFrame,
        feature_columns: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Check for feature drift between reference and current data.

        Args:
            current_data: Current production data
            feature_columns: Specific columns to check (None = all numeric)

        Returns:
            Dict with drift results
        """
        if not EVIDENTLY_AVAILABLE:
            raise DataProcessingError("Evidently AI not installed")

        if self.reference_data is None:
            raise DataProcessingError("No reference data set")

        self.metrics['checks_performed'] = (self.metrics.get('checks_performed', 0) or 0) + 1  # type: ignore
        self.metrics['last_check_time'] = datetime.now()  # type: ignore

        # Select feature columns
        if feature_columns is None:
            # Use all numeric columns except targets
            feature_columns = [
                col for col in current_data.select_dtypes(include=[np.number]).columns
                # audit-ignore: ARCHITECTURAL_USAGE
                if not col.startswith('target_') and col not in ['hash', 'interval']
            ]

        # Ensure columns exist in both datasets
        common_columns = list(set(feature_columns) & set(self.reference_data.columns) & set(current_data.columns))

        if not common_columns:
            raise DataProcessingError("No common columns between reference and current data")

        logger.info(f"🔍 Checking drift for {len(common_columns)} features...")

        # Prepare data
        ref_data = self.reference_data[common_columns].copy()
        cur_data = current_data[common_columns].copy()

        # Create Evidently report
        try:
            report = Report(metrics=[
                DataDriftPreset(),
                DatasetDriftMetric()
            ])

            report.run(reference_data=ref_data, current_data=cur_data)

            # Extract results.
            #
            # Look metrics up BY NAME. This read `report_dict['metrics'][1]`,
            # but DataDriftPreset expands into several metrics, so index 1 is
            # DataDriftTable rather than DatasetDriftMetric. DataDriftTable
            # has no `drift_share` key, so `.get('drift_share', 0.0)` always
            # returned the default -- the monitor reported "0.0% of features
            # drifted" on every run, including runs where drift was real.
            report_dict = report.as_dict()
            metrics_by_name: dict[str, dict] = {}
            for metric in report_dict.get('metrics', []):
                metrics_by_name.setdefault(metric.get('metric'), metric.get('result', {}) or {})

            dataset_drift = (
                metrics_by_name.get('DatasetDriftMetric')
                or metrics_by_name.get('DataDriftTable')
                or {}
            )
            # `share_of_drifted_columns` is the OBSERVED share; `drift_share`
            # is the configured THRESHOLD (0.5 by default) and is constant
            # whether or not anything drifted -- reporting it as the score
            # showed "50.0% of features drifted" on clean data too.
            drift_score = dataset_drift.get(
                'share_of_drifted_columns', dataset_drift.get('drift_share', 0.0)
            )
            drift_detected = dataset_drift.get('dataset_drift', False)

            self.metrics['last_drift_score'] = drift_score

            if drift_detected:
                self.metrics['drifts_detected'] = (self.metrics.get('drifts_detected', 0) or 0) + 1  # type: ignore

            # Per-column drift lives inside DataDriftTable's `drift_by_columns`.
            # The old loop looked for standalone 'ColumnDriftMetric' entries,
            # which DataDriftPreset does not emit at the top level, so
            # column_drifts was always empty and every report claimed
            # "Drifted features: 0/0".
            column_drifts = {}
            drift_table = metrics_by_name.get('DataDriftTable', {})
            for col_name, col_result in (drift_table.get('drift_by_columns') or {}).items():
                if not isinstance(col_result, dict):
                    continue
                column_drifts[col_name] = {
                    'drift_detected': bool(col_result.get('drift_detected', False)),
                    'drift_score': col_result.get('drift_score', 0.0),
                }
            # Older/other layouts may still emit standalone per-column metrics.
            for metric in report_dict.get('metrics', []):
                if metric.get('metric') == 'ColumnDriftMetric':
                    result = metric.get('result', {}) or {}
                    name = result.get('column_name')
                    if name and name not in column_drifts:
                        column_drifts[name] = {
                            'drift_detected': bool(result.get('drift_detected', False)),
                            'drift_score': result.get('drift_score', 0.0),
                        }

            # Save report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.report_dir / f"drift_report_{timestamp}.html"
            report.save_html(str(report_path))

            # Log results
            if drift_detected:
                logger.warning(f"⚠️ DRIFT DETECTED: {drift_score:.1%} of features drifted")
                logger.warning(f"   Drifted features: {sum(1 for d in column_drifts.values() if d['drift_detected'])}/{len(column_drifts)}")
            else:
                logger.info(f"✅ No significant drift detected (drift score: {drift_score:.1%})")

            result = {
                'status': 'OK',
                'drift_detected': drift_detected,
                'drift_score': drift_score,
                'drifted_features_count': sum(1 for d in column_drifts.values() if d['drift_detected']),
                'total_features': len(column_drifts),
                'column_drifts': column_drifts,
                'report_path': str(report_path),
                'timestamp': timestamp
            }

            self.drift_history.append(result)

            return result

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f"❌ Error checking drift: {e}")
            raise DataProcessingError(f"Error checking drift: {e}") from e

    def get_drift_summary(self) -> dict[str, Any]:
        """Get summary of drift monitoring."""
        if not self.drift_history:
            return {
                'total_checks': self.metrics['checks_performed'],
                'drifts_detected': 0,
                'drift_rate': 0.0,
                'last_check': self.metrics['last_check_time']
            }

        checks_performed = self.metrics.get('checks_performed', 0)
        drifts_detected = self.metrics.get('drifts_detected', 0)

        return {
            'total_checks': checks_performed,
            'drifts_detected': drifts_detected,
            'drift_rate': drifts_detected / checks_performed if checks_performed > 0 else 0.0,  # type: ignore
            'last_check': self.metrics['last_check_time'],
            'last_drift_score': self.metrics['last_drift_score'],
            'avg_drift_score': np.mean([h['drift_score'] for h in self.drift_history if h['status'] == 'OK'])  # type: ignore
        }

    def get_metrics(self) -> dict[str, Any]:
        """Get monitoring metrics."""
        return self.metrics.copy()


def check_feature_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    feature_columns: list[str] | None = None,
    drift_threshold: float = 0.5
) -> dict[str, Any]:
    """
    Quick function to check feature drift.

    Args:
        reference_data: Reference dataset (training data)
        current_data: Current production data
        feature_columns: Specific columns to check
        drift_threshold: Threshold for drift detection

    Returns:
        Drift check result
    """
    monitor = FeatureDriftMonitor(
        reference_data=reference_data,
        drift_threshold=drift_threshold
    )
    return monitor.check_drift(current_data, feature_columns)


# Singleton instance
_feature_drift_monitor_instance: FeatureDriftMonitor | None = None


def get_feature_drift_monitor(
    reference_data: pd.DataFrame | None = None,
    drift_threshold: float = 0.5,
    report_dir: str = "reports/drift"
) -> FeatureDriftMonitor:
    """
    Get or create singleton FeatureDriftMonitor instance.

    Args:
        reference_data: Reference dataset (training data)
        drift_threshold: Threshold for drift detection
        report_dir: Directory to save drift reports

    Returns:
        FeatureDriftMonitor instance
    """
    global _feature_drift_monitor_instance

    if _feature_drift_monitor_instance is None:
        _feature_drift_monitor_instance = FeatureDriftMonitor(
            reference_data=reference_data,
            drift_threshold=drift_threshold,
            report_dir=report_dir
        )
    elif reference_data is not None:
        # Update reference data if provided
        _feature_drift_monitor_instance.set_reference_data(reference_data)

    return _feature_drift_monitor_instance
