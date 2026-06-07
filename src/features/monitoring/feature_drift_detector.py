"""
Feature Drift Detector using Evidently AI
Monitors feature distribution changes over time.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("FeatureDriftDetector")

# Try to import Evidently AI
try:
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    logger.warning("⚠️ Evidently AI not installed. Install with: pip install evidently")


class FeatureDriftDetector:
    """
    Detects feature drift using Evidently AI.

    Audit Point: FEATURE LAYER → Feature Drift
    """

    def __init__(
        self,
        drift_threshold: float = 0.5,
        output_dir: str = "reports/drift"
    ):
        """
        Initialize drift detector.

        Args:
            drift_threshold: Threshold for drift detection (0-1)
            output_dir: Directory to save drift reports
        """
        self.drift_threshold = drift_threshold
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = {
            'checks_performed': 0,
            'drifts_detected': 0,
            'last_check_time': None,
            'drift_history': []
        }

        if not EVIDENTLY_AVAILABLE:
            logger.error("❌ Evidently AI not available. Drift detection disabled.")

    def detect_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        feature_columns: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Detect feature drift between reference and current data.

        Args:
            reference_data: Reference (training) data
            current_data: Current (production) data
            feature_columns: Columns to check (None = all numeric)

        Returns:
            Dict with drift detection results
        """
        self.metrics['checks_performed'] += 1
        self.metrics['last_check_time'] = pd.Timestamp.now()

        if not EVIDENTLY_AVAILABLE:
            return {
                'status': 'ERROR',
                'message': 'Evidently AI not installed',
                'drift_detected': False
            }

        # Validate inputs
        if reference_data.empty or current_data.empty:
            logger.error("❌ Empty DataFrame provided")
            return {
                'status': 'ERROR',
                'message': 'Empty DataFrame',
                'drift_detected': False
            }

        # Select feature columns
        if feature_columns is None:
            feature_columns = reference_data.select_dtypes(include=[np.number]).columns.tolist()

        # Filter columns that exist in both datasets
        common_columns = [col for col in (feature_columns or []) if col in reference_data.columns and col in current_data.columns]

        if not common_columns:
            logger.error("❌ No common columns found")
            return {
                'status': 'ERROR',
                'message': 'No common columns',
                'drift_detected': False
            }

        logger.info(f"Checking drift for {len(common_columns)} features")

        try:
            # Create Evidently report
            report = Report(metrics=[DataDriftPreset()])

            # Run report
            report.run(
                reference_data=reference_data[common_columns],
                current_data=current_data[common_columns]
            )

            # Extract results
            report_dict = report.as_dict()

            # Parse drift results
            drift_results = self._parse_drift_results(report_dict)

            # Save report
            report_path = self._save_report(report, drift_results)

            # Update metrics
            if drift_results['drift_detected']:
                self.metrics['drifts_detected'] = self.metrics.get('drifts_detected', 0) + 1

            if isinstance(self.metrics.get('drift_history'), list):
                self.metrics['drift_history'].append({
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'drift_detected': drift_results['drift_detected'],
                    'drift_share': drift_results['drift_share'],
                    'drifted_features': len(drift_results['drifted_features'])
                })

            # Log results
            if drift_results['drift_detected']:
                logger.warning(
                    f"⚠️ DRIFT DETECTED: {drift_results['drift_share']:.1%} of features drifted"
                )
                logger.warning(f"   Drifted features: {drift_results['drifted_features'][:5]}")
            else:
                logger.info("✅ No significant drift detected")

            return {
                'status': 'OK',
                'drift_detected': drift_results['drift_detected'],
                'drift_share': drift_results['drift_share'],
                'drifted_features': drift_results['drifted_features'],
                'report_path': str(report_path),
                'details': drift_results
            }

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f"❌ Drift detection failed: {e}", exc_info=True)
            return {
                'status': 'ERROR',
                'message': str(e),
                'drift_detected': False
            }

    def _parse_drift_results(self, report_dict: dict) -> dict:
        """Parse Evidently report results."""
        try:
            # Extract metrics from report
            metrics = report_dict.get('metrics', [])

            drift_detected = False
            drift_share = 0.0
            drifted_features = []

            for metric in metrics:
                metric_type = metric.get('metric', '')

                if 'DatasetDriftMetric' in metric_type:
                    result = metric.get('result', {})
                    drift_detected = result.get('dataset_drift', False)
                    drift_share = result.get('drift_share', 0.0)

                    # Get drifted features
                    drift_by_columns = result.get('drift_by_columns', {})
                    drifted_features = [
                        col for col, info in drift_by_columns.items()
                        if info.get('drift_detected', False)
                    ]

            return {
                'drift_detected': drift_detected,
                'drift_share': drift_share,
                'drifted_features': drifted_features,
                'total_features': len(metrics)
            }

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f"Error parsing drift results: {e}")
            return {
                'drift_detected': False,
                'drift_share': 0.0,
                'drifted_features': [],
                'total_features': 0
            }

    def _save_report(self, report: Any, drift_results: dict) -> Path:
        """Save drift report to file."""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        drift_status = "DRIFT" if drift_results['drift_detected'] else "OK"

        # Save HTML report
        html_path = self.output_dir / f"drift_report_{drift_status}_{timestamp}.html"
        report.save_html(str(html_path))

        # Save JSON summary
        json_path = self.output_dir / f"drift_summary_{drift_status}_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(drift_results, f, indent=2)

        logger.info(f"📄 Drift report saved: {html_path}")

        return html_path

    def get_metrics(self) -> dict:
        """Get drift detector metrics."""
        return self.metrics.copy()

    def reset_metrics(self):
        """Reset metrics."""
        self.metrics = {
            'checks_performed': 0,
            'drifts_detected': 0,
            'last_check_time': None,
            'drift_history': []
        }


def check_feature_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    feature_columns: list[str] | None = None,
    drift_threshold: float = 0.5
) -> dict[str, any]:
    """
    Quick function to check feature drift.

    Args:
        reference_data: Reference (training) data
        current_data: Current (production) data
        feature_columns: Columns to check
        drift_threshold: Drift threshold

    Returns:
        Drift detection results
    """
    detector = FeatureDriftDetector(drift_threshold=drift_threshold)
    return detector.detect_drift(reference_data, current_data, feature_columns)
