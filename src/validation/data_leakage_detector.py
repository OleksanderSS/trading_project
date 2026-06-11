#!/usr/bin/env python3
"""
Data Leakage Detector - Comprehensive Analysis
A robust utility to identify and prevent information leakage in financial datasets.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("DataLeakageDetector")

class LeakageType(Enum):
    """Types of data leakage"""
    LOOKAHEAD_BIAS = "lookahead_bias"
    FUTURE_INFORMATION = "future_information"
    TARGET_LEAKAGE = "target_leakage"
    TEMPORAL_LEAKAGE = "temporal_leakage"
    INFORMATION_LEAKAGE = "information_leakage"
    SELECTION_BIAS = "selection_bias"
    SURVIVORSHIP_BIAS = "survivorship_bias"
    MULTIPLE_COMPARISON = "multiple_comparison"

@dataclass
class LeakageReport:
    """Report on detected data leakage"""
    leakage_detected: bool
    leakage_types: list[LeakageType]
    severity: str  # low, medium, high, critical
    affected_features: list[str]
    recommendations: list[str]
    confidence: float
    detailed_analysis: dict[str, Any]

class DataLeakageDetector:
    """
    Comprehensive Data Leakage Detector for identifying information leakage in financial machine learning pipelines.
    Provides methods for correlation analysis, temporal overlap detection, and lookahead bias identification.
    """

    def __init__(self):
        """Initializes the DataLeakageDetector."""
        pass

    @staticmethod
    def detect_correlation_leakage(df: pd.DataFrame, target_col: str, threshold: float = 0.99) -> dict[str, float]:
        """
        Identifies features that have a suspiciously high correlation with the target variable.

        Args:
            df: The DataFrame containing features and target.
            target_col: The name of the target column.
            threshold: Correlation coefficient threshold (absolute value) to flag as leakage.

        Returns:
            Dict mapping feature names to their absolute correlation coefficient.
        """
        if target_col not in df.columns:
            logger.error(f"Target column '{target_col}' not found for correlation analysis.")
            return {}

        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return {}

        correlations = numeric_df.corr()[target_col].abs()
        # Filter out target itself and features below threshold
        leakage_features = correlations[(correlations >= threshold) & (correlations.index != target_col)]

        if not leakage_features.empty:
            logger.warning(f"Detected {len(leakage_features)} features with correlation >= {threshold} against {target_col}.")

        return leakage_features.to_dict()

    @staticmethod
    def detect_temporal_overlap(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[Any]:
        """
        Detects if there are overlapping indices between training and testing datasets,
        which is a common source of data leakage in time-series forecasting.

        Args:
            train_df: The training dataset.
            test_df: The testing dataset.

        Returns:
            List of overlapping indices.
        """
        train_indices = set(train_df.index)
        test_indices = set(test_df.index)

        overlap = list(train_indices.intersection(test_indices))

        if overlap:
            logger.critical(f"Temporal Overlap Detected: {len(overlap)} indices exist in both train and test sets.")
        else:
            logger.info("No temporal overlap detected between train and test sets.")

        return overlap

    @staticmethod
    def detect_future_data_in_features(df: pd.DataFrame, features: list[str], target_col: str) -> dict[str, float]:
        """
        Checks if features contain information that could only be known in the future relative to the timestamp.
        Specifically looks for features that correlate perfectly with a shifted target.

        Args:
            df: The dataset to analyze.
            features: List of feature columns to check.
            target_col: The target column.

        Returns:
            Dict containing flagged features and their correlation with future target values.
        """
        if target_col not in df.columns:
            return {}

        results = {}
        # Check correlations with target shifted forward (future values)
        # Shift -1 means the target at T+1
        future_target = df[target_col].shift(-1)

        for feature in features:
            if feature in df.columns and feature != target_col:
                # Check absolute correlation with future target
                correlation = abs(df[feature].corr(future_target))
                if correlation > 0.95:
                    logger.warning(f"Feature '{feature}' suspiciously correlates with future target (r={correlation:.4f}).")
                    results[feature] = correlation

        return results

    def run_comprehensive_audit(self, df: pd.DataFrame, target_col: str,
                                 train_df: pd.DataFrame | None = None,
                                 test_df: pd.DataFrame | None = None) -> LeakageReport:
        """
        Runs all available leakage detection methods on a dataset and generates a summary report.
        """
        logger.info(f"Starting comprehensive leakage audit for target: {target_col}")

        corr_leakage = self.detect_correlation_leakage(df, target_col)
        lookahead_leakage = self.detect_future_data_in_features(df, df.columns.tolist(), target_col)

        temporal_overlap: list[Any] = []
        if train_df is not None and test_df is not None:
            temporal_overlap = self.detect_temporal_overlap(train_df, test_df)

        leakage_detected = bool(corr_leakage or lookahead_leakage or temporal_overlap)
        leakage_types = []
        if corr_leakage:
            leakage_types.append(LeakageType.TARGET_LEAKAGE)
        if lookahead_leakage:
            leakage_types.append(LeakageType.LOOKAHEAD_BIAS)
        if temporal_overlap:
            leakage_types.append(LeakageType.TEMPORAL_LEAKAGE)

        affected_features = list(set(list(corr_leakage.keys()) + list(lookahead_leakage.keys())))

        recommendations = []
        if leakage_detected:
            recommendations.append("Investigate and remove flagged features.")
            recommendations.append("Ensure target calculation does not overlap with feature windows.")
        if temporal_overlap:
            recommendations.append(f"Fix train/test split: {len(temporal_overlap)} overlapping indices found.")

        severity = "low"
        if temporal_overlap or lookahead_leakage:
            severity = "critical"
        elif corr_leakage:
            severity = "high"

        return LeakageReport(
            leakage_detected=leakage_detected,
            leakage_types=leakage_types,
            severity=severity,
            affected_features=affected_features,
            recommendations=recommendations,
            confidence=0.95,
            detailed_analysis={
                "correlation_leakage": corr_leakage,
                "lookahead_leakage": lookahead_leakage,
                "temporal_overlap_count": len(temporal_overlap),
            }
        )

def main():
    """Example usage of the DataLeakageDetector."""
    detector = DataLeakageDetector()

    # ✅ Ensuring example determinism
    rng = np.random.default_rng(42)

    # Create dummy data with leakage
    data = {
        'feature_clean': rng.standard_normal(100),
        'feature_leaky': np.arange(100) + 0.01 * rng.standard_normal(100),
        'target': np.arange(100)
    }
    df = pd.DataFrame(data)

    report = detector.run_comprehensive_audit(df, target_col='target')

    logger.info(f"Leakage Detected: {report.leakage_detected}")
    logger.info(f"Severity: {report.severity}")
    logger.info(f"Affected Features: {report.affected_features}")
    logger.info(f"Recommendations: {report.recommendations}")

if __name__ == "__main__":
    main()
