import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional
from src.core.logging.logger import ProjectLogger
from src.core.file_management.file_manager import FileManager
from .time_series_validator import TimeSeriesValidator
from .data_leakage_detector import DataLeakageDetector

logger = ProjectLogger.get_logger("UnifiedValidator")

class DataValidationError(Exception):
    """Custom exception raised when data fails validation checks."""
    pass

class UnifiedValidator:
    """
    Facade class that unifies project-wide validation logic.
    Provides a high-level interface for ProcessingStage and EvaluationStage.
    """

    def __init__(self, file_manager: Optional[FileManager] = None):
        self.fm = file_manager or FileManager()
        self.ts_validator = TimeSeriesValidator()
        self.leakage_detector = DataLeakageDetector()
        
        # Configuration for validation thresholds
        self.nan_threshold = 0.1  # Max 10% NaNs allowed
        self.inf_threshold = 0.01 # Max 1% Infinite values allowed
        self.essential_columns = ['open', 'high', 'low', 'close', 'volume']

    def validate_cleaned_data(self, data_map: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates the output of the ProcessingStage.
        Checks for data integrity, continuity, and statistical health.
        
        Args:
            data_map: Dictionary containing dataframes (e.g., {'market_data': df})
            
        Returns:
            Dict containing 'is_valid' (bool) and 'issues' (list of strings).
        """
        logger.info("Starting cleaned data validation...")
        issues = []
        is_valid = True

        if not data_map:
            return {"is_valid": False, "issues": ["Received empty data map."]}

        for data_key, df in data_map.items():
            if not isinstance(df, pd.DataFrame):
                continue
            
            if df.empty:
                issues.append(f"[{data_key}] DataFrame is empty.")
                continue

            # 1. Essential Columns Check
            if data_key == 'market_data':
                missing_cols = [col for col in self.essential_columns if col not in df.columns]
                if missing_cols:
                    issues.append(f"[{data_key}] Missing essential columns: {missing_cols}")

            # 2. NaN Ratio Check
            nan_ratio = df.isna().mean().max()
            if nan_ratio > self.nan_threshold:
                issues.append(f"[{data_key}] Critical NaN ratio detected: {nan_ratio:.2%}")

            # 3. Infinite Values Check
            inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
            if inf_count > 0:
                issues.append(f"[{data_key}] Detected {inf_count} infinite values.")

            # 4. Time Continuity Check
            if isinstance(df.index, pd.DatetimeIndex):
                gaps = self.ts_validator.validate_time_gaps(df)
                if gaps.get('has_gaps', False):
                    issues.append(f"[{data_key}] Time series contains gaps: {gaps.get('gap_count')} missing periods.")

        # 5. Leakage Check (Cross-referencing market and features if present)
        if 'market_data' in data_map and 'target' in df.columns:
            leakage_report = self.leakage_detector.detect_correlation_leakage(df, 'target')
            if leakage_report:
                issues.append(f"Potential data leakage detected in columns: {list(leakage_report.keys())}")

        if issues:
            is_valid = False
            logger.warning(f"Validation failed with {len(issues)} issues.")
        else:
            logger.info("Validation passed successfully.")

        return {
            "is_valid": is_valid,
            "issues": issues,
            "summary": {
                "rows": len(df) if 'market_data' in data_map else 0,
                "columns": len(df.columns) if 'market_data' in data_map else 0
            }
        }

    def validate_training_ready(self, X_train: pd.DataFrame, X_val: pd.DataFrame) -> bool:
        """
        Pre-modeling validation to ensure train/val sets are clean and separated.
        """
        logger.info("Performing pre-training validation...")
        
        # Check for temporal overlap
        overlap = self.ts_validator.check_leakage(X_train, X_val)
        if overlap:
            logger.error("Data leakage detected: Overlapping indices between train and validation sets.")
            return False
            
        # Check for shape consistency
        if X_train.shape[1] != X_val.shape[1]:
            logger.error(f"Feature count mismatch: Train({X_train.shape[1]}) vs Val({X_val.shape[1]})")
            return False

        return True

    def run_system_health_check(self) -> bool:
        """
        Delegates file-system health checks.
        """
        from src.devtools.system_validator import SystemValidator
        sys_val = SystemValidator(self.fm)
        return sys_val.run_all_checks()