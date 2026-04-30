#!/usr/bin/env python3
"""
Validation Protocols - Advanced ML Validation Methods
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Generator
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

class ValidationType(Enum):
    """Validation types for financial ML"""
    WALK_FORWARD = "walk_forward"
    PURGED_CV = "purged_cv"
    EMBARGO_CV = "embargo_cv"
    PURGED_WALK_FORWARD = "purged_walk_forward"
    CONFORMAL_PREDICTION = "conformal_prediction"
    STRESS_TESTING = "stress_testing"
    REGIME_ANALYSIS = "regime_analysis"
    CONSENSUS_STABILITY = "consensus_stability"

@dataclass
class ValidationResult:
    """Standardized validation output"""
    validation_type: ValidationType
    is_valid: bool
    confidence: float
    performance_metrics: Dict[str, float]
    issues_found: List[str]
    recommendations: List[str]
    detailed_results: Dict[str, Any]

class PurgedTimeSeriesSplit:
    """
    Advanced Time Series Cross-Validator with Purging and Embargo.
    Prevents data leakage by removing overlapping data points.
    """
    def __init__(self, n_splits: int = 5, purge_window: int = 0, embargo_period: int = 0):
        self.n_splits = n_splits
        self.purge_window = purge_window
        self.embargo_period = embargo_period

    def split(self, X: pd.DataFrame) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        n_samples = len(X)
        test_size = n_samples // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            train_end = (i + 1) * test_size
            test_start = train_end
            test_end = test_start + test_size
            
            if test_end > n_samples:
                test_end = n_samples

            purged_train_end = train_end - self.purge_window
            if purged_train_end <= 0:
                continue
                
            embargoed_test_start = test_start + self.embargo_period
            if embargoed_test_start >= test_end:
                continue

            train_indices = np.arange(0, purged_train_end)
            test_indices = np.arange(embargoed_test_start, test_end)
            
            yield train_indices, test_indices

class ValidationProtocolsEngine:
    """
    Unified validation engine integrating TimeSeriesValidator logic 
    with advanced financial ML protocols.
    """
    
    def __init__(self):
        self.protocols = self._initialize_protocols()
        self.logger = ProjectLogger.get_logger("ValidationEngine")
        
    def _initialize_protocols(self) -> Dict[str, Dict[str, Any]]:
        return {
            "purged_walk_forward": {
                "description": "Purged Walk-forward with Embargo",
                "parameters": {"purge_window": 5, "embargo_period": 10, "n_splits": 5}
            },
            "purged_cv": {
                "description": "Purged cross-validation",
                "parameters": {"purge_window": 5, "cv_folds": 5}
            },
            "embargo_cv": {
                "description": "Embargo cross-validation",
                "parameters": {"embargo_period": 10, "cv_folds": 5}
            },
            "consensus_stability": {
                "description": "Heavy vs Light model agreement analysis",
                "parameters": {"agreement_threshold": 0.7, "min_test_size": 100}
            }
        }

    def run_comprehensive_validation(self, data: pd.DataFrame, 
                                  features: List[str], target: str,
                                  model: Any, heavy_model: Optional[Any] = None, 
                                  light_model: Optional[Any] = None) -> Dict[str, ValidationResult]:
        """
        Main entry point for Pipeline stages to get a full quality report.
        """
        self.logger.info(f"Executing comprehensive validation for target: {target}")
        results = {}
        
        # 1. Advanced Purged Walk-Forward
        results["purged_walk_forward"] = self._run_purged_walk_forward(data, features, target, model)
        
        # 2. Purged CV
        results["purged_cv"] = self._run_cv_variant(data, features, target, model, 
                                                  v_type=ValidationType.PURGED_CV)
        
        # 3. Embargo CV
        results["embargo_cv"] = self._run_cv_variant(data, features, target, model, 
                                                   v_type=ValidationType.EMBARGO_CV)

        # 4. Consensus Stability
        if heavy_model and light_model:
            results["consensus_stability"] = self._run_consensus_validation(data, features, target, heavy_model, light_model)
        
        return results

    def _run_purged_walk_forward(self, data: pd.DataFrame, features: List[str], 
                               target: str, model: Any) -> ValidationResult:
        params = self.protocols["purged_walk_forward"]["parameters"]
        ps = PurgedTimeSeriesSplit(
            n_splits=params["n_splits"], 
            purge_window=params["purge_window"], 
            embargo_period=params["embargo_period"]
        )
        
        metrics, predictions, actuals = [], [], []
        for train_idx, test_idx in ps.split(data):
            X_train, y_train = data.iloc[train_idx][features], data.iloc[train_idx][target]
            X_test, y_test = data.iloc[test_idx][features], data.iloc[test_idx][target]
            
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                metrics.append(mean_squared_error(y_test, y_pred))
                predictions.extend(y_pred)
                actuals.extend(y_test)
            except Exception as e:
                self.logger.warning(f"Fold failed in purged walk-forward: {e}")
                
        is_valid = len(metrics) > 0
        avg_mse = np.mean(metrics) if is_valid else 1.0
        
        return ValidationResult(
            validation_type=ValidationType.PURGED_WALK_FORWARD,
            is_valid=is_valid and avg_mse < 0.05,
            confidence=0.9 if is_valid else 0.0,
            performance_metrics={"mse": avg_mse, "folds": len(metrics)},
            issues_found=["Low performance" if avg_mse > 0.05 else ""],
            recommendations=["Check for remaining leakage" if avg_mse < 0.001 else "Stable"],
            detailed_results={"fold_errors": metrics}
        )

    def _run_cv_variant(self, data: pd.DataFrame, features: List[str], target: str, 
                        model: Any, v_type: ValidationType) -> ValidationResult:
        """Runs CV with either purging or embargo based on ValidationType."""
        p_cfg = self.protocols["purged_cv" if v_type == ValidationType.PURGED_CV else "embargo_cv"]["parameters"]
        
        ps = PurgedTimeSeriesSplit(
            n_splits=p_cfg["cv_folds"],
            purge_window=p_cfg.get("purge_window", 0),
            embargo_period=p_cfg.get("embargo_period", 0)
        )
        
        scores = []
        for train_idx, test_idx in ps.split(data):
            try:
                model.fit(data.iloc[train_idx][features], data.iloc[train_idx][target])
                scores.append(model.score(data.iloc[test_idx][features], data.iloc[test_idx][target]))
            except Exception as e:
                self.logger.warning(f"CV Fold failed for {v_type.value}: {e}")
                continue
            
        mean_score = np.mean(scores) if scores else 0.0
        return ValidationResult(
            validation_type=v_type,
            is_valid=mean_score > 0.1,
            confidence=0.8,
            performance_metrics={"mean_r2_or_acc": mean_score},
            issues_found=[], recommendations=[], detailed_results={}
        )

    def _run_consensus_validation(self, data: pd.DataFrame, features: List[str], 
                                target: str, heavy_model: Any, light_model: Any) -> ValidationResult:
        params = self.protocols["consensus_stability"]["parameters"]
        if len(data) < params["min_test_size"]:
            return ValidationResult(ValidationType.CONSENSUS_STABILITY, False, 0.0, {}, ["Data small"], [], {})

        split_idx = int(len(data) * 0.8)
        X_train, y_train = data.iloc[:split_idx][features], data.iloc[:split_idx][target]
        X_test, _ = data.iloc[split_idx:][features], data.iloc[split_idx:][target]

        try:
            heavy_model.fit(X_train, y_train)
            light_model.fit(X_train, y_train)
            h_pred, l_pred = heavy_model.predict(X_test), light_model.predict(X_test)
            
            agreement = np.mean(np.sign(h_pred) == np.sign(l_pred))
            return ValidationResult(
                validation_type=ValidationType.CONSENSUS_STABILITY,
                is_valid=agreement >= params["agreement_threshold"],
                confidence=agreement,
                performance_metrics={"agreement_rate": agreement},
                issues_found=[] if agreement >= params["agreement_threshold"] else ["Low agreement"],
                recommendations=[], detailed_results={}
            )
        except Exception as e:
            self.logger.error(f"Consensus validation error: {e}")
            return ValidationResult(ValidationType.CONSENSUS_STABILITY, False, 0.0, {}, [str(e)], [], {})

def main():
    """Engine test"""
    rng = np.random.default_rng(42)
    engine = ValidationProtocolsEngine()
    data = pd.DataFrame({'target': rng.normal(0, 1, 200), 'f1': rng.normal(0, 1, 200)}, 
                        index=pd.date_range('2023-01-01', periods=200))
    from sklearn.linear_model import Ridge
    report = engine.run_comprehensive_validation(data, ['f1'], 'target', Ridge())
    for k, v in report.items():
        logger.info(f"Protocol {k}: Valid={v.is_valid}, Metrics={v.performance_metrics}")

if __name__ == "__main__":
    main()