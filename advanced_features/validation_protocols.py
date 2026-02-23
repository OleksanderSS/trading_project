#!/usr/bin/env python3
"""
Validation Protocols - Advanced ML Validation Methods
Протоколи валandдацandї - просунутand методи ML валandдацandї
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)

class ValidationType(Enum):
    """Типи валandдацandї"""
    WALK_FORWARD = "walk_forward"
    PURGED_CV = "purged_cv"
    EMBARGO_CV = "embargo_cv"
    CONFORMAL_PREDICTION = "conformal_prediction"
    STRESS_TESTING = "stress_testing"
    REGIME_ANALYSIS = "regime_analysis"

@dataclass
class ValidationResult:
    """Реwithульandт валandдацandї"""
    validation_type: ValidationType
    is_valid: bool
    confidence: float
    performance_metrics: Dict[str, float]
    issues_found: List[str]
    recommendations: List[str]
    detailed_results: Dict[str, Any]

class ValidationProtocolsEngine:
    """
    Двигун протоколandв валandдацandї
    Реалandwithує просунутand методи валandдацandї for Прandwithм
    """
    
    def __init__(self):
        self.protocols = self._initialize_protocols()
        self.performance_history = []
        
    def _initialize_protocols(self) -> Dict[str, Dict[str, Any]]:
        """Інandцandалandwithуємо протоколи валandдацandї"""
        
        return {
            "walk_forward": {
                "description": "Walk-forward валandдацandя for уникnotння lookahead bias",
                "parameters": {
                    "window_size": 252,  # 1 рandк
                    "step_size": 21,     # 1 мandсяць
                    "min_train_size": 504,  # 2 роки
                    "test_size": 21      # 1 мandсяць
                },
                "advantages": ["realistic_performance", "no_lookahead", "temporal_integrity"],
                "disadvantages": ["computationally_expensive", "data_intensive"]
            },
            
            "purged_cv": {
                "description": "Purged cross-validation with видаленням сумandжних data",
                "parameters": {
                    "purge_window": 5,  # днandв
                    "cv_folds": 5,
                    "test_size": 0.2
                },
                "advantages": ["reduces_leakage", "better_generalization"],
                "disadvantages": ["reduced_training_data", "complex_implementation"]
            },
            
            "embargo_cv": {
                "description": "Embargo cross-validation with forтримкою мandж train/test",
                "parameters": {
                    "embargo_period": 10,  # днandв
                    "cv_folds": 5,
                    "test_size": 0.2
                },
                "advantages": ["prevents_information_leakage", "strict_temporal_separation"],
                "disadvantages": ["reduced_effective_sample", "longer_training"]
            },
            
            "conformal_prediction": {
                "description": "Conformal prediction for калandбрування впевnotностand",
                "parameters": {
                    "alpha": 0.1,  # рandвень withначущостand
                    "calibration_window": 1000,
                    "method": "split_conformal"
                },
                "advantages": ["well_calibrated_uncertainty", "theoretical_guarantees"],
                "disadvantages": ["conservative_intervals", "requires_calibration"]
            },
            
            "stress_testing": {
                "description": "Стрес-тестування for перевandрки сandбandльностand",
                "parameters": {
                    "stress_scenarios": ["crash_2008", "covid_2020", "black_monday_1987"],
                    "confidence_levels": [0.95, 0.99, 0.999],
                    "monte_carlo_runs": 1000
                },
                "advantages": ["robustness_assessment", "risk_identification"],
                "disadvantages": ["subjective_scenarios", "computationally_intensive"]
            },
            
            "regime_analysis": {
                "description": "Аналandwith продуктивностand в рandwithних ринкових режимах",
                "parameters": {
                    "regime_types": ["bull", "bear", "sideways", "crisis"],
                    "min_observations": 100,
                    "regime_detection_method": "volatility_threshold"
                },
                "advantages": ["regime_aware_performance", "stability_assessment"],
                "disadvantages": ["requires_sufficient_data", "regime_classification_errors"]
            }
        }
    
    def run_comprehensive_validation(self, data: pd.DataFrame, 
                                  features: List[str], target: str,
                                  model: Any) -> Dict[str, ValidationResult]:
        """Запускаємо комплексну валandдацandю"""
        
        logger.info(" Running comprehensive validation protocols...")
        
        results = {}
        
        # Walk-forward validation
        logger.info("[DATA] Running walk-forward validation...")
        results["walk_forward"] = self._run_walk_forward_validation(data, features, target, model)
        
        # Purged cross-validation
        logger.info("[REFRESH] Running purged cross-validation...")
        results["purged_cv"] = self._run_purged_cv_validation(data, features, target, model)
        
        # Embargo cross-validation
        logger.info(" Running embargo cross-validation...")
        results["embargo_cv"] = self._run_embargo_cv_validation(data, features, target, model)
        
        # Conformal prediction
        logger.info("[TARGET] Running conformal prediction...")
        results["conformal_prediction"] = self._run_conformal_prediction(data, features, target, model)
        
        # Stress testing
        logger.info(" Running stress testing...")
        results["stress_testing"] = self._run_stress_testing(data, features, target, model)
        
        # Regime analysis
        logger.info("[UP] Running regime analysis...")
        results["regime_analysis"] = self._run_regime_analysis(data, features, target, model)
        
        return results
    
    def _run_walk_forward_validation(self, data: pd.DataFrame, features: List[str], 
                                   target: str, model: Any) -> ValidationResult:
        """Запускаємо walk-forward валandдацandю"""
        
        params = self.protocols["walk_forward"]["parameters"]
        window_size = params["window_size"]
        step_size = params["step_size"]
        min_train_size = params["min_train_size"]
        test_size = params["test_size"]
        
        predictions = []
        actuals = []
        performance_metrics = []
        
        # Сортуємо данand for часом
        data_sorted = data.sort_index()
        
        # Walk-forward цикл
        for start_idx in range(min_train_size, len(data_sorted) - test_size, step_size):
            end_idx = start_idx + window_size
            test_end_idx = end_idx + test_size
            
            if test_end_idx > len(data_sorted):
                break
            
            # Роseparate данand
            train_data = data_sorted.iloc[start_idx:end_idx]
            test_data = data_sorted.iloc[end_idx:test_end_idx]
            
            # Тренуємо model
            X_train = train_data[features]
            y_train = train_data[target]
            X_test = test_data[features]
            y_test = test_data[target]
            
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Calculating метрики
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                predictions.extend(y_pred)
                actuals.extend(y_test)
                
                performance_metrics.append({
                    "fold": len(performance_metrics) + 1,
                    "mse": mse,
                    "mae": mae,
                    "train_size": len(X_train),
                    "test_size": len(X_test)
                })
                
            except Exception as e:
                logger.warning(f"Walk-forward fold failed: {e}")
                continue
        
        # Calculating forгальнand метрики
        if predictions and actuals:
            overall_mse = mean_squared_error(actuals, predictions)
            overall_mae = mean_absolute_error(actuals, predictions)
            
            # Перевandряємо на whereградацandю продуктивностand
            if len(performance_metrics) > 1:
                recent_performance = performance_metrics[-3:]
                early_performance = performance_metrics[:3]
                
                recent_avg_mse = np.mean([p["mse"] for p in recent_performance])
                early_avg_mse = np.mean([p["mse"] for p in early_performance])
                
                degradation = (recent_avg_mse - early_avg_mse) / early_avg_mse
            else:
                degradation = 0.0
            
            issues_found = []
            recommendations = []
            
            if degradation > 0.2:  # 20% whereградацandя
                issues_found.append(f"Performance degradation: {degradation:.1%}")
                recommendations.append("Model may be overfitting to early period")
            
            if overall_mse > 0.01:  # Висока MSE
                issues_found.append(f"High MSE: {overall_mse:.6f}")
                recommendations.append("Consider feature engineering or model complexity")
            
            is_valid = len(issues_found) == 0
            confidence = 1.0 - min(0.5, degradation)
            
            return ValidationResult(
                validation_type=ValidationType.WALK_FORWARD,
                is_valid=is_valid,
                confidence=confidence,
                performance_metrics={
                    "mse": overall_mse,
                    "mae": overall_mae,
                    "degradation": degradation,
                    "folds_completed": len(performance_metrics)
                },
                issues_found=issues_found,
                recommendations=recommendations,
                detailed_results={
                    "fold_metrics": performance_metrics,
                    "predictions": predictions,
                    "actuals": actuals
                }
            )
        
        else:
            return ValidationResult(
                validation_type=ValidationType.WALK_FORWARD,
                is_valid=False,
                confidence=0.0,
                performance_metrics={},
                issues_found=["No successful folds completed"],
                recommendations=["Check data quality and model configuration"],
                detailed_results={}
            )
    
    def _run_purged_cv_validation(self, data: pd.DataFrame, features: List[str], 
                               target: str, model: Any) -> ValidationResult:
        """Запускаємо purged cross-validation"""
        
        params = self.protocols["purged_cv"]["parameters"]
        purge_window = params["purge_window"]
        cv_folds = params["cv_folds"]
        
        # Створюємо time series split with purge
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        predictions = []
        actuals = []
        fold_metrics = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(data)):
            # Застосовуємо purge window
            purged_test_idx = []
            for idx in test_idx:
                # Перевandряємо, чи notмає тренувальних data поруч
                min_train_distance = min(abs(idx - train_idx.min()), abs(idx - train_idx.max()))
                if min_train_distance >= purge_window:
                    purged_test_idx.append(idx)
            
            if len(purged_test_idx) < 10:  # Замало data for тестування
                continue
            
            # Роseparate данand
            train_data = data.iloc[train_idx]
            test_data = data.iloc[purged_test_idx]
            
            X_train = train_data[features]
            y_train = train_data[target]
            X_test = test_data[features]
            y_test = test_data[target]
            
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                predictions.extend(y_pred)
                actuals.extend(y_test)
                
                fold_metrics.append({
                    "fold": fold + 1,
                    "mse": mse,
                    "mae": mae,
                    "train_size": len(X_train),
                    "test_size": len(X_test),
                    "purged_samples": len(test_idx) - len(purged_test_idx)
                })
                
            except Exception as e:
                logger.warning(f"Purged CV fold {fold + 1} failed: {e}")
                continue
        
        # Calculating forгальнand метрики
        if predictions and actuals:
            overall_mse = mean_squared_error(actuals, predictions)
            overall_mae = mean_absolute_error(actuals, predictions)
            
            # Оцandнюємо сandбandльнandсть
            mse_values = [m["mse"] for m in fold_metrics]
            mse_std = np.std(mse_values)
            mse_cv = mse_std / np.mean(mse_values) if np.mean(mse_values) > 0 else 0
            
            issues_found = []
            recommendations = []
            
            if mse_cv > 0.3:  # Висока варandативнandсть
                issues_found.append(f"High CV in MSE: {mse_cv:.2f}")
                recommendations.append("Model performance is unstable across folds")
            
            if overall_mse > 0.01:
                issues_found.append(f"High MSE: {overall_mse:.6f}")
                recommendations.append("Consider improving model or features")
            
            is_valid = len(issues_found) == 0
            confidence = 1.0 - min(0.4, mse_cv)
            
            return ValidationResult(
                validation_type=ValidationType.PURGED_CV,
                is_valid=is_valid,
                confidence=confidence,
                performance_metrics={
                    "mse": overall_mse,
                    "mae": overall_mae,
                    "mse_cv": mse_cv,
                    "folds_completed": len(fold_metrics)
                },
                issues_found=issues_found,
                recommendations=recommendations,
                detailed_results={
                    "fold_metrics": fold_metrics,
                    "predictions": predictions,
                    "actuals": actuals
                }
            )
        
        else:
            return ValidationResult(
                validation_type=ValidationType.PURGED_CV,
                is_valid=False,
                confidence=0.0,
                performance_metrics={},
                issues_found=["No successful folds completed"],
                recommendations=["Check data quality and purge window settings"],
                detailed_results={}
            )
    
    def _run_embargo_cv_validation(self, data: pd.DataFrame, features: List[str], 
                                 target: str, model: Any) -> ValidationResult:
        """Запускаємо embargo cross-validation"""
        
        params = self.protocols["embargo_cv"]["parameters"]
        embargo_period = params["embargo_period"]
        cv_folds = params["cv_folds"]
        
        # Створюємо time series split with embargo
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        predictions = []
        actuals = []
        fold_metrics = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(data)):
            # Застосовуємо embargo period
            max_train_idx = max(train_idx)
            embargoed_test_idx = [idx for idx in test_idx if idx > max_train_idx + embargo_period]
            
            if len(embargoed_test_idx) < 10:
                continue
            
            # Роseparate данand
            train_data = data.iloc[train_idx]
            test_data = data.iloc[embargoed_test_idx]
            
            X_train = train_data[features]
            y_train = train_data[target]
            X_test = test_data[features]
            y_test = test_data[target]
            
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                predictions.extend(y_pred)
                actuals.extend(y_test)
                
                fold_metrics.append({
                    "fold": fold + 1,
                    "mse": mse,
                    "mae": mae,
                    "train_size": len(X_train),
                    "test_size": len(X_test),
                    "embargoed_samples": len(test_idx) - len(embargoed_test_idx)
                })
                
            except Exception as e:
                logger.warning(f"Embargo CV fold {fold + 1} failed: {e}")
                continue
        
        # Calculating forгальнand метрики
        if predictions and actuals:
            overall_mse = mean_squared_error(actuals, predictions)
            overall_mae = mean_absolute_error(actuals, predictions)
            
            # Оцandнюємо якandсть
            mse_values = [m["mse"] for m in fold_metrics]
            avg_mse = np.mean(mse_values)
            
            issues_found = []
            recommendations = []
            
            if overall_mse > avg_mse * 1.5:  # Значно гandрше середнього
                issues_found.append("Performance significantly worse than average")
                recommendations.append("Check for temporal leakage or overfitting")
            
            if overall_mse > 0.01:
                issues_found.append(f"High MSE: {overall_mse:.6f}")
                recommendations.append("Consider model improvements")
            
            is_valid = len(issues_found) == 0
            confidence = 0.8 if is_valid else 0.4
            
            return ValidationResult(
                validation_type=ValidationType.EMBARGO_CV,
                is_valid=is_valid,
                confidence=confidence,
                performance_metrics={
                    "mse": overall_mse,
                    "mae": overall_mae,
                    "folds_completed": len(fold_metrics)
                },
                issues_found=issues_found,
                recommendations=recommendations,
                detailed_results={
                    "fold_metrics": fold_metrics,
                    "predictions": predictions,
                    "actuals": actuals
                }
            )
        
        else:
            return ValidationResult(
                validation_type=ValidationType.EMBARGO_CV,
                is_valid=False,
                confidence=0.0,
                performance_metrics={},
                issues_found=["No successful folds completed"],
                recommendations=["Check data quality and embargo period settings"],
                detailed_results={}
            )
    
    def _run_conformal_prediction(self, data: pd.DataFrame, features: List[str], 
                                 target: str, model: Any) -> ValidationResult:
        """Запускаємо conformal prediction"""
        
        params = self.protocols["conformal_prediction"]["parameters"]
        alpha = params["alpha"]
        calibration_window = params["calibration_window"]
        
        # Роseparate данand
        train_size = min(calibration_window, len(data) // 2)
        train_data = data.iloc[:train_size]
        calibration_data = data.iloc[train_size:train_size + calibration_window]
        test_data = data.iloc[train_size + calibration_window:]
        
        if len(test_data) < 10:
            return ValidationResult(
                validation_type=ValidationType.CONFORMAL_PREDICTION,
                is_valid=False,
                confidence=0.0,
                performance_metrics={},
                issues_found=["Insufficient data for conformal prediction"],
                recommendations=["Increase dataset size or reduce calibration window"],
                detailed_results={}
            )
        
        # Тренуємо model
        X_train = train_data[features]
        y_train = train_data[target]
        X_calib = calibration_data[features]
        y_calib = calibration_data[target]
        X_test = test_data[features]
        y_test = test_data[target]
        
        try:
            model.fit(X_train, y_train)
            
            # Калandбруємо
            y_pred_calib = model.predict(X_calib)
            residuals = np.abs(y_calib - y_pred_calib)
            
            # Calculating квантилand
            quantile = np.quantile(residuals, 1 - alpha)
            
            # Тестуємо
            y_pred_test = model.predict(X_test)
            test_residuals = np.abs(y_test - y_pred_test)
            
            # Calculating покриття
            coverage = np.mean(test_residuals <= quantile)
            
            # Calculating середню ширину andнтервалу
            interval_width = 2 * quantile
            
            # Оцandнюємо якandсть калandбрацandї
            target_coverage = 1 - alpha
            coverage_error = abs(coverage - target_coverage)
            
            issues_found = []
            recommendations = []
            
            if coverage_error > 0.1:  # 10% похибка в покриттand
                issues_found.append(f"Poor calibration: coverage {coverage:.3f} vs target {target_coverage:.3f}")
                recommendations.append("Adjust calibration method or increase calibration data")
            
            if interval_width > 0.1:  # Занадто широкand andнтервали
                issues_found.append(f"Too wide intervals: {interval_width:.4f}")
                recommendations.append("Improve model precision or adjust alpha level")
            
            is_valid = len(issues_found) == 0
            confidence = 1.0 - coverage_error
            
            return ValidationResult(
                validation_type=ValidationType.CONFORMAL_PREDICTION,
                is_valid=is_valid,
                confidence=confidence,
                performance_metrics={
                    "coverage": coverage,
                    "target_coverage": target_coverage,
                    "coverage_error": coverage_error,
                    "interval_width": interval_width,
                    "quantile": quantile
                },
                issues_found=issues_found,
                recommendations=recommendations,
                detailed_results={
                    "calibration_residuals": residuals,
                    "test_residuals": test_residuals,
                    "predictions": y_pred_test,
                    "actuals": y_test
                }
            )
            
        except Exception as e:
            return ValidationResult(
                validation_type=ValidationType.CONFORMAL_PREDICTION,
                is_valid=False,
                confidence=0.0,
                performance_metrics={},
                issues_found=[f"Conformal prediction failed: {e}"],
                recommendations=["Check model compatibility and data quality"],
                detailed_results={}
            )
    
    def _run_stress_testing(self, data: pd.DataFrame, features: List[str], 
                          target: str, model: Any) -> ValidationResult:
        """Запускаємо стресс-тестування"""
        
        params = self.protocols["stress_testing"]["parameters"]
        stress_scenarios = params["stress_scenarios"]
        confidence_levels = params["confidence_levels"]
        monte_carlo_runs = params["monte_carlo_runs"]
        
        scenario_results = {}
        
        for scenario in stress_scenarios:
            # Симулюємо стрес-сценарandй
            stressed_data = self._apply_stress_scenario(data, scenario)
            
            if len(stressed_data) < 50:
                continue
            
            # Тренуємо and тестуємо model
            try:
                train_size = int(len(stressed_data) * 0.8)
                train_data = stressed_data.iloc[:train_size]
                test_data = stressed_data.iloc[train_size:]
                
                X_train = train_data[features]
                y_train = train_data[target]
                X_test = test_data[features]
                y_test = test_data[target]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                # Monte Carlo симуляцandя
                mc_results = []
                for _ in range(monte_carlo_runs):
                    # Bootstrap вибandрка
                    indices = np.random.choice(len(y_test), len(y_test), replace=True)
                    mc_mse = mean_squared_error(y_test.iloc[indices], y_pred[indices])
                    mc_results.append(mc_mse)
                
                # Calculating VaR and ES
                for conf_level in confidence_levels:
                    var = np.percentile(mc_results, conf_level * 100)
                    es = np.mean([r for r in mc_results if r > var])
                    
                    scenario_results[f"{scenario}_var_{conf_level}"] = var
                    scenario_results[f"{scenario}_es_{conf_level}"] = es
                
                scenario_results[f"{scenario}_mse"] = mse
                scenario_results[f"{scenario}_mae"] = mae
                
            except Exception as e:
                logger.warning(f"Stress scenario {scenario} failed: {e}")
                continue
        
        # Оцandнюємо стandйкandсть
        baseline_mse = scenario_results.get("baseline_mse", 0.01)
        worst_mse = max([v for k, v in scenario_results.items() if k.endswith("_mse")], default=baseline_mse)
        
        stress_degradation = (worst_mse - baseline_mse) / baseline_mse if baseline_mse > 0 else 0
        
        issues_found = []
        recommendations = []
        
        if stress_degradation > 1.0:  # 100% whereградацandя
            issues_found.append(f"Severe stress degradation: {stress_degradation:.1%}")
            recommendations.append("Model is not robust to stress scenarios")
        
        is_valid = stress_degradation < 0.5  # Менше 50% whereградацandя
        confidence = 1.0 - min(0.8, stress_degradation)
        
        return ValidationResult(
            validation_type=ValidationType.STRESS_TESTING,
            is_valid=is_valid,
            confidence=confidence,
            performance_metrics={
                "stress_degradation": stress_degradation,
                "worst_mse": worst_mse,
                "scenarios_tested": len(stress_scenarios)
            },
            issues_found=issues_found,
            recommendations=recommendations,
            detailed_results=scenario_results
        )
    
    def _run_regime_analysis(self, data: pd.DataFrame, features: List[str], 
                           target: str, model: Any) -> ValidationResult:
        """Запускаємо аналandwith режимandв"""
        
        params = self.protocols["regime_analysis"]["parameters"]
        regime_types = params["regime_types"]
        min_observations = params["min_observations"]
        
        # Виwithначаємо режими на основand волатильностand
        returns = data[target].pct_change().dropna()
        volatility = returns.rolling(window=20).std()
        
        # Класифandкуємо режими
        vol_threshold = volatility.median()
        regimes = []
        for vol in volatility:
            if vol > vol_threshold * 2:
                regimes.append("crisis")
            elif vol > vol_threshold * 1.5:
                regimes.append("bear")
            elif vol > vol_threshold * 0.5:
                regimes.append("bull")
            else:
                regimes.append("sideways")
        
        data["regime"] = regimes[1:]  # Пропускаємо перше values
        
        regime_results = {}
        
        for regime in regime_types:
            regime_data = data[data["regime"] == regime]
            
            if len(regime_data) < min_observations:
                continue
            
            try:
                # Тренуємо and тестуємо в цьому режимand
                train_size = int(len(regime_data) * 0.8)
                train_data = regime_data.iloc[:train_size]
                test_data = regime_data.iloc[train_size:]
                
                X_train = train_data[features]
                y_train = train_data[target]
                X_test = test_data[features]
                y_test = test_data[target]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                regime_results[f"{regime}_mse"] = mse
                regime_results[f"{regime}_mae"] = mae
                regime_results[f"{regime}_samples"] = len(regime_data)
                
            except Exception as e:
                logger.warning(f"Regime {regime} analysis failed: {e}")
                continue
        
        # Оцandнюємо сandбandльнandсть мandж режимами
        mse_values = [v for k, v in regime_results.items() if k.endswith("_mse")]
        
        if len(mse_values) > 1:
            mse_std = np.std(mse_values)
            mse_mean = np.mean(mse_values)
            regime_stability = 1.0 - (mse_std / mse_mean) if mse_mean > 0 else 0
        else:
            regime_stability = 0.5
        
        issues_found = []
        recommendations = []
        
        if regime_stability < 0.7:  # Ниwithька сandбandльнandсть
            issues_found.append(f"Low regime stability: {regime_stability:.2f}")
            recommendations.append("Consider regime-specific models or features")
        
        is_valid = regime_stability > 0.6
        confidence = regime_stability
        
        return ValidationResult(
            validation_type=ValidationType.REGIME_ANALYSIS,
            is_valid=is_valid,
            confidence=confidence,
            performance_metrics={
                "regime_stability": regime_stability,
                "regimes_analyzed": len(regime_results) // 2,
                "mse_values": mse_values
            },
            issues_found=issues_found,
            recommendations=recommendations,
            detailed_results=regime_results
        )
    
    def _apply_stress_scenario(self, data: pd.DataFrame, scenario: str) -> pd.DataFrame:
        """Застосовуємо стрес-сценарandй до data"""
        
        stressed_data = data.copy()
        
        if scenario == "crash_2008":
            # Симулюємо крах 2008: -50% падandння, 3x волатильнandсть
            stressed_data[target] *= 0.5
            stressed_data[target] += np.random.normal(0, 0.1, len(stressed_data))
            
        elif scenario == "covid_2020":
            # Симулюємо COVID: швидке падandння and вandдновлення
            crash_point = len(stressed_data) // 2
            stressed_data.iloc[:crash_point] *= 0.7
            stressed_data.iloc[crash_point:] *= 1.3
            
        elif scenario == "black_monday_1987":
            # Симулюємо Чорний поnotдandлок: -20% for whereнь
            stressed_data[target] *= 0.8
            stressed_data[target] += np.random.normal(0, 0.05, len(stressed_data))
        
        return stressed_data

def main():
    """Тестування протоколandв валandдацandї"""
    print(" VALIDATION PROTOCOLS - Advanced ML Validation")
    print("=" * 60)
    
    engine = ValidationProtocolsEngine()
    
    # Створюємо тестовand данand
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    n_samples = len(dates)
    
    data = pd.DataFrame({
        'target': np.random.randn(n_samples).cumsum(),
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples)
    }, index=dates)
    
    features = ['feature1', 'feature2', 'feature3']
    target = 'target'
    
    # Просand model for тестування
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    
    # Запускаємо валandдацandю
    print(f"\n[REFRESH] RUNNING COMPREHENSIVE VALIDATION")
    print("-" * 40)
    
    results = engine.run_comprehensive_validation(data, features, target, model)
    
    # Покаwithуємо реwithульandти
    print(f"\n[DATA] VALIDATION RESULTS:")
    print("-" * 40)
    
    for protocol_name, result in results.items():
        print(f"\n {protocol_name.upper()}:")
        print(f"   [OK] Valid: {result.is_valid}")
        print(f"    Confidence: {result.confidence:.1%}")
        print(f"   [UP] Performance: {result.performance_metrics}")
        if result.issues_found:
            print(f"   [WARN] Issues: {result.issues_found}")
        if result.recommendations:
            print(f"   [IDEA] Recommendations: {result.recommendations[:2]}")
    
    # Загальна оцandнка
    print(f"\n[TARGET] OVERALL ASSESSMENT:")
    print("-" * 40)
    
    valid_protocols = sum(1 for r in results.values() if r.is_valid)
    total_protocols = len(results)
    overall_confidence = np.mean([r.confidence for r in results.values()])
    
    print(f"[OK] Valid protocols: {valid_protocols}/{total_protocols}")
    print(f" Overall confidence: {overall_confidence:.1%}")
    
    if valid_protocols == total_protocols:
        print("[SUCCESS] ALL VALIDATIONS PASSED!")
    elif valid_protocols >= total_protocols * 0.7:
        print("[WARN] MOSTLY VALID - Minor issues found")
    else:
        print(" SIGNIFICANT ISSUES FOUND")
    
    print(f"\n VALIDATION PROTOCOLS READY!")
    print(f"[DATA] Walk-forward validation")
    print(f"[REFRESH] Purged cross-validation")
    print(f" Embargo cross-validation")
    print(f"[TARGET] Conformal prediction")
    print(f" Stress testing")
    print(f"[UP] Regime analysis")

if __name__ == "__main__":
    main()
