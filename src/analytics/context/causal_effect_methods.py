#!/usr/bin/env python3
"""
Causal Effect Methods - Implementation of Causal Effect Estimation

This module contains the implementation of causal effect estimation methods.
"""


import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

from src.core.logging.logger import ProjectLogger


class CausalEffectMethods:
    """
    Implementation of causal effect estimation methods.

    This class encapsulates the logic for causal effect estimation using
    linear regression, random forest, and double machine learning.
    """

    def __init__(self, confidence_level: float = 0.95, logger=None):
        """
        Initialize the causal effect methods.

        Args:
            confidence_level: Confidence level for statistical tests
            logger: Logger instance
        """
        self.confidence_level = confidence_level
        self.logger = logger or ProjectLogger.get_logger(self.__class__.__name__)

    def linear_estimation(self, X, y, treatment_col):
        """Linear regression causal estimation."""
        x_values = X.astype(float).to_numpy()
        y_values = pd.Series(y).astype(float).to_numpy()
        design = np.column_stack([np.ones(len(x_values)), x_values])
        coefficients = np.linalg.lstsq(design, y_values, rcond=None)[0]
        treatment_idx = X.columns.get_loc(treatment_col) + 1
        treatment_coef = float(coefficients[treatment_idx])

        fitted = design @ coefficients
        residuals = y_values - fitted
        dof = max(len(y_values) - design.shape[1], 1)
        mse = float(np.sum(residuals ** 2) / dof)
        covariance = mse * np.linalg.pinv(design.T @ design)
        standard_error = float(np.sqrt(max(covariance[treatment_idx, treatment_idx], 0.0)))

        if standard_error > 0:
            t_statistic = treatment_coef / standard_error
            p_value = float(2 * (1 - stats.t.cdf(abs(t_statistic), dof)))
            critical = float(stats.t.ppf(1 - (1 - self.confidence_level) / 2, dof))
        else:
            t_statistic = 0.0
            p_value = 1.0
            critical = 0.0

        return {
            'treatment_effect': treatment_coef,
            'standard_error': standard_error,
            't_statistic': float(t_statistic),
            'p_value': p_value,
            'confidence_intervals': (
                treatment_coef - critical * standard_error,
                treatment_coef + critical * standard_error,
            ),
            'effect_size': abs(treatment_coef)
        }

    def rf_estimation(self, X, y, treatment_col):
        """Random forest causal estimation."""
        model = RandomForestRegressor(random_state=42)
        model.fit(X, y)
        treated = X.copy()
        untreated = X.copy()
        treated[treatment_col] = 1
        untreated[treatment_col] = 0
        individual_effects = model.predict(treated) - model.predict(untreated)
        treatment_effect = float(np.mean(individual_effects))

        if len(individual_effects) > 1 and np.std(individual_effects, ddof=1) > 0:
            t_statistic, p_value = stats.ttest_1samp(individual_effects, popmean=0.0, nan_policy='omit')
            standard_error = float(np.std(individual_effects, ddof=1) / np.sqrt(len(individual_effects)))
            critical = float(stats.t.ppf(1 - (1 - self.confidence_level) / 2, len(individual_effects) - 1))
        else:
            t_statistic = 0.0
            p_value = 1.0
            standard_error = 0.0
            critical = 0.0

        return {
            'treatment_effect': treatment_effect,
            'standard_error': standard_error,
            't_statistic': float(t_statistic) if np.isfinite(t_statistic) else 0.0,
            'p_value': float(p_value) if np.isfinite(p_value) else 1.0,
            'confidence_intervals': (
                treatment_effect - critical * standard_error,
                treatment_effect + critical * standard_error,
            ),
            'effect_size': abs(treatment_effect)
        }

    def double_ml_estimation(self, X, y, treatment_col, covariates):
        """Double Machine Learning causal estimation."""
        if not covariates:
            return self.linear_estimation(X, y, treatment_col)

        x_covariates = X[covariates].astype(float)
        treatment = X[treatment_col].astype(float)
        outcome = pd.Series(y).astype(float)

        outcome_model = RandomForestRegressor(random_state=42)
        treatment_model = RandomForestRegressor(random_state=43)
        outcome_model.fit(x_covariates, outcome)
        treatment_model.fit(x_covariates, treatment)

        outcome_residuals = outcome - outcome_model.predict(x_covariates)
        treatment_residuals = treatment - treatment_model.predict(x_covariates)
        denominator = float(np.dot(treatment_residuals, treatment_residuals))

        if denominator <= 0:
            treatment_effect = 0.0
            standard_error = 0.0
            p_value = 1.0
            t_statistic = 0.0
            critical = 0.0
        else:
            treatment_effect = float(np.dot(treatment_residuals, outcome_residuals) / denominator)
            residuals = outcome_residuals - treatment_effect * treatment_residuals
            dof = max(len(outcome_residuals) - 1, 1)
            residual_variance = float(np.sum(residuals ** 2) / dof)
            standard_error = float(np.sqrt(residual_variance / denominator))

            if standard_error > 0:
                t_statistic = treatment_effect / standard_error
                p_value = float(2 * (1 - stats.t.cdf(abs(t_statistic), dof)))
                critical = float(stats.t.ppf(1 - (1 - self.confidence_level) / 2, dof))
            else:
                t_statistic = 0.0
                p_value = 1.0
                critical = 0.0

        return {
            'treatment_effect': treatment_effect,
            'standard_error': standard_error,
            't_statistic': float(t_statistic),
            'p_value': p_value,
            'confidence_intervals': (
                treatment_effect - critical * standard_error,
                treatment_effect + critical * standard_error,
            ),
            'effect_size': abs(treatment_effect)
        }

    def robustness_checks(self, data, treatment_col, outcome_col, covariates):
        """Perform robustness checks for causal estimates."""
        model_columns = covariates + [treatment_col]

        try:
            base_effect = self.linear_estimation(
                data[model_columns],
                data[outcome_col],
                treatment_col,
            )
            shuffled = data.copy()
            rng = np.random.default_rng(42)
            shuffled[treatment_col] = rng.permutation(shuffled[treatment_col].to_numpy())
            placebo_effect = self.linear_estimation(
                shuffled[model_columns],
                shuffled[outcome_col],
                treatment_col,
            )

            if len(data) >= 4:
                subsample = data.sample(frac=0.5, random_state=42)
                subsample_effect = self.linear_estimation(
                    subsample[model_columns],
                    subsample[outcome_col],
                    treatment_col,
                )
                effect_delta = abs(
                    base_effect['treatment_effect']
                    - subsample_effect['treatment_effect']
                )
                consistent = effect_delta <= max(abs(base_effect['treatment_effect']) * 0.5, 1e-12)
            else:
                subsample_effect = base_effect
                effect_delta = 0.0
                consistent = True
        except Exception as exc:
            self.logger.error(f"Robustness check failed: {exc}", exc_info=True)
            raise RuntimeError(f"Robustness check failed: {exc}") from exc

        return {
            'placebo_test': {
                'treatment_effect': placebo_effect['treatment_effect'],
                'p_value': placebo_effect['p_value'],
                'passes': placebo_effect['p_value'] >= 0.05,
            },
            'sensitivity_analysis': {
                'effect_delta': effect_delta,
                'base_effect': base_effect['treatment_effect'],
                'subsample_effect': subsample_effect['treatment_effect'],
            },
            'subsample_analysis': {'consistent': consistent}
        }
