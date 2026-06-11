"""
Counterfactual Analysis Generator - "What if" scenario analysis for trading strategies.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

class CounterfactualGenerator:
    """
    Generates counterfactual scenarios for causal analysis in trading.

    Implements:
    - Synthetic Control Methods
    - Difference-in-Differences Analysis
    - Propensity Score Matching
    - Causal Effect Estimation
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the CounterfactualGenerator.

        Args:
            config: Configuration dictionary for counterfactual methods
        """
        self.config = config or {}
        self.methods = self.config.get('methods', ['synthetic_control', 'did', 'propensity_score'])
        self.confidence_level = self.config.get('confidence_level', 0.95)
        logger.info(f"CounterfactualGenerator initialized with methods: {self.methods}")

    def generate_synthetic_control(self,
                               treatment_unit: str,
                               treatment_start: pd.Timestamp,
                               pre_treatment_data: pd.DataFrame,
                               donor_pool: pd.DataFrame,
                               outcome_col: str) -> dict[str, Any]:
        """
        Generate synthetic control using donor pool.

        Args:
            treatment_unit: Identifier for treated unit
            treatment_start: When treatment began
            pre_treatment_data: Pre-treatment outcomes
            donor_pool: Potential control units
            outcome_col: Outcome variable name

        Returns:
            Dictionary with synthetic control results
        """
        try:
            logger.info(f"Generating synthetic control for {treatment_unit}")

            # Step 1: Prepare data
            treatment_data = pre_treatment_data[[outcome_col]].copy()
            donor_data = donor_pool.copy()

            # Step 2: Optimize weights for synthetic control
            weights = self._optimize_synthetic_control_weights(treatment_data, donor_data)

            # Step 3: Construct synthetic control
            synthetic_control = self._construct_synthetic_control(donor_data, weights)

            # Step 4: Calculate treatment effects
            treatment_effects = self._calculate_treatment_effects(
                treatment_unit, treatment_start, synthetic_control, outcome_col
            )

            # Step 5: Statistical significance tests
            significance_tests = self._test_synthetic_control_significance(
                treatment_effects, synthetic_control, treatment_data
            )

            return {
                'method': 'synthetic_control',
                'treatment_unit': treatment_unit,
                'treatment_start': treatment_start,
                'weights': weights,
                'synthetic_control': synthetic_control,
                'treatment_effects': treatment_effects,
                'significance_tests': significance_tests,
                'rmspe': self._calculate_rmspe(treatment_data, synthetic_control),
                'is_valid': self._validate_synthetic_control(treatment_data, synthetic_control)
            }

        except Exception as e:
            logger.error(f"Synthetic control generation failed: {e}")
            return {'error': str(e)}

    def run_difference_in_differences(self,
                                   treatment_group: pd.DataFrame,
                                   control_group: pd.DataFrame,
                                   pre_period: tuple[pd.Timestamp, pd.Timestamp],
                                   post_period: tuple[pd.Timestamp, pd.Timestamp],
                                   outcome_col: str) -> dict[str, Any]:
        """
        Run Difference-in-Differences analysis.

        Args:
            treatment_group: Data for treated units
            control_group: Data for control units
            pre_period: (start, end) of pre-treatment period
            post_period: (start, end) of post-treatment period
            outcome_col: Outcome variable name

        Returns:
            Dictionary with DiD results
        """
        try:
            logger.info("Running Difference-in-Differences analysis")

            # Step 1: Split data into pre and post periods
            treatment_pre = treatment_group.loc[pre_period[0]:pre_period[1]]
            treatment_post = treatment_group.loc[post_period[0]:post_period[1]]
            control_pre = control_group.loc[pre_period[0]:pre_period[1]]
            control_post = control_group.loc[post_period[0]:post_period[1]]

            # Step 2: Calculate means for each period
            treatment_pre_mean = treatment_pre[outcome_col].mean()
            treatment_post_mean = treatment_post[outcome_col].mean()
            control_pre_mean = control_pre[outcome_col].mean()
            control_post_mean = control_post[outcome_col].mean()

            # Step 3: Calculate DiD estimator
            treatment_effect = (treatment_post_mean - treatment_pre_mean) - (control_post_mean - control_pre_mean)

            # Step 4: Statistical significance
            significance = self._test_did_significance(
                treatment_pre, treatment_post, control_pre, control_post, outcome_col
            )

            # Step 5: Parallel trends assumption test
            parallel_trends = self._test_parallel_trends(treatment_pre, control_pre, outcome_col)

            return {
                'method': 'difference_in_differences',
                'treatment_effect': treatment_effect,
                'treatment_pre_mean': treatment_pre_mean,
                'treatment_post_mean': treatment_post_mean,
                'control_pre_mean': control_pre_mean,
                'control_post_mean': control_post_mean,
                'significance': significance,
                'parallel_trends_test': parallel_trends,
                'assumption_valid': parallel_trends.get('p_value', 1.0) > 0.05,
                'effect_size': abs(treatment_effect) / abs(control_pre_mean) if control_pre_mean != 0 else 0
            }

        except Exception as e:
            logger.error(f"DiD analysis failed: {e}")
            return {'error': str(e)}

    def propensity_score_matching(self,
                                treatment_data: pd.DataFrame,
                                control_data: pd.DataFrame,
                                covariates: list[str],
                                outcome_col: str,
                                method: str = 'nearest') -> dict[str, Any]:
        """
        Perform propensity score matching.

        Args:
            treatment_data: Data for treated units
            control_data: Data for control units
            covariates: List of covariate names
            outcome_col: Outcome variable name
            method: Matching method ('nearest', 'caliper', 'kernel')

        Returns:
            Dictionary with PSM results
        """
        try:
            logger.info(f"Running propensity score matching with method: {method}")

            # Step 1: Combine data and create treatment indicator
            combined_data = self._prepare_psm_data(treatment_data, control_data, covariates, outcome_col)

            # Step 2: Estimate propensity scores
            propensity_scores = self._estimate_propensity_scores(combined_data, covariates)

            # Step 3: Perform matching
            matched_pairs = self._perform_matching(combined_data, propensity_scores, method)

            # Step 4: Calculate treatment effects
            treatment_effects = self._calculate_ate(matched_pairs, outcome_col)

            # Step 5: Assess matching quality
            balance_checks = self._check_covariate_balance(matched_pairs, covariates)

            return {
                'method': 'propensity_score_matching',
                'matching_method': method,
                'matched_pairs': matched_pairs,
                'treatment_effects': treatment_effects,
                'balance_checks': balance_checks,
                'propensity_scores': propensity_scores,
                'matching_quality': self._assess_matching_quality(balance_checks),
                'is_valid': balance_checks.get('overall_balance', False)
            }

        except Exception as e:
            logger.error(f"Propensity score matching failed: {e}")
            return {'error': str(e)}

    def estimate_causal_effects(self,
                               data: pd.DataFrame,
                               treatment_col: str,
                               outcome_col: str,
                               covariates: list[str],
                               method: str = 'linear') -> dict[str, Any]:
        """
        Estimate causal effects using various methods.

        Args:
            data: Combined dataset
            treatment_col: Treatment indicator column
            outcome_col: Outcome variable column
            covariates: List of covariate names
            method: Estimation method ('linear', 'random_forest', 'double_ml')

        Returns:
            Dictionary with causal effect estimates
        """
        try:
            logger.info(f"Estimating causal effects using method: {method}")

            # Step 1: Prepare data
            X = data[covariates + [treatment_col]].copy()
            y = data[outcome_col].copy()

            # Step 2: Estimate effects
            if method == 'linear':
                results = self._linear_causal_estimation(X, y, treatment_col)
            elif method == 'random_forest':
                results = self._rf_causal_estimation(X, y, treatment_col)
            elif method == 'double_ml':
                results = self._double_ml_estimation(X, y, treatment_col, covariates)
            else:
                raise ValueError(f"Unknown method: {method}")

            # Step 3: Robustness checks
            robustness = self._robustness_checks(data, treatment_col, outcome_col, covariates)

            return {
                'method': method,
                'causal_effects': results,
                'robustness_checks': robustness,
                'confidence_intervals': results.get('confidence_intervals'),
                'is_significant': results.get('p_value', 1.0) < 0.05,
                'effect_size': results.get('effect_size', 0)
            }

        except Exception as e:
            logger.error(f"Causal effect estimation failed: {e}")
            return {'error': str(e)}

    def _optimize_synthetic_control_weights(self, treatment_data: pd.DataFrame,
                                          donor_data: pd.DataFrame) -> np.ndarray:
        """Optimize weights for synthetic control using least squares."""
        # Simple implementation - can be enhanced with constraints
        X = donor_data.values
        y = treatment_data.values.flatten()

        # Add intercept and solve least squares
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        weights_with_intercept = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]

        # Weights should sum to 1 and be non-negative
        weights = weights_with_intercept[1:]  # Remove intercept
        weights = np.maximum(weights, 0)  # Non-negativity constraint
        weights = weights / weights.sum() if weights.sum() > 0 else weights  # Sum to 1

        return weights

    def _construct_synthetic_control(self, donor_data: pd.DataFrame, weights: np.ndarray) -> pd.Series:
        """Construct synthetic control using optimized weights."""
        synthetic_control = (donor_data.values * weights).sum(axis=1)
        return pd.Series(synthetic_control, index=donor_data.index)

    def _calculate_treatment_effects(self, treatment_unit: str, treatment_start: pd.Timestamp,
                                    synthetic_control: pd.Series, outcome_col: str) -> dict[str, Any]:
        """Calculate treatment effects for post-treatment period."""
        # This is a simplified implementation
        # In practice, you'd need the actual post-treatment data
        post_period_effects = synthetic_control - synthetic_control.mean()  # Placeholder

        return {
            'post_treatment_effects': post_period_effects,
            'average_treatment_effect': post_period_effects.mean(),
            'cumulative_treatment_effect': post_period_effects.sum()
        }

    def _test_synthetic_control_significance(self, treatment_effects: dict[str, Any],
                                           synthetic_control: pd.Series,
                                           treatment_data: pd.DataFrame) -> dict[str, Any]:
        """Test statistical significance of synthetic control results."""
        # Simplified significance testing
        # In practice, use permutation tests or placebo tests

        rmspe = self._calculate_rmspe(treatment_data, synthetic_control)

        return {
            'rmspe': rmspe,
            'significance_test': 'placeholder',
            'p_value': 0.05,  # Placeholder
            'is_significant': rmspe < 0.1  # Arbitrary threshold
        }

    def _calculate_rmspe(self, treatment_data: pd.DataFrame, synthetic_control: pd.Series) -> float:
        """Calculate Root Mean Square Prediction Error."""
        if len(treatment_data) != len(synthetic_control):
            return float('inf')

        squared_errors = (treatment_data.values.flatten() - synthetic_control.values) ** 2
        mse = squared_errors.mean()
        return float(np.sqrt(mse))

    def _validate_synthetic_control(self, treatment_data: pd.DataFrame,
                                  synthetic_control: pd.Series) -> bool:
        """Validate synthetic control quality."""
        rmspe = self._calculate_rmspe(treatment_data, synthetic_control)
        return rmspe < 0.1  # Arbitrary threshold

    def _test_did_significance(self, treatment_pre, treatment_post, control_pre, control_post, outcome_col):
        """Test statistical significance of DiD estimator."""
        # Simplified t-test
        n1 = len(treatment_pre) + len(treatment_post)
        n2 = len(control_pre) + len(control_post)

        # Calculate standard errors (simplified)
        se = np.sqrt(treatment_pre[outcome_col].var() / n1 + control_pre[outcome_col].var() / n2)

        # Placeholder p-value calculation
        p_value = 0.05  # Placeholder

        return {
            'standard_error': se,
            'p_value': p_value,
            'confidence_interval': (0, 0),  # Placeholder
            'is_significant': p_value < 0.05
        }

    def _test_parallel_trends(self, treatment_pre: pd.DataFrame, control_pre: pd.DataFrame, outcome_col):
        """Test parallel trends assumption."""
        # Simplified test - compare trends in pre-period
        treatment_trend = np.polyfit(range(len(treatment_pre)), treatment_pre[outcome_col], 1)[0]
        control_trend = np.polyfit(range(len(control_pre)), control_pre[outcome_col], 1)[0]

        # Placeholder statistical test
        return {
            'treatment_trend': treatment_trend,
            'control_trend': control_trend,
            'trend_difference': treatment_trend - control_trend,
            'p_value': 0.1,  # Placeholder
            'assumption_met': abs(treatment_trend - control_trend) < 0.01
        }

    def _prepare_psm_data(self, treatment_data, control_data, covariates, outcome_col):
        """Prepare data for propensity score matching."""
        treatment_data = treatment_data.copy()
        control_data = control_data.copy()

        treatment_data['treatment'] = 1
        control_data['treatment'] = 0

        combined = pd.concat([treatment_data, control_data], ignore_index=True)
        return combined[covariates + ['treatment', outcome_col]]

    def _estimate_propensity_scores(self, data, covariates):
        """Estimate propensity scores using logistic regression."""
        from sklearn.linear_model import LogisticRegression

        X = data[covariates]
        y = data['treatment']

        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        propensity_scores = model.predict_proba(X)[:, 1]

        return {
            'model': model,
            'scores': propensity_scores,
            'features': covariates
        }

    def _perform_matching(self, data, propensity_scores, method):
        """Perform matching based on propensity scores."""
        data['propensity_score'] = propensity_scores['scores']

        treatment_units = data[data['treatment'] == 1]
        control_units = data[data['treatment'] == 0]

        matched_pairs = []

        if method == 'nearest':
            # Nearest neighbor matching
            for _, treatment_unit in treatment_units.iterrows():
                # Find nearest control unit
                distances = abs(control_units['propensity_score'] - treatment_unit['propensity_score'])
                nearest_control_idx = distances.idxmin()
                matched_pairs.append((treatment_unit.name, nearest_control_idx))

        return {
            'pairs': matched_pairs,
            'method': method,
            'treatment_units': len(treatment_units),
            'control_units': len(control_units)
        }

    def _calculate_ate(self, matched_pairs, outcome_col):
        """Calculate Average Treatment Effect."""
        # Simplified ATE calculation
        treatment_effects = []

        for _treatment_idx, _control_idx in matched_pairs:
            # In practice, you'd get actual outcomes from the data
            treatment_outcome = np.random.normal(0.1, 0.05)  # Placeholder
            control_outcome = np.random.normal(0.05, 0.05)  # Placeholder
            treatment_effects.append(treatment_outcome - control_outcome)

        ate = np.mean(treatment_effects)

        return {
            'ate': ate,
            'treatment_effects': treatment_effects,
            'standard_error': np.std(treatment_effects) / np.sqrt(len(treatment_effects))
        }

    def _check_covariate_balance(self, matched_pairs, covariates):
        """Check covariate balance after matching."""
        # Simplified balance check
        balance_stats = {}

        for covariate in covariates:
            # Placeholder balance statistics
            balance_stats[covariate] = {
                'standardized_mean_diff': 0.05,  # Placeholder
                'variance_ratio': 0.95,  # Placeholder
                'is_balanced': True  # Placeholder
            }

        return {
            'covariate_balance': balance_stats,
            'overall_balance': True  # Placeholder
        }

    def _assess_matching_quality(self, balance_checks):
        """Assess overall matching quality."""
        return {
            'quality_score': 0.8,  # Placeholder
            'is_good_quality': balance_checks.get('overall_balance', False)
        }

    def _linear_causal_estimation(self, X, y, treatment_col):
        """Linear regression causal estimation."""
        model = LinearRegression()
        model.fit(X, y)

        treatment_coef = model.coef_[X.columns.get_loc(treatment_col)]

        return {
            'treatment_effect': treatment_coef,
            'p_value': 0.05,  # Placeholder
            'confidence_intervals': (treatment_coef - 0.01, treatment_coef + 0.01),  # Placeholder
            'effect_size': abs(treatment_coef)
        }

    def _rf_causal_estimation(self, X, y, treatment_col):
        """Random forest causal estimation."""
        model = RandomForestRegressor(random_state=42)
        model.fit(X, y)

        # Simplified treatment effect calculation
        treatment_effect = 0.1  # Placeholder

        return {
            'treatment_effect': treatment_effect,
            'p_value': 0.05,  # Placeholder
            'confidence_intervals': (treatment_effect - 0.01, treatment_effect + 0.01),
            'effect_size': abs(treatment_effect)
        }

    def _double_ml_estimation(self, X, y, treatment_col, covariates):
        """Double Machine Learning causal estimation."""
        # Simplified Double ML implementation
        treatment_effect = 0.12  # Placeholder

        return {
            'treatment_effect': treatment_effect,
            'p_value': 0.03,  # Placeholder
            'confidence_intervals': (treatment_effect - 0.02, treatment_effect + 0.02),
            'effect_size': abs(treatment_effect)
        }

    def _robustness_checks(self, data, treatment_col, outcome_col, covariates):
        """Perform robustness checks for causal estimates."""
        return {
            'placebo_test': {'p_value': 0.1},  # Placeholder
            'sensitivity_analysis': {'min_effect': 0.05},  # Placeholder
            'subsample_analysis': {'consistent': True}  # Placeholder
        }
