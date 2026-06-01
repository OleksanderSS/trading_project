"""
Counterfactual Analysis Generator - "What if" scenario analysis for trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
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
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
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
                               outcome_col: str) -> Dict[str, Any]:
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
            logger.error(f"Synthetic control generation failed: {e}", exc_info=True)
            raise RuntimeError(f"Synthetic control generation failed: {e}") from e
    
    def run_difference_in_differences(self,
                                   treatment_group: pd.DataFrame,
                                   control_group: pd.DataFrame,
                                   pre_period: Tuple[pd.Timestamp, pd.Timestamp],
                                   post_period: Tuple[pd.Timestamp, pd.Timestamp],
                                   outcome_col: str) -> Dict[str, Any]:
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
            logger.error(f"DiD analysis failed: {e}", exc_info=True)
            raise RuntimeError(f"DiD analysis failed: {e}") from e
    
    def propensity_score_matching(self,
                                treatment_data: pd.DataFrame,
                                control_data: pd.DataFrame,
                                covariates: List[str],
                                outcome_col: str,
                                method: str = 'nearest') -> Dict[str, Any]:
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
            treatment_effects = self._calculate_ate(matched_pairs, outcome_col, combined_data)
            
            # Step 5: Assess matching quality
            balance_checks = self._check_covariate_balance(matched_pairs, covariates, combined_data)
            
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
            logger.error(f"Propensity score matching failed: {e}", exc_info=True)
            raise RuntimeError(f"Propensity score matching failed: {e}") from e
    
    def estimate_causal_effects(self,
                               data: pd.DataFrame,
                               treatment_col: str,
                               outcome_col: str,
                               covariates: List[str],
                               method: str = 'linear') -> Dict[str, Any]:
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
            logger.error(f"Causal effect estimation failed: {e}", exc_info=True)
            raise RuntimeError(f"Causal effect estimation failed: {e}") from e
    
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
                                    synthetic_control: pd.Series, outcome_col: str) -> Dict[str, Any]:
        """Calculate treatment effects for post-treatment period."""
        post_period_effects = pd.Series(dtype=float)
        
        return {
            'post_treatment_effects': post_period_effects,
            'average_treatment_effect': None,
            'cumulative_treatment_effect': None,
            'requires_post_treatment_data': True
        }
    
    def _test_synthetic_control_significance(self, treatment_effects: Dict[str, Any],
                                           synthetic_control: pd.Series,
                                           treatment_data: pd.DataFrame) -> Dict[str, Any]:
        """Test statistical significance of synthetic control results."""
        rmspe = self._calculate_rmspe(treatment_data, synthetic_control)
        effects = pd.Series(treatment_effects.get('post_treatment_effects', []), dtype=float).dropna()
        if len(effects) > 1 and effects.std(ddof=1) > 0:
            t_statistic, p_value = stats.ttest_1samp(effects, popmean=0.0, nan_policy='omit')
            t_statistic = float(t_statistic) if np.isfinite(t_statistic) else 0.0
            p_value = float(p_value) if np.isfinite(p_value) else 1.0
        else:
            t_statistic = 0.0
            p_value = 1.0
        
        return {
            'rmspe': rmspe,
            'significance_test': 'one_sample_t_test',
            't_statistic': t_statistic,
            'p_value': p_value,
            'is_significant': p_value < (1 - self.confidence_level)
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
        treatment_pre_mean = treatment_pre[outcome_col].mean()
        control_pre_mean = control_pre[outcome_col].mean()
        treatment_post_mean = treatment_post[outcome_col].mean()
        control_post_mean = control_post[outcome_col].mean()
        treatment_delta = (treatment_post[outcome_col] - treatment_pre_mean).dropna()
        control_delta = (control_post[outcome_col] - control_pre_mean).dropna()
        effect = float(
            (treatment_post_mean - treatment_pre_mean)
            - (control_post_mean - control_pre_mean)
        )

        if len(treatment_delta) > 1 and len(control_delta) > 1:
            t_statistic, p_value = stats.ttest_ind(
                treatment_delta,
                control_delta,
                equal_var=False,
                nan_policy='omit',
            )
            se = np.sqrt(
                treatment_delta.var(ddof=1) / len(treatment_delta)
                + control_delta.var(ddof=1) / len(control_delta)
            )
        else:
            t_statistic = 0.0
            p_value = 1.0
            se = 0.0

        t_statistic = float(t_statistic) if np.isfinite(t_statistic) else 0.0
        p_value = float(p_value) if np.isfinite(p_value) else 1.0
        se = float(se) if np.isfinite(se) else 0.0
        alpha = 1 - self.confidence_level
        dof = max(len(treatment_delta) + len(control_delta) - 2, 1)
        critical = float(stats.t.ppf(1 - alpha / 2, dof)) if se > 0 else 0.0
        
        return {
            'standard_error': se,
            't_statistic': t_statistic,
            'p_value': p_value,
            'confidence_interval': (effect - critical * se, effect + critical * se),
            'is_significant': p_value < alpha
        }
    
    def _test_parallel_trends(self, treatment_pre: pd.DataFrame, control_pre: pd.DataFrame, outcome_col):
        """Test parallel trends assumption."""
        treatment_trend = np.polyfit(range(len(treatment_pre)), treatment_pre[outcome_col], 1)[0]
        control_trend = np.polyfit(range(len(control_pre)), control_pre[outcome_col], 1)[0]
        treatment_changes = treatment_pre[outcome_col].diff().dropna()
        control_changes = control_pre[outcome_col].diff().dropna()
        if len(treatment_changes) > 1 and len(control_changes) > 1:
            _, p_value = stats.ttest_ind(
                treatment_changes,
                control_changes,
                equal_var=False,
                nan_policy='omit',
            )
            p_value = float(p_value) if np.isfinite(p_value) else 1.0
        else:
            p_value = 1.0
        
        return {
            'treatment_trend': treatment_trend,
            'control_trend': control_trend,
            'trend_difference': treatment_trend - control_trend,
            'p_value': p_value,
            'assumption_met': p_value > 0.05
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
        distances_used = []
        if treatment_units.empty or control_units.empty:
            return {
                'pairs': matched_pairs,
                'distances': distances_used,
                'method': method,
                'treatment_units': len(treatment_units),
                'control_units': len(control_units)
            }
        
        if method == 'nearest':
            # Nearest neighbor matching
            for _, treatment_unit in treatment_units.iterrows():
                # Find nearest control unit
                distances = abs(control_units['propensity_score'] - treatment_unit['propensity_score'])
                nearest_control_idx = distances.idxmin()
                distances_used.append(float(distances.loc[nearest_control_idx]))
                matched_pairs.append((treatment_unit.name, nearest_control_idx))
        elif method == 'caliper':
            caliper = float(self.config.get('caliper', 0.05))
            for _, treatment_unit in treatment_units.iterrows():
                distances = abs(control_units['propensity_score'] - treatment_unit['propensity_score'])
                nearest_control_idx = distances.idxmin()
                nearest_distance = float(distances.loc[nearest_control_idx])
                if nearest_distance <= caliper:
                    distances_used.append(nearest_distance)
                    matched_pairs.append((treatment_unit.name, nearest_control_idx))
        else:
            raise ValueError(f"Unsupported matching method: {method}")
        
        return {
            'pairs': matched_pairs,
            'distances': distances_used,
            'method': method,
            'treatment_units': len(treatment_units),
            'control_units': len(control_units)
        }
    
    def _calculate_ate(self, matched_pairs, outcome_col, data=None):
        """Calculate Average Treatment Effect."""
        treatment_effects = []
        
        if data is None:
            return {'ate': 0.0, 'treatment_effects': [], 'standard_error': 0.0}

        pairs = matched_pairs.get('pairs', []) if isinstance(matched_pairs, dict) else matched_pairs
        for treatment_idx, control_idx in pairs:
            treatment_outcome = data.loc[treatment_idx, outcome_col]
            control_outcome = data.loc[control_idx, outcome_col]
            if pd.notna(treatment_outcome) and pd.notna(control_outcome):
                treatment_effects.append(float(treatment_outcome - control_outcome))
        
        if not treatment_effects:
            return {'ate': 0.0, 'treatment_effects': [], 'standard_error': 0.0}

        ate = float(np.mean(treatment_effects))
        standard_error = float(np.std(treatment_effects, ddof=1) / np.sqrt(len(treatment_effects))) if len(treatment_effects) > 1 else 0.0
        
        return {
            'ate': ate,
            'treatment_effects': treatment_effects,
            'standard_error': standard_error
        }
    
    def _check_covariate_balance(self, matched_pairs, covariates, data=None):
        """Check covariate balance after matching."""
        balance_stats = {}
        pairs = matched_pairs.get('pairs', []) if isinstance(matched_pairs, dict) else matched_pairs
        if data is None or not pairs:
            return {'covariate_balance': balance_stats, 'overall_balance': False}

        treatment_indices = [pair[0] for pair in pairs]
        control_indices = [pair[1] for pair in pairs]
        treatment_matched = data.loc[treatment_indices]
        control_matched = data.loc[control_indices]
        
        for covariate in covariates:
            treatment_values = treatment_matched[covariate].astype(float).dropna()
            control_values = control_matched[covariate].astype(float).dropna()
            treatment_var = treatment_values.var(ddof=1) if len(treatment_values) > 1 else 0.0
            control_var = control_values.var(ddof=1) if len(control_values) > 1 else 0.0
            degrees_of_freedom = len(treatment_values) + len(control_values) - 2
            pooled_var = (
                ((len(treatment_values) - 1) * treatment_var + (len(control_values) - 1) * control_var)
                / degrees_of_freedom
                if degrees_of_freedom > 0
                else 0.0
            )
            pooled_std = np.sqrt(pooled_var)
            treatment_mean = treatment_values.mean()
            control_mean = control_values.mean()
            standardized_mean_diff = (
                (treatment_mean - control_mean) / pooled_std
                if pooled_std > 0 else 0.0
            )
            variance_ratio = treatment_var / control_var if control_var > 0 else 1.0
            balance_stats[covariate] = {
                'standardized_mean_diff': float(standardized_mean_diff),
                'variance_ratio': float(variance_ratio),
                'is_balanced': abs(standardized_mean_diff) < 0.1 and 0.5 <= variance_ratio <= 2.0
            }
        
        return {
            'covariate_balance': balance_stats,
            'overall_balance': all(item['is_balanced'] for item in balance_stats.values())
        }
    
    def _assess_matching_quality(self, balance_checks):
        """Assess overall matching quality."""
        balance_stats = balance_checks.get('covariate_balance', {})
        if not balance_stats:
            return {'quality_score': 0.0, 'is_good_quality': False}
        avg_abs_smd = np.mean([
            abs(stats_['standardized_mean_diff'])
            for stats_ in balance_stats.values()
        ])
        return {
            'quality_score': float(max(0.0, 1.0 - avg_abs_smd)),
            'is_good_quality': balance_checks.get('overall_balance', False)
        }
    
    def _linear_causal_estimation(self, X, y, treatment_col):
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
    
    def _rf_causal_estimation(self, X, y, treatment_col):
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
    
    def _double_ml_estimation(self, X, y, treatment_col, covariates):
        """Double Machine Learning causal estimation."""
        if not covariates:
            return self._linear_causal_estimation(X, y, treatment_col)

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
    
    def _robustness_checks(self, data, treatment_col, outcome_col, covariates):
        """Perform robustness checks for causal estimates."""
        model_columns = covariates + [treatment_col]
        try:
            base_effect = self._linear_causal_estimation(
                data[model_columns],
                data[outcome_col],
                treatment_col,
            )
            shuffled = data.copy()
            rng = np.random.default_rng(42)
            shuffled[treatment_col] = rng.permutation(shuffled[treatment_col].to_numpy())
            placebo_effect = self._linear_causal_estimation(
                shuffled[model_columns],
                shuffled[outcome_col],
                treatment_col,
            )

            if len(data) >= 4:
                subsample = data.sample(frac=0.5, random_state=42)
                subsample_effect = self._linear_causal_estimation(
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
            logger.error(f"Robustness check failed: {exc}", exc_info=True)
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
