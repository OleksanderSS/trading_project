"""
Counterfactual Analysis Generator - "What if" scenario analysis for trading strategies.
"""

from typing import Any

import pandas as pd

from src.analytics.context.causal_effect_methods import CausalEffectMethods
from src.analytics.context.difference_in_differences_methods import DifferenceInDifferencesMethods
from src.analytics.context.propensity_score_methods import PropensityScoreMethods
from src.analytics.context.synthetic_control_methods import SyntheticControlMethods
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

        # Initialize method-specific classes
        self.synthetic_control = SyntheticControlMethods(self.confidence_level, logger)
        self.did_methods = DifferenceInDifferencesMethods(self.confidence_level, logger)
        self.psm_methods = PropensityScoreMethods(self.config, logger)
        self.causal_effects = CausalEffectMethods(self.confidence_level, logger)

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
            weights = self.synthetic_control.optimize_weights(treatment_data, donor_data)

            # Step 3: Construct synthetic control
            synthetic_control = self.synthetic_control.construct_synthetic_control(donor_data, weights)

            # Step 4: Calculate treatment effects
            treatment_effects = self.synthetic_control.calculate_treatment_effects(
                treatment_unit, treatment_start, synthetic_control, outcome_col
            )

            # Step 5: Statistical significance tests
            significance_tests = self.synthetic_control.test_significance(
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
                'rmspe': self.synthetic_control.calculate_rmspe(treatment_data, synthetic_control),
                'is_valid': self.synthetic_control.validate_synthetic_control(treatment_data, synthetic_control)
            }

        except Exception as e:
            logger.error(f"Synthetic control generation failed: {e}", exc_info=True)
            raise RuntimeError(f"Synthetic control generation failed: {e}") from e

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
            significance = self.did_methods.test_significance(
                treatment_pre, treatment_post, control_pre, control_post, outcome_col
            )

            # Step 5: Parallel trends assumption test
            parallel_trends = self.did_methods.test_parallel_trends(treatment_pre, control_pre, outcome_col)

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
            combined_data = self.psm_methods.prepare_data(treatment_data, control_data, covariates, outcome_col)

            # Step 2: Estimate propensity scores
            propensity_scores = self.psm_methods.estimate_propensity_scores(combined_data, covariates)

            # Step 3: Perform matching
            matched_pairs = self.psm_methods.perform_matching(combined_data, propensity_scores, method)

            # Step 4: Calculate treatment effects
            treatment_effects = self.psm_methods.calculate_ate(matched_pairs, outcome_col, combined_data)

            # Step 5: Assess matching quality
            balance_checks = self.psm_methods.check_covariate_balance(matched_pairs, covariates, combined_data)

            return {
                'method': 'propensity_score_matching',
                'matching_method': method,
                'matched_pairs': matched_pairs,
                'treatment_effects': treatment_effects,
                'balance_checks': balance_checks,
                'propensity_scores': propensity_scores,
                'matching_quality': self.psm_methods.assess_matching_quality(balance_checks),
                'is_valid': balance_checks.get('overall_balance', False)
            }

        except Exception as e:
            logger.error(f"Propensity score matching failed: {e}", exc_info=True)
            raise RuntimeError(f"Propensity score matching failed: {e}") from e

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
                results = self.causal_effects.linear_estimation(X, y, treatment_col)
            elif method == 'random_forest':
                results = self.causal_effects.rf_estimation(X, y, treatment_col)
            elif method == 'double_ml':
                results = self.causal_effects.double_ml_estimation(X, y, treatment_col, covariates)
            else:
                raise ValueError(f"Unknown method: {method}")

            # Step 3: Robustness checks
            robustness = self.causal_effects.robustness_checks(data, treatment_col, outcome_col, covariates)

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
