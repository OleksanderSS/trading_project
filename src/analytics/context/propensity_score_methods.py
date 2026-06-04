#!/usr/bin/env python3
"""
Propensity Score Methods - Implementation of Propensity Score Matching

This module contains the implementation of propensity score matching methods for causal analysis.
"""

from typing import Any

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger


class PropensityScoreMethods:
    """
    Implementation of propensity score matching methods.

    This class encapsulates the logic for PSM, including data preparation,
    score estimation, matching, and balance checking.
    """

    def __init__(self, config: dict[str, Any] = None, logger=None):
        """
        Initialize the propensity score methods.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config or {}
        self.logger = logger or ProjectLogger.get_logger(self.__class__.__name__)

    def prepare_data(self, treatment_data, control_data, covariates, outcome_col):
        """Prepare data for propensity score matching."""
        treatment_data = treatment_data.copy()
        control_data = control_data.copy()

        treatment_data['treatment'] = 1
        control_data['treatment'] = 0

        combined = pd.concat([treatment_data, control_data], ignore_index=True)
        return combined[covariates + ['treatment', outcome_col]]

    def estimate_propensity_scores(self, data, covariates):
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

    def perform_matching(self, data, propensity_scores, method):
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

    def calculate_ate(self, matched_pairs, outcome_col, data=None):
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

    def check_covariate_balance(self, matched_pairs, covariates, data=None):
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

    def assess_matching_quality(self, balance_checks):
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
