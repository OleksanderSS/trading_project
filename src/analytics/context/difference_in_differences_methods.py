#!/usr/bin/env python3
"""
Difference-in-Differences Methods - Implementation of DiD Analysis

This module contains the implementation of difference-in-differences methods for causal analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from scipy import stats

from src.core.logging.logger import ProjectLogger


class DifferenceInDifferencesMethods:
    """
    Implementation of difference-in-differences methods.
    
    This class encapsulates the logic for DiD analysis, significance testing,
    and parallel trends assumption validation.
    """
    
    def __init__(self, confidence_level: float = 0.95, logger=None):
        """
        Initialize the DiD methods.
        
        Args:
            confidence_level: Confidence level for statistical tests
            logger: Logger instance
        """
        self.confidence_level = confidence_level
        self.logger = logger or ProjectLogger.get_logger(self.__class__.__name__)
    
    def test_significance(self, treatment_pre, treatment_post, control_pre, control_post, outcome_col):
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
    
    def test_parallel_trends(self, treatment_pre: pd.DataFrame, control_pre: pd.DataFrame, outcome_col):
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
