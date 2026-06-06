#!/usr/bin/env python3
"""
Metrics Calculator - Financial metrics calculation for evaluation
Handles calculation of professional financial metrics for portfolio evaluation.
"""

from typing import Any

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("MetricsCalculator")


class MetricsCalculator:
    """
    Financial metrics calculator for portfolio evaluation.

    Calculates:
    - Total return
    - Sharpe ratio
    - Maximum drawdown
    - Volatility
    - CAGR
    - Win rate
    - Pattern-specific metrics
    - Chaos efficiency metrics
    """

    def __init__(self, metrics_calculator=None):
        """
        Initialize Metrics Calculator.

        Args:
            metrics_calculator: Optional PortfolioMetricsCalculator instance
        """
        self.logger = logger
        self.metrics_calculator = metrics_calculator
        self.logger.info("✅ MetricsCalculator initialized")

    def calculate_financial_metrics(self, portfolio_history: pd.DataFrame) -> dict[str, Any]:
        """
        Calculate professional financial metrics from portfolio history.

        Args:
            portfolio_history: DataFrame with portfolio history

        Returns:
            Dictionary with financial metrics
        """
        try:
            if portfolio_history is None or portfolio_history.empty:
                return {}

            if 'total_value' not in portfolio_history.columns:
                self.logger.error("❌ 'total_value' column not found in portfolio_history")
                return {}

            # Use PortfolioMetricsCalculator if available
            if self.metrics_calculator:
                financial_metrics = self.metrics_calculator.calculate(portfolio_history[['total_value']])
                return financial_metrics

            # Fallback to manual calculation
            return self._calculate_basic_metrics(portfolio_history)

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error calculating financial metrics: {e}")
            return {'status': 'error', 'error': str(e)}

    def _calculate_basic_metrics(self, portfolio_history: pd.DataFrame) -> dict[str, Any]:
        """Calculate basic financial metrics manually."""
        try:
            values = portfolio_history['total_value']
            returns = values.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()

            if len(returns) == 0:
                return {}

            # Calculate metrics
            total_return = (values.iloc[-1] / values.iloc[0]) - 1
            volatility = returns.std()
            sharpe_ratio = (
                returns.mean() / volatility
                if np.isfinite(volatility) and volatility > 1e-12
                else np.nan
            )

            # Calculate drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            # Calculate CAGR
            days = len(portfolio_history)
            years = days / 252  # Trading days per year
            cagr = (values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1 if years > 0 else 0

            return {
                'total_return': total_return,
                'total_return_pct': total_return * 100,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'cagr': cagr
            }

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error calculating basic metrics: {e}")
            return {'status': 'error', 'error': str(e)}

    def calculate_pattern_specific_metrics(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Calculate financial metrics for each specific pattern.

        Args:
            df: DataFrame with signals and pattern information

        Returns:
            Dictionary with pattern-specific metrics
        """
        try:
            if 'context_pattern_id' not in df.columns:
                return {}

            scorecard = {}
            for pattern_id, group in df.groupby('context_pattern_id'):
                # Simplified calculation for each pattern
                # In real backtest, this would calculate PnL specifically at these moments
                returns = group['predictions'].values if 'predictions' in group.columns else []
                if len(returns) > 0:
                    win_rate = np.mean(returns > 0)
                    avg_ret = np.mean(returns)
                    scorecard[str(pattern_id)] = {
                        'samples': len(group),
                        'win_rate': float(win_rate),
                        'avg_return': float(avg_ret),
                        'chaos_level': float(group['context_velocity'].mean()) if 'context_velocity' in group.columns else 0
                    }
            return scorecard

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error calculating pattern-specific metrics: {e}")
            return {'status': 'error', 'error': str(e)}

    def analyze_chaos_efficiency(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Analyze how much loss was avoided thanks to Kill-Switch.

        Args:
            df: DataFrame with signals and chaos information

        Returns:
            Dictionary with chaos efficiency metrics
        """
        try:
            if 'context_velocity' not in df.columns or 'confidence' not in df.columns:
                return {'status': 'insufficient_data'}

            high_chaos = df[df['context_velocity'] > 0.7]
            if high_chaos.empty:
                return {'avoided_drawdown': 0.0, 'status': 'low_volatility_period'}

            # Calculate difference between "model confidence" and "penalized confidence"
            potential_exposure = len(high_chaos)
            actual_exposure = (high_chaos['confidence'] > 0.3).sum()

            return {
                'chaos_samples': int(potential_exposure),
                'reduced_exposure_trades': int(potential_exposure - actual_exposure),
                'protection_factor': float(1 - (actual_exposure / potential_exposure)) if potential_exposure > 0 else 0.0
            }

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error analyzing chaos efficiency: {e}")
            return {'status': 'error', 'error': str(e)}

    def generate_expertise_map(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Determine which architectures work best in which patterns.

        Args:
            df: DataFrame with signals and model information

        Returns:
            Dictionary with expertise map
        """
        try:
            if 'context_pattern_id' not in df.columns or 'selected_primary_model' not in df.columns:
                return {}

            # Create map: Pattern -> Most popular model (and its success)
            expertise = {}
            for pattern_id, group in df.groupby('context_pattern_id'):
                best_models = group['selected_primary_model'].value_counts()
                if not best_models.empty:
                    expertise[str(pattern_id)] = {
                        'top_expert': str(best_models.index[0]),
                        'expert_usage_pct': float(best_models.iloc[0] / len(group))
                    }
            return expertise

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error generating expertise map: {e}")
            return {'status': 'error', 'error': str(e)}


# Factory function
def get_metrics_calculator(metrics_calculator=None) -> MetricsCalculator:
    """Factory function to get MetricsCalculator instance."""
    return MetricsCalculator(metrics_calculator)
