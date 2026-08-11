#!/usr/bin/env python3
"""
Evaluation Metrics Calculator - Financial metrics calculation for evaluation
Handles calculation of professional financial metrics for portfolio evaluation.
"""

from typing import Any

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("EvaluationMetricsCalculator")


class EvaluationMetricsCalculator:
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
        Initialize Evaluation Metrics Calculator.

        Args:
            metrics_calculator: Optional PortfolioMetricsCalculator instance
        """
        self.logger = logger
        self.metrics_calculator = metrics_calculator
        self.logger.info("✅ EvaluationMetricsCalculator initialized")

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
                # PortfolioMetricsCalculator.calculate() requires a Series
                # (validate_input rejects a DataFrame outright, returning {}
                # silently) - portfolio_history[['total_value']] (double
                # brackets) is a one-column DataFrame, not a Series.
                financial_metrics = self.metrics_calculator.calculate(portfolio_history['total_value'])
                return financial_metrics

            # Fallback to manual calculation
            return self._calculate_basic_metrics(portfolio_history)

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error calculating financial metrics: {e}")
            return {'status': 'error', 'error': str(e)}

    def _calculate_basic_metrics(
        self,
        portfolio_history: pd.DataFrame,
        periods_per_year: int | None = None,
    ) -> dict[str, Any]:
        """Calculate basic financial metrics manually.

        Args:
            portfolio_history: Portfolio history DataFrame.
            periods_per_year: Annualisation factor (e.g. 252 for daily,
                1440 for 1-min intraday). When *None* the factor is inferred
                automatically from the DatetimeIndex of portfolio_history.
                Pass an explicit value when the index is not a DatetimeIndex.
        """
        try:
            values = portfolio_history['total_value']
            returns = values.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()

            if len(returns) == 0:
                return {}

            from src.metrics.financial.financial_metrics_library import (
                FinancialMetricsLibrary,
                get_risk_free_rate,
                infer_periods_per_year,
            )

            # Determine annualisation factor
            ppy = periods_per_year if periods_per_year is not None else infer_periods_per_year(returns)
            risk_free_rate = get_risk_free_rate()

            # Calculate metrics
            total_return = (values.iloc[-1] / values.iloc[0]) - 1
            volatility = returns.std()
            # Delegated rather than recomputed: this used to hardcode
            # risk_free_rate = 0.0 while the backtest engine's Sharpe over the
            # SAME equity curve used 0.02, which is the whole reason one
            # summary carried two different Sharpe ratios.
            sharpe_ratio = FinancialMetricsLibrary.calculate_sharpe_ratio(
                returns,
                risk_free_rate=risk_free_rate,
                trading_days_per_year=ppy,
                on_error=np.nan,
            )

            # Drawdown straight off the equity curve. It used to be measured on
            # `(1 + returns).cumprod()`, which is rebased to 1.0 at the SECOND
            # observation -- so a peak on the first bar was invisible and the
            # drawdown came out shallower than the truth. That is the second
            # half of the same disagreement: -0.003749 here against -0.005443
            # from the engine, on one curve.
            max_drawdown = FinancialMetricsLibrary.calculate_max_drawdown(values)

            # Calculate CAGR — use actual number of periods, not raw row count
            n_periods = len(portfolio_history)
            years = n_periods / ppy
            cagr = (values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1 if years > 0 else 0

            return {
                'total_return': total_return,
                'total_return_pct': total_return * 100,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'cagr': cagr,
                'periods_per_year_used': ppy,
                # A Sharpe cannot be compared with another Sharpe unless both
                # rates are known, so the convention travels with the number.
                'risk_free_rate_used': risk_free_rate,
            }

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error calculating basic metrics: {e}")
            return {'status': 'error', 'error': str(e)}

    #: Columns that carry a REALIZED outcome, in preference order. The frame
    #: this class receives is Stage 5's prediction frame (evaluation
    #: orchestrator: `signals = kwargs.get('predictions')` when Stage 6 is
    #: skipped, which is the normal review path), so none of these is
    #: normally present -- and that is exactly what has to be said out loud.
    _REALIZED_RETURN_COLUMNS = ('realized_return', 'actual_return', 'outcome_return')

    def calculate_pattern_specific_metrics(self, df: pd.DataFrame) -> dict[str, Any]:
        """Per-pattern scorecard, labelled by what was actually measured.

        This used to publish ``mean(predictions)`` under the key
        ``avg_return`` and ``mean(predictions > 0)`` under ``win_rate``. Those
        are not returns and not a win rate: a prediction frame stacks targets
        of different units (fractional returns, price levels, indicator
        values), so their mean is dimensionless noise. In
        ``summary_20260810_123512.json`` it read **2.66e14** while the
        portfolio's real return was 0.17%.

        Realized returns are used when the frame carries them; otherwise the
        prediction statistics are reported under prediction names, with a
        status saying no outcome was available. A number that quietly means
        something else is worse than a missing one.
        """
        try:
            if 'context_pattern_id' not in df.columns:
                return {}

            realized_col = next(
                (c for c in self._REALIZED_RETURN_COLUMNS if c in df.columns),
                None,
            )
            if realized_col is None and 'predictions' not in df.columns:
                return {'status': 'no_outcome_or_prediction_column'}

            scorecard: dict[str, Any] = {}
            for pattern_id, group in df.groupby('context_pattern_id'):
                chaos_level = (
                    float(group['context_velocity'].mean())
                    if 'context_velocity' in group.columns
                    else 0.0
                )
                if realized_col is not None:
                    returns = pd.to_numeric(
                        group[realized_col], errors='coerce'
                    ).dropna().to_numpy()
                    if returns.size == 0:
                        continue
                    scorecard[str(pattern_id)] = {
                        'samples': int(returns.size),
                        'win_rate': float(np.mean(returns > 0)),
                        'avg_return': float(np.mean(returns)),
                        'chaos_level': chaos_level,
                        'measured_from': realized_col,
                    }
                    continue

                preds = pd.to_numeric(
                    group['predictions'], errors='coerce'
                ).dropna().to_numpy()
                if preds.size == 0:
                    continue
                scorecard[str(pattern_id)] = {
                    'samples': int(preds.size),
                    # Deliberately NOT win_rate/avg_return: these describe the
                    # model's output, not money.
                    'positive_prediction_rate': float(np.mean(preds > 0)),
                    'avg_prediction': float(np.mean(preds)),
                    'chaos_level': chaos_level,
                    'status': 'no_realized_returns_available',
                    'reason': (
                        'Frame carries predictions only; realized outcomes '
                        f'({", ".join(self._REALIZED_RETURN_COLUMNS)}) absent. '
                        'Predictions of different targets are on different '
                        'scales, so their mean is not a return.'
                    ),
                }
            return scorecard

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error calculating pattern-specific metrics: {e}")
            return {'status': 'error', 'error': str(e)}

    def analyze_chaos_efficiency(self, df: pd.DataFrame) -> dict[str, Any]:
        """Exposure reduction in high-chaos bars. NOT avoided loss.

        The docstring used to promise "how much loss was avoided thanks to
        Kill-Switch" and the empty branch returned ``avoided_drawdown: 0.0``,
        but nothing here touches prices or equity: it counts how many
        high-chaos rows carried confidence above the threshold. Avoided
        drawdown cannot be computed without the counterfactual equity curve of
        the trades that were suppressed, and that curve does not exist.

        Args:
            df: DataFrame with signals and chaos information

        Returns:
            Exposure-reduction counts, or a status explaining what is missing.
        """
        try:
            if 'context_velocity' not in df.columns or 'confidence' not in df.columns:
                return {'status': 'insufficient_data'}

            high_chaos = df[df['context_velocity'] > 0.7]
            if high_chaos.empty:
                # No `avoided_drawdown: 0.0` here: a zero under that name reads
                # as "the kill-switch saved nothing", when the truth is that no
                # bar was chaotic enough for it to have anything to do.
                return {
                    'chaos_samples': 0,
                    'status': 'no_high_chaos_bars',
                    'reason': 'No row exceeded context_velocity > 0.7.',
                }

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
        """Which model was USED most per pattern. Usage, not skill.

        Two defects fixed here. The result was published as ``top_expert``
        while `value_counts()` ranks by frequency — the most-used model is the
        one the router reached for, which says nothing about whether it was
        right. And blank labels were counted: in
        ``summary_20260810_123512.json`` the "top expert" was the empty
        string, with a 14.5% share, because `selected_primary_model` is unset
        on most rows.

        Args:
            df: DataFrame with signals and model information

        Returns:
            Per-pattern usage map, or a status when no model is labelled.
        """
        try:
            if 'context_pattern_id' not in df.columns or 'selected_primary_model' not in df.columns:
                return {}

            expertise: dict[str, Any] = {}
            for pattern_id, group in df.groupby('context_pattern_id'):
                labels = group['selected_primary_model'].astype('string').str.strip()
                labels = labels[labels.notna() & (labels != '')]
                if labels.empty:
                    expertise[str(pattern_id)] = {
                        'status': 'no_model_labelled',
                        'samples': int(len(group)),
                    }
                    continue
                counts = labels.value_counts()
                expertise[str(pattern_id)] = {
                    # Renamed from top_expert/expert_usage_pct: this ranks by
                    # how often a model was selected, not by its outcome.
                    'most_used_model': str(counts.index[0]),
                    'usage_pct_of_labelled': float(counts.iloc[0] / len(labels)),
                    'labelled_rows': int(len(labels)),
                    'unlabelled_rows': int(len(group) - len(labels)),
                    'ranked_by': 'selection_frequency_not_performance',
                }
            return expertise

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error generating expertise map: {e}")
            return {'status': 'error', 'error': str(e)}


# Factory function
def get_evaluation_metrics_calculator(metrics_calculator=None) -> EvaluationMetricsCalculator:
    """Factory function to get EvaluationMetricsCalculator instance."""
    return EvaluationMetricsCalculator(metrics_calculator)


# Backward compatibility alias
MetricsCalculator = EvaluationMetricsCalculator
