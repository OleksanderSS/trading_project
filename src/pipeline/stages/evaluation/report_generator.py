#!/usr/bin/env python3
"""
Report Generator - Evaluation report generation
Handles generation of comprehensive evaluation reports.
"""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("ReportGenerator")


class ReportGenerator:
    """
    Evaluation report generator.

    Handles:
    - Equity curve plotting
    - Summary report generation
    - Notification message generation
    - Report saving
    """

    def __init__(self, reports_dir: Path | None = None):
        """
        Initialize Report Generator.

        Args:
            reports_dir: Directory for saving reports
        """
        self.logger = logger
        self.reports_dir = reports_dir or Path('reports/evaluation')
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("✅ ReportGenerator initialized")

    def plot_equity_curve(self, portfolio_history: pd.DataFrame,
                          financial_metrics: dict[str, Any],
                          simulated: bool = False) -> str:
        """Draw the equity curve, and say on the image when it is not real.

        THIS METHOD DID NOT ACCEPT `simulated` UNTIL 2026-09-07, AND STAGE 7
        HAS BEEN CALLING IT WITH `simulated=` SINCE 2026-08-30.

        There were two implementations of this drawing. `reporting.
        plot_equity_curve` is the one that was fixed when a run wrote
        `equity_curve.png` from randomly generated data, finished with
        "Pipeline completed successfully", and left a person looking at a
        picture with no way to know. The orchestrator was pointed at THIS one,
        which had no such marking and no such parameter -- so the call raised
        TypeError, the broad `except (ValueError, TypeError, ...)` around the
        comprehensive evaluation caught it, and every run degraded to the basic
        path with `notification_status: basic_evaluation_not_sent`. The fix of
        2026-08-30 never once took effect.

        Two implementations of one thing, the fix applied to one and the caller
        pointed at the other: it is the half-landed shape this project keeps
        naming. So there is one implementation now and this delegates to it,
        passing `out_dir` so the file stays exactly where this method has
        always written it.
        """
        from src.pipeline.stages.evaluation.reporting import (
            plot_equity_curve as draw,
        )

        try:
            plot_path = draw(portfolio_history, financial_metrics,
                             simulated=simulated, out_dir=self.reports_dir)
            self.logger.info(
                "Equity curve saved to %s%s", plot_path,
                " (MARKED as simulated)" if simulated else "")
            return str(plot_path)

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error plotting equity curve: {e}")
            return ""

    def create_evaluation_summary(self, financial_metrics: dict[str, Any],
                                  backtest_results: dict[str, Any],
                                  analysis_results: dict[str, Any],
                                  signals_df: pd.DataFrame | None = None) -> dict[str, Any]:
        """
        Create comprehensive evaluation summary.

        Args:
            financial_metrics: Financial metrics dictionary
            backtest_results: Backtest results dictionary
            analysis_results: Analysis results dictionary
            signals_df: Optional signals DataFrame for pattern analysis

        Returns:
            Dictionary with evaluation summary
        """
        try:
            summary = {
                'metrics': financial_metrics,
                'backtest_stats': backtest_results.get('performance', {}),
                'analysis': analysis_results,
                'timestamp': pd.Timestamp.now().isoformat()
            }

            if signals_df is not None and not signals_df.empty:
                # Pattern-specific analysis
                from src.pipeline.stages.evaluation.metrics_calculator import MetricsCalculator
                metrics_calc = MetricsCalculator()

                summary['regime_scorecard'] = metrics_calc.calculate_pattern_specific_metrics(signals_df)
                summary['chaos_efficiency'] = metrics_calc.analyze_chaos_efficiency(signals_df)
                summary['expertise_map'] = metrics_calc.generate_expertise_map(signals_df)

            return summary

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error creating evaluation summary: {e}")
            return {
                'metrics': financial_metrics,
                'backtest_stats': backtest_results.get('performance', {}),
                'analysis': analysis_results,
                'timestamp': pd.Timestamp.now().isoformat(),
                'error': str(e)
            }

    def generate_notification_message(self, metrics: dict[str, Any]) -> str:
        """
        Generate notification message for UniversalNotifier.

        Args:
            metrics: Financial metrics dictionary

        Returns:
            Formatted notification message
        """
        try:
            message = f"""🏁 **Pipeline Execution Finished**

📈 Total Return: {metrics.get('total_return_pct', 0):+.2%}
🛡 Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
📉 Max Drawdown: {metrics.get('max_drawdown', 0):.2%}
🗓 CAGR: {metrics.get('cagr', 0):.2%}"""

            return message

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error generating notification message: {e}")
            return "Pipeline execution finished (error generating details)"

    def save_summary(self, summary: dict[str, Any],
                    results_dir: Path | None = None) -> str:
        """
        Save evaluation summary to JSON file.

        Args:
            summary: Summary dictionary
            results_dir: Optional directory for saving (uses default if not provided)

        Returns:
            Path to saved file
        """
        try:
            save_dir = results_dir or Path('data/results')
            save_dir.mkdir(parents=True, exist_ok=True)

            file_path = save_dir / f"summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=4, default=str)

            self.logger.info(f"Summary saved to {file_path}")
            return str(file_path)

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error saving summary: {e}")
            return ""


# Factory function
def get_report_generator(reports_dir: Path | None = None) -> ReportGenerator:
    """Factory function to get ReportGenerator instance."""
    return ReportGenerator(reports_dir)
