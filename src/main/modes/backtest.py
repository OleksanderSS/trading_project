"""
Backtest mode - повноцінний, ІНТЕГРОВАНИЙ бектестинг стратегій на основі ML.
(ОНОВЛЕНО для використання нової архітектури PipelineOrchestrator)
"""
from typing import Any

import pandas as pd

from src.backtesting.advanced.advanced_engine import WalkForwardOptimizer
from src.core.logging.logger import ProjectLogger
from src.pipeline.pipeline_orchestrator import PipelineOrchestrator

from .base import BaseMode

logger = ProjectLogger.get_logger(__name__)


class BacktestMode(BaseMode):
    """Режим бектестингу, що використовує нову архітектуру пайплайну."""

    def run(self, **kwargs) ->dict[str, Any]:
        """
        Запускає повний цикл бектестингу: від збору даних до аналізу прибутковості,
        використовуючи PipelineOrchestrator з walk-forward validation.

        Args:
            **kwargs: Additional parameters (tickers, timeframes, etc.) for compatibility
                      with SystemOrchestrator. Currently unused but accepted for future extension.
        """
        self.logger.info(
            '--- Starting Integrated Backtesting Mode using PipelineOrchestrator ---'
            )
        try:
            backtest_config = self.config_manager.get_config('backtest', {})
            use_walk_forward = backtest_config.get('walk_forward_validation',
                False)
            if use_walk_forward:
                return self._run_walk_forward_validation()
            else:
                return self._run_standard_backtest()
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.exception(f'[Backtest] A critical error occurred: {e}')
            return {'status': 'failed', 'error': str(e)}

    def _run_standard_backtest(self) ->dict[str, Any]:
        """Standard backtest execution.

        Stage 7 (EvaluationStage) already runs as part of
        execute_full_pipeline() and performs the real backtest itself
        (signal generation from predictions, per-ticker pivot, and
        AdvancedBacktestEngine simulation with its own built-in
        look-ahead-bias detection) - this mode just reads that result
        instead of re-simulating it separately.
        """
        orchestrator = PipelineOrchestrator(self.config_manager)
        final_data = self._execute_pipeline(orchestrator)
        evaluation_summary = self._validate_evaluation_summary(final_data)
        performance_metrics = evaluation_summary.get('metrics', {})
        performance_metrics['backtest_stats'] = evaluation_summary.get(
            'backtest_stats', {})
        self._log_results(performance_metrics)
        return {'status': 'success', 'metrics': performance_metrics,
            'evaluation_summary': evaluation_summary}

    def _run_walk_forward_validation(self) ->dict[str, Any]:
        """Walk-forward validation execution."""
        self.logger.info('[Backtest] Running walk-forward validation...')
        backtest_config = self.config_manager.get(
            'backtest.walk_forward', {})
        in_sample_months = backtest_config.get('in_sample_months', 12)
        out_sample_months = backtest_config.get('out_sample_months', 3)
        walk_forward = WalkForwardOptimizer(self.config_manager)
        orchestrator = PipelineOrchestrator(self.config_manager)
        historical_data = self._get_historical_data(orchestrator)

        def optimization_function(train_data):
            """Optimize model parameters on training data."""
            return {'learning_rate': 0.01, 'n_estimators': 100}
        wf_results = walk_forward.walk_forward_optimization(data=
            historical_data, optimization_func=optimization_function,
            in_sample_months=in_sample_months, out_sample_months=
            out_sample_months)
        performance_metrics = self._aggregate_walk_forward_results(wf_results)
        self._log_walk_forward_results(performance_metrics, wf_results)
        return {'status': 'success', 'metrics': performance_metrics,
            'walk_forward_results': wf_results}

    def _get_historical_data(self, orchestrator: PipelineOrchestrator
        ) ->pd.DataFrame:
        """Get historical data for walk-forward validation."""
        final_data = orchestrator.execute_full_pipeline()
        price_data = final_data.get('processed_data')
        return price_data if price_data is not None else pd.DataFrame()

    def _aggregate_walk_forward_results(self, wf_results: dict[str, Any]
        ) ->dict[str, Any]:
        """Aggregate results from walk-forward validation windows."""
        if not wf_results or 'windows' not in wf_results:
            return {'error': 'No walk-forward results to aggregate'}
        windows = wf_results['windows']
        total_return = 0
        sharpe_ratios = []
        max_drawdowns = []
        for window in windows:
            metrics = window.get('performance_metrics', {})
            total_return += metrics.get('total_return', 0)
            if 'sharpe_ratio' in metrics:
                sharpe_ratios.append(metrics['sharpe_ratio'])
            if 'max_drawdown' in metrics:
                max_drawdowns.append(metrics['max_drawdown'])
        aggregated_metrics = {'walk_forward_total_return': total_return /
            len(windows) if windows else 0, 'walk_forward_avg_sharpe': sum(
            sharpe_ratios) / len(sharpe_ratios) if sharpe_ratios else 0,
            'walk_forward_avg_drawdown': sum(max_drawdowns) / len(
            max_drawdowns) if max_drawdowns else 0, 'walk_forward_windows':
            len(windows), 'validation_type': 'walk_forward'}
        return aggregated_metrics

    def _log_walk_forward_results(self, performance_metrics: dict[str, Any],
        wf_results: dict[str, Any]):
        """Log walk-forward validation results."""
        self.logger.info('--- Walk-Forward Validation Completed ---')
        self.logger.info(
            f"Windows analyzed: {wf_results.get('windows', []).__len__()}")
        self.logger.info(
            f"Average return: {performance_metrics.get('walk_forward_total_return', 0):.2%}"
            )
        self.logger.info(
            f"Average Sharpe ratio: {performance_metrics.get('walk_forward_avg_sharpe', 0):.2f}"
            )
        self.logger.info(
            f"Average max drawdown: {performance_metrics.get('walk_forward_avg_drawdown', 0):.2%}"
            )

    def _execute_pipeline(self, orchestrator: PipelineOrchestrator) ->dict[
        str, Any]:
        """Виконує повний пайплайн для генерації прогнозів."""
        self.logger.info(
            '[Backtest] Running the full data and prediction pipeline...')
        return orchestrator.execute_full_pipeline()

    def _validate_evaluation_summary(self, final_data: dict[str, Any]
        ) ->dict[str, Any]:
        """Перевіряє, що Stage 7 (EvaluationStage) реально відпрацював."""
        evaluation_summary = final_data.get('evaluation_summary')
        if not evaluation_summary:
            raise ValueError(
                'Pipeline did not produce an evaluation summary (Stage 7). '
                'Backtesting cannot proceed.'
                )
        return evaluation_summary

    def _log_results(self, performance_metrics: dict[str, float]) ->None:
        """Логує результати бектестингу."""
        self.logger.info('--- Backtest Completed Successfully ---')
        if 'final_equity' not in performance_metrics:
            self.logger.warning(
                '[Backtest] Stage 7 returned a basic (non-simulated) '
                'evaluation - no portfolio metrics available.'
                )
            return
        self.logger.info(
            f"Final Portfolio Value: ${performance_metrics.get('final_equity', 0.0):,.2f}"
            )
        self.logger.info(
            f"Total Return: {performance_metrics.get('total_return_pct', 0.0):.2%}"
            )
        self.logger.info(
            f"Sharpe Ratio: {performance_metrics.get('sharpe_ratio', 0.0):.2f}"
            )
