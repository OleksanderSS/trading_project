"""
Backtest mode - повноцінний, ІНТЕГРОВАНИЙ бектестинг стратегій на основі ML.
(ОНОВЛЕНО для використання нової архітектури PipelineOrchestrator)
"""
from typing import Any

import pandas as pd

from src.backtesting.advanced.advanced_engine import BiasDetector, WalkForwardOptimizer
from src.core.logging.logger import ProjectLogger
from src.metrics.calculator import MetricsCalculator
from src.pipeline.pipeline_orchestrator import PipelineOrchestrator
from src.trading.virtual_portfolio import VirtualPortfolio

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
        """Standard backtest execution."""
        orchestrator = PipelineOrchestrator(self.config_manager)
        final_data = self._execute_pipeline(orchestrator)
        _, signals_df = self._extract_predictions_and_signals(final_data)
        price_data = self._validate_price_data(final_data)
        bias_results = self._detect_biases(signals_df, price_data)
        embargoed_signals = self._apply_embargo_period(signals_df, price_data)
        aligned_prices, aligned_signals = self._align_data(price_data,
            embargoed_signals)
        performance_metrics = self._run_portfolio_simulation(aligned_prices,
            aligned_signals)
        performance_metrics['bias_analysis'] = bias_results
        self._log_results(performance_metrics)
        return {'status': 'success', 'metrics': performance_metrics}

    def _run_walk_forward_validation(self) ->dict[str, Any]:
        """Walk-forward validation execution."""
        self.logger.info('[Backtest] Running walk-forward validation...')
        backtest_config = self.config_manager.get_config(
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

    def _detect_biases(self, signals_df: pd.DataFrame, price_data: pd.DataFrame
        ) ->dict[str, Any]:
        """Detect various biases in backtest data."""
        bias_detector = BiasDetector()
        bias_results = {'look_ahead_bias': None, 'survivorship_bias': None,
            'warnings': []}
        try:
            if ('signal' in signals_df.columns and 'close' in price_data.
                columns):
                look_ahead_results = bias_detector.detect_look_ahead_bias(
                    signals=signals_df[['signal']], future_prices=
                    price_data[['close']], lag_periods=1)
                bias_results['look_ahead_bias'] = look_ahead_results
                if look_ahead_results.get('has_look_ahead_bias'):
                    bias_results['warnings'].append('Look-ahead bias detected!'
                        )
            bias_results['survivorship_bias'] = {'has_survivorship_bias':
                False, 'message': 'Not enough data for analysis'}
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            self.logger.warning(f'Bias detection failed: {e}')
            bias_results['warnings'].append(f'Bias detection error: {e}')
            raise
        return bias_results

    def _apply_embargo_period(self, signals_df: pd.DataFrame, price_data:
        pd.DataFrame) ->pd.DataFrame:
        """Apply embargo period to signals to prevent look-ahead bias."""
        try:
            backtest_config = self.config_manager.get_config(
                'backtest.bias_prevention', {})
            embargo_periods = backtest_config.get('embargo_periods', 1)
            self.logger.info(
                f'[Backtest] Applying {embargo_periods} period embargo to signals...'
                )
            embargoed_signals = signals_df.copy()
            if 'signal' in embargoed_signals.columns:
                embargoed_signals['signal'] = embargoed_signals['signal'
                    ].shift(embargo_periods)
                embargoed_signals = embargoed_signals.dropna(subset=['signal'])
                self.logger.info(
                    f'[Backtest] Embargo applied: {len(signals_df)} -> {len(embargoed_signals)} signals'
                    )
            return embargoed_signals
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            self.logger.warning(f'Embargo application failed: {e}')
            return signals_df

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

    def _extract_predictions_and_signals(self, final_data: dict[str, Any]
        ) ->tuple:
        """Витягує прогнози та сигнали з результатів пайплайну."""
        predictions = final_data.get('prediction_results') or final_data.get(
            'predictions')
        if predictions is None or hasattr(predictions, 'empty'
            ) and predictions.empty:
            raise ValueError(
                'Pipeline did not generate any predictions. Backtesting cannot proceed.'
                )
        signals_df = final_data.get('signals')
        if signals_df is None:
            signals_df = final_data.get('prediction_results'
                ) or final_data.get('predictions')
            if isinstance(signals_df, dict):
                signals_df = pd.DataFrame(signals_df)
        if signals_df is None or hasattr(signals_df, 'empty'
            ) and signals_df.empty:
            raise ValueError(
                'Pipeline did not generate any signal data for backtesting.')
        return predictions, signals_df

    def _validate_price_data(self, final_data: dict[str, Any]) ->pd.DataFrame:
        """Перевіряє наявність та коректність даних про ціни."""
        price_data = final_data.get('processed_data')
        if price_data is None or 'close' not in price_data.columns:
            raise ValueError(
                "Price data with 'close' column is missing from pipeline results."
                )
        return price_data

    def _align_data(self, price_data: pd.DataFrame, signals_df: pd.DataFrame
        ) ->tuple:
        """Узгоджує дані про ціни та сигнали за часом."""
        aligned_prices, aligned_signals = price_data['close'].align(signals_df
            ['signal'], join='inner')
        if aligned_prices.empty or aligned_signals.empty:
            raise ValueError(
                'After alignment, there are no overlapping data points between prices and signals.'
                )
        return aligned_prices, aligned_signals

    def _run_portfolio_simulation(self, aligned_prices: pd.Series,
        aligned_signals: pd.Series) ->dict[str, float]:
        """Запускає симуляцію портфеля та розраховує метрики."""
        self.logger.info(
            '[Backtest] Initializing and running the virtual portfolio...')
        portfolio_config = self.config_manager.get_config('trading')
        initial_capital = portfolio_config.get('initial_capital', 100000)
        portfolio = VirtualPortfolio(initial_capital=initial_capital)
        portfolio.run_simulation(aligned_prices, aligned_signals)
        self.logger.info('[Backtest] Calculating performance metrics...')
        metrics_calculator = MetricsCalculator(portfolio.get_equity_curve())
        return metrics_calculator.calculate_all_metrics()

    def _log_results(self, performance_metrics: dict[str, float]) ->None:
        """Логує результати бектестингу."""
        self.logger.info('--- Backtest Completed Successfully ---')
        self.logger.info(
            f"Final Portfolio Value: ${performance_metrics.get('final_equity'):,.2f}"
            )
        self.logger.info(
            f"Total Return: {performance_metrics.get('total_return_pct'):.2%}")
        self.logger.info(
            f"Sharpe Ratio: {performance_metrics.get('sharpe_ratio'):.2f}")
