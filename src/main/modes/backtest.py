#!/usr/bin/env python3
"""
Backtest mode - повноцінний, ІНТЕГРОВАНИЙ бектестинг стратегій на основі ML.
(ОНОВЛЕНО для використання нової архітектури PipelineOrchestrator)
"""

import pandas as pd
from typing import Dict, Any

from .base import BaseMode
from src.pipeline.pipeline_orchestrator import PipelineOrchestrator
from src.config.unified_config_manager import UnifiedConfigManager
from src.trading.virtual_portfolio import VirtualPortfolio
from src.metrics.calculator import MetricsCalculator
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

class BacktestMode(BaseMode):
    """Режим бектестингу, що використовує нову архітектуру пайплайну."""

    def run(self) -> Dict[str, Any]:
        """
        Запускає повний цикл бектестингу: від збору даних до аналізу прибутковості,
        використовуючи PipelineOrchestrator.
        """
        self.logger.info("--- Starting Integrated Backtesting Mode using PipelineOrchestrator ---")
        try:
            orchestrator = PipelineOrchestrator(self.config_manager)
            final_data = self._execute_pipeline(orchestrator)
            _, signals_df = self._extract_predictions_and_signals(final_data)
            price_data = self._validate_price_data(final_data)
            aligned_prices, aligned_signals = self._align_data(price_data, signals_df)
            performance_metrics = self._run_portfolio_simulation(aligned_prices, aligned_signals)
            self._log_results(performance_metrics)
            
            return {'status': 'success', 'metrics': performance_metrics}

        except Exception as e:
            self.logger.exception(f"[Backtest] A critical error occurred: {e}")
            return {'status': 'failed', 'error': str(e)}

    def _execute_pipeline(self, orchestrator: PipelineOrchestrator) -> Dict[str, Any]:
        """Виконує повний пайплайн для генерації прогнозів."""
        self.logger.info("[Backtest] Running the full data and prediction pipeline...")
        return orchestrator.execute_full_pipeline()

    def _extract_predictions_and_signals(self, final_data: Dict[str, Any]) -> tuple:
        """Витягує прогнози та сигнали з результатів пайплайну."""
        predictions = final_data.get('prediction_results') or final_data.get('predictions')
        if predictions is None or (hasattr(predictions, 'empty') and predictions.empty):
            raise ValueError("Pipeline did not generate any predictions. Backtesting cannot proceed.")

        signals_df = final_data.get('signals')
        if signals_df is None:
            signals_df = final_data.get('prediction_results') or final_data.get('predictions')
            if isinstance(signals_df, dict):
                signals_df = pd.DataFrame(signals_df)

        if signals_df is None or (hasattr(signals_df, 'empty') and signals_df.empty):
            raise ValueError("Pipeline did not generate any signal data for backtesting.")
            
        return predictions, signals_df

    def _validate_price_data(self, final_data: Dict[str, Any]) -> pd.DataFrame:
        """Перевіряє наявність та коректність даних про ціни."""
        price_data = final_data.get('processed_data')
        if price_data is None or 'close' not in price_data.columns:
            raise ValueError("Price data with 'close' column is missing from pipeline results.")
        return price_data

    def _align_data(self, price_data: pd.DataFrame, signals_df: pd.DataFrame) -> tuple:
        """Узгоджує дані про ціни та сигнали за часом."""
        aligned_prices, aligned_signals = price_data['close'].align(signals_df['signal'], join='inner')
        
        if aligned_prices.empty or aligned_signals.empty:
            raise ValueError("After alignment, there are no overlapping data points between prices and signals.")
            
        return aligned_prices, aligned_signals

    def _run_portfolio_simulation(self, aligned_prices: pd.Series, aligned_signals: pd.Series) -> Dict[str, float]:
        """Запускає симуляцію портфеля та розраховує метрики."""
        self.logger.info("[Backtest] Initializing and running the virtual portfolio...")
        portfolio_config = self.config_manager.get_config('trading')
        initial_capital = portfolio_config.get('initial_capital', 100000)
        
        portfolio = VirtualPortfolio(initial_capital=initial_capital)
        portfolio.run_simulation(aligned_prices, aligned_signals)
        
        self.logger.info("[Backtest] Calculating performance metrics...")
        metrics_calculator = MetricsCalculator(portfolio.get_equity_curve())
        return metrics_calculator.calculate_all_metrics()

    def _log_results(self, performance_metrics: Dict[str, float]) -> None:
        """Логує результати бектестингу."""
        self.logger.info("--- Backtest Completed Successfully ---")
        self.logger.info(f"Final Portfolio Value: ${performance_metrics.get('final_equity'):,.2f}")
        self.logger.info(f"Total Return: {performance_metrics.get('total_return_pct'):.2%}")
        self.logger.info(f"Sharpe Ratio: {performance_metrics.get('sharpe_ratio'):.2f}")
