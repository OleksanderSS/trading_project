#!/usr/bin/env python3
"""
Backtest mode - повноцінний, ІНТЕГРОВАНИЙ бектестинг стратегій на основі ML.
(ОНОВЛЕНО для використання нової архітектури PipelineOrchestrator)
"""

import logging
import pandas as pd
from typing import Dict, Any

from .base import BaseMode
from src.pipeline.pipeline_orchestrator import PipelineOrchestrator
from src.config.unified_config_manager import UnifiedConfigManager
from src.trading.virtual_portfolio import VirtualPortfolio
from src.metrics.calculator import MetricsCalculator

logger = logging.getLogger(__name__)

class BacktestMode(BaseMode):
    """Режим бектестингу, що використовує нову архітектуру пайплайну."""

    def run(self) -> Dict[str, Any]:
        """
        Запускає повний цикл бектестингу: від збору даних до аналізу прибутковості,
        використовуючи PipelineOrchestrator.
        """
        self.logger.info("--- Starting Integrated Backtesting Mode using PipelineOrchestrator ---")
        try:
            # 1. ІНІЦІАЛІЗАЦІЯ ОРКЕСТРАТОРА ПАЙПЛАЙНУ
            # UnifiedConfigManager вже доступний як self.config_manager з BaseMode
            orchestrator = PipelineOrchestrator(self.config_manager)

            # 2. ЗАПУСК ПОВНОГО ПАЙПЛАЙНУ ДЛЯ ГЕНЕРАЦІЇ ПРОГНОЗІВ
            self.logger.info("[Backtest] Running the full data and prediction pipeline...")
            final_data = orchestrator.execute_full_pipeline()

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

            # 3. ПІДГОТОВКА ДО БЕКТЕСТИНГУ
            # Отримання потрібних даних з результатів пайплайну
            price_data = final_data.get('processed_data') # Або інший відповідний ключ
            if price_data is None or 'close' not in price_data.columns:
                raise ValueError("Price data with 'close' column is missing from pipeline results.")

            # Узгодження сигналів та цін
            aligned_prices, aligned_signals = price_data['close'].align(signals_df['signal'], join='inner')

            if aligned_prices.empty or aligned_signals.empty:
                raise ValueError("After alignment, there are no overlapping data points between prices and signals.")

            # 4. ІНІЦІАЛІЗАЦІЯ ТА ЗАПУСК ВІРТУАЛЬНОГО ПОРТФЕЛЯ
            self.logger.info("[Backtest] Initializing and running the virtual portfolio...")
            portfolio_config = self.config_manager.get_config('trading')
            initial_capital = portfolio_config.get('initial_capital', 100000)
            
            portfolio = VirtualPortfolio(initial_capital=initial_capital)
            
            # Виконання симуляції торгівлі
            portfolio.run_simulation(aligned_prices, aligned_signals)
            
            # 5. РОЗРАХУНОК ТА АНАЛІЗ РЕЗУЛЬТАТІВ
            self.logger.info("[Backtest] Calculating performance metrics...")
            metrics_calculator = MetricsCalculator(portfolio.get_equity_curve())
            performance_metrics = metrics_calculator.calculate_all_metrics()
            
            self.logger.info("--- Backtest Completed Successfully ---")
            self.logger.info(f"Final Portfolio Value: ${performance_metrics.get('final_equity'):,.2f}")
            self.logger.info(f"Total Return: {performance_metrics.get('total_return_pct'):.2%}")
            self.logger.info(f"Sharpe Ratio: {performance_metrics.get('sharpe_ratio'):.2f}")

            # Повертаємо фінальні метрики
            return {'status': 'success', 'metrics': performance_metrics}

        except Exception as e:
            self.logger.exception(f"[Backtest] A critical error occurred: {e}")
            return {'status': 'failed', 'error': str(e)}
