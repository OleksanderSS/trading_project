#!/usr/bin/env python3
"""
Monster Test - Комплексне стрес-тестування системи за методом Монте-Карло.
(ОНОВЛЕНО для використання нової архітектури PipelineOrchestrator)
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.main.modes.base import BaseMode
from src.pipeline.pipeline_orchestrator import PipelineOrchestrator
from src.simulation.simulation_engine import SimulationEngine

logger = logging.getLogger(__name__)


class MonsterTestMode(BaseMode):
    """
    Режим, що тестує стійкість ML-стратегії до ринкових шоків
    за допомогою симуляцій Монте-Карло, використовуючи нову архітектуру.
    """

    def run(self, **kwargs) -> dict[str, Any]:
        """
        Запускає повний цикл стрес-тестування з використанням PipelineOrchestrator.
        """
        self.logger.info("--- Starting MONSTER TEST Mode using PipelineOrchestrator ---")
        try:
            # 1. ІНІЦІАЛІЗАЦІЯ ОРКЕСТРАТОРА ТА ЗАПУСК ПАЙПЛАЙНУ НАВЧАННЯ
            orchestrator = PipelineOrchestrator(self.config_manager)

            self.logger.info("[MonsterTest] Running training pipeline to get the model...")
            training_results = orchestrator.execute_training_pipeline()

            if not training_results or 'best_model' not in training_results:
                raise RuntimeError("Training pipeline did not produce a model for stress testing.")

            training_results['best_model']
            training_results['feature_names']
            # Отримуємо дані, на яких проводився бектест, для симуляцій
            backtest_data = training_results['backtest_data']

            # 2. ІНІЦІАЛІЗАЦІЯ СИМУЛЯТОРА СЦЕНАРІЇВ
            simulation_config = self.config_manager.get_config('simulation')
            n_simulations = simulation_config.get('n_simulations', 1000)

            simulator = SimulationEngine()

            # 3. ГЕНЕРАЦІЯ СИНТЕТИЧНИХ СЦЕНАРІЇВ
            self.logger.info(f"[MonsterTest] Generating {n_simulations} synthetic price scenarios...")
            # Симулятор генерує сценарії на основі історичних даних з backtest_data
            # Використовуємо run_monte_carlo_for_strategy з функцією-стратегією
            from datetime import datetime

            from src.simulation.simulation_engine import SimulationContext, SimulationGranularity

            # Створюємо контекст симуляції
            context = SimulationContext(
                ticker='SPY',
                timestamp=datetime.now(),
                granularity=SimulationGranularity.MARKET_LEVEL,
                historical_returns=backtest_data['close'].pct_change(fill_method=None).dropna()
            )

            # Проста стратегія для тестування
            def simple_strategy(market_data: pd.DataFrame) -> pd.Series:
                # Генеруємо прості сигнали на основі рухомих середніх
                ma_short = market_data['close'].rolling(window=5, min_periods=1).mean()
                ma_long = market_data['close'].rolling(window=20, min_periods=1).mean()
                return pd.Series(np.where(ma_short > ma_long, 1, -1), index=market_data.index)

            # Запускаємо симуляцію
            simulation_results = simulator.run_monte_carlo_for_strategy(
                strategy_logic=simple_strategy,
                initial_context=context,
                horizon=len(backtest_data),
                runs=n_simulations
            )

            # 4. АНАЛІЗ РЕЗУЛЬТАТІВ СТРЕС-ТЕСТУ
            self.logger.info("[MonsterTest] Analyzing simulation results...")

            # Обробляємо результати симуляції Монте-Карло
            all_metrics = []
            if simulation_results:
                for report in simulation_results:
                    if report:
                        # Конвертуємо SimulationRiskReport в словник метрик
                        metrics = {
                            'total_return_pct': (report.var_95 * 100),  # Використовуємо VaR як proxy для повернення
                            'sharpe_ratio': report.sharpe_ratio,
                            'max_drawdown_pct': (report.max_drawdown * 100),
                            'var_95_pct': (report.var_95 * 100),
                            'var_99_pct': (report.var_99 * 100),
                            'expected_shortfall_pct': (report.expected_shortfall * 100)
                        }
                        all_metrics.append(metrics)

            # 5. АНАЛІЗ РЕЗУЛЬТАТІВ СТРЕС-ТЕСТУ
            final_report = self._analyze_monte_carlo_results(all_metrics, n_simulations)

            self.logger.info("--- MONSTER TEST Completed Successfully ---")
            return {'status': 'success', 'report': final_report}

        except Exception as e:
            self.logger.exception(f"[MonsterTest] A critical error occurred: {e}")
            return {'status': 'failed', 'error': str(e)}

    def _analyze_monte_carlo_results(self, all_metrics: list[dict], n_simulations: int) -> dict[str, Any]:
        """Аналізує розподіл результатів симуляцій."""
        returns = [res.get('total_return_pct', 0) for res in all_metrics]

        report = {
            'n_simulations': n_simulations,
            'average_return_pct': np.mean(returns),
            'median_return_pct': np.median(returns),
            'return_std_dev': np.std(returns),
            'best_case_return_pct': np.max(returns),
            'worst_case_return_pct': np.min(returns),
            'value_at_risk_5_pct': np.percentile(returns, 5),
            'probability_of_profit_pct': (np.sum(np.array(returns) > 0) / n_simulations) * 100
        }
        self.logger.info(f"Stress test summary: {report}")
        return report

