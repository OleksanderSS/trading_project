#!/usr/bin/env python3
"""
Monster Test - Комплексне стрес-тестування системи за методом Монте-Карло.
(ОНОВЛЕНО для використання нової архітектури PipelineOrchestrator)
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List

from .base import BaseMode
from src.pipeline.pipeline_orchestrator import PipelineOrchestrator
from src.config.unified_config_manager import UnifiedConfigManager
from src.simulation.simulation_engine import SimulationEngine
from src.trading.virtual_portfolio import VirtualPortfolio
from src.metrics.calculator import MetricsCalculator

logger = logging.getLogger(__name__)

class MonsterTestMode(BaseMode):
    """
    Режим, що тестує стійкість ML-стратегії до ринкових шоків
    за допомогою симуляцій Монте-Карло, використовуючи нову архітектуру.
    """

    def run(self) -> Dict[str, Any]:
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

            model = training_results['best_model']
            feature_names = training_results['feature_names']
            # Отримуємо дані, на яких проводився бектест, для симуляцій
            backtest_data = training_results['backtest_data'] 

            # 2. ІНІЦІАЛІЗАЦІЯ СИМУЛЯТОРА СЦЕНАРІЇВ
            simulation_config = self.config_manager.get_config('simulation')
            n_simulations = simulation_config.get('n_simulations', 1000)
            
            simulator = SimulationEngine(simulation_config)

            # 3. ГЕНЕРАЦІЯ СИНТЕТИЧНИХ СЦЕНАРІЇВ
            self.logger.info(f"[MonsterTest] Generating {n_simulations} synthetic price scenarios...")
            # Симулятор генерує сценарії на основі історичних даних з backtest_data
            synthetic_scenarios = simulator.generate_monte_carlo_scenarios(backtest_data['close'], n_simulations)

            # 4. ЗАПУСК БЕКТЕСТУ НА КОЖНОМУ СЦЕНАРІЇ
            self.logger.info("[MonsterTest] Running backtests on all synthetic scenarios...")
            all_metrics = []
            initial_capital = self.config_manager.get_config('trading').get('initial_capital', 100000)

            for i in range(n_simulations):
                scenario_prices = synthetic_scenarios[f'sim_{i}']
                
                # Створюємо копію даних для цього сценарію, щоб не змінювати оригінал
                scenario_df = backtest_data.copy()
                scenario_df['close'] = scenario_prices
                
                # Генеруємо прогнози моделі на основі ОРИГІНАЛЬНИХ фіч, але нової ціни
                # Це симулює, як модель реагує на шокові рухи цін
                predictions = model.predict(scenario_df[feature_names])
                signals = np.sign(predictions - np.mean(predictions)) # Проста стратегія

                # Запуск симуляції портфеля для цього сценарію
                portfolio = VirtualPortfolio(initial_capital)
                portfolio.run_simulation(scenario_prices, pd.Series(signals, index=scenario_prices.index))
                
                # Розрахунок метрик для цього сценарію
                metrics_calculator = MetricsCalculator(portfolio.get_equity_curve())
                performance_metrics = metrics_calculator.calculate_all_metrics()
                all_metrics.append(performance_metrics)
            
            # 5. АНАЛІЗ РЕЗУЛЬТАТІВ СТРЕС-ТЕСТУ
            self.logger.info("[MonsterTest] Analyzing simulation results...")
            final_report = self._analyze_monte_carlo_results(all_metrics, n_simulations)

            self.logger.info("--- MONSTER TEST Completed Successfully ---")
            return {'status': 'success', 'report': final_report}

        except Exception as e:
            self.logger.exception(f"[MonsterTest] A critical error occurred: {e}")
            return {'status': 'failed', 'error': str(e)}

    def _analyze_monte_carlo_results(self, all_metrics: List[Dict], n_simulations: int) -> Dict[str, Any]:
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
