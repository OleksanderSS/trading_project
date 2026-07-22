#!/usr/bin/env python3
"""
Monster Test - Комплексне стрес-тестування системи за методом Монте-Карло.

Логіка:
  1. ПРОПУСКАЄ навчання — завантажує вже натреновані CHAMP-моделі з диску.
  2. Завантажує ринкові дані з бази (DuckDB) для побудови контексту симуляцій.
  3. Запускає Монте-Карло симуляції (Black Swan сценарій).
  4. Виводить фінальний звіт по метриках ризику.

Щоб примусово перетренувати — запустіть accumulation-режим окремо.
"""

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.main.modes.base import BaseMode
from src.simulation.simulation_engine import SimulationEngine

logger = logging.getLogger(__name__)

_MODELS_DIR = Path("data/trained_models")
_DEFAULT_TICKER = "SPY"


class MonsterTestMode(BaseMode):
    """
    Режим стрес-тестування ML-стратегії.
    Завантажує вже натреновані моделі з диску і запускає симуляції Монте-Карло.
    """

    def run(self, ticker: str = _DEFAULT_TICKER, **kwargs) -> dict[str, Any]:
        self.logger.info("--- Starting MONSTER TEST Mode (using pre-trained models) ---")
        try:
            # ═══════════════════════════════════════════════════
            # КРОК 1: Завантажуємо CHAMP-модель з диску
            # ═══════════════════════════════════════════════════
            self.logger.info(f"[MonsterTest] STEP 1/3: Loading pre-trained champion model for {ticker}...")
            model = self._load_champion_model(ticker)
            if model is None:
                raise RuntimeError(
                    f"No champion model found for '{ticker}' in {_MODELS_DIR}. "
                    "Run accumulation mode first to train models."
                )
            self.logger.info(f"[MonsterTest] ✅ Model loaded: {type(model).__name__}"
                             f" from data/trained_models/CHAMP_{ticker}_target_up_1d.joblib")

            # ═══════════════════════════════════════════════════
            # КРОК 3: Запускаємо Монте-Карло симуляції по всіх таймфреймах
            # ═══════════════════════════════════════════════════
            simulation_config = self.config_manager.get_config('simulation') or {}
            n_simulations = simulation_config.get('n_simulations', 50)
            horizon = simulation_config.get('horizon', 100)

            # Всі таймфрейми що є в базі
            TIMEFRAMES = ['1d', '1h', '15m']

            self.logger.info(
                f"[MonsterTest] STEP 3/3: Running Monte Carlo across {TIMEFRAMES} "
                f"({n_simulations} runs × {horizon} bars each, scenario=black_swan)"
            )

            from datetime import datetime
            from src.simulation.simulation_engine import SimulationContext, SimulationGranularity

            simulator = SimulationEngine()
            all_metrics: list[dict] = []

            for tf in TIMEFRAMES:
                self.logger.info(f"[MonsterTest] Simulating timeframe: {tf}...")
                tf_returns = self._load_historical_returns(ticker, interval=tf)
                if tf_returns is None or tf_returns.empty:
                    self.logger.warning(f"[MonsterTest] No data for {ticker}/{tf}, skipping.")
                    continue

                context = SimulationContext(
                    ticker=ticker,
                    timestamp=datetime.now(),
                    granularity=SimulationGranularity.MARKET_LEVEL,
                    historical_returns=tf_returns,
                )

                def ml_strategy(market_data: pd.DataFrame) -> pd.Series:
                    try:
                        if hasattr(model, 'model') and hasattr(model.model, 'feature_names_'):
                            expected_features = model.model.feature_names_
                            features = market_data.reindex(columns=expected_features, fill_value=0)
                        else:
                            features = market_data.select_dtypes(include=[np.number])
                            
                        if features.empty or not hasattr(model, 'predict'):
                            return pd.Series(0, index=market_data.index)
                        predictions = model.predict(features)
                        return pd.Series(predictions, index=market_data.index)
                    except Exception as e:
                        self.logger.warning(f"Strategy failed: {e}. Returning neutral signal.")
                        return pd.Series(0, index=market_data.index)

                tf_results = simulator.run_monte_carlo_for_strategy(
                    strategy_logic=ml_strategy,
                    initial_context=context,
                    horizon=horizon,
                    runs=n_simulations,
                    scenario_name='black_swan',
                )

                for report in (tf_results or []):
                    if report:
                        all_metrics.append({
                            'timeframe': tf,
                            'total_return_pct': report.var_95 * 100,
                            'sharpe_ratio': report.sharpe_ratio,
                            'max_drawdown_pct': report.max_drawdown * 100,
                            'var_95_pct': report.var_95 * 100,
                            'var_99_pct': report.var_99 * 100,
                            'expected_shortfall_pct': report.expected_shortfall * 100,
                        })
                self.logger.info(f"[MonsterTest] ✅ {tf}: {len(tf_results or [])} paths simulated.")

            final_report = self._analyze_monte_carlo_results(all_metrics, n_simulations)
            self.logger.info("--- MONSTER TEST Completed Successfully ---")
            return {'status': 'success', 'report': final_report}

        except (ValueError, TypeError, AttributeError, KeyError, RuntimeError) as e:
            self.logger.exception(f"[MonsterTest] Critical error: {e}")
            return {'status': 'failed', 'error': str(e)}

    # ─────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────

    def _load_champion_model(self, ticker: str) -> Any | None:
        """
        Шукає найкращу CHAMP-модель для тікера.
        Пріоритет: target_up_1d → target_return_1d → будь-яка.
        """
        priorities = [
            f"CHAMP_{ticker}_target_up_1d.joblib",
            f"CHAMP_{ticker}_target_return_1d.joblib",
        ]
        for name in priorities:
            path = _MODELS_DIR / name
            if path.exists():
                self.logger.info(f"Loading champion model: {path}")
                return joblib.load(path)

        # fallback: будь-яка CHAMP-модель для цього тікера
        candidates = list(_MODELS_DIR.glob(f"CHAMP_{ticker}_*.joblib"))
        if candidates:
            path = candidates[0]
            self.logger.info(f"Loading fallback champion model: {path}")
            return joblib.load(path)

        self.logger.warning(f"No CHAMP model found for {ticker} in {_MODELS_DIR}")
        return None

    def _load_historical_returns(self, ticker: str, interval: str = '1d') -> pd.Series | None:
        """
        Завантажує ціни закриття з DuckDB і повертає доходності для заданого таймфрейму.
        """
        try:
            from src.data.management.data_manager import DataManager
            db = DataManager(self.config_manager)
            con = db.get_connection("data/trading_data.duckdb")
            df = con.execute(
                "SELECT datetime, close FROM market_data_raw "
                f"WHERE ticker = '{ticker}' AND interval = '{interval}' "
                "ORDER BY datetime ASC"
            ).df()
            if df is None or df.empty or 'close' not in df.columns:
                self.logger.warning(f"No price data for {ticker} in DB (interval={interval}).")
                return None
            returns = df['close'].pct_change(fill_method=None).dropna()
            self.logger.info(f"Loaded {len(returns)} returns for {ticker}/{interval}")
            return returns
        except Exception as e:
            self.logger.warning(f"Could not load historical data ({ticker}/{interval}) from DB: {e}")
            return None

    def _analyze_monte_carlo_results(self, all_metrics: list[dict], n_simulations: int) -> dict[str, Any]:
        """Аналізує розподіл результатів симуляцій з розбивкою по таймфреймах."""
        returns = [r.get('total_return_pct', 0) for r in all_metrics]
        n = len(returns)

        # Розбивка по таймфреймах
        tf_breakdown: dict[str, dict] = {}
        for tf in ['1d', '1h', '15m']:
            tf_returns = [r.get('total_return_pct', 0) for r in all_metrics if r.get('timeframe') == tf]
            if tf_returns:
                tf_breakdown[tf] = {
                    'n': len(tf_returns),
                    'avg_return_pct': float(np.mean(tf_returns)),
                    'var_95_pct': float(np.percentile(tf_returns, 5)),
                    'max_drawdown_pct': float(np.mean([r.get('max_drawdown_pct', 0) for r in all_metrics if r.get('timeframe') == tf])),
                    'avg_sharpe': float(np.mean([r.get('sharpe_ratio', 0) for r in all_metrics if r.get('timeframe') == tf])),
                }

        report = {
            'n_simulations_requested': n_simulations,
            'n_simulations_completed': n,
            'average_return_pct': float(np.mean(returns)) if n else 0.0,
            'median_return_pct': float(np.median(returns)) if n else 0.0,
            'return_std_dev': float(np.std(returns)) if n else 0.0,
            'best_case_return_pct': float(np.max(returns)) if n else 0.0,
            'worst_case_return_pct': float(np.min(returns)) if n else 0.0,
            'value_at_risk_5_pct': float(np.percentile(returns, 5)) if n else 0.0,
            'probability_of_profit_pct': float(np.sum(np.array(returns) > 0) / max(n, 1) * 100),
            'by_timeframe': tf_breakdown,
        }
        self.logger.info(f"[MonsterTest] ✅ Stress test summary: avg_return={report['average_return_pct']:.2f}%, "
                         f"VaR95={report['value_at_risk_5_pct']:.2f}%, "
                         f"prob_profit={report['probability_of_profit_pct']:.1f}%")
        return report
