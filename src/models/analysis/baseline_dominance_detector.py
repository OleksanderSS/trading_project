#!/usr/bin/env python3
"""
Baseline Dominance Detector - Detects When Simple Baselines Outperform Complex Models
Analyzes whether complex models provide real value over simple baselines.
"""

from datetime import datetime
from typing import Any

import pandas as pd

from src.core.exceptions import DataProcessingError
from src.core.logging.logger import ProjectLogger
from src.models.analysis.baseline.comparison import BaselineComparisonEngine
from src.models.analysis.baseline.recommendations import BaselineRecommendationEngine

logger = ProjectLogger.get_logger("BaselineDominanceDetector")

class BaselineDominanceDetector:
    """
    Виявляє та аналізує домінування базових моделей над складними моделями.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Ініціалізує Baseline Dominance Detector.
        """
        self.logger = logger
        self.config = config or {}

        self.BASELINE_MODELS = {
            'linear_regression': {'description': 'Simple linear regression', 'complexity_score': 1},
            'moving_average': {'description': 'Moving average strategy', 'complexity_score': 0.5},
            'buy_and_hold': {'description': 'Buy and hold strategy', 'complexity_score': 0.1},
            'random_forest_simple': {'description': 'Simple Random Forest', 'complexity_score': 3},
            'mean_reversion': {'description': 'Mean reversion strategy', 'complexity_score': 2}
        }

        self.min_samples = self.config.get('min_samples', 100)
        self.dominance_threshold = self.config.get('dominance_threshold', 0.05)

        self._init_components()
        self.logger.info("✅ BaselineDominanceDetector initialized")

    def _init_components(self):
        """Ініціалізує модульні компоненти."""
        from .baselines import (
            BuyAndHoldBaseline,
            LinearRegressionBaseline,
            MeanReversionBaseline,
            MovingAverageBaseline,
            SimpleRandomForestBaseline,
        )
        self.baseline_implementations = {
            'buy_and_hold': BuyAndHoldBaseline(self.BASELINE_MODELS['buy_and_hold']['complexity_score']),
            'moving_average': MovingAverageBaseline(self.BASELINE_MODELS['moving_average']['complexity_score']),
            'linear_regression': LinearRegressionBaseline(self.BASELINE_MODELS['linear_regression']['complexity_score'], self.min_samples),
            'random_forest_simple': SimpleRandomForestBaseline(self.BASELINE_MODELS['random_forest_simple']['complexity_score'], self.min_samples),
            'mean_reversion': MeanReversionBaseline(self.BASELINE_MODELS['mean_reversion']['complexity_score'])
        }
        self.comparison_engine = BaselineComparisonEngine(self.dominance_threshold)
        self.recommendation_engine = BaselineRecommendationEngine(
            self.config.get('complexity_penalty', 0.02),
            self.BASELINE_MODELS
        )

    async def analyze_baseline_dominance(self,
                                      complex_model_results: dict[str, Any],
                                      market_data: pd.DataFrame,
                                      features_df: pd.DataFrame | None = None,
                                      target_series: pd.Series | None = None) -> dict[str, Any]:
        """Аналізує домінування базових моделей."""
        self.logger.info("🔍 Analyzing baseline dominance...")

        results = {
            'timestamp': datetime.now(),
            'complex_model_info': complex_model_results,
            'baseline_results': {},
            'dominance_analysis': {},
            'cost_benefit_analysis': {},
            'recommendations': []
        }

        try:
            baseline_results = await self._train_baseline_models(market_data, features_df, target_series)
            results['baseline_results'] = baseline_results

            dominance_analysis = self.comparison_engine.compare(
                complex_model_results.get('metrics', {}),
                baseline_results
            )
            results['dominance_analysis'] = dominance_analysis

            if self.config.get('enable_cost_benefit', True):
                cost_benefit = self.recommendation_engine.perform_cost_benefit_analysis(
                    complex_model_results, dominance_analysis
                )
                results['cost_benefit_analysis'] = cost_benefit

            results['recommendations'] = self.recommendation_engine.generate_simplification_recommendations(
                dominance_analysis, results.get('cost_benefit_analysis', {})
            )

            self.logger.info("✅ Baseline dominance analysis complete")
            return results

        except Exception as e:
            self.logger.error(f"Error in baseline dominance analysis: {e}", exc_info=True)
            raise DataProcessingError(f"Baseline dominance analysis failed: {e}") from e

    async def _train_baseline_models(self,
                                  market_data: pd.DataFrame,
                                  features_df: pd.DataFrame | None,
                                  target_series: pd.Series | None) -> dict[str, Any]:
        """Навчає та оцінює всі доступні базові моделі."""
        baseline_results = {}
        for name, model in self.baseline_implementations.items():
            try:
                result = model.train_and_evaluate(market_data, features_df, target_series)
                if result.get('status') != 'no_data':
                    baseline_results[name] = result
            except Exception as e:
                self.logger.error(f"Error training baseline {name}: {e}", exc_info=True)

        if not baseline_results:
            raise DataProcessingError("Failed to train any baseline models")

        self.logger.info(f"📊 Trained {len(baseline_results)} baseline models")
        return baseline_results
