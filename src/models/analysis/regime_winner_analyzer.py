"""
Regime Winner Analyzer - Analyzes Model Winner Consistency Across Market Regimes
Tracks and analyzes model performance patterns across different market regimes.
"""
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger('RegimeWinnerAnalyzer')


class RegimeWinnerAnalyzer:
    """
    Аналізує стабільність переможців серед моделей у різних режимах ринку.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Ініціалізує RegimeWinnerAnalyzer.
        """
        self.logger = logger
        self.config = config or {}

        self.REGIME_TYPES = {
            'bull': {
                'description': 'Strong upward trend',
                'volatility_range': (0.0, 0.02),
                'trend_strength': (0.01, 0.1),
                'typical_winners': ['trend_follower', 'ensemble']
            },
            'bear': {
                'description': 'Strong downward trend',
                'volatility_range': (0.02, 0.05),
                'trend_strength': (-0.1, -0.01),
                'typical_winners': ['mean_reversion', 'short_biased']
            },
            'ranging': {
                'description': 'Sideways market',
                'volatility_range': (0.0, 0.015),
                'trend_strength': (-0.01, 0.01),
                'typical_winners': ['mean_reversion', 'oscillator']
            },
            'volatile': {
                'description': 'High volatility conditions',
                'volatility_range': (0.03, 0.06),
                'trend_strength': (-0.05, 0.05),
                'typical_winners': ['rf', 'svm', 'ensemble']
            },
            'crisis': {
                'description': 'Market crisis conditions',
                'volatility_range': (0.04, 0.1),
                'trend_strength': (-0.01, 0.01),
                'typical_winners': ['rf', 'svm', 'conservative']
            }
        }

        self.thresholds = {
            'min_samples_per_regime': 30,
            'consistency_threshold': 0.6,
            'switch_detection_window': 5,
            'performance_gap_threshold': 0.05,
            'stability_window_days': 30
        }
        self.thresholds.update(self.config.get('thresholds', {}))

        self.regime_performance_history = []
        self.regime_switch_points = []

        self.storage_path = Path(self.config.get('storage_path', 'data/analysis/regime_winners'))
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self._init_components()
        self.logger.info("✅ RegimeWinnerAnalyzer initialized")

    def _init_components(self):
        """Ініціалізує модульні компоненти."""
        from .regime import (
            MarketRegimeDetector,
            RegimeMetrics,
            RegimePatternAnalyzer,
            RegimeRecommendationEngine,
            RegimeStabilityAnalyzer,
        )
        self.detector = MarketRegimeDetector(self.REGIME_TYPES)
        self.metrics_calculator = RegimeMetrics()
        self.pattern_analyzer = RegimePatternAnalyzer(self.REGIME_TYPES)
        self.stability_analyzer = RegimeStabilityAnalyzer()
        self.recommendation_engine = RegimeRecommendationEngine(self.REGIME_TYPES)

    async def analyze_regime_consistency(self,
                                      model_results: dict[str, Any],
                                      market_data: pd.DataFrame,
                                      current_time: datetime | None = None) -> dict[str, Any]:
        """
        Аналізує консистентність переможців.
        """
        current_time = current_time or datetime.now()
        self.logger.info(f"📊 Analyzing regime consistency at {current_time}")

        try:
            current_regime = self.detector.detect_regime(market_data)

            regime_winners = await self._analyze_winners_by_regime(
                model_results, market_data, current_regime
            )

            consistency_metrics = await self._calculate_consistency_metrics(
                regime_winners, current_regime
            )

            winner_patterns = self.pattern_analyzer.analyze_winner_patterns(
                regime_winners, current_regime
            )

            switching_analysis = await self._detect_regime_switching(
                current_regime, current_time
            )

            recommendations = self.recommendation_engine.generate_regime_recommendations(
                current_regime, consistency_metrics, winner_patterns
            )

            results = {
                'timestamp': current_time,
                'current_regime': current_regime,
                'regime_performance': regime_winners,
                'consistency_analysis': consistency_metrics,
                'winner_patterns': winner_patterns,
                'switching_analysis': switching_analysis,
                'recommendations': recommendations
            }

            self._store_analysis_results(results)
            self.logger.info(f"✅ Regime consistency analysis complete. Regime: {current_regime}")

            return results

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error in regime consistency analysis: {e}", exc_info=True)
            return {'status': 'error', 'error': str(e), 'timestamp': current_time}

    async def _analyze_winners_by_regime(self,
                                      model_results: dict[str, Any],
                                      market_data: pd.DataFrame,
                                      current_regime: str) -> dict[str, Any]:
        """Аналізує лідерів для поточного режиму."""
        regime_analysis = {
            'current_regime': current_regime,
            'model_performance': {},
            'winner_ranking': [],
            'regime_specific_metrics': {}
        }

        try:
            model_performance = {}
            if isinstance(model_results, dict):
                for model_name, result in model_results.items():
                    if isinstance(result, dict) and 'metrics' in result:
                        metrics = result['metrics']
                        score = self.metrics_calculator.calculate_performance_score(metrics)
                        model_performance[model_name] = {
                            'performance_score': score,
                            'metrics': metrics
                        }

            sorted_winners = sorted(
                model_performance.items(),
                key=lambda x: x[1]['performance_score'],
                reverse=True
            )

            regime_analysis['model_performance'] = model_performance
            regime_analysis['winner_ranking'] = [
                {'model_name': name, 'performance_score': info['performance_score']}
                for name, info in sorted_winners
            ]

            regime_analysis['regime_specific_metrics'] = {
                'score_gap': self.metrics_calculator.calculate_score_gap(sorted_winners)
            }

            return regime_analysis

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error analyzing winners: {e}")
            return {'current_regime': current_regime, 'model_performance': {}, 'winner_ranking': [], 'regime_specific_metrics': {}, 'error': str(e)}

    async def _calculate_consistency_metrics(self,
                                          regime_winners: dict[str, Any],
                                          current_regime: str) -> dict[str, Any]:
        """Розраховує метрики стабільності результатів."""
        history = [r for r in self.regime_performance_history if r['regime'] == current_regime]

        model_consistency = self.pattern_analyzer.calculate_model_consistency(history)

        overall_consistency = np.mean(list(model_consistency.values())) if model_consistency else 1.0

        return {
            'current_regime': current_regime,
            'model_consistency': model_consistency,
            'overall_consistency': float(overall_consistency),
            'sample_count': len(history)
        }

    async def _detect_regime_switching(self, current_regime: str, current_time: datetime) -> dict[str, Any]:
        """Аналізує зміни режимів ринку."""
        if self.regime_performance_history and self.regime_performance_history[-1]['regime'] != current_regime:
            switch = {
                'timestamp': current_time,
                'from_regime': self.regime_performance_history[-1]['regime'],
                'to_regime': current_regime
            }
            self.regime_switch_points.append(switch)
            self.logger.info(f"🔄 Market regime switch detected: {switch['from_regime']} -> {switch['to_regime']}")

        recent_history = self.regime_performance_history[-50:]

        return {
            'current_regime': current_regime,
            'stability_index': self.stability_analyzer.calculate_regime_stability(recent_history),
            'most_frequent_switch': self.stability_analyzer.get_most_frequent_switch(self.regime_switch_points),
            'avg_stable_period': self.stability_analyzer.calculate_average_stable_period(self.regime_switch_points)
        }

    async def _generate_regime_recommendations(self,
                                            current_regime: str,
                                            consistency_metrics: dict[str, Any],
                                            winner_patterns: dict[str, Any]) -> list[str]:
        """Генерує рекомендації на основі аналізу режимів."""
        recommendations = []

        consistency = consistency_metrics.get('overall_consistency', 0.0)
        if consistency < 0.5:
            recommendations.append(f"⚠️ Low model consistency ({consistency:.2f}). Consider ensemble methods.")
        elif consistency < 0.7:
            recommendations.append(f"📊 Moderate model consistency ({consistency:.2f}). Monitor closely.")

        deviations = winner_patterns.get('pattern_deviations', [])
        if len(deviations) > 2:
            recommendations.append(f"🚨 High pattern deviations ({len(deviations)}). Review model selection.")

        insights = winner_patterns.get('regime_specific_insights', {})
        recommendations.extend(insights.get('recommendations', []))

        return recommendations

    def _store_analysis_results(self, results: dict[str, Any]) -> None:
        """Зберігає результати аналізу."""
        try:
            # Update history
            self.regime_performance_history.append({
                'timestamp': results['timestamp'],
                'regime': results['current_regime'],
                'winner': results['regime_performance']['winner_ranking'][0]['model_name'] if results['regime_performance']['winner_ranking'] else None,
                'score': results['regime_performance']['winner_ranking'][0]['performance_score'] if results['regime_performance']['winner_ranking'] else 0.0,
                'model_name': results['regime_performance']['winner_ranking'][0]['model_name'] if results['regime_performance']['winner_ranking'] else None # For consistency logic
            })

            timestamp = results['timestamp'].strftime('%Y%m%d_%H%M%S')
            filepath = self.storage_path / f"regime_analysis_{timestamp}.json"

            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error storing results: {e}")

    def get_regime_summary(self, days: int = 30) -> dict[str, Any]:
        """Повертає підсумок аналізу за останні дні."""
        cutoff = datetime.now() - timedelta(days=days)
        recent_history = [r for r in self.regime_performance_history if r['timestamp'] >= cutoff]

        return {
            'days': days,
            'total_records': len(recent_history),
            'regime_distribution': self._calculate_distribution(recent_history),
            'regime_stability': self.stability_analyzer.calculate_regime_stability(recent_history)
        }

    def _calculate_distribution(self, history: list[dict[str, Any]]) -> dict[str, float]:
        if not history:
            return {}
        counts = defaultdict(int)
        for r in history:
            counts[r['regime']] += 1
        return {k: v / len(history) for k, v in counts.items()}
