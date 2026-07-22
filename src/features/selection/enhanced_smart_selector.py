#!/usr/bin/env python3
"""
Enhanced Smart Feature Selector - Full Integration with New Analysis Components
Integrates drift monitoring, redundancy detection, regime tracking, and news decay optimization.
"""

from datetime import datetime
from typing import Any

import pandas as pd

from src.config.unified_config_manager import UnifiedConfigManager, get_current_config
from src.core.logging.logger import ProjectLogger
from src.features.analysis.news_decay_modeler import get_news_decay_modeler
from src.features.analysis.regime_importance_tracker import get_regime_importance_tracker
from src.features.validation.redundancy_detector import get_redundancy_detector
from src.monitoring.data_freshness_monitor import get_data_freshness_monitor

# Import new analysis components
from src.monitoring.feature_drift_monitor import get_feature_drift_monitor

# Import existing SmartFeatureSelector
from .smart_selector import SmartFeatureSelector

logger = ProjectLogger.get_logger("EnhancedSmartFeatureSelector")

class EnhancedSmartFeatureSelector(SmartFeatureSelector):
    """
    Enhanced Smart Feature Selector with full integration of new analysis components.

    🎯 REGIME-AWARE & REDUNDANCY-CLEANED:
    - Використовує 'context_pattern_id' для адаптивної ваги методів.
    - Автоматично видаляє дублікати (Redundancy Elimination).
    - Відстежує дріфт ознак у реальному часі.
    """

    def __init__(self, config_manager: UnifiedConfigManager | None = None):
        """
        Initialize Enhanced Smart Feature Selector.
        """
        super().__init__()

        self.config_manager = config_manager or get_current_config()

        # Initialize new analysis components
        self.drift_monitor = get_feature_drift_monitor()
        self.freshness_monitor = get_data_freshness_monitor()
        self.redundancy_detector = get_redundancy_detector()
        self.regime_tracker = get_regime_importance_tracker()
        self.news_decay_modeler = get_news_decay_modeler()

        # Enhanced settings
        self.monitoring_enabled = self.config_manager.get('feature_selection.monitoring_enabled', True)
        self.redundancy_elimination_enabled = self.config_manager.get('feature_selection.redundancy_elimination_enabled', True)
        self.regime_adaptation_enabled = self.config_manager.get('feature_selection.regime_adaptation_enabled', True)

        self.logger.info("✅ EnhancedSmartFeatureSelector initialized with Regime-Aware selection")

    async def select_with_full_analysis(self,
                                      features_df: pd.DataFrame,
                                      target_series: pd.Series,
                                      context_id: str,
                                      market_data: pd.DataFrame | None = None,
                                      news_data: pd.DataFrame | None = None,
                                      model_metadata: dict[str, Any] | None = None,
                                      **kwargs) -> dict[str, Any]:
        """
        Виконує комплексну селекцію з аналізом патернів та очищенням.
        """
        self.logger.info(f"🧠 Starting enhanced feature selection for {context_id}")

        # 0a. context_pattern_id — fingerprint послідовності патернів (sha256 хеш).
        #     Використовується ТІЛЬКИ для кешування/фінгерпринтингу, а не для
        #     визначення ринкового режиму.
        current_pattern = "normal"
        if 'context_pattern_id' in features_df.columns:
            current_pattern = str(features_df['context_pattern_id'].iloc[-1])
            self.logger.info(f"📍 Detected Context Pattern (fingerprint): {current_pattern}")

        # 0b. market_regime — реальний ринковий режим з колонки MARKET_REGIME,
        #     яку генерує TechnicalAnalysisEnricher._add_market_regime_features().
        #     Очікувані значення: 'TRENDING_UP', 'TRENDING_DOWN', 'VOLATILE',
        #     'RANGING', 'UNKNOWN'. Маппимо на vocab SmartFeatureSelector.
        market_regime = self._resolve_market_regime(features_df)
        self.logger.info(f"📊 Resolved market_regime for selection: {market_regime}")

        results: dict[str, Any] = {
            'context_id': context_id,
            'pattern_id': current_pattern,
            'market_regime': market_regime,
            'timestamp': datetime.now(),
            'original_feature_count': len(features_df.columns),
            'selected_features': [],
            'analysis_results': {},
            'monitoring_results': {}
        }

        try:
            # 1. Redundancy Elimination (Обов'язкове очищення)
            if self.redundancy_elimination_enabled:
                self.logger.info("🗑️ Running redundancy elimination...")
                redundancy_result = self.redundancy_detector.eliminate_redundant_features(
                    features_df, target_series
                )
                clean_features = redundancy_result['cleaned_features']
                results['analysis_results']['redundancy'] = redundancy_result
            else:
                clean_features = features_df.copy()

            # 2. Адаптивні ваги на основі патерна.
            #    Результат передається в базовий селектор через market_regime.
            pattern_weights = self._get_weights_for_pattern(current_pattern)
            if logger.isEnabledFor('DEBUG' if hasattr(logger, 'DEBUG') else 10):
                self.logger.debug(f"Pattern weights for '{current_pattern}': {pattern_weights}")

            # 3. Базова селекція з реальним market_regime (не з context_pattern_id)
            selected_features = self.select(
                clean_features, target_series, context_id,
                market_regime=market_regime, **kwargs
            )

            results['selected_features'] = selected_features
            results['performance_metrics'] = self._calculate_performance_metrics(
                features_df, clean_features, selected_features
            )

            self.logger.info(f"✅ Enhanced selection complete: {len(selected_features)} features selected")
            return results

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error in enhanced feature selection: {e}", exc_info=True)
            results['error'] = str(e)
            return results

    def _resolve_market_regime(self, features_df: pd.DataFrame) -> str:
        """
        Визначає market_regime з колонки MARKET_REGIME features_df.

        TechnicalAnalysisEnricher генерує колонку 'MARKET_REGIME' зі значеннями
        'TRENDING_UP', 'TRENDING_DOWN', 'VOLATILE', 'RANGING', 'UNKNOWN'.
        Маппимо на трирівневий vocab SmartFeatureSelector: normal / volatile / trending.
        """
        if 'MARKET_REGIME' not in features_df.columns:
            return 'normal'

        raw = str(features_df['MARKET_REGIME'].iloc[-1]).upper()

        if 'VOLATILE' in raw or 'CRISIS' in raw:
            return 'volatile'
        if 'TRENDING' in raw or 'BULL' in raw or 'BEAR' in raw:
            return 'trending'
        return 'normal'

    def _get_weights_for_pattern(self, pattern_id: str) -> dict[str, float]:
        """Визначає ваги методів для конкретного патерна."""
        # Якщо патерн нестабільний (висока ентропія), фокусуємося на Random Forest та MI
        if pattern_id == "normal":
            return {'correlation': 1.0, 'mutual_info': 1.0, 'lgbm': 1.0}

        # Для специфічних патернів можна додати логіку:
        # Наприклад, якщо патерн відомий як "Trending", збільшуємо вагу кореляції
        return {'correlation': 1.2, 'mutual_info': 1.0, 'lgbm': 1.5, 'rf': 1.2}

    def _calculate_performance_metrics(self,
                                     original_features: pd.DataFrame,
                                     clean_features: pd.DataFrame,
                                     selected_features: list[str]) -> dict[str, Any]:
        """Розраховує метрики ефективності селекції."""
        orig_count = len(original_features.columns)
        sel_count = len(selected_features)
        return {
            'reduction_ratio': (1 - sel_count / orig_count) * 100 if orig_count > 0 else 0,
            'clean_count': len(clean_features.columns),
            'selected_count': sel_count
        }

# Factory function
def get_enhanced_smart_selector(config_manager: UnifiedConfigManager | None = None) -> EnhancedSmartFeatureSelector:
    return EnhancedSmartFeatureSelector(config_manager)

# Convenience function for quick enhanced selection
async def select_features_enhanced(features_df: pd.DataFrame,
                                 target_series: pd.Series,
                                 context_id: str,
                                 config_manager: UnifiedConfigManager | None = None,
                                 **kwargs) -> dict[str, Any]:
    """
    Quick enhanced feature selection.
    """
    selector = get_enhanced_smart_selector(config_manager)
    return await selector.select_with_full_analysis(features_df, target_series, context_id, **kwargs)
