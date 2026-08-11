#!/usr/bin/env python3
"""
Enhanced Smart Feature Selector - regime-aware selection with redundancy cleaning.

Drift monitoring, regime tracking and news-decay modelling were imported and
constructed here but never called; the imports went with them. See the class
docstring for where drift/freshness checks actually live.
"""

from datetime import datetime
from typing import Any

import pandas as pd

from src.config.unified_config_manager import UnifiedConfigManager, get_current_config
from src.core.logging.logger import ProjectLogger
from src.features.validation.redundancy_detector import get_redundancy_detector

# Import existing SmartFeatureSelector
from .smart_selector import SmartFeatureSelector

logger = ProjectLogger.get_logger("EnhancedSmartFeatureSelector")

class EnhancedSmartFeatureSelector(SmartFeatureSelector):
    """
    Enhanced Smart Feature Selector with full integration of new analysis components.

    🎯 REGIME-AWARE & REDUNDANCY-CLEANED:
    - Використовує реальний MARKET_REGIME для адаптивної селекції.
    - Автоматично видаляє дублікати (Redundancy Elimination).

    NOT done here, despite what this docstring used to claim: feature drift is
    NOT tracked in this class. It constructed `drift_monitor`,
    `freshness_monitor`, `regime_tracker` and `news_decay_modeler` and called
    a method on NONE of them -- four of five components were initialisation
    cost and a false impression of monitoring. Drift and freshness checks do
    exist, in `src.pipeline.stages.monitoring.feature_monitoring.
    FeatureEngineeringMonitor`, which is the class written to run them around
    feature engineering; it currently has no callers either. Wiring it is a
    deliberate decision, not a cleanup: `check_drift` over a full feature
    frame is the computation that produced a five-hour hang in Stage 7.
    """

    def __init__(self, config_manager: UnifiedConfigManager | None = None):
        """
        Initialize Enhanced Smart Feature Selector.
        """
        super().__init__()

        self.config_manager = config_manager or get_current_config()

        # The only analysis component this class actually calls.
        self.redundancy_detector = get_redundancy_detector()

        # Enhanced settings
        self.redundancy_elimination_enabled = self.config_manager.get('feature_selection.redundancy_elimination_enabled', True)
        # `feature_selection.monitoring_enabled` and
        # `feature_selection.regime_adaptation_enabled` were read into
        # attributes here and consulted by nothing -- two config switches an
        # operator could toggle for ever without changing behaviour. Regime
        # adaptation is not optional in this class (the regime always reaches
        # `select`), and monitoring does not happen here at all, so neither
        # flag has a truthful meaning to preserve.

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
            # Was an empty dict that nothing ever wrote to, which reads
            # downstream as "monitored, nothing found" rather than "never
            # ran". Say which it is.
            'monitoring_results': {
                'status': 'not_performed',
                'reason': (
                    'Feature drift/freshness monitoring is not part of '
                    'selection. See FeatureEngineeringMonitor '
                    '(src/pipeline/stages/monitoring/feature_monitoring.py), '
                    'which implements these checks and is currently unwired.'
                ),
            },
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

            # 2. Базова селекція з реальним market_regime (не з context_pattern_id)
            #
            # `_get_weights_for_pattern(current_pattern)` used to be computed
            # here, logged at DEBUG, and then dropped: `select()` receives
            # market_regime, never the weights. It was dead in a second way
            # too -- `current_pattern` is a sha256 fingerprint (see 0a above),
            # and the function only returns its first branch when that string
            # is literally "normal". Measured on the 2026-08-06 export the
            # fingerprint is near-unique per row (7,112 distinct values over
            # 7,128 daily rows), so the branch could essentially never be
            # taken and the function returned one constant dict that was then
            # discarded. Weighting selection methods by regime belongs in
            # `select()`, which already knows the regime.
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
