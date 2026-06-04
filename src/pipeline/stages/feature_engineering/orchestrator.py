from datetime import datetime
from typing import Any

import pandas as pd

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.error_handling.error_handler import ErrorHandler
from src.core.logging.logger import ProjectLogger
from src.features.selection.enhanced_smart_selector import get_enhanced_smart_selector
from src.pipeline.stages.base_stage import BaseStage

from .enricher import FeatureEnricher
from .guards import FeatureGuards
from .targets import TargetGenerator


class FeatureEngineeringStage(BaseStage):
    """
    Modular Stage 3: Advanced Feature Engineering Hub.
    Delegates to specialized components for enrichment, target generation, and safety.
    """

    def __init__(self, config_manager: UnifiedConfigManager, error_handler: ErrorHandler, **kwargs):
        super().__init__(config_manager, error_handler, **kwargs)
        self.logger = ProjectLogger.get_logger('FeatureEngineeringStage')

        # Initialize Core Components
        self.selector = get_enhanced_smart_selector(config_manager)

        # Initialize Specialized Modular Components
        self.guards = FeatureGuards(mode=kwargs.get('mode', 'full'))
        self.enricher = FeatureEnricher(config_manager)
        self.target_gen = TargetGenerator(config_manager)

        self.logger.info("✅ FeatureEngineeringStage (Modular) initialized")

    async def run(self, **kwargs) -> dict[str, Any]:
        """Runs the feature engineering cycle."""
        self.logger.info('Starting modular feature engineering stage...')

        cleaned_data, market_data_dict = self._validate_and_prepare_market_data(**kwargs)
        if not market_data_dict:
            return {'status': 'failed', 'reason': 'no_data'}

        enriched_data: dict[str, pd.DataFrame] = {}
        all_targets: dict[str, pd.DataFrame] = {}

        # 1. Enrichment for each timeframe
        for tf, df in market_data_dict.items():
            enriched_df = self.enricher.enrich_features(df, timeframe=tf)

            # 2. Target Generation (usually on 1d)
            if tf == '1d':
                targets_df = self.target_gen.generate_targets(enriched_df)
                all_targets[tf] = targets_df
                target_cols = [col for col in targets_df.columns if col.startswith('target_')]
                for col in target_cols:
                    enriched_df[col] = targets_df[col].reindex(enriched_df.index)

            # 3. Apply Safety Guards
            enriched_df = self.guards.apply_guards(enriched_df)

            enriched_data[tf] = enriched_df

        # 4. Feature Selection (on the primary timeframe)
        final_features = enriched_data.get('1d', pd.DataFrame())
        selected_features = list(final_features.columns) if not final_features.empty else []
        feature_importance: dict[str, float] = {}
        if not final_features.empty:
            target_col = kwargs.get('target_column', 'target_up_1d')
            if target_col in final_features.columns:
                selected_features, feature_importance = await self._select_features(
                    final_features,
                    target_col,
                    kwargs,
                )

        return {
            'status': 'success',
            'features': final_features,
            'enriched_data': final_features,
            'all_timeframes': enriched_data,
            'enriched_prices': enriched_data,
            'all_targets': all_targets,
            'combined_features': final_features,
            'selected_features': selected_features,
            'feature_importance': feature_importance,
            'timestamp': datetime.now().isoformat()
        }

    def _validate_and_prepare_market_data(self, **kwargs):
        cleaned_data = kwargs.get('cleaned_data', {})
        market_data_raw = cleaned_data.get('prices') or cleaned_data.get('market_data') or kwargs.get('market_data')

        if isinstance(market_data_raw, pd.DataFrame):
            market_data_raw = {'1d': market_data_raw}

        return cleaned_data, market_data_raw

    async def _select_features(
        self,
        final_features: pd.DataFrame,
        target_col: str,
        kwargs: dict[str, Any],
    ) -> tuple[list[str], dict[str, float]]:
        target_cols = [col for col in final_features.columns if col.startswith('target_')]
        metadata_cols = {'datetime', 'date', 'timestamp', 'ticker', 'interval'}
        candidate_features = final_features.drop(columns=target_cols, errors='ignore')
        candidate_features = candidate_features.drop(
            columns=[col for col in metadata_cols if col in candidate_features.columns],
            errors='ignore',
        )
        candidate_features = candidate_features.select_dtypes(include='number')
        target_series = final_features[target_col]
        # Ensure we do not leak the target into features
        candidate_features = candidate_features.drop(columns=[target_col], errors='ignore')
        valid_index = candidate_features.index.intersection(target_series.dropna().index)

        if candidate_features.empty or len(valid_index) < 5 or target_series.loc[valid_index].nunique() < 2:
            fallback = list(candidate_features.columns)
            return fallback, dict.fromkeys(fallback, 1.0)

        try:
            selection_result = await self.selector.select_with_full_analysis(
                candidate_features.loc[valid_index],
                target_series.loc[valid_index],
                context_id=kwargs.get('context_id', f'stage3_{target_col}'),
                market_data=final_features.loc[valid_index],
                max_features=kwargs.get('max_features'),
            )
            selected = selection_result.get('selected_features') or []
            selected = [feature for feature in selected if feature in candidate_features.columns]
            if not selected:
                selected = list(candidate_features.columns)
            importance = {
                feature: 1.0 / (rank + 1)
                for rank, feature in enumerate(selected)
            }
            return selected, importance
        except Exception as e:
            self.logger.error(f'Feature selection failed critically: {e}', exc_info=True)
            fallback = list(candidate_features.columns)
            return fallback, dict.fromkeys(fallback, 1.0)
