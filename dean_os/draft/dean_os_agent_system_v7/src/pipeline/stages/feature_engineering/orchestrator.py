from datetime import datetime
from typing import Any

import pandas as pd

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.error_handling.error_handler import ErrorHandler
from src.core.logging.logger import ProjectLogger
from src.features.selection.enhanced_smart_selector import get_enhanced_smart_selector
from src.pipeline.stages.base_stage import BaseStage
from src.pipeline.target_column_utils import is_target_like_column
from src.pipeline.timeframe_lineage import (
    normalize_timeframe,
    partition_market_frame_by_timeframe,
    timeframe_lineage_report,
)

from .enricher import FeatureEnricher
from .guards import FeatureGuards
from .targets import TargetGenerator
from .timeframe_context import BackwardTimeframeContextAssembler


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
        self.timeframe_context_assembler = BackwardTimeframeContextAssembler()
        self._last_timeframe_context_report: dict[str, Any] = {}

        self.logger.info("✅ FeatureEngineeringStage (Modular) initialized")

    async def run(self, **kwargs) -> dict[str, Any]:
        """Runs the feature engineering cycle."""
        self.logger.info('Starting modular feature engineering stage...')

        cleaned_data, market_data_dict = self._validate_and_prepare_market_data(**kwargs)
        if not market_data_dict:
            raise ValueError(
                "FeatureEngineeringStage: No market data available. "
                "Ensure Stage 2 (ProcessingStage) produced 'prices' in cleaned_data."
            )

        enriched_data: dict[str, pd.DataFrame] = {}
        all_targets: dict[str, pd.DataFrame] = {}

        # 1. Enrichment for each timeframe
        for tf, df in market_data_dict.items():
            # ✅ FIX: pass macro_data and news from cleaned_data to enrichers via kwargs
            enrich_kwargs = {}
            if 'macro_data' in cleaned_data:
                enrich_kwargs['macro_data'] = cleaned_data['macro_data']
            if 'news' in cleaned_data:
                enrich_kwargs['news'] = cleaned_data['news']
            if kwargs.get('offline_only'):
                enrich_kwargs['offline_only'] = True

            enriched_df = self.enricher.enrich_features(df, timeframe=tf, **enrich_kwargs)
            enriched_df = self._restore_service_columns(enriched_df, df)

            # 2. Target Generation (for all timeframes, not just 1d)
            targets_df = self.target_gen.generate_targets(
                enriched_df,
                timeframe=tf,
            )
            all_targets[tf] = targets_df
            target_cols = [col for col in targets_df.columns if col.startswith('target_')]
            for col in target_cols:
                enriched_df[col] = targets_df[col].reindex(enriched_df.index)

            # 3. Apply Safety Guards
            enriched_df = self.guards.apply_guards(enriched_df)
            enriched_df = self._restore_service_columns(enriched_df, df)

            enriched_data[tf] = enriched_df

        # 4. Feature Selection (on the primary timeframe)
        final_features = self._combine_timeframes(enriched_data)
        selected_features = self._initial_feature_columns(final_features)
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
            'timeframe_context_report': self._last_timeframe_context_report,
            'timestamp': datetime.now().isoformat()
        }

    def _validate_and_prepare_market_data(self, **kwargs):
        cleaned_data = kwargs.get('cleaned_data', {})
        stage_logger = getattr(self, "logger", None)
        if stage_logger is not None:
            stage_logger.info(
                f"Stage3 input keys: {list(kwargs.keys())}; "
                f"cleaned_data keys: {list(cleaned_data.keys()) if isinstance(cleaned_data, dict) else type(cleaned_data).__name__}"
            )
        market_data_raw = cleaned_data.get('prices')
        if market_data_raw is None:
            market_data_raw = cleaned_data.get('market_data')
        if market_data_raw is None:
            market_data_raw = kwargs.get('market_data')

        if market_data_raw is None:
            if stage_logger is not None:
                stage_logger.error("No market data found in cleaned_data['prices'], cleaned_data['market_data'], or kwargs['market_data']")
        elif isinstance(market_data_raw, dict):
            inner_types = {k: type(v).__name__ for k, v in market_data_raw.items()}
            if stage_logger is not None:
                stage_logger.info(f"Market data dict contents: {inner_types}")
        else:
            if stage_logger is not None:
                stage_logger.info(f"Market data type: {type(market_data_raw).__name__}")

        if isinstance(market_data_raw, pd.DataFrame):
            market_data_raw = partition_market_frame_by_timeframe(
                market_data_raw
            )
        elif isinstance(market_data_raw, dict):
            validated = {}
            for raw_timeframe, frame in market_data_raw.items():
                if not isinstance(frame, pd.DataFrame) or frame.empty:
                    continue
                declared = normalize_timeframe(raw_timeframe)
                report = timeframe_lineage_report(
                    frame,
                    declared_timeframe=declared,
                )
                if report.get("status") in {
                    "timeframe_cadence_mismatch",
                    "timeframe_cadence_ambiguous",
                }:
                    raise ValueError(
                        f"Market frame {raw_timeframe} conflicts with "
                        f"observed {report.get('observed_timeframe')} "
                        "cadence"
                    )
                resolved = report.get("resolved_timeframe")
                if not resolved:
                    raise ValueError(
                        f"Market frame {raw_timeframe} has no "
                        "cadence-validated timeframe"
                    )
                candidate = frame.copy()
                candidate["interval"] = resolved
                candidate.attrs["timeframe_lineage"] = report
                candidate.attrs["timeframe_source"] = (
                    "stage3_input_mapping_key"
                )
                validated[resolved] = candidate
            market_data_raw = validated

        return cleaned_data, market_data_raw

    def _combine_timeframes(self, enriched_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Build point-in-time higher-timeframe context for every base frame."""
        # Filter each dataframe to only contain its own interval
        filtered_data = {}
        for tf, df in enriched_data.items():
            if 'interval' in df.columns:
                filtered_df = df[df['interval'] == tf].copy()
                self.logger.info(f"Filtered {tf} timeframe: {len(df)} -> {len(filtered_df)} rows")
                filtered_data[tf] = filtered_df
            else:
                filtered_data[tf] = df

        assembler = getattr(
            self,
            "timeframe_context_assembler",
            BackwardTimeframeContextAssembler(),
        )
        combined, report = assembler.assemble(filtered_data)
        self._last_timeframe_context_report = report
        self.logger.info(
            "Assembled %s causal timeframe contexts: %s total rows",
            report["summary"]["base_context_count"],
            report["summary"]["output_rows"],
        )
        return combined

    def _initial_feature_columns(self, frame: pd.DataFrame) -> list[str]:
        """Return numeric model candidates without labels or service metadata."""
        if frame.empty:
            return []
        metadata_columns = {
            "datetime",
            "date",
            "timestamp",
            "ticker",
            "interval",
            "timeframe",
        }
        return [
            column
            for column in frame.select_dtypes(include="number").columns
            if column not in metadata_columns and not is_target_like_column(column)
        ]

    def _restore_service_columns(
        self,
        enriched_df: pd.DataFrame,
        source_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Preserve row-level source identity when enrichers drop service columns."""
        if len(enriched_df) != len(source_df):
            return enriched_df
        result = enriched_df.copy()
        service_columns = (
            "datetime",
            "timestamp",
            "date",
            "ticker",
            "interval",
            "timeframe",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "hash",
        )
        for column in service_columns:
            if column not in result.columns and column in source_df.columns:
                result[column] = source_df[column].to_numpy(copy=True)
        return result

    async def _select_features(
        self,
        final_features: pd.DataFrame,
        target_col: str,
        kwargs: dict[str, Any],
    ) -> tuple[list[str], dict[str, float]]:
        target_cols = [col for col in final_features.columns if is_target_like_column(col)]
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
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Feature selection failed critically: {e}', exc_info=True)
            fallback = list(candidate_features.columns)
            return fallback, dict.fromkeys(fallback, 1.0)
