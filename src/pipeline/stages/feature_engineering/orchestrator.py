from datetime import datetime
from typing import Any

import pandas as pd

from src.analytics.calculators.advanced_econometrics_calculator import AdvancedEconometricsCalculator
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
        self._last_causal_evidence: dict[str, Any] = {}

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
            'causal_evidence': self._last_causal_evidence,
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

    # Held-out fraction (chronological tail, per ticker) that feature selection
    # must never see. Matches the test_size=0.2 convention already used by
    # src/models/adapters/data_preparation.py's train/val/test split, so the
    # rows Stage 4's walk-forward validator eventually treats as unseen were
    # never used to decide which columns exist.
    _SELECTION_HOLDOUT_FRACTION = 0.2

    def _train_only_index(self, frame: pd.DataFrame) -> pd.Index:
        """Chronological prefix of `frame`, excluding the final holdout tail.

        Grouped per ticker (when a 'ticker' column is present) so a
        multi-symbol frame doesn't leak one ticker's future into another's
        selection window. Falls back to positional order (frame is expected
        to already be time-sorted per timeframe) when no explicit time
        column is available.
        """
        time_col = next((c for c in ('datetime', 'timestamp', 'date') if c in frame.columns), None)

        def _prefix(group: pd.DataFrame) -> pd.Index:
            ordered = group.sort_values(time_col) if time_col else group
            cutoff = int(len(ordered) * (1.0 - self._SELECTION_HOLDOUT_FRACTION))
            return ordered.index[:cutoff]

        if 'ticker' in frame.columns:
            parts = [_prefix(g) for _, g in frame.groupby('ticker', sort=False)]
            return parts[0].append(parts[1:]) if parts else frame.index[:0]
        return _prefix(frame)

    # Column-name prefixes/substrings that identify an "external" predictor
    # (macro, sentiment, news, economic-calendar) as opposed to a technical
    # indicator derived from the ticker's own price — Granger-testing a
    # technical indicator against the price it was computed from is close
    # to circular, so those are deliberately excluded from this diagnostic.
    _EXTERNAL_PREDICTOR_MARKERS: tuple[str, ...] = (
        'FRED_', 'sentiment', 'nlp_sentiment', 'finbert_', 'news_', 'economic_', 'macro_', 'surprise_index',
    )
    _MAX_CAUSAL_DIAGNOSTIC_PREDICTORS = 15

    def _diagnose_external_predictor_causality(
        self,
        candidate_features: pd.DataFrame,
        target_series: pd.Series,
        target_col: str,
    ) -> dict[str, Any]:
        """Diagnostic-only Granger/stationarity/cointegration check for
        external (macro/sentiment/news) predictors against the target.

        Does NOT filter or reweight feature selection — this is attached to
        the Stage 3 output as `causal_evidence` purely so a human (or a
        later, explicitly-approved change) can see which external
        predictors show real lead-lag structure versus which are along for
        the ride on correlation alone. Best-effort: any failure returns an
        empty/partial result rather than blocking feature selection, and a
        predictor count cap keeps this from dominating Stage 3 runtime,
        since each predictor fits a VAR model plus stationarity/
        cointegration/impulse-response/variance-decomposition tests.
        """
        external_cols = [
            col
            for col in candidate_features.columns
            if any(marker in col for marker in self._EXTERNAL_PREDICTOR_MARKERS)
        ][: self._MAX_CAUSAL_DIAGNOSTIC_PREDICTORS]

        if not external_cols:
            return {}

        frame = candidate_features[external_cols].copy()
        frame[target_col] = target_series

        try:
            causality_results = AdvancedEconometricsCalculator.run_comprehensive_causal_analysis(
                frame, target_col, external_cols, maxlag=10, lag_selection='aic',
            )
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.warning(f"Causal diagnostic failed, skipping: {e}")
            return {}

        evidence = {
            col: {
                'is_significant': result.get('is_significant'),
                'causality_strength': result.get('causality_strength'),
                'p_value': (result.get('granger_test') or {}).get('p_value'),
                'is_cointegrated': (result.get('cointegration') or {}).get('is_cointegrated'),
            }
            for col, result in causality_results.items()
            if col != '_summary' and isinstance(result, dict) and 'error' not in result
        }

        if evidence:
            significant = sum(1 for v in evidence.values() if v.get('is_significant'))
            self.logger.info(
                f"Causal diagnostic: {significant}/{len(evidence)} external predictors show "
                f"significant Granger causality vs '{target_col}' (diagnostic only, selection unaffected)."
            )
        return evidence

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

        # LEAKAGE FIX: restrict the index used for selection to a chronological
        # train-only prefix. Selecting against the full dataset (including the
        # rows Stage 4 later holds out) lets feature choice "see" future
        # target values, inflating apparent walk-forward skill.
        train_only_index = self._train_only_index(final_features).intersection(valid_index)
        if len(train_only_index) < 5:
            # Not enough train-only rows (e.g. tiny fixture data) — fall back
            # to the full valid_index rather than failing selection outright.
            train_only_index = valid_index

        if candidate_features.empty or len(valid_index) < 5 or target_series.loc[valid_index].nunique() < 2:
            fallback = list(candidate_features.columns)
            return fallback, dict.fromkeys(fallback, 1.0)

        self._last_causal_evidence = self._diagnose_external_predictor_causality(
            candidate_features.loc[train_only_index],
            target_series.loc[train_only_index],
            target_col,
        )

        try:
            selection_result = await self.selector.select_with_full_analysis(
                candidate_features.loc[train_only_index],
                target_series.loc[train_only_index],
                context_id=kwargs.get('context_id', f'stage3_{target_col}'),
                market_data=final_features.loc[train_only_index],
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
