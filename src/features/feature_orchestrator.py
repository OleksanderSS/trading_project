import importlib
import inspect
import logging
import os
import pkgutil
from typing import Any

import pandas as pd

from src.config.unified_config_manager import get_current_config
from src.core.logging.logger import ProjectLogger
from src.features.enrichers.base import BaseEnricher
from src.features.selection.volatility_driver_selector import VolatilityDriverSelector

logger = ProjectLogger.get_logger('FeatureOrchestrator')

# ✅ Lineage tracking — enabled only when diagnostic mode is active
_LINEAGE_TRACKER = None

def get_lineage_tracker():
    """Get the global lineage tracker instance (None if not enabled)."""
    return _LINEAGE_TRACKER

def enable_lineage_tracking(save_path: str = "diagnostic_reports/feature_lineage_report.json"):
    """Enable lineage tracking for the current session."""
    global _LINEAGE_TRACKER
    try:
        import sys
        if 'diagnostics' not in sys.path:
            sys.path.insert(0, '.')
        from diagnostics.feature_lineage_tracker import FeatureLineageTracker
        _LINEAGE_TRACKER = FeatureLineageTracker()
        _LINEAGE_TRACKER._save_path = save_path
        logger.info(f"✅ FeatureLineageTracker enabled → {save_path}")
        return _LINEAGE_TRACKER
    except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
        logger.warning(f"FeatureLineageTracker not available: {e}")
        return None

def disable_lineage_tracking():
    """Save and disable lineage tracking."""
    global _LINEAGE_TRACKER
    if _LINEAGE_TRACKER:
        try:
            _LINEAGE_TRACKER.save(getattr(_LINEAGE_TRACKER, '_save_path',
                "diagnostic_reports/feature_lineage_report.json"))
            logger.info("✅ FeatureLineageTracker saved")
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.warning(f"Could not save lineage tracker: {e}")
        _LINEAGE_TRACKER = None


class FeatureOrchestrator:
    """
    Orchestrates the feature enrichment process, now with an explicit, configurable
    step for dynamic context feature selection.
    """

    def __init__(self, enrichers: list[BaseEnricher], config_manager: (Any |
        None)=None):
        self.config_manager = config_manager or get_current_config()
        self.enrichers = sorted(enrichers, key=lambda e: (e.priority,
            getattr(e, 'name', type(e).__name__)))
        final_order = [e.name for e in self.enrichers]
        logger.info(
            f'FeatureOrchestrator initialized. Execution order: {final_order}')

    @staticmethod
    def create_from_config(config_manager: Any) ->'FeatureOrchestrator':
        """
        Dynamically discovers and instantiates enabled enrichers from the config.
        """
        enabled_enrichers = []
        package_path = os.path.join(os.path.dirname(__file__), 'enrichers')
        package_name = 'src.features.enrichers'
        logger.info(f"🔍 Discovering enrichers in '{package_name}'...")
        for _, module_name, _ in pkgutil.iter_modules([package_path]):
            full_module_name = f'{package_name}.{module_name}'
            enrichers_from_module = (FeatureOrchestrator.
                _discover_enrichers_in_module(full_module_name, config_manager)
                )
            enabled_enrichers.extend(enrichers_from_module)
        logger.info(f'✅ Discovered {len(enabled_enrichers)} enabled enrichers')
        enabled_enrichers = FeatureOrchestrator._dedupe_enrichers(
            enabled_enrichers)
        return FeatureOrchestrator(enabled_enrichers, config_manager=
            config_manager)

    @staticmethod
    def _discover_enrichers_in_module(full_module_name: str, config_manager:
        Any) ->list[BaseEnricher]:
        """Discover and instantiate enrichers in a single module."""
        enrichers = []
        try:
            module = importlib.import_module(full_module_name)
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, BaseEnricher) and obj is not BaseEnricher:
                    enricher = FeatureOrchestrator._process_enricher_class(obj,
                        name, config_manager)
                    if enricher:
                        enrichers.append(enricher)
        except ImportError as e:
            logger.error(f'Failed to import module {full_module_name}: {e}')
        return enrichers

    @staticmethod
    def _process_enricher_class(obj: type, name: str, config_manager: Any) ->(
        BaseEnricher | None):
        """Process a single enricher class and return instance if enabled."""
        try:
            enricher_id = FeatureOrchestrator._get_enricher_id(obj, name)
            if not FeatureOrchestrator._is_enricher_enabled(enricher_id,
                config_manager):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"⏭️  Enricher '{enricher_id}' is DISABLED in config.")
                return None
            instance = FeatureOrchestrator._instantiate_enricher(obj,
                enricher_id, config_manager)
            if instance:
                logger.info(
                    f"✅ Enricher '{enricher_id}' (class: {name}) is ENABLED and instantiated."
                    )
            return instance
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f'Failed to instantiate enricher {name}: {e}')
            raise RuntimeError(f"Failed to instantiate enabled enricher {name}") from e

    @staticmethod
    def _get_enricher_id(obj: type, name: str) -> str:
        """Get enricher ID from class without full instantiation."""
        # ✅ FIX: Check for class-level NAME attribute first (no side effects)
        class_name_attr = getattr(obj, 'NAME', None)
        if class_name_attr and isinstance(class_name_attr, str):
            return class_name_attr

        # Try reading the property via __dict__ on the class (no instance needed)
        # 'name' is usually an @property that returns a constant string
        name_prop = obj.__dict__.get('name')
        if name_prop and isinstance(name_prop, property):
            try:
                # Call fget with a dummy object to avoid __init__ side effects
                dummy = object.__new__(obj)
                result = name_prop.fget(dummy)
                if result and isinstance(result, str):
                    return result
            except (AttributeError, TypeError, RuntimeError) as e:
                logger.warning(f'Could not get enricher ID for {name} using fget: {e}')
                pass  # fallback to full instantiation below

        # Fallback: full instantiation (original behavior)
        try:
            temp_instance = obj()
            return str(temp_instance.name)
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.debug(f'Could not get enricher ID for {name}: {e}')
            return name.lower()

    @staticmethod
    def _is_enricher_enabled(enricher_id: str, config_manager: Any) ->bool:
        """Check if enricher is enabled in config (old or new format)."""
        old_config = config_manager.get_config('features', {}).get('enrichers',
            {}).get(enricher_id, {})
        new_config = config_manager.get_config('features', {}).get(
            'enabled_enrichers', {}).get(enricher_id, False)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"📋 Enricher '{enricher_id}': old_config={old_config}, new_config={new_config}"
                )
        old_enabled = old_config.get('enabled', False) if isinstance(old_config
            , dict) else False
        new_enabled = new_config if isinstance(new_config, bool) else False
        enabled = old_enabled or new_enabled
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"📋 Enricher '{enricher_id}': old_enabled={old_enabled}, new_enabled={new_enabled}, enabled={enabled}"
                )
        return enabled

    @staticmethod
    def _instantiate_enricher(obj: type, enricher_id: str, config_manager: Any
        ) ->(BaseEnricher | None):
        """Instantiate enricher with or without config parameters."""
        try:
            sig = inspect.signature(obj)
            expects_args = len(sig.parameters) > 0
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f'Виникла помилка під час перевірки сигнатури {enricher_id}: {e}', exc_info=True)
            expects_args = False
        if expects_args:
            old_config = config_manager.get_config('features', {}).get(
                'enrichers', {}).get(enricher_id, {})
            enricher_config = old_config if isinstance(old_config, dict) else {
                }
            return obj(enricher_config)
        else:
            return obj()

    @staticmethod
    def _dedupe_enrichers(enrichers: list[BaseEnricher]) ->list[BaseEnricher]:
        seen_names = set()
        cleaned = []
        for enricher in enrichers:
            if enricher.name in seen_names:
                logger.warning(
                    f"Duplicate enricher name '{enricher.name}' found. Ignoring duplicate instance of {type(enricher).__name__}."
                    )
                continue
            seen_names.add(enricher.name)
            cleaned.append(enricher)
        if len(cleaned) != len(enrichers):
            logger.info(
                f'Removed {len(enrichers) - len(cleaned)} duplicate enrichers by name.'
                )
        return cleaned

    def _process_single_enricher(self, enricher, df: pd.DataFrame, kwargs: dict) -> tuple:
        """Process a single enricher and return (df_enriched, stats_dict)."""
        start_time = pd.Timestamp.now()
        initial_shape = df.shape

        logger.info(f"🔄 Running enricher '{enricher.name}' (Priority: {enricher.priority})...")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'   Enricher class: {enricher.__class__.__name__}')
            logger.debug(f'   Input shape: {df.shape}')

        df_enriched = enricher.enrich(df, **kwargs)

        # ✅ Lineage tracking — captures what each enricher adds
        tracker = get_lineage_tracker()
        if tracker is not None:
            try:
                tracker.capture_component_output(
                    enricher.__class__.__name__,
                    before=df,
                    after=df_enriched,
                )
            except (AttributeError, TypeError, RuntimeError, KeyError) as e:
                logger.warning(f"Lineage tracking failed: {e}") # never block pipeline due to tracking

        end_time = pd.Timestamp.now()
        duration = (end_time - start_time).total_seconds()
        final_shape = df_enriched.shape
        cols_added = final_shape[1] - initial_shape[1]

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'   Output shape: {df_enriched.shape}')

        logger.info(f"✅ Enricher '{enricher.name}' completed: +{cols_added} columns in {duration:.2f}s")

        stats = {
            'enricher': enricher.name,
            'duration_sec': duration,
            'cols_added': cols_added,
            'final_cols': final_shape[1]
        }

        return df_enriched, stats

    def _handle_duplicate_columns(self, df: pd.DataFrame, enricher_name: str) -> pd.DataFrame:
        """Handle duplicate columns created by enricher."""
        if df.columns.duplicated().any():
            duplicated_cols = df.columns[df.columns.duplicated()].tolist()
            logger.warning(f"⚠️ Enricher '{enricher_name}' created duplicate columns: {duplicated_cols}")
            logger.warning('   Removing duplicates (keeping first occurrence)...')
            df = df.loc[:, ~df.columns.duplicated()]
        return df

    def _optimize_dataframe_memory(self, df: pd.DataFrame, iteration: int) -> pd.DataFrame:
        """Optimize dataframe memory periodically."""
        if df.shape[1] > 100 and iteration % 3 == 0:
            df = df.copy()
        return df

    def _log_enrichment_summary(self, enrichment_stats: list) -> None:
        """Log enrichment summary statistics."""
        total_duration = sum(s['duration_sec'] for s in enrichment_stats)
        logger.info(f'⏱️ Total enrichment time: {total_duration:.2f}s')
        for stat in enrichment_stats:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"   {stat['enricher']}: {stat['duration_sec']:.2f}s, +{stat['cols_added']} cols")

    def run(self, df: pd.DataFrame, add_timeframe_suffix: bool=False, **kwargs
        ) ->pd.DataFrame:
        """
        Runs the full enrichment pipeline, including dynamic context selection.

        Args:
            df: Input dataframe
            add_timeframe_suffix: If True, adds timeframe suffix to feature columns (e.g., SMA_5 -> SMA_5_15m)
                                 DEFAULT: False (disabled by default - only needed for event-centric/news-based data)
                                 For time-series data with 'interval' column, suffixes are NOT needed
            **kwargs: Additional arguments passed to enrichers

        🎯 ВАЖНО: Суфікси таймфреймів АКТИВОВАНІ за замовчуванням!

        Це означає:
        - SMA_5 → SMA_5_15m, SMA_5_1d, SMA_5_60m (окремі колонки)
        - SmartFeatureSelector розуміє різницю між таймфреймами
        - Аналіз по таймфреймах працює правильно
        - Моделі тренуються на правильних даних
        """
        logger.info(f'🔄 Starting enrichment: {df.shape[0]} rows, {df.shape[1]} columns')
        df_enriched = df.copy()
        run_kwargs = kwargs.copy()

        context_selection_config = self.config_manager.get_config('features', {}).get('context_selection', {})
        if context_selection_config.get('enabled', False):
            df_enriched = self._run_dynamic_context_selection(df_enriched, context_selection_config, run_kwargs)

        enrichment_stats = []
        for i, enricher in enumerate(self.enrichers):
            try:
                df_enriched, stats = self._process_single_enricher(enricher, df_enriched, run_kwargs)
                enrichment_stats.append(stats)

                df_enriched = self._handle_duplicate_columns(df_enriched, enricher.name)
                df_enriched = self._optimize_dataframe_memory(df_enriched, i)
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                logger.error(f"❌ Error in enricher '{enricher.name}': {e}", exc_info=True)
                raise

        if add_timeframe_suffix:
            df_enriched = self._add_timeframe_suffix(df_enriched)

        logger.info('✅ Feature enrichment pipeline completed.')
        logger.info(f'📊 Final result: {df_enriched.shape[0]} rows, {df_enriched.shape[1]} columns')
        self._log_enrichment_summary(enrichment_stats)

        return df_enriched

    def _add_timeframe_suffix(self, df: pd.DataFrame) ->pd.DataFrame:
        """
        Adds timeframe suffix to feature columns based on 'interval' column.

        Example:
            Before: SMA_5, RSI_14, interval='15m'
            After:  SMA_5_15m, RSI_14_15m, interval='15m'
        """
        if 'interval' not in df.columns:
            logger.warning(
                "⚠️ Cannot add timeframe suffix: 'interval' column not found")
            return df
        logger.info('🏷️ Adding timeframe suffixes to feature columns...')
        service_cols = ['ticker', 'datetime', 'timestamp', 'interval',
            'open', 'high', 'low', 'close', 'volume', 'hash']
        # audit-ignore: ARCHITECTURAL_USAGE
        target_cols = [col for col in df.columns if col.startswith('target_')]
        # audit-ignore: ARCHITECTURAL_USAGE
        exclude_cols = set(service_cols + target_cols)
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        logger.info(
            f'   Found {len(feature_cols)} feature columns to add suffix')
        results = []
        for interval, group in df.groupby('interval'):
            group_copy = group.copy()
            rename_dict = {col: f'{col}_{interval}' for col in feature_cols}
            group_copy = group_copy.rename(columns=rename_dict)
            results.append(group_copy)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Added suffix '_{interval}' to {len(rename_dict)} columns")
        df_with_suffix = pd.concat(results, ignore_index=True)
        logger.info(
            f'Timeframe suffixes added: {df.shape[1]} → {df_with_suffix.shape[1]} columns'
            )
        return df_with_suffix

    def _run_dynamic_context_selection(self, df: pd.DataFrame, config: dict
        [str, Any], run_kwargs: dict[str, Any]) ->pd.DataFrame:
        """
        Runs the VolatilityDriverSelector to find features for the context map.
        """
        try:
            logger.info('Running dynamic context feature selection...')
            selector_config = config.get('selector_config', {})
            aux_pool = selector_config.get('auxiliary_pool_cols')
            # audit-ignore: ARCHITECTURAL_USAGE
            target_col = selector_config.get('target_col')
            # audit-ignore: ARCHITECTURAL_USAGE
            if not aux_pool or not target_col:
                logger.error(
                    # audit-ignore: ARCHITECTURAL_USAGE
                    "'auxiliary_pool_cols' and 'target_col' must be configured for context selection."
                    )
                return df
            volatility_selector = VolatilityDriverSelector(top_n=
                selector_config.get('top_n', 10))
            dynamic_context_features = volatility_selector.select(df,
                # audit-ignore: ARCHITECTURAL_USAGE
                aux_pool, target_col)
            if dynamic_context_features:
                run_kwargs['selected_features'] = dynamic_context_features
                logger.info(
                    f'Added {len(dynamic_context_features)} dynamic features to kwargs for ContextMapEnricher.'
                    )
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f'Dynamic context feature selection failed: {e}',
                exc_info=True)
        return df
