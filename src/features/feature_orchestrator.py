import importlib
import inspect
import os
import pkgutil
from typing import Any

import pandas as pd

from src.config.unified_config_manager import get_current_config
from src.core.logging.logger import ProjectLogger
from src.features.enrichers.base import BaseEnricher
from src.features.selection.volatility_driver_selector import VolatilityDriverSelector

logger = ProjectLogger.get_logger("FeatureOrchestrator")

class FeatureOrchestrator:
    """
    Orchestrates the feature enrichment process, now with an explicit, configurable
    step for dynamic context feature selection.
    """

    def __init__(self, enrichers: list[BaseEnricher], config_manager: Any | None = None):
        self.config_manager = config_manager or get_current_config()
        self.enrichers = sorted(enrichers, key=lambda e: (e.priority, getattr(e, 'name', type(e).__name__)))
        final_order = [e.name for e in self.enrichers]
        logger.info(f"FeatureOrchestrator initialized. Execution order: {final_order}")

    @staticmethod
    def create_from_config(config_manager: Any) -> 'FeatureOrchestrator':
        """
        Dynamically discovers and instantiates enabled enrichers from the config.
        """
        enabled_enrichers = []
        package_path = os.path.join(os.path.dirname(__file__), 'enrichers')
        package_name = 'src.features.enrichers'

        logger.info(f"🔍 Discovering enrichers in '{package_name}'...")

        for _, module_name, _ in pkgutil.iter_modules([package_path]):
            full_module_name = f"{package_name}.{module_name}"
            enrichers_from_module = FeatureOrchestrator._discover_enrichers_in_module(
                full_module_name, config_manager
            )
            enabled_enrichers.extend(enrichers_from_module)

        logger.info(f"✅ Discovered {len(enabled_enrichers)} enabled enrichers")
        enabled_enrichers = FeatureOrchestrator._dedupe_enrichers(enabled_enrichers)
        return FeatureOrchestrator(enabled_enrichers, config_manager=config_manager)

    @staticmethod
    def _discover_enrichers_in_module(full_module_name: str, config_manager: Any) -> list[BaseEnricher]:
        """Discover and instantiate enrichers in a single module."""
        enrichers = []

        try:
            module = importlib.import_module(full_module_name)
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, BaseEnricher) and obj is not BaseEnricher:
                    enricher = FeatureOrchestrator._process_enricher_class(
                        obj, name, config_manager
                    )
                    if enricher:
                        enrichers.append(enricher)
        except ImportError as e:
            logger.error(f"Failed to import module {full_module_name}: {e}")

        return enrichers

    @staticmethod
    def _process_enricher_class(obj: type, name: str, config_manager: Any) -> BaseEnricher | None:
        """Process a single enricher class and return instance if enabled."""
        try:
            enricher_id = FeatureOrchestrator._get_enricher_id(obj, name)

            if not FeatureOrchestrator._is_enricher_enabled(enricher_id, config_manager):
                logger.debug(f"⏭️  Enricher '{enricher_id}' is DISABLED in config.")
                return None

            instance = FeatureOrchestrator._instantiate_enricher(obj, enricher_id, config_manager)
            if instance:
                logger.info(f"✅ Enricher '{enricher_id}' (class: {name}) is ENABLED and instantiated.")
            return instance

        except Exception as e:
            logger.error(f"Failed to instantiate enricher {name}: {e}", exc_info=True)
            return None

    @staticmethod
    def _get_enricher_id(obj: type, name: str) -> str:
        """Get enricher ID from class instance or fallback to class name."""
        try:
            temp_instance = obj()
            enricher_name: str = str(temp_instance.name)
            return enricher_name
        except Exception:
            return name.lower()

    @staticmethod
    def _is_enricher_enabled(enricher_id: str, config_manager: Any) -> bool:
        """Check if enricher is enabled in config (old or new format)."""
        old_config = config_manager.get_config('features', {}).get('enrichers', {}).get(enricher_id, {})
        new_config = config_manager.get_config('features', {}).get('enabled_enrichers', {}).get(enricher_id, False)

        # Debug logging
        logger.debug(f"📋 Enricher '{enricher_id}': old_config={old_config}, new_config={new_config}")

        # Check if enabled in either config
        old_enabled = old_config.get('enabled', False) if isinstance(old_config, dict) else False
        new_enabled = new_config if isinstance(new_config, bool) else False
        enabled = old_enabled or new_enabled

        logger.debug(f"📋 Enricher '{enricher_id}': old_enabled={old_enabled}, new_enabled={new_enabled}, enabled={enabled}")
        return enabled

    @staticmethod
    def _instantiate_enricher(obj: type, enricher_id: str, config_manager: Any) -> BaseEnricher | None:
        """Instantiate enricher with or without config parameters."""
        try:
            sig = inspect.signature(obj)
            expects_args = len(sig.parameters) > 0
        except Exception:
            expects_args = False

        if expects_args:
            # Enricher expects config parameter
            old_config = config_manager.get_config('features', {}).get('enrichers', {}).get(enricher_id, {})
            enricher_config = old_config if isinstance(old_config, dict) else {}
            return obj(enricher_config)  # type: ignore[no-any-return]
        else:
            # Enricher has no-argument constructor
            return obj()  # type: ignore[no-any-return]

    @staticmethod
    def _dedupe_enrichers(enrichers: list[BaseEnricher]) -> list[BaseEnricher]:
        seen_names = set()
        cleaned = []
        for enricher in enrichers:
            if enricher.name in seen_names:
                logger.warning(f"Duplicate enricher name '{enricher.name}' found. Ignoring duplicate instance of {type(enricher).__name__}.")
                continue
            seen_names.add(enricher.name)
            cleaned.append(enricher)
        if len(cleaned) != len(enrichers):
            logger.info(f"Removed {len(enrichers) - len(cleaned)} duplicate enrichers by name.")
        return cleaned

    def run(self, df: pd.DataFrame, add_timeframe_suffix: bool = False, **kwargs) -> pd.DataFrame:
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
        logger.info(f"🔄 Starting enrichment: {df.shape[0]} rows, {df.shape[1]} columns")

        df_enriched = df.copy()
        run_kwargs = kwargs.copy()

        # 1. Dynamic Context Feature Selection (if enabled)
        context_selection_config = self.config_manager.get_config('features', {}).get('context_selection', {})
        if context_selection_config.get('enabled', False):
            df_enriched = self._run_dynamic_context_selection(df_enriched, context_selection_config, run_kwargs)

        # 2. Run All Enabled Enrichers in Prioritized Order
        enrichment_stats = []
        for i, enricher in enumerate(self.enrichers):
            try:
                start_time = pd.Timestamp.now()
                initial_shape = df_enriched.shape

                logger.info(f"🔄 Running enricher '{enricher.name}' (Priority: {enricher.priority})...")
                logger.debug(f"   Enricher class: {enricher.__class__.__name__}")
                logger.debug(f"   Input shape: {df_enriched.shape}")

                df_enriched = enricher.enrich(df_enriched, **run_kwargs)

                end_time = pd.Timestamp.now()
                duration = (end_time - start_time).total_seconds()
                final_shape = df_enriched.shape
                cols_added = final_shape[1] - initial_shape[1]

                logger.debug(f"   Output shape: {df_enriched.shape}")
                logger.info(f"✅ Enricher '{enricher.name}' completed: +{cols_added} columns in {duration:.2f}s")

                enrichment_stats.append({
                    'enricher': enricher.name,
                    'duration_sec': duration,
                    'cols_added': cols_added,
                    'final_cols': final_shape[1]
                })

                # Check for duplicate columns after each enricher
                if df_enriched.columns.duplicated().any():
                    duplicated_cols = df_enriched.columns[df_enriched.columns.duplicated()].tolist()
                    logger.warning(f"⚠️ Enricher '{enricher.name}' created duplicate columns: {duplicated_cols}")
                    logger.warning("   Removing duplicates (keeping first occurrence)...")
                    df_enriched = df_enriched.loc[:, ~df_enriched.columns.duplicated()]

                # Prevent Memory Fragmentation (Pandas PerformanceWarning)
                if df_enriched.shape[1] > 100 and i % 3 == 0:
                    df_enriched = df_enriched.copy()

            except Exception as e:
                logger.error(f"❌ Error in enricher '{enricher.name}': {e}", exc_info=True)
                raise

        # 3. Add timeframe suffix if requested
        if add_timeframe_suffix:
            df_enriched = self._add_timeframe_suffix(df_enriched)

        # 4. Final Model Feature Selection (e.g., using SmartFeatureSelector)
        # This step is conceptually separate and would be called by a higher-level process
        # after the main enrichment is complete.

        logger.info("✅ Feature enrichment pipeline completed.")
        logger.info(f"📊 Final result: {df_enriched.shape[0]} rows, {df_enriched.shape[1]} columns")

        # Log enrichment stats
        total_duration = sum(s['duration_sec'] for s in enrichment_stats)
        logger.info(f"⏱️ Total enrichment time: {total_duration:.2f}s")
        for stat in enrichment_stats:
            logger.debug(f"   {stat['enricher']}: {stat['duration_sec']:.2f}s, +{stat['cols_added']} cols")

        return df_enriched

    def _add_timeframe_suffix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds timeframe suffix to feature columns based on 'interval' column.

        Example:
            Before: SMA_5, RSI_14, interval='15m'
            After:  SMA_5_15m, RSI_14_15m, interval='15m'
        """
        if 'interval' not in df.columns:
            logger.warning("⚠️ Cannot add timeframe suffix: 'interval' column not found")
            return df

        logger.info("🏷️ Adding timeframe suffixes to feature columns...")

        # Service columns that should NOT get suffix
        service_cols = [
            'ticker', 'datetime', 'timestamp', 'interval',
            'open', 'high', 'low', 'close', 'volume', 'hash'
        ]

        # Target columns that should NOT get suffix
        target_cols = [col for col in df.columns if col.startswith('target_')]

        # Columns to exclude from suffix
        exclude_cols = set(service_cols + target_cols)

        # Feature columns that need suffix
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        logger.info(f"   Found {len(feature_cols)} feature columns to add suffix")

        # Group by interval and add suffix
        results = []
        for interval, group in df.groupby('interval'):
            group_copy = group.copy()

            # Add suffix to feature columns
            rename_dict = {col: f'{col}_{interval}' for col in feature_cols}
            group_copy = group_copy.rename(columns=rename_dict)

            results.append(group_copy)

            logger.debug(f"Added suffix '_{interval}' to {len(rename_dict)} columns")

        df_with_suffix = pd.concat(results, ignore_index=True)

        logger.info(f"Timeframe suffixes added: {df.shape[1]} → {df_with_suffix.shape[1]} columns")

        return df_with_suffix

    def _run_dynamic_context_selection(self, df: pd.DataFrame, config: dict[str, Any], run_kwargs: dict[str, Any]) -> pd.DataFrame:
        """
        Runs the VolatilityDriverSelector to find features for the context map.
        """
        try:
            logger.info("Running dynamic context feature selection...")
            selector_config = config.get('selector_config', {})
            aux_pool = selector_config.get('auxiliary_pool_cols')
            target_col = selector_config.get('target_col')

            if not aux_pool or not target_col:
                logger.error("'auxiliary_pool_cols' and 'target_col' must be configured for context selection.")
                return df

            volatility_selector = VolatilityDriverSelector(top_n=selector_config.get('top_n', 10))
            dynamic_context_features = volatility_selector.select(df, aux_pool, target_col)

            if dynamic_context_features:
                # This key 'selected_features' is specifically used by ContextMapEnricher
                run_kwargs['selected_features'] = dynamic_context_features
                logger.info(f"Added {len(dynamic_context_features)} dynamic features to kwargs for ContextMapEnricher.")

        except Exception as e:
            logger.error(f"Dynamic context feature selection failed: {e}", exc_info=True)

        return df
