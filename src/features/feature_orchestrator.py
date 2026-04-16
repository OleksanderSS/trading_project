import pandas as pd
import logging
import importlib
import pkgutil
import inspect
import os
from typing import List, Any, Optional, Dict, Type

from src.features.enrichers.base import BaseEnricher
from src.config.unified_config_manager import UnifiedConfigManager
from src.core.logging.logger import ProjectLogger

# Import the necessary selectors and enrichers to make the orchestrator aware of them
from src.features.selection.volatility_driver_selector import VolatilityDriverSelector
from src.features.enrichers.context_map_enricher import ContextMapEnricher

logger = ProjectLogger.get_logger("FeatureOrchestrator")

class FeatureOrchestrator:
    """
    Orchestrates the feature enrichment process, now with an explicit, configurable
    step for dynamic context feature selection.
    """

    def __init__(self, enrichers: List[BaseEnricher], config_manager: Optional[UnifiedConfigManager] = None):
        self.enrichers = sorted(enrichers, key=lambda e: (e.priority, getattr(e, 'name', type(e).__name__)))
        self.config_manager = config_manager or UnifiedConfigManager()
        final_order = [e.name for e in self.enrichers]
        logger.info(f"FeatureOrchestrator initialized. Execution order: {final_order}")

    @staticmethod
    def create_from_config(config_manager: UnifiedConfigManager) -> 'FeatureOrchestrator':
        """
        Dynamically discovers and instantiates enabled enrichers from the config.
        """
        enabled_enrichers = []
        package_path = os.path.join(os.path.dirname(__file__), 'enrichers')
        package_name = 'src.features.enrichers'

        logger.info(f"🔍 Discovering enrichers in '{package_name}'...")
        for _, module_name, _ in pkgutil.iter_modules([package_path]):
            full_module_name = f"{package_name}.{module_name}"
            try:
                module = importlib.import_module(full_module_name)
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, BaseEnricher) and obj is not BaseEnricher:
                        try:
                            # Use getattr to safely access the 'name' class attribute
                            # Create instance to get the name property correctly
                            try:
                                temp_instance = obj()
                                enricher_id = temp_instance.name
                            except Exception:
                                # Fallback to class name
                                enricher_id = name.lower()
                            # Check both old and new config paths for compatibility
                            old_config = config_manager.get_config('features', {}).get('enrichers', {}).get(enricher_id, {})
                            new_config = config_manager.get_config('features', {}).get('enabled_enrichers', {}).get(enricher_id, False)
                            
                            # Debug logging
                            logger.debug(f"📋 Enricher '{enricher_id}': old_config={old_config}, new_config={new_config}")
                            
                            # Check if enabled in either config
                            old_enabled = old_config.get('enabled', False) if isinstance(old_config, dict) else False
                            new_enabled = new_config if isinstance(new_config, bool) else False
                            enabled = old_enabled or new_enabled
                            
                            logger.debug(f"📋 Enricher '{enricher_id}': old_enabled={old_enabled}, new_enabled={new_enabled}, enabled={enabled}")
                            if enabled:
                                try:
                                    try:
                                        sig = inspect.signature(obj.__init__)
                                        expects_args = len(sig.parameters) > 1
                                    except Exception:
                                        expects_args = False
                                        
                                    if expects_args:
                                        # Enricher expects config parameter
                                        # ✅ FIX: Використовуємо old_config якщо він є dict, інакше порожній dict
                                        enricher_config = old_config if isinstance(old_config, dict) else {}
                                        instance = obj(enricher_config)
                                    else:
                                        # Enricher has no-argument constructor
                                        instance = obj()
                                    enabled_enrichers.append(instance)
                                    logger.info(f"✅ Enricher '{enricher_id}' (class: {name}) is ENABLED and instantiated.")
                                except Exception as e:
                                    logger.error(f"❌ Failed to instantiate enricher {name}: {e}", exc_info=True)
                            else:
                                logger.debug(f"⏭️  Enricher '{enricher_id}' is DISABLED in config.")
                        except Exception as e:
                            logger.error(f"Failed to instantiate enricher {name}: {e}", exc_info=True)
            except ImportError as e:
                logger.error(f"Failed to import module {full_module_name}: {e}")

        logger.info(f"✅ Discovered {len(enabled_enrichers)} enabled enrichers")
        enabled_enrichers = FeatureOrchestrator._dedupe_enrichers(enabled_enrichers)
        return FeatureOrchestrator(enabled_enrichers, config_manager=config_manager)

    @staticmethod
    def _dedupe_enrichers(enrichers: List[BaseEnricher]) -> List[BaseEnricher]:
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

    def run(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Runs the full enrichment pipeline, including dynamic context selection.
        """
        logger.info(f"🔄 Початок збагачення: {df.shape[0]} рядків, {df.shape[1]} колонок")
        
        df_enriched = df.copy()
        run_kwargs = kwargs.copy()

        # 1. Dynamic Context Feature Selection (if enabled)
        context_selection_config = self.config_manager.get_config('features', {}).get('context_selection', {})
        if context_selection_config.get('enabled', False):
            df_enriched = self._run_dynamic_context_selection(df_enriched, context_selection_config, run_kwargs)

        # 2. Run All Enabled Enrichers in Prioritized Order
        enrichment_stats = []
        for enricher in self.enrichers:
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
                logger.info(f"✅ Enricher '{enricher.name}' завершено: +{cols_added} колонок за {duration:.2f}s")
                
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
                    logger.warning(f"   Removing duplicates (keeping first occurrence)...")
                    df_enriched = df_enriched.loc[:, ~df_enriched.columns.duplicated()]
            except Exception as e:
                logger.error(f"❌ Error in enricher '{enricher.name}': {e}", exc_info=True)
                raise

        # 3. Final Model Feature Selection (e.g., using SmartFeatureSelector)
        # This step is conceptually separate and would be called by a higher-level process
        # after the main enrichment is complete.

        logger.info("✅ Feature enrichment pipeline completed.")
        logger.info(f"📊 Фінальний результат: {df_enriched.shape[0]} рядків, {df_enriched.shape[1]} колонок")
        
        # Логуємо статистику збагачення
        total_duration = sum(s['duration_sec'] for s in enrichment_stats)
        logger.info(f"⏱️ Загальний час збагачення: {total_duration:.2f}s")
        for stat in enrichment_stats:
            logger.debug(f"   {stat['enricher']}: {stat['duration_sec']:.2f}s, +{stat['cols_added']} cols")
        
        return df_enriched

    def _run_dynamic_context_selection(self, df: pd.DataFrame, config: Dict[str, Any], run_kwargs: Dict[str, Any]) -> pd.DataFrame:
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
