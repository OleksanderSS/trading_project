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
        self.enrichers = sorted(enrichers, key=lambda e: e.priority)
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

        logger.info(f"Discovering enrichers in '{package_name}'...")
        for _, module_name, _ in pkgutil.iter_modules([package_path]):
            full_module_name = f"{package_name}.{module_name}"
            try:
                module = importlib.import_module(full_module_name)
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, BaseEnricher) and obj is not BaseEnricher:
                        try:
                            # Use getattr to safely access the 'name' class attribute
                            enricher_id = getattr(obj, 'name', name.lower())
                            config = config_manager.get_config('features', {}).get('enrichers', {}).get(enricher_id, {})
                            if config.get('enabled', False):
                                enabled_enrichers.append(obj(config))
                                logger.info(f"Enricher '{enricher_id}' (class: {name}) is ENABLED.")
                            else:
                                logger.debug(f"Enricher '{enricher_id}' is DISABLED in config.")
                        except Exception as e:
                            logger.error(f"Failed to instantiate enricher {name}: {e}", exc_info=True)
            except ImportError as e:
                logger.error(f"Failed to import module {full_module_name}: {e}")

        return FeatureOrchestrator(enabled_enrichers, config_manager=config_manager)

    def run(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Runs the full enrichment pipeline, including dynamic context selection.
        """
        df_enriched = df.copy()
        run_kwargs = kwargs.copy()

        # 1. Dynamic Context Feature Selection (if enabled)
        context_selection_config = self.config_manager.get_config('features', {}).get('context_selection', {})
        if context_selection_config.get('enabled', False):
            df_enriched = self._run_dynamic_context_selection(df_enriched, context_selection_config, run_kwargs)

        # 2. Run All Enabled Enrichers in Prioritized Order
        for enricher in self.enrichers:
            try:
                logger.info(f"Running enricher '{enricher.name}' (Priority: {enricher.priority})...")
                df_enriched = enricher.enrich(df_enriched, **run_kwargs)
            except Exception as e:
                logger.error(f"Error in enricher '{enricher.name}': {e}", exc_info=True)
                raise

        # 3. Final Model Feature Selection (e.g., using SmartFeatureSelector)
        # This step is conceptually separate and would be called by a higher-level process
        # after the main enrichment is complete.

        logger.info("Feature enrichment pipeline completed.")
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
