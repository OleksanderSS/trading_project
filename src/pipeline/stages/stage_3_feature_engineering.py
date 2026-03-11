# src/pipeline/stages/stage_3_feature_engineering.py

import logging
import os
from typing import Optional, Any, Dict, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import psutil
from pathlib import Path

from src.pipeline.stages.base_stage import BaseStage
from src.config.unified_config_manager import UnifiedConfigManager
from src.core.error_handling.error_handler import ErrorHandler
from src.features.feature_orchestrator import FeatureOrchestrator
from src.features.selection.smart_selector import SmartFeatureSelector
from src.targets.target_orchestrator import TargetOrchestrator
from src.utils.trading_calendar import TradingCalendar
from src.core.logging.logger import ProjectLogger

# Advanced financial and context modules
from src.analytics.calculators.fama_french_factors import FamaFrenchFactors
from src.analytics.analyzers.hedge_fund_analyzer import HedgeFundAnalyzer
from src.analytics.analyzers.news_impact_analyzer import NewsImpactAnalyzer
from src.meta_learning.awareness.context_engine import ContextAwarenessEngine
from src.analytics.detectors.critical_signal_detector import CriticalSignalDetector
from src.analytics.analyzers.knn_similarity_finder import KnnSimilarityFinder
from src.analytics.analyzers.causal_event_finder import CausalEngine

logger = ProjectLogger.get_logger("FeatureEngineeringStage")

class FeatureEngineeringStage(BaseStage):
    """
    Stage 3: Advanced Feature Engineering Hub.
    Uses FeatureOrchestrator for modular enrichment and TargetOrchestrator for unified labeling.
    Leverages SmartFeatureSelector for final model feature selection.
    """
    def __init__(self, config_manager: UnifiedConfigManager, error_handler: ErrorHandler, **kwargs):
        super().__init__(config_manager, error_handler, **kwargs)
        self.feature_config = self.config_manager.get_config('features', default={})
        self.calendar = TradingCalendar()
        
        # Initialize dynamic Feature Orchestrator
        self.orchestrator = FeatureOrchestrator.create_from_config(config_manager)
        
        # Initialize TargetOrchestrator with the list of targets
        targets_list = self.config_manager.get('targets').as_dict() if hasattr(self.config_manager.get('targets'), 'as_dict') else self.config_manager.get('targets')
        self.target_orchestrator = TargetOrchestrator(targets_list=targets_list)

        self.selector = SmartFeatureSelector()

        self.output_dir = Path('data/processed/features')
        self.reports_dir = Path('reports/features')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.master_features_path = self.output_dir / "enriched_features.parquet"

    async def run(self, **kwargs) -> Dict[str, Any]:
        """
        Runs the streamlined feature engineering pipeline:
        1. Enrichment via FeatureOrchestrator.
        2. Target Generation.
        3. Final Feature Selection for the model.
        """
        cleaned_data = kwargs.get('cleaned_data')
        if not cleaned_data:
            logger.warning("Missing cleaned data for feature generation.")
            return {}

        logger.info(f"Starting Feature Engineering Pipeline. System RAM: {psutil.virtual_memory().percent}%")
        market_data = cleaned_data.get('market_data')

        if market_data is None or market_data.empty:
            logger.error("Market data is required for feature engineering.")
            return {}

        try:
            logger.info("Running Feature Orchestrator for data enrichment...")
            df_enriched = self.orchestrator.run(
                market_data, 
                **cleaned_data
            )

            logger.info("Generating targets based on unified configuration...")
            df_with_targets = self._generate_targets(df_enriched)

            # Load existing features if available
            if self.master_features_path.exists():
                logger.info(f"Loading existing master features from {self.master_features_path}")
                master_features_df = pd.read_parquet(self.master_features_path)
                
                # Combine new data with existing
                df_with_targets = pd.concat([master_features_df, df_with_targets], ignore_index=True)
                df_with_targets.drop_duplicates(subset=['ticker', 'datetime'], keep='last', inplace=True)
                logger.info(f"Combined new and existing data. Total rows: {len(df_with_targets)}")

            processed_tickers = []
            version = datetime.now().strftime("%Y%m%d_%H%M")

            for ticker, group in df_with_targets.groupby('ticker'):
                logger.info(f"Processing final feature selection for Ticker: {ticker}")
                
                target_cols = [c for c in group.columns if c.startswith('target_')]
                if not target_cols:
                    logger.warning(f"No targets found for {ticker}, skipping selection.")
                    continue
                
                primary_target = target_cols[0]
                context_id = f"{ticker}_{primary_target}"
                
                df_task = group.dropna(subset=[primary_target]).fillna(0)
                
                exclude_metadata = ['datetime', 'ticker'] + target_cols
                feature_pool = [c for c in df_task.columns if c not in exclude_metadata]
                
                X = df_task[feature_pool]
                y = df_task[primary_target]
                
                selected_features = self.selector.select(X, y, context_id=context_id)
                
                final_cols = selected_features + target_cols + ['ticker', 'context_fingerprint']
                final_cols_exist = [c for c in final_cols if c in df_task.columns]

                final_df = df_task[final_cols_exist]
                processed_tickers.append(final_df)

            if processed_tickers:
                master_features_df = pd.concat(processed_tickers)
                master_features_df.to_parquet(self.master_features_path)
                
                logger.info(f"Feature Engineering complete. Master file saved: {self.master_features_path}")
                
                return {
                    'enriched_data': master_features_df,
                    'feature_version': version
                }

        except Exception as e:
            logger.error(f"Critical error in FeatureEngineeringStage: {e}", exc_info=True)
            raise

        return {"status": "failed", "reason": "no_data_processed"}

    def _generate_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generates targets using the configured TargetOrchestrator."""
        return self.target_orchestrator.generate_targets(df)
