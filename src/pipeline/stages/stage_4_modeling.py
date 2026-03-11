
"""
Stage 4: Modeling and Selection (Unified Advanced ML Arena)

Implements a hybrid workflow using all available architectures,
automated training management, and deep context analytics through the UnifiedTrainingManager.
"""

import logging
import os
import datetime
from typing import Optional, Any, Dict, Tuple, List
from pathlib import Path

import pandas as pd
import psutil

from src.pipeline.stages.base_stage import BaseStage
from src.config.unified_config_manager import UnifiedConfigManager
from src.core.logging.logger import ProjectLogger
from src.training.unified_training_manager import UnifiedTrainingManager, UnifiedConfig, TrainingStrategy
from src.models.adapters.data_preparation import prepare_data_for_models
from src.analytics.analyzers.model_comparison_analyzer import ModelComparisonAnalyzer

logger = ProjectLogger.get_logger("ModelingStage")

class ModelingStage(BaseStage):
    """
    Modeling stage: uses UnifiedTrainingManager for training orchestration
    and prepare_data_for_models for unified dataset formatting.
    Supports dynamic switching between Light (local) and Heavy (Colab) training.
    """
    def __init__(self, config_manager: UnifiedConfigManager, brain: Dict[str, Any], **kwargs):
        super().__init__(config_manager, brain)
        self.modeling_config = self.config_manager.get_config('modeling') or {}
        self.system_config = self.config_manager.get_config('system') or {}
        
        # Initialize strategy based on config
        strategy_str = self.modeling_config.get('strategy', 'hybrid').upper()
        strategy = TrainingStrategy[strategy_str] if strategy_str in TrainingStrategy.__members__ else TrainingStrategy.HYBRID
        
        training_config = UnifiedConfig(
            strategy=strategy,
            batch_size=self.modeling_config.get('batch_size', 10),
            max_memory_gb=self.modeling_config.get('max_memory_gb', 12.0)
        )
        
        self.training_manager = UnifiedTrainingManager(training_config)
        self.comparison_analyzer = ModelComparisonAnalyzer()
        
        # Get paths from config
        self.models_dir = Path(self.system_config.get('models_path', 'src/trained_models'))
        self.diary_path = Path(self.system_config.get('diary_path', 'logs/experience_diary.csv'))

        self._init_infrastructure()

    def _init_infrastructure(self):
        """Initializes the environment for training artifacts."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.diary_path.exists():
            self.diary_path.parent.mkdir(parents=True, exist_ok=True)
            columns = [
                'timestamp', 'ticker', 'tf', 'target', 'model_name', 'context_fingerprint', 
                'is_champion', 'cpu_usage', 'ram_usage'
            ]
            pd.DataFrame(columns=columns).to_csv(self.diary_path, index=False)

    async def run(self, **kwargs) -> Dict[str, Any]:
        """Runs the full training cycle using UnifiedTrainingManager."""
        enriched_data = kwargs.get('enriched_data')
        if not enriched_data:
            logger.error("Enriched data not found in pipeline_data. Skipping Modeling Stage.")
            return {}

        champions = {}
        logger.info("--- [Modeling Stage] Starting Unified Training Flow ---")

        ticker_groups = enriched_data.groupby('ticker') if isinstance(enriched_data, pd.DataFrame) else enriched_data.items()

        for ticker, df in ticker_groups:
            try:
                target_cols = [c for c in df.columns if c.startswith('target_')]
                timeframe = df['timeframe'].iloc[-1] if 'timeframe' in df.columns else "1d"
                
                for target_name in target_cols:
                    context_key = f"{ticker}_{target_name}"
                    
                    # 1. Unified ML Data Preparation
                    prepared_data = prepare_data_for_models(
                        df=df,
                        ticker=ticker,
                        timeframe=timeframe,
                        target_cols=[target_name],
                        test_size=self.modeling_config.get('test_size', 0.2)
                    )

                    if not prepared_data:
                        logger.warning(f"Data preparation failed for {context_key}. Skipping.")
                        continue

                    # 2. Execute Training via Manager 
                    training_results = self.training_manager.execute_unified_training(
                        tickers=[ticker], 
                        data_context=prepared_data
                    )
                    
                    # 3. Model Comparison and Champion Selection
                    comparison_report = self.comparison_analyzer.compare_models(
                        training_results, 
                        market_context=self.brain.get('market_regime', 'neutral')
                    )
                    
                    # 4. Process Results
                    ticker_result = training_results.get('tickers_results', {}).get(ticker, {})
                    if ticker_result.get('status') == 'success':
                        context_fingerprint = df['context_fingerprint'].iloc[-1] if 'context_fingerprint' in df.columns else "unknown"
                        
                        champion_info = {
                            "ticker": ticker,
                            "target": target_name,
                            "winner": comparison_report.get('champion_model', ticker_result.get('winner')),
                            "champion_reason": comparison_report.get('selection_reason', 'Top accuracy'),
                            "context": context_fingerprint,
                            "timestamp": datetime.datetime.now().isoformat(),
                            "metrics": ticker_result.get('metrics', {}),
                            "model_path": ticker_result.get('model_path')
                        }
                        champions[context_key] = champion_info
                        
                        # 5. Log to Experience Diary
                        self._log_to_diary(champion_info, timeframe)

            except Exception as e:
                logger.error(f"Error during modeling for ticker {ticker}: {e}", exc_info=True)

        logger.info(f"Modeling Stage complete. Trained {len(champions)} champion models.")
        
        return {
            'models_metadata': champions,
            'processed_data': enriched_data
        }

    def _log_to_diary(self, info: Dict[str, Any], tf: str):
        """Records champion selection metadata."""
        entry = {
            'timestamp': info['timestamp'],
            'ticker': info['ticker'], 
            'tf': tf,
            'target': info['target'], 
            'model_name': info['winner'],
            'context_fingerprint': info['context'], 
            'is_champion': True,
            'cpu_usage': psutil.cpu_percent(), 
            'ram_usage': psutil.virtual_memory().percent
        }
        pd.DataFrame([entry]).to_csv(self.diary_path, mode='a', header=False, index=False)
