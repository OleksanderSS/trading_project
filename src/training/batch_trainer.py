# src/training/batch_trainer.py

import pandas as pd
import numpy as np
import os
import time
import joblib
from datetime import datetime
from typing import List, Dict, Any, Optional
from joblib import Parallel, delayed
from pathlib import Path

from src.core.logging.logger import ProjectLogger
from src.config.unified_config_manager import UnifiedConfigManager
from src.factories.model_factory import ModelFactory
from src.meta_learning.memory.diary_engine import DiaryEngine
from src.metrics.model.ml_evaluator import MLEvaluator
from src.training.constants import (
    BATCH_TRAINER_DEFAULT_BATCH_SIZE,
    BATCH_TRAINER_DEFAULT_MAX_MEMORY_GB
)
from src.training.base_trainer import BaseTrainer, TrainerConfig

logger = ProjectLogger.get_logger("BatchTrainer")

class BatchTrainer(BaseTrainer):
    """
    Advanced parallelized batch trainer.
    
    Trains all tickers in a single batch using parallel execution.
    Extends BaseTrainer with batch-specific grouping and parallel training logic.
    """

    def __init__(self, config: Optional[TrainerConfig] = None):
        super().__init__(config or TrainerConfig(
            batch_size=BATCH_TRAINER_DEFAULT_BATCH_SIZE,
            max_memory_gb=BATCH_TRAINER_DEFAULT_MAX_MEMORY_GB
        ))
    
    def _prepare_ticker_groups(self, plan: Dict[str, Any]) -> List[List[str]]:
        """
        Batch trainer: All tickers in one group.
        
        The entire set of tickers is trained in parallel as a single batch.
        """
        tickers = plan.get('tickers', [])
        if not tickers:
            return []
        return [tickers]  # All tickers in one group
    
    def _train_ticker_group(self, ticker_group: List[str], data_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train a group of tickers in parallel.
        
        Uses joblib.Parallel with delayed execution for parallelization.
        Automatically uses all cores (-1) when multiple tickers, single core (1) otherwise.
        """
        if not ticker_group:
            return {}
        
        n_jobs = -1 if len(ticker_group) > 1 else 1
        self.logger.info(f"Parallel training {len(ticker_group)} tickers (n_jobs={n_jobs})")
        
        batch_results = Parallel(n_jobs=n_jobs)(
            delayed(self._train_ticker_suite)(
                ticker=ticker,
                data=data_context
            )
            for ticker in ticker_group
        )
        
        return dict(zip(ticker_group, batch_results))

    def create_batch_plan(self, tickers: List[str], strategy: str = 'batch') -> Dict[str, Any]:
        """
        Create a batch training plan.
        
        Args:
            tickers: List of tickers to train
            strategy: Training strategy name
        
        Returns:
            Training plan dictionary
        """
        return {
            "tickers": tickers,
            "strategy": strategy,
            "context_fingerprint": "batch_training"
        }
