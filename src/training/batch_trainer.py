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
        self.model_factory = ModelFactory()
        self.diary = DiaryEngine()
        self.evaluator = MLEvaluator()
    
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
        
        return {ticker: result for ticker, result in zip(ticker_group, batch_results)}

    def _train_ticker_suite(self, ticker: str, data: Dict[str, Any]) -> Dict:
        """
        Train all configured models for a specific ticker.
        
        Args:
            ticker: Ticker symbol
            data: Prepared data with X_train, y_train, X_test, y_test, target_name, etc.
        
        Returns:
            Training result with winner, metrics, and best_score
        """
        ticker_results = {"status": "success", "models": [], "metrics": {}, "ticker": ticker}
        
        X_train = data.get('X_train')
        y_train = data.get('y_train')
        X_test = data.get('X_test')
        y_test = data.get('y_test')
        target_name = data.get('target_name', 'unknown')

        if X_train is None or y_train is None:
            ticker_results["status"] = "failed"
            ticker_results["reason"] = "incomplete_data"
            return ticker_results
        
        try:
            is_classification = 'classification' in data.get('target_type', '')
            model_types = self.config_manager.get_config('models.enabled_types', ['lgbm', 'rf', 'xgb', 'linear'])
            
            best_score = -np.inf
            winner_name = None
            
            for m_type in model_types:
                try:
                    # Create and train model
                    model_instance = self.model_factory.create_model(
                        model_type=m_type,
                        is_classification=is_classification,
                        params=self.config_manager.get_config(f"models.{m_type}", {})
                    )
                    
                    model_instance.fit(X_train, y_train)
                    predictions = model_instance.predict(X_test)
                    score = self.evaluator.evaluate(y_test, predictions, is_classification)
                    
                    # Store metrics
                    metrics_dict = {
                        'score': float(score),
                        'accuracy': float(score) if is_classification else float(-score),
                        'mse': float(score) if not is_classification else None
                    }
                    ticker_results['metrics'][m_type] = metrics_dict
                    
                    # Track best model
                    if score > best_score:
                        best_score = score
                        winner_name = m_type
                        self._save_champion(model_instance, ticker, target_name)
                    
                    # Log to diary
                    self.diary.log_event(
                        ticker=ticker,
                        model_name=m_type,
                        target=target_name,
                        metrics=float(score),
                        context_fingerprint=data.get('context_fingerprint', 'default')
                    )

                except Exception as e:
                    self.logger.error(f"Failed to train {m_type} for {ticker}: {e}")
                    continue
            
            ticker_results['winner'] = winner_name
            ticker_results['best_score'] = float(best_score) if best_score > -np.inf else None
            ticker_results['winner_metrics'] = ticker_results['metrics'].get(winner_name, {})
            
            return ticker_results
        
        except Exception as e:
            self.logger.error(f"Error during training for {ticker}: {e}")
            return {"status": "failed", "ticker": ticker, "reason": str(e)}

    def _save_champion(self, model: Any, ticker: str, target: str):
        """Save best model to disk"""
        filename = f"CHAMP_{ticker}_{target}.joblib"
        path = self.output_dir / filename
        try:
            joblib.dump(model, path)
            self.logger.debug(f"Champion saved: {path}")
        except Exception as e:
            self.logger.error(f"Error saving champion {filename}: {e}")

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
