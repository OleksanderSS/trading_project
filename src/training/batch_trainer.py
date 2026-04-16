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

logger = ProjectLogger.get_logger("BatchTrainer")

class BatchConfig:
    """Configuration for Batch Training"""
    def __init__(self, batch_size: int = 10, max_memory_gb: float = 12.0):
        self.batch_size = batch_size
        self.max_memory_gb = max_memory_gb

class BatchTrainer:
    """Advanced parallelized batch trainer"""

    def __init__(self, config: Optional[BatchConfig] = None):
        self.config = config or BatchConfig()
        self.config_manager = UnifiedConfigManager()
        self.model_factory = ModelFactory()
        self.diary = DiaryEngine()
        self.evaluator = MLEvaluator()
        
        models_path = self.config_manager.get('paths.models', None) or self.config_manager.get_config('system', {}).get('models_path', 'data/trained_models')
        self.output_dir = Path(models_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def execute_batch_training(self, plan: Dict[str, Any], data_context: Dict[str, Any]) -> Dict[str, Any]:
        """Executes the batch training process based on the provided plan and data context."""
        tickers = plan.get('tickers', [])
        if not tickers or not data_context:
            logger.warning("No tickers or data context found in training plan.")
            return {"status": "failed", "reason": "no_tickers_or_data"}

        logger.info(f"Starting batch training for {len(tickers)} tickers. Strategy: {plan.get('strategy')}")

        n_jobs = -1 if len(tickers) > 1 else 1
        
        batch_results = Parallel(n_jobs=n_jobs)(
            delayed(self._train_ticker_suite)(
                ticker=ticker,
                data=data_context,  # Pass the prepared data
                plan=plan
            )
            for ticker in tickers
        )

        results = {ticker: result for ticker, result in zip(tickers, batch_results)}

        summary = self._generate_summary(results)
        return {
            "status": "success",
            "tickers_results": results,
            "training_summary": summary
        }

    def _train_ticker_suite(self, ticker: str, data: Dict[str, Any], plan: Dict) -> Dict:
        """Trains all configured models and targets for a specific ticker using prepared data."""
        ticker_results = {"status": "success", "models": [], "metrics": {}}
        
        X_train = data.get('X_train')
        y_train = data.get('y_train')
        X_test = data.get('X_test')
        y_test = data.get('y_test')
        target_name = data.get('target_name')

        if X_train is None or y_train is None:
            return {"status": "failed", "reason": "incomplete_data"}
        
        is_classification = 'classification' in data.get('target_type', '')
        model_types = self.config_manager.get_config('models.enabled_types', ['lgbm', 'rf', 'xgb', 'linear'])
        
        best_score = -np.inf
        winner_name = None
        
        for m_type in model_types:
            try:
                model_instance = self.model_factory.create_model(
                    model_type=m_type,
                    is_classification=is_classification,
                    params=self.config_manager.get_config(f"models.{m_type}", {})
                )
                
                # Real training on prepared data
                model_instance.fit(X_train, y_train)
                
                predictions = model_instance.predict(X_test)
                score = self.evaluator.evaluate(y_test, predictions, is_classification)
                
                # ✅ Зберігаємо metrics для кожної моделі
                # Для regression: accuracy = -mse (чим менше mse, тим краще)
                # Для classification: accuracy = accuracy_score
                if is_classification:
                    metrics_dict = {
                        'score': float(score),
                        'accuracy': float(score),
                        'mse': None
                    }
                else:
                    # For regression, use -mse as accuracy (higher is better)
                    metrics_dict = {
                        'score': float(score),
                        'accuracy': float(-score),  # -(-mse) = mse
                        'mse': float(score)
                    }
                
                ticker_results['metrics'][m_type] = metrics_dict
                
                if score > best_score:
                    best_score = score
                    winner_name = m_type
                    self._save_champion(model_instance, ticker, target_name)

                self.diary.log_event(
                    ticker=ticker,
                    model_name=m_type,
                    target=target_name,
                    metrics=score,
                    context_fingerprint=plan.get('context_fingerprint', 'default')
                )

            except Exception as e:
                logger.error(f"Failed to train {m_type} for {ticker} on {target_name}: {e}")

        ticker_results['winner'] = winner_name
        ticker_results['best_score'] = best_score
        # ✅ Додаємо metrics чемпіона
        ticker_results['winner_metrics'] = ticker_results['metrics'].get(winner_name, {})
        
        return ticker_results

    def _save_champion(self, model: Any, ticker: str, target: str):
        filename = f"CHAMP_{ticker}_{target}.joblib"
        path = self.output_dir / filename
        try:
            joblib.dump(model, path)
            logger.info(f"Champion saved: {path}")
        except Exception as e:
            logger.error(f"Error saving champion {filename}: {e}")

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        total = len(results)
        successful = sum(1 for r in results.values() if r.get('status') == 'success')
        
        return {
            "total_tickers": total,
            "successful_tickers": successful,
            "failed_tickers": total - successful,
            "average_score": np.mean([r.get('best_score', 0) for r in results.values()]),
            "timestamp": datetime.now().isoformat()
        }

    def create_batch_plan(self, tickers: List[str], strategy: str) -> Dict[str, Any]:
        # Placeholder for plan creation
        return {
            "tickers": tickers,
            "strategy": strategy,
        }
