"""
Unified Training Manager for Large Ticker Sets
"""

import os
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.config.unified_config_manager import UnifiedConfigManager
from src.training.batch_trainer import BatchTrainer, BatchConfig
from src.training.progressive_trainer import ProgressiveTrainer, ProgressiveConfig
from src.scripts.colab.auto_colab_sync import ColabOptimizer, ColabConfig
from src.analytics.context.contextual_model_selector import ContextSubModelSelector
from src.analytics.arena.arena_battle import ArenaBattle, get_trading_arena

logger = ProjectLogger.get_logger("UnifiedTrainingManager")

class TrainingStrategy(Enum):
    BATCH = "batch"
    PROGRESSIVE = "progressive"
    COLAB = "colab"
    HYBRID = "hybrid"

@dataclass
class UnifiedConfig:
    strategy: TrainingStrategy = TrainingStrategy.HYBRID
    batch_size: int = 10
    max_memory_gb: float = 12.0
    initial_batch_size: int = 5
    max_batch_size: int = 20
    growth_factor: float = 1.5
    max_tickers_per_run: int = 20
    max_time_hours: float = 10.0
    min_accuracy: float = 0.75
    max_loss: float = 0.5
    max_total_time_hours: float = 24.0
    checkpoint_interval: int = 5

class UnifiedTrainingManager:
    
    def __init__(self, config: UnifiedConfig = None):
        self.config = config or UnifiedConfig()
        self.config_manager = UnifiedConfigManager()
        self.logger = logger
        
        system_config = self.config_manager.get_config('system') or {}
        self.base_dir = Path(system_config.get('unified_models_path', 'models/unified'))
        self.plans_dir = self.base_dir / "plans"
        self.results_dir = self.base_dir / "results"
        self.checkpoints_dir = self.base_dir / "checkpoints"
        
        for dir_path in [self.base_dir, self.plans_dir, self.results_dir, self.checkpoints_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.trainers = {}
        self.context_selector = ContextSubModelSelector()
        self.arena = get_trading_arena()
        self._initialize_trainers()
    
    def _initialize_trainers(self):
        batch_config = BatchConfig(batch_size=self.config.batch_size, max_memory_gb=self.config.max_memory_gb)
        self.trainers[TrainingStrategy.BATCH] = BatchTrainer(batch_config)
        
        progressive_config = ProgressiveConfig(
            initial_batch_size=self.config.initial_batch_size,
            max_batch_size=self.config.max_batch_size,
            growth_factor=self.config.growth_factor,
            min_accuracy_threshold=self.config.min_accuracy,
            max_loss_threshold=self.config.max_loss,
            max_time_hours=self.config.max_total_time_hours
        )
        self.trainers[TrainingStrategy.PROGRESSIVE] = ProgressiveTrainer(progressive_config)
        
        colab_config = ColabConfig(
            max_tickers_per_run=self.config.max_tickers_per_run,
            max_memory_usage_gb=self.config.max_memory_gb,
            cleanup_memory=True
        )
        self.trainers[TrainingStrategy.COLAB] = ColabOptimizer(colab_config)
    
    def execute_unified_training(self, tickers: List[str], data_context: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info(f"Starting unified DEAN-aware training for {len(tickers)} tickers")
        
        plan = self.create_unified_plan(tickers)
        self.save_unified_plan(plan)
        strategy = TrainingStrategy(plan["strategy"])
        
        results = {"strategy": strategy.value, "tickers_results": {}, "training_summary": {}}
        
        for ticker in tickers:
            models_to_train = self._select_models_for_ticker(ticker)
            plan["ticker_plans"][ticker] = {"models": models_to_train}
            self.logger.info(f"Selected models for {ticker}: {models_to_train}")

        if strategy == TrainingStrategy.BATCH:
            results.update(self.trainers[strategy].execute_batch_training(plan, data_context=data_context))
        elif strategy == TrainingStrategy.PROGRESSIVE:
            results.update(self.trainers[strategy].execute_progressive_training(tickers, data_context=data_context))
        elif strategy == TrainingStrategy.COLAB:
            results.update(self._execute_colab_training(plan))
        elif strategy == TrainingStrategy.HYBRID:
            results.update(self._execute_hybrid_training(plan, data_context=data_context))
        
        if results.get("tickers_results"):
            self.logger.info("Initiating Arena Battle for trained models...")
            battle_results = self.arena.run_battle(results["tickers_results"])
            results["arena_rankings"] = battle_results

        self.save_unified_results(results)
        return results

    def _select_models_for_ticker(self, ticker: str) -> List[str]:
        assets_config = self.config_manager.get_config('assets', {}).get(ticker, {})
        regime = self.config_manager.get_config('market_regime', 'neutral')
        
        models = ["random_forest", "lightgbm"]
        
        if assets_config.get('sector') == 'tech' or regime == 'volatile':
            models.append("xgboost")
            
        models.append("cnn")
        
        return models

    def create_unified_plan(self, tickers: List[str]) -> Dict[str, Any]:
        analysis = self.analyze_ticker_set(tickers)
        strategy_str = self.config.strategy.value if self.config.strategy != TrainingStrategy.HYBRID else analysis["recommended_strategy"]
        strategy = TrainingStrategy(strategy_str)

        plan = {}
        if strategy == TrainingStrategy.BATCH:
            plan = self.trainers[strategy].create_batch_plan(tickers, "balanced")
        elif strategy == TrainingStrategy.PROGRESSIVE:
            plan = self._create_progressive_plan(tickers)
        elif strategy == TrainingStrategy.COLAB:
            plan = self.trainers[strategy].create_colab_training_plan(tickers)
        else:
            plan = self._create_hybrid_plan(tickers, analysis)
        
        plan.update({
            "analysis": analysis, 
            "strategy": strategy.value, 
            "timestamp": datetime.now().isoformat(),
            "ticker_plans": {}
        })
        return plan

    def analyze_ticker_set(self, tickers: List[str]) -> Dict[str, Any]:
        # This is a placeholder for a more complex analysis
        return {
            "recommended_strategy": "hybrid"
        }

    def _create_progressive_plan(self, tickers: List[str]) -> Dict[str, Any]:
        trainer = self.trainers[TrainingStrategy.PROGRESSIVE]
        batches = trainer.create_progressive_batches(tickers)
        return {
            "total_tickers": len(tickers), "total_batches": len(batches), "strategy": "progressive",
            "batches": [{"batch_id": i+1, "tickers": b} for i, b in enumerate(batches)],
        }

    def _create_hybrid_plan(self, tickers: List[str], analysis: Dict[str, Any]) -> Dict[str, Any]:
        # This is a placeholder for a more complex analysis
        return {
            "total_tickers": len(tickers),
            "strategy": "hybrid",
            "phases": []
        }

    def _execute_colab_training(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        notebook_path = self.trainers[TrainingStrategy.COLAB].create_colab_notebook(plan)
        return {"strategy": "colab", "notebook_path": notebook_path, "plan": plan, "status": "notebook_created"}

    def _execute_hybrid_training(self, plan: Dict[str, Any], data_context: Dict[str, Any]) -> Dict[str, Any]:
        # This is a placeholder for a more complex execution
        return {
            "strategy": "hybrid",
            "phases": [],
            "plan": plan,
        }

    def save_unified_plan(self, plan: Dict[str, Any]) -> str:
        filepath = self.plans_dir / f"unified_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filepath, 'w') as f: json.dump(plan, f, indent=2)
        return str(filepath)

    def save_unified_results(self, results: Dict[str, Any]) -> str:
        filepath = self.results_dir / f"unified_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filepath, 'w') as f: json.dump(results, f, indent=2)
        return str(filepath)
