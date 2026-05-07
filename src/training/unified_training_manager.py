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

from src.config.unified_config_manager import get_current_config
from src.training.batch_trainer import BatchTrainer
from src.training.progressive_trainer import ProgressiveTrainer
from src.training.base_trainer import TrainerConfig

# Dummy classes to replace missing ColabOptimizer removed
from src.analytics.context.contextual_model_selector import ContextualModelSelector
from src.analytics.arena.arena_battle import get_trading_arena

logger = ProjectLogger.get_logger("UnifiedTrainingManager")

class TrainingStrategy(Enum):
    BATCH = "batch"
    PROGRESSIVE = "progressive"
    # COLAB removed
    HYBRID = "hybrid"

class UnifiedTrainingManager:
    
    def __init__(self, config: Optional[TrainerConfig] = None):
        self.config = config or TrainerConfig()
        self.config_manager = get_current_config()
        self.logger = logger
        
        system_config = self.config_manager.get_config('system') or {}
        self.base_dir = Path(system_config.get('unified_models_path', 'models/unified'))
        self.plans_dir = self.base_dir / "plans"
        self.results_dir = self.base_dir / "results"
        self.checkpoints_dir = self.base_dir / "checkpoints"
        
        for dir_path in [self.base_dir, self.plans_dir, self.results_dir, self.checkpoints_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.trainers: Dict[str, Any] = {}
        self.context_selector = ContextualModelSelector(['LSTM', 'RandomForest'])
        self.arena = get_trading_arena()
        self._initialize_trainers()
    
    def _initialize_trainers(self):
        batch_config = TrainerConfig(
            batch_size=self.config.batch_size, 
            max_memory_gb=self.config.max_memory_gb
        )
        self.trainers[TrainingStrategy.BATCH.value] = BatchTrainer(batch_config)
        
        progressive_config = TrainerConfig(
            initial_batch_size=self.config.initial_batch_size,
            max_batch_size=self.config.max_batch_size,
            growth_factor=self.config.growth_factor,
            min_accuracy_threshold=self.config.min_accuracy_threshold,  # Fixed: was min_accuracy
            max_loss_threshold=self.config.max_loss_threshold,  # Fixed: was max_loss
            max_time_hours=self.config.max_time_hours  # Fixed: was max_total_time_hours
        )
        self.trainers[TrainingStrategy.PROGRESSIVE.value] = ProgressiveTrainer(progressive_config)
        
        # Colab strategy replaced by external orchestrator
    
    def execute_unified_training(self, tickers: List[str], data_context: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info(f"Starting unified DEAN-aware training for {len(tickers)} tickers")
        
        plan = self.create_unified_plan(tickers)
        self.save_unified_plan(plan)
        strategy = TrainingStrategy(plan["strategy"])
        
        results: dict[str, Any] = {"strategy": strategy.value, "tickers_results": {}, "training_summary": {}}
        
        for ticker in tickers:
            models_to_train = self._select_models_for_ticker(ticker)
            plan["ticker_plans"][ticker] = {"models": models_to_train}
            self.logger.info(f"Selected models for {ticker}: {models_to_train}")

        if strategy == TrainingStrategy.BATCH:
            results.update(self.trainers[strategy.value].execute_batch_training(plan, data_context=data_context))
        elif strategy == TrainingStrategy.PROGRESSIVE:
            results.update(self.trainers[strategy.value].execute_progressive_training(plan, data_context=data_context))
        # Removed Colab training fallback
        elif strategy == TrainingStrategy.HYBRID:
            results.update(self._execute_hybrid_training(plan, data_context=data_context))
        
        if results.get("tickers_results"):
            self.logger.info("Initiating Arena Battle for trained models...")
            # TODO: Fix run_battle signature - needs actual_targets parameter
            # battle_results = self.arena.run_battle(results["tickers_results"], actual_targets)
            # results["arena_rankings"] = battle_results
            self.logger.warning("Arena battle skipped - actual_targets parameter needed")

        self.save_unified_results(results)
        return results

    def _select_models_for_ticker(self, ticker: str) -> List[str]:
        """Select models to train for a specific ticker based on config and context."""
        # Get light models from config
        models_config = self.config_manager.get_config('models') or {}
        light_models = models_config.get('categories', {}).get('light', [])
        
        if not light_models:
            # Fallback to defaults if not configured
            light_models = ["random_forest", "lightgbm", "catboost", "xgboost"]
            
        self.logger.debug(f"Selected models for {ticker}: {light_models}")
        return list(light_models)  # Ensure it returns a list

    def create_unified_plan(self, tickers: List[str]) -> Dict[str, Any]:
        analysis = self.analyze_ticker_set()
        strategy_str = self.config.strategy.value if self.config.strategy != TrainingStrategy.HYBRID else analysis["recommended_strategy"]
        strategy = TrainingStrategy(strategy_str)

        plan = {}
        if strategy == TrainingStrategy.BATCH:
            plan = self.trainers[strategy.value].create_batch_plan(tickers, "balanced")
        elif strategy == TrainingStrategy.PROGRESSIVE:
            plan = self._create_progressive_plan(tickers)
        # Removed Colab plan logic
        else:
            plan = self._create_hybrid_plan(tickers)
        
        plan.update({
            "analysis": analysis, 
            "strategy": strategy.value, 
            "timestamp": datetime.now().isoformat(),
            "ticker_plans": {}
        })
        return plan

    def analyze_ticker_set(self) -> Dict[str, Any]:
        # This is a placeholder for a more complex analysis
        return {
            "recommended_strategy": "hybrid"
        }

    def _create_progressive_plan(self, tickers: List[str]) -> Dict[str, Any]:
        trainer = self.trainers[TrainingStrategy.PROGRESSIVE.value]
        batches = trainer.create_progressive_batches(tickers)
        return {
            "total_tickers": len(tickers), "total_batches": len(batches), "strategy": "progressive",
            "batches": [{"batch_id": i+1, "tickers": b} for i, b in enumerate(batches)],
        }

    def _create_hybrid_plan(self, tickers: List[str]) -> Dict[str, Any]:
        # This is a placeholder for a more complex analysis
        return {
            "strategy": "hybrid",
            "phases": [],
            "tickers": tickers
        }


    def _execute_hybrid_training(self, plan: Dict[str, Any], data_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute hybrid training. Currently falls back to batch training for local execution."""
        self.logger.info("Executing HYBRID training (falling back to BATCH for local components)")
        result = self.trainers[TrainingStrategy.BATCH.value].execute_batch_training(plan, data_context=data_context)
        return dict(result) if result else {}

    def save_unified_plan(self, plan: Dict[str, Any]) -> str:
        filepath = self.plans_dir / f"unified_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filepath, 'w') as f: json.dump(plan, f, indent=2)
        return str(filepath)

    def save_unified_results(self, results: Dict[str, Any]) -> str:
        filepath = self.results_dir / f"unified_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filepath, 'w') as f: json.dump(results, f, indent=2)
        return str(filepath)
