"""
Unified Training Manager for Large Ticker Sets
"""

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from src.analytics.arena.arena_battle import get_trading_arena
from src.analytics.context.contextual_model_selector import ContextualModelSelector
from src.config.unified_config_manager import get_current_config
from src.core.logging.logger import ProjectLogger
from src.training.base_trainer import TrainerConfig
from src.training.batch_trainer import BatchTrainer
from src.training.progressive_trainer import ProgressiveTrainer

logger = ProjectLogger.get_logger("UnifiedTrainingManager")

class TrainingStrategy(Enum):
    BATCH = "batch"
    PROGRESSIVE = "progressive"
    HYBRID = "hybrid"

class UnifiedTrainingManager:
    """
    Unified manager for coordinating different training strategies (Batch, Progressive, Hybrid).
    Acts as a high-level orchestrator for the Modeling Stage.
    """

    def __init__(self, config: TrainerConfig | None = None):
        self.config = config or TrainerConfig()
        self.config_manager = get_current_config()
        self.logger = logger

        system_config = self.config_manager.get_config('system', {})
        self.base_dir = Path(system_config.get('unified_models_path', 'models/unified'))
        self.plans_dir = self.base_dir / "plans"
        self.results_dir = self.base_dir / "results"
        self.checkpoints_dir = self.base_dir / "checkpoints"

        for dir_path in [self.base_dir, self.plans_dir, self.results_dir, self.checkpoints_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.trainers: dict[str, Any] = {}
        # Contextual selector for intelligent model type selection
        all_models = [
            'lightgbm', 'xgboost', 'catboost', 'random_forest',
            'linear', 'svm', 'knn', 'mlp', 'lstm', 'gru',
            'transformer', 'cnn', 'tabnet', 'autoencoder', 'ensemble'
        ]
        self.context_selector = ContextualModelSelector(available_models=all_models)
        self.arena = get_trading_arena()
        self._initialize_trainers()

    def _initialize_trainers(self):
        """Initialize all available trainer implementations."""
        # Batch Trainer
        batch_config = TrainerConfig(
            batch_size=self.config.batch_size,
            max_memory_gb=self.config.max_memory_gb
        )
        self.trainers[TrainingStrategy.BATCH.value] = BatchTrainer(batch_config)

        # Progressive Trainer (Preferred for large datasets)
        progressive_config = TrainerConfig(
            initial_batch_size=self.config.initial_batch_size,
            max_batch_size=self.config.max_batch_size,
            growth_factor=self.config.growth_factor,
            min_accuracy_threshold=self.config.min_accuracy_threshold,
            max_loss_threshold=self.config.max_loss_threshold,
            max_time_hours=self.config.max_time_hours,
            enable_adaptive_batching=self.config.enable_adaptive_batching
        )
        self.trainers[TrainingStrategy.PROGRESSIVE.value] = ProgressiveTrainer(progressive_config)

    def execute_unified_training(self, tickers: list[str], data_context: dict[str, Any]) -> dict[str, Any]:
        """
        Execute the training cycle based on the best strategy for the given ticker set.
        """
        self.logger.info(f"🚀 Starting unified training for {len(tickers)} tickers")

        plan = self.create_unified_plan(tickers)
        self.save_unified_plan(plan)
        strategy_val = plan["strategy"]
        strategy = TrainingStrategy(strategy_val)

        # Prepare individual ticker plans (which models to train for each)
        for ticker in tickers:
            models_to_train = self._select_models_for_ticker(ticker, data_context)
            plan["ticker_plans"][ticker] = {"models": models_to_train}

        results: dict[str, Any] = {
            "strategy": strategy.value,
            "tickers_results": {},
            "training_summary": {},
            "timestamp": datetime.now().isoformat()
        }

        # Delegate execution to the selected trainer
        try:
            if strategy == TrainingStrategy.BATCH:
                results.update(self.trainers[strategy.value].execute_batch_training(plan, data_context))
            elif strategy == TrainingStrategy.PROGRESSIVE:
                results.update(self.trainers[strategy.value].execute_progressive_training(plan, data_context))
            elif strategy == TrainingStrategy.HYBRID:
                results.update(self._execute_hybrid_training(plan, data_context))
        except Exception as e:
            self.logger.error(f"❌ Training execution failed: {e}", exc_info=True)
            results["status"] = "failed"
            results["error"] = str(e)
            return results

        # Run Arena Battle for benchmarking if we have results and targets
        if results.get("tickers_results") and "y_test" in data_context:
            self.logger.info("⚔️ Initiating Arena Battle for benchmarking...")
            try:
                # Actual targets are needed for the arena to score models
                battle_results = self.arena.run_battle(
                    results["tickers_results"],
                    actual_targets=data_context["y_test"]
                )
                results["arena_rankings"] = battle_results
                self.logger.info("✅ Arena Battle completed.")
            except Exception as e:
                self.logger.warning(f"⚠️ Arena battle failed: {e}")

        self.save_unified_results(results)
        return results

    def _select_models_for_ticker(self, ticker: str, data: dict[str, Any] | None = None) -> list[str]:
        """Select optimal models for a ticker using contextual analysis."""
        # Use ContextualModelSelector if possible, otherwise fallback to config
        context_fingerprint = data.get('context_fingerprint', 'default') if data else 'default'

        try:
            recommended = self.context_selector.select_models(ticker, context_fingerprint)
            if recommended:
                return recommended
        except Exception:
            pass

        # Fallback to config
        models_config = self.config_manager.get_config('models', {})
        return models_config.get('enabled_types', ["lightgbm", "catboost", "xgboost", "random_forest"])

    def create_unified_plan(self, tickers: list[str]) -> dict[str, Any]:
        """Analyze the task and create a training plan."""
        analysis = self._analyze_ticker_set(tickers)

        # Override strategy if requested in config, otherwise use recommendation
        strategy_str = self.config.strategy if hasattr(self.config, 'strategy') and self.config.strategy else analysis["recommended_strategy"]
        if isinstance(strategy_str, TrainingStrategy):
            strategy_str = strategy_str.value

        strategy = TrainingStrategy(strategy_str)

        if strategy == TrainingStrategy.BATCH:
            plan = self.trainers[strategy.value]._prepare_ticker_groups({"tickers": tickers}) # Simplistic
            plan = {"strategy": "batch", "tickers": tickers, "groups": plan}
        elif strategy == TrainingStrategy.PROGRESSIVE:
            plan = self._create_progressive_plan(tickers)
        else:
            plan = {"strategy": strategy.value, "tickers": tickers}

        plan.update({
            "analysis": analysis,
            "strategy": strategy.value,
            "timestamp": datetime.now().isoformat(),
            "ticker_plans": {}
        })
        return plan

    def _analyze_ticker_set(self, tickers: list[str]) -> dict[str, Any]:
        """Analyze the ticker set to recommend the best strategy."""
        count = len(tickers)
        # Recommendation logic: progressive for many tickers, batch for few
        recommended = "progressive" if count > 5 else "batch"
        return {
            "ticker_count": count,
            "recommended_strategy": recommended,
            "complexity_estimate": "high" if count > 20 else "medium"
        }

    def _create_progressive_plan(self, tickers: list[str]) -> dict[str, Any]:
        trainer = self.trainers[TrainingStrategy.PROGRESSIVE.value]
        batches = trainer.create_progressive_batches(tickers)
        return {
            "total_tickers": len(tickers),
            "total_batches": len(batches),
            "strategy": "progressive",
            "batches": [{"batch_id": i+1, "tickers": b} for i, b in enumerate(batches)],
        }

    def _create_hybrid_plan(self, tickers: list[str]) -> dict[str, Any]:
        # This is a placeholder for a more complex analysis
        return {
            "strategy": "hybrid",
            "phases": [],
            "tickers": tickers
        }


    def _execute_hybrid_training(self, plan: dict[str, Any], data_context: dict[str, Any]) -> dict[str, Any]:
        """Execute hybrid training. Автоматично вибирає кращу локальну стратегію."""
        num_tickers = len(plan.get("ticker_plans", {}))
        strategy = TrainingStrategy.BATCH if num_tickers < 50 else TrainingStrategy.PROGRESSIVE

        self.logger.info(f"Executing HYBRID training: використовую {strategy.value} для локальної оптимізації.")
        result = self.trainers[strategy.value].execute_batch_training(plan, data_context=data_context)
        return dict(result) if result else {}

    def save_unified_plan(self, plan: dict[str, Any]) -> str:
        filepath = self.plans_dir / f"unified_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filepath, 'w') as f: json.dump(plan, f, indent=2)
        return str(filepath)

    def save_unified_results(self, results: dict[str, Any]) -> str:
        filepath = self.results_dir / f"unified_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filepath, 'w') as f: json.dump(results, f, indent=2)
        return str(filepath)
