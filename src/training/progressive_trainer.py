# src/training/progressive_trainer.py
"""
Progressive Training System for Large Ticker Sets
Enables adaptive batch processing and quality-controlled model training evolution.
"""

import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from src.config.unified_config_manager import get_current_config
from src.core.exceptions import ModelTrainingError
from src.core.logging.logger import ProjectLogger
from src.training.base_trainer import BaseTrainer, TrainerConfig
from src.training.batch.batch_processor import BatchProcessor
from src.training.constants import (
    PROGRESSIVE_BATCH_GROWTH_FACTOR,
    PROGRESSIVE_CHECKPOINT_INTERVAL,
    PROGRESSIVE_INITIAL_BATCH_SIZE,
    PROGRESSIVE_MAX_BATCH_SIZE,
    PROGRESSIVE_MAX_LOSS_THRESHOLD,
    PROGRESSIVE_MAX_MEMORY_GB,
    PROGRESSIVE_MAX_TIME_HOURS,
    PROGRESSIVE_MIN_ACCURACY_THRESHOLD,
)
from src.training.security.path_security_validator import PathSecurityValidator
from src.training.state.training_state_manager import TrainingStateManager

logger = ProjectLogger.get_logger("ProgressiveTrainer")

class ProgressiveTrainer(BaseTrainer):
    """
    Adaptive trainer designed for large-scale ticker universes.
    Iteratively evolves batch sizes based on real-time hardware performance and model stability.
    """

    def __init__(self, config: TrainerConfig | None = None):
        """Initializes the progressive training environment and local persistence paths."""
        # Create config with progressive defaults if not provided
        if config is None:
            config = TrainerConfig(
                initial_batch_size=PROGRESSIVE_INITIAL_BATCH_SIZE,
                max_batch_size=PROGRESSIVE_MAX_BATCH_SIZE,
                growth_factor=PROGRESSIVE_BATCH_GROWTH_FACTOR,
                min_accuracy_threshold=PROGRESSIVE_MIN_ACCURACY_THRESHOLD,
                max_loss_threshold=PROGRESSIVE_MAX_LOSS_THRESHOLD,
                max_memory_gb=PROGRESSIVE_MAX_MEMORY_GB,
                max_time_hours=PROGRESSIVE_MAX_TIME_HOURS,
                checkpoint_interval=PROGRESSIVE_CHECKPOINT_INTERVAL
            )

        super().__init__(config=config)

        # Directory structure initialization
        self.system_config = get_current_config().get_config('models', {}).get('progressive', {})
        self.checkpoints_dir = Path("models/progressive/checkpoints")
        self.results_dir = Path("results/progressive")
        self.analytics_dir = Path("analytics/progressive")

        # Ensure directories exist
        for dir_path in [self.checkpoints_dir, self.results_dir, self.analytics_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        if hasattr(self, 'progress_dir'):
            self.progress_dir.mkdir(parents=True, exist_ok=True)

        # Initialize modular components
        self.path_validator = PathSecurityValidator()
        self.state_manager = TrainingStateManager(self.checkpoints_dir)
        self.state_manager.initialize_state(self.config.initial_batch_size or PROGRESSIVE_INITIAL_BATCH_SIZE)
        self.batch_processor = BatchProcessor(self.config.enable_adaptive_batching)

        # Analytics buffers
        self.analytics: dict[str, list[Any]] = defaultdict(list)
        self.performance_history: list[dict[str, Any]] = []

    def _prepare_ticker_groups(self, plan: dict[str, Any]) -> list[list[str]]:
        """
        Calculates adaptive ticker batches for the training deployment.
        Batch sizes scale dynamically based on historical success rates.
        """
        tickers = plan.get('tickers', [])
        if not tickers:
            return []
        return self.batch_processor.create_progressive_batches(
            tickers,
            self.state_manager.state.processed_tickers,
            self.config.initial_batch_size or PROGRESSIVE_INITIAL_BATCH_SIZE,
            self.config.max_batch_size or PROGRESSIVE_MAX_BATCH_SIZE,
            self.config.growth_factor or PROGRESSIVE_BATCH_GROWTH_FACTOR
        )

    def _train_ticker_group(self, ticker_group: list[str], data_context: dict[str, Any]) -> dict[str, Any]:
        """
        Sequentially trains a group of assets, adapting logic mid-batch if instability is detected.
        """
        results = {}
        for ticker in ticker_group:
            try:
                # Execution of the primary training suite (defined in BaseTrainer or subclass)
                result = self._train_ticker_suite(ticker, data_context)
                results[ticker] = result
                self.state_manager.state.processed_tickers.add(ticker)

                if result.get('status') == 'success':
                    self.state_manager.state.successful_tickers.add(ticker)
                else:
                    self.state_manager.state.failed_tickers.add(ticker)
            except Exception as e:
                self.logger.error(f"Inference Failure for {ticker}: {e}")
                results[ticker] = {"status": "failed", "reason": str(e)}
        return results

    def execute_progressive_training(self, plan_or_tickers: list[str] | dict[str, Any], data_context: dict[str, Any] | None = None) -> dict[str, Any]:
        """The main deployment entry point for the progressive trainer."""
        plan, tickers = self._prepare_training_plan(plan_or_tickers)
        self.logger.info(f"Initiating progressive training cycle: {len(tickers)} assets.")

        if data_context is not None:
            data_context = data_context.copy()
            data_context['plan'] = plan
            self._current_data_context = data_context

        batches = self.batch_processor.create_progressive_batches(
            tickers,
            self.state_manager.state.processed_tickers,
            self.config.initial_batch_size or PROGRESSIVE_INITIAL_BATCH_SIZE,
            self.config.max_batch_size or PROGRESSIVE_MAX_BATCH_SIZE,
            self.config.growth_factor or PROGRESSIVE_BATCH_GROWTH_FACTOR
        )
        batch_results = []

        for i, batch in enumerate(batches):
            if not self._check_resources():
                self.logger.warning("Resource saturation reached. Suspending training cycle.")
                break

            batch_result = self._process_single_batch_lifecycle(i + 1, batch)
            if batch_result:
                batch_results.append(batch_result)

        final_results = self._create_final_results(batch_results)
        self._save_final_results(final_results)
        return final_results

    def _prepare_training_plan(self, plan_or_tickers: list[str] | dict[str, Any]) -> tuple[dict, list[str]]:
        """Extracts plan and tickers from input."""
        if isinstance(plan_or_tickers, dict):
            return plan_or_tickers, plan_or_tickers.get('tickers', [])
        return {"tickers": plan_or_tickers, "strategy": "progressive"}, plan_or_tickers

    def _process_single_batch_lifecycle(self, batch_id: int, batch: list[str]) -> dict[str, Any] | None:
        """Handles the complete processing cycle for a single batch."""
        filtered_batch = [t for t in batch if not self.state_manager.should_skip_ticker(t)]
        if not filtered_batch:
            self.logger.info(f"Batch {batch_id}: No eligible tickers. Skipping.")
            return None

        self.logger.info(f"Processing Batch {batch_id}: {filtered_batch}")
        try:
            difficulty = self.batch_processor.estimate_batch_difficulty(filtered_batch)
            batch_result = self._train_progressive_batch(batch_id, filtered_batch, difficulty)

            self._update_state(batch_result)
            if batch_id % self.config.checkpoint_interval == 0:
                self.state_manager.save_checkpoint(batch_id, dict(self.analytics))

            if self.config.enable_quality_filtering:
                self._analyze_batch_quality(batch_result)

            if self.config.enable_smart_scheduling:
                self._adjust_training_strategy(batch_result)

            self.logger.info(f"Batch {batch_id} deployment successful.")
            return batch_result

        except Exception as e:
            self.logger.error(f"Critical failure in Batch {batch_id}: {e}")
            return {"batch_id": batch_id, "status": "failed", "tickers": filtered_batch, "error": str(e)}

    def _train_progressive_batch(self, batch_id: int, batch: list[str], difficulty: dict[str, float]) -> dict[str, Any]:
        """Executes actual model training for an asset group."""
        start_time = time.time()
        data_context = getattr(self, '_current_data_context', None)
        if data_context is None:
            self.logger.error(f"Batch {batch_id}: data_context is undefined.")
            raise ModelTrainingError(f"Batch {batch_id}: data_context is undefined.")

        try:
            group_results = self._train_ticker_group(batch, data_context)
            batch_info = {"id": batch_id, "tickers": batch, "diff": difficulty}
            return cast(dict[str, Any], self.batch_processor.aggregate_batch_metrics(batch_info, group_results, start_time))
        except Exception as e:
            self.logger.error(f"Training Protocol Error in Batch {batch_id}: {e}")
            raise ModelTrainingError(f"Training Protocol Error in Batch {batch_id}: {e}") from e

    def _update_state(self, batch_result: dict[str, Any]):
        """Synchronizes batch completion results with the global training ledger."""
        tickers = batch_result["tickers"]
        status = batch_result["status"]

        self.state_manager.state.processed_tickers.update(tickers)

        if status == "completed":
            self.state_manager.state.successful_tickers.update(tickers)
        else:
            self.state_manager.state.failed_tickers.update(tickers)

        self.state_manager.state.total_batches_processed += 1
        self.state_manager.state.last_checkpoint = time.time()

        # Buffer for persistent analytics
        self.analytics["batch_results"].append(batch_result)
        self.analytics["success_rate"].append(1.0 if status == "completed" else 0.0)
        self.analytics["accuracy"].append(batch_result.get("accuracy", 0.0))
        self.analytics["loss"].append(batch_result.get("loss", 1.0))

    def _analyze_batch_quality(self, batch_result: dict[str, Any]):
        """Audits the statistical quality of the model outputs in a finished batch."""
        accuracy = batch_result.get("accuracy", 0.0)
        loss = batch_result.get("loss", 1.0)

        min_accuracy = self.config.min_accuracy_threshold or PROGRESSIVE_MIN_ACCURACY_THRESHOLD
        max_loss = self.config.max_loss_threshold or PROGRESSIVE_MAX_LOSS_THRESHOLD

        if accuracy < min_accuracy:
            self.logger.warning(f"Quality Alert (Accuracy) in Batch {batch_result['batch_id']}: {accuracy:.4f}")

        if loss > max_loss:
            self.logger.warning(f"Quality Alert (Loss) in Batch {batch_result['batch_id']}: {loss:.4f}")

        self.performance_history.append({
            "batch_id": batch_result["batch_id"],
            "timestamp": time.time(),
            "accuracy": accuracy,
            "loss": loss,
            "status": batch_result["status"]
        })

    def _adjust_training_strategy(self, batch_result: dict[str, Any]):
        """Modifies future scaling factors based on current batch stability."""
        accuracy = batch_result.get("accuracy", 0.0)
        status = batch_result["status"]

        min_accuracy = self.config.min_accuracy_threshold or PROGRESSIVE_MIN_ACCURACY_THRESHOLD
        initial_batch = self.config.initial_batch_size or PROGRESSIVE_INITIAL_BATCH_SIZE
        max_batch = self.config.max_batch_size or PROGRESSIVE_MAX_BATCH_SIZE

        # Scale back in case of degradation
        if accuracy < min_accuracy and status == "completed":
            self.state_manager.adjust_batch_size(
                max(initial_batch, int(self.state_manager.state.current_batch_size * 0.8))
            )
            self.logger.info(f"Stability Control: Reducing next batch capacity to {self.state_manager.state.current_batch_size}")

        # Aggressive scaling for high-confidence sectors
        elif accuracy > 0.9 and status == "completed":
            self.state_manager.adjust_batch_size(
                int(min(max_batch, int(self.state_manager.state.current_batch_size * 1.1)))
            )
            self.logger.info(f"Optimization: Increasing next batch capacity to {self.state_manager.state.current_batch_size}")

    def _check_resources(self) -> bool:
        """Hardware telemetry audit."""
        max_time = self.config.max_time_hours or PROGRESSIVE_MAX_TIME_HOURS
        elapsed_time = time.time() - self.state_manager.state.start_time
        if elapsed_time > max_time * 3600:
            self.logger.warning("Progressive training time budget exceeded.")
            return False

        max_memory_gb = self.config.max_memory_gb or PROGRESSIVE_MAX_MEMORY_GB
        try:
            import psutil

            memory = psutil.virtual_memory()
            used_gb = memory.used / 1024 ** 3
            if used_gb > max_memory_gb:
                self.logger.warning(
                    f"Memory budget exceeded: {used_gb:.2f}GB used > {max_memory_gb:.2f}GB limit"
                )
                return False

            cpu_percent = psutil.cpu_percent(interval=0.0)
            if cpu_percent >= 95.0 and memory.percent >= 90.0:
                self.logger.warning(
                    f"Resource saturation: cpu={cpu_percent:.1f}%, memory={memory.percent:.1f}%"
                )
                return False
        except ImportError:
            self.logger.debug("psutil unavailable; skipping OS resource checks.")

        return True

    def _create_final_results(self, batch_results: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregates individual batch metrics into a finalized deployment report."""
        total_time = time.time() - self.state_manager.state.start_time

        successful_batches = [r for r in batch_results if r.get("status") == "completed"]
        failed_batches = [r for r in batch_results if r.get("status") == "failed"]

        return {
            "training_summary": {
                "total_tickers": len(self.state_manager.state.processed_tickers),
                "successful_tickers": len(self.state_manager.state.successful_tickers),
                "failed_tickers": len(self.state_manager.state.failed_tickers),
                "total_batches": len(batch_results),
                "successful_batches": len(successful_batches),
                "failed_batches": len(failed_batches),
                "total_time_hours": total_time / 3600,
                "average_accuracy": float(np.mean([r.get("accuracy", 0) for r in successful_batches])) if successful_batches else 0.0,
                "average_loss": float(np.mean([r.get("loss", 1) for r in successful_batches])) if successful_batches else 1.0
            },
            "batch_results": batch_results,
            "performance_history": self.performance_history,
            "final_state": {
                "processed_tickers": list(self.state_manager.state.processed_tickers),
                "successful_tickers": list(self.state_manager.state.successful_tickers),
                "failed_tickers": list(self.state_manager.state.failed_tickers),
                "current_batch_size": self.state_manager.state.current_batch_size
            },
            "timestamp": datetime.now().isoformat()
        }

    def _save_final_results(self, results: dict[str, Any]):
        """Persists the final report and analytics matrix into the project archive."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        results_file = self.results_dir / f"progressive_results_{timestamp}.json"
        analytics_file = self.analytics_dir / f"progressive_analytics_{timestamp}.json"

        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)

            with open(analytics_file, 'w') as f:
                json.dump(dict(self.analytics), f, indent=2)

            self.logger.info(f"Cycle intelligence saved to {results_file}")
        except Exception as e:
            self.logger.error(f"Sync error during cycle conclusion: {e}")

    def load_checkpoint(self, checkpoint_file: str) -> bool:
        """Restores training state from a localized checkpoint file."""
        return self.state_manager.load_checkpoint(checkpoint_file, self.path_validator)

if __name__ == "__main__":
    # Internal validation logic
    import argparse

    parser = argparse.ArgumentParser(description='Progressive Training Controller')
    parser.add_argument('--tickers', default='all', help='Target sectors or list')
    parser.add_argument('--initial-batch', type=int, default=5)
    parser.add_argument('--max-batch', type=int, default=20)
    parser.add_argument('--resume', help='Checkpoint URI for recovery')

    args = parser.parse_args()

    config = TrainerConfig(
        initial_batch_size=args.initial_batch,
        max_batch_size=args.max_batch
    )
    trainer = ProgressiveTrainer(config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.logger.info("Progressive training controller initialized via CLI.")
