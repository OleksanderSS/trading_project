# src/training/progressive_trainer.py
"""
Progressive Training System for Large Ticker Sets
Enables adaptive batch processing and quality-controlled model training evolution.
"""

import os
import json
import time
import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime

def sanitize_path_input(path_input: str) -> str:
    """
    Sanitize path input to prevent path traversal attacks.
    
    Args:
        path_input: Input string that will be used in file paths
        
    Returns:
        Sanitized string safe for path construction
    """
    if not path_input:
        return ""
    
    # Remove path traversal characters
    sanitized = re.sub(r'[./\\]', '_', path_input)
    
    # Remove null bytes and other dangerous characters
    sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', sanitized)
    
    # Limit length to prevent path overflow
    sanitized = sanitized[:100]
    
    return sanitized

from src.core.logging.logger import ProjectLogger
from src.config.unified_config_manager import get_current_config
from src.training.constants import (
    PROGRESSIVE_INITIAL_BATCH_SIZE,
    PROGRESSIVE_MAX_BATCH_SIZE,
    PROGRESSIVE_BATCH_GROWTH_FACTOR,
    PROGRESSIVE_MIN_ACCURACY_THRESHOLD,
    PROGRESSIVE_MAX_LOSS_THRESHOLD,
    PROGRESSIVE_CHECKPOINT_INTERVAL,
    PROGRESSIVE_MAX_TIME_HOURS,
    PROGRESSIVE_MAX_MEMORY_GB
)
from src.training.base_trainer import BaseTrainer

logger = ProjectLogger.get_logger("ProgressiveTrainer")

@dataclass
class ProgressiveConfig:
    """Configuration for the adaptive progressive training pipeline."""
    # Base batch settings
    initial_batch_size: int = PROGRESSIVE_INITIAL_BATCH_SIZE
    max_batch_size: int = PROGRESSIVE_MAX_BATCH_SIZE
    growth_factor: float = PROGRESSIVE_BATCH_GROWTH_FACTOR
    
    # Quality thresholds
    min_accuracy_threshold: float = PROGRESSIVE_MIN_ACCURACY_THRESHOLD
    max_loss_threshold: float = PROGRESSIVE_MAX_LOSS_THRESHOLD       
    
    # Adaptive logic toggles
    enable_adaptive_batching: bool = True
    enable_quality_filtering: bool = True
    enable_smart_scheduling: bool = True

    # Persistence settings
    save_intermediate_results: bool = True
    checkpoint_interval: int = PROGRESSIVE_CHECKPOINT_INTERVAL
    
    # Resource management
    max_memory_gb: float = PROGRESSIVE_MAX_MEMORY_GB
    max_time_hours: float = PROGRESSIVE_MAX_TIME_HOURS

@dataclass
class TrainingState:
    """Persistence object for tracking training progress and lifecycle."""
    processed_tickers: Set[str] = field(default_factory=set)
    successful_tickers: Set[str] = field(default_factory=set)
    failed_tickers: Set[str] = field(default_factory=set)
    current_batch_size: int = PROGRESSIVE_INITIAL_BATCH_SIZE
    total_batches_processed: int = 0
    start_time: float = field(default_factory=time.time)
    last_checkpoint: float = field(default_factory=time.time)
    
class ProgressiveTrainer(BaseTrainer):
    """
    Adaptive trainer designed for large-scale ticker universes.
    Iteratively evolves batch sizes based on real-time hardware performance and model stability.
    """
    
    def __init__(self, config: Optional[ProgressiveConfig] = None):
        """Initializes the progressive training environment and local persistence paths."""
        self.progressive_config = config or ProgressiveConfig()
        super().__init__(config=self.progressive_config)
        
        # Directory structure initialization
        self.system_config = get_current_config().models.get('progressive', {})
        self.checkpoints_dir = Path("models/progressive/checkpoints")
        self.results_dir = Path("results/progressive")
        self.analytics_dir = Path("analytics/progressive")
        
        # Ensure self.progress_dir (from BaseTrainer) and others exist
        for dir_path in [self.checkpoints_dir, self.results_dir, self.analytics_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        if hasattr(self, 'progress_dir'):
            self.progress_dir.mkdir(parents=True, exist_ok=True)
        
        # Internal state tracking
        self.state = TrainingState()
        
        # Analytics buffers
        self.analytics = defaultdict(list)
        self.performance_history = []
    
    def _prepare_ticker_groups(self, plan: Dict[str, Any]) -> List[List[str]]:
        """
        Calculates adaptive ticker batches for the training deployment.
        Batch sizes scale dynamically based on historical success rates.
        """
        tickers = plan.get('tickers', [])
        if not tickers:
            return []
        return self.create_progressive_batches(tickers)
    
    def _train_ticker_group(self, ticker_group: List[str], data_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sequentially trains a group of assets, adapting logic mid-batch if instability is detected.
        """
        results = {}
        for ticker in ticker_group:
            try:
                # Execution of the primary training suite (defined in BaseTrainer or subclass)
                result = self._train_ticker_suite(ticker, data_context)
                results[ticker] = result
                self.state.processed_tickers.add(ticker)
                
                if result.get('status') == 'success':
                    self.state.successful_tickers.add(ticker)
                else:
                    self.state.failed_tickers.add(ticker)
            except Exception as e:
                self.logger.error(f"Inference Failure for {ticker}: {e}")
                results[ticker] = {"status": "failed", "reason": str(e)}
        return results
        
    def create_progressive_batches(self, tickers: List[str]) -> List[List[str]]:
        """
        Segments the ticker universe into progressive batches.
        
        Args:
            tickers: Full list of assets to be processed.
            
        Returns:
            Nested list of adaptive asset batches.
        """
        if not self.progressive_config.enable_adaptive_batching:
            return self._create_fixed_batches(tickers)
        
        # Hierarchy-aware ticker prioritization
        prioritized_tickers = self._prioritize_tickers(tickers)
        
        batches = []
        current_batch = []
        current_batch_size = self.progressive_config.initial_batch_size
        
        for ticker in prioritized_tickers:
            # Skip assets already synchronized in previous cycles
            if ticker in self.state.processed_tickers:
                continue
            
            current_batch.append(ticker)
            
            # Finalize current batch and compute scaling for the next
            batch_ready = len(current_batch) >= current_batch_size
            if batch_ready:
                batches.append(current_batch)
                current_batch = []
                
                # Scale batch size progressively towards the ceiling
                current_batch_size = min(
                    int(current_batch_size * self.progressive_config.growth_factor),
                    self.progressive_config.max_batch_size
                )
        
        # Append remaining assets
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _create_fixed_batches(self, tickers: List[str]) -> List[List[str]]:
        """Fallback for fixed-size batch segmentation."""
        batch_size = self.progressive_config.initial_batch_size
        batches = []
        
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            if batch:
                batches.append(batch)
        
        return batches
    
    def _prioritize_tickers(self, tickers: List[str]) -> List[str]:
        """Assigns training priority based on asset category and market significance."""
        # Sensitivity matrix for training order
        category_priority = {
            'core': 10,      # Maximum priority (High liquidity indices)
            'tech': 9,       # High volatility tech sector
            'etf': 8,        # Diversified trackers
            'finance': 7,    
            'sp500': 6,      
            'healthcare': 5, 
            'consumer': 4,   
            'energy': 3,     
            'industrial': 2, 
            'other': 1       
        }
        
        assets_config = get_current_config().get_config('assets') or {}
        
        def get_ticker_priority(ticker):
            asset_info = assets_config.get(ticker, {})
            category = asset_info.get('sector', 'other')
            return category_priority.get(category, 1)
        
        # Sort using the categorical weightings
        prioritized = sorted(tickers, key=get_ticker_priority, reverse=True)
        
        self.logger.info(f"Prioritization sequence established for {len(tickers)} tickers.")
        return prioritized
    
    def estimate_batch_difficulty(self, batch: List[str]) -> Dict[str, float]:
        """
        Predicts computational overhead and expected stability of a proposed batch.
        
        Args:
            batch: Proposed asset group.
            
        Returns:
            Resource demand and difficulty projections.
        """
        # Linear base overhead
        base_difficulty = len(batch)
        
        # Weighting for sector-specific feature complexity
        category_difficulty = {
            'tech': 1.5,      # High noise/variability
            'finance': 1.3,   
            'etf': 0.8,       # Low noise / high correlation
            'core': 1.0,      
            'other': 1.0      
        }
        
        assets_config = get_current_config().get_config('assets') or {}

        total_difficulty = 0
        for ticker in batch:
            asset_info = assets_config.get(ticker, {})
            category = asset_info.get('sector', 'other')
            difficulty = category_difficulty.get(category, 1.0)
            total_difficulty += difficulty
        
        return {
            "base_difficulty": float(base_difficulty),
            "category_difficulty": float(total_difficulty),
            "estimated_time_hours": total_difficulty * 0.5,
            "estimated_memory_gb": len(batch) * 0.4,
            "success_probability": min(0.95, 1.0 - (total_difficulty * 0.05))
        }
    
    def should_skip_ticker(self, ticker: str) -> bool:
        """Determines if an asset should be bypassed in the current cycle."""
        if ticker in self.state.processed_tickers:
            return True
        
        if ticker in self.state.failed_tickers:
            # Note: Critical failures might justify an extended cooldown period
            return False
        
        return False
    
    def execute_progressive_training(self, tickers: List[str], data_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        The main deployment entry point for the progressive trainer.
        
        Args:
            tickers: Full target asset list.
            data_context: Payload containing pricing and feature datasets.
            
        Returns:
            Unified training report with adaptive metrics.
        """
        self.logger.info(f"Initiating progressive training cycle: {len(tickers)} assets.")
        self._current_data_context = data_context
        
        # Generate the adaptive batch schedule
        batches = self.create_progressive_batches(tickers)
        
        batch_results = []
        
        for i, batch in enumerate(batches):
            batch_id = i + 1
            
            # Telemetry check for hardware saturation
            if not self._check_resources():
                self.logger.warning("Resource saturation threshold reached. Suspending training cycle.")
                break
            
            # Filter batch based on state and historical stability
            filtered_batch = [t for t in batch if not self.should_skip_ticker(t)]
            
            if not filtered_batch:
                self.logger.info(f"Batch {batch_id}: No eligible tickers for processing. Skipping.")
                continue
            
            self.logger.info(f"Processing Batch {batch_id}: {filtered_batch}")
            
            try:
                # Pre-run difficulty analysis
                difficulty = self.estimate_batch_difficulty(filtered_batch)
                
                # Execution with metric aggregation
                batch_result = self._train_progressive_batch(batch_id, filtered_batch, difficulty)
                batch_results.append(batch_result)
                
                # Synchronization of global state
                self._update_state(batch_result)
                
                # Persistent checkpoint for disaster recovery
                if batch_id % self.progressive_config.checkpoint_interval == 0:
                    self._save_checkpoint(batch_id)
                
                # Quality audit for adaptive scaling
                if self.progressive_config.enable_quality_filtering:
                    self._analyze_batch_quality(batch_result)
                
                # Dynamic strategy adjustment
                if self.progressive_config.enable_smart_scheduling:
                    self._adjust_training_strategy(batch_result)
                
                self.logger.info(f"Batch {batch_id} deployment successful.")
                
            except Exception as e:
                self.logger.error(f"Critical failure in Batch {batch_id}: {e}")
                batch_results.append({
                    "batch_id": batch_id,
                    "status": "failed",
                    "tickers": filtered_batch,
                    "error": str(e)
                })
        
        # Completion and report generation
        final_results = self._create_final_results(batch_results)
        self._save_final_results(final_results)
        
        return final_results
    
    def _train_progressive_batch(self, batch_id: int, batch: List[str], difficulty: Dict[str, float]) -> Dict[str, Any]:
        """
        Executes actual model training for an asset group.
        Standardizes metrics across heterogeneous model architectures.
        """
        start_time = time.time()

        data_context = getattr(self, '_current_data_context', None)
        if data_context is None:
            self.logger.error(f"Batch {batch_id}: data_context is undefined. Deployment aborted.")
            return {
                "batch_id": batch_id, "status": "skipped", "tickers": batch,
                "difficulty": difficulty, "training_time": 0.0,
                "accuracy": 0.0, "loss": 1.0, "memory_used": 0.0,
                "models_trained": 0, "success_rate": 0.0,
                "error": "Missing data payload.",
            }

        try:
            group_results = self._train_ticker_group(batch, data_context)
        except Exception as e:
            self.logger.error(f"Training Protocol Error in Batch {batch_id}: {e}")
            end_time = time.time()
            return {
                "batch_id": batch_id, "status": "failed", "tickers": batch,
                "difficulty": difficulty, "training_time": end_time - start_time,
                "accuracy": 0.0, "loss": 1.0, "error": str(e),
                "success_rate": 0.0
            }

        end_time = time.time()
        successful = [r for r in group_results.values() if r.get("status") == "success"]
        n_failed = len(batch) - len(successful)

        # Accuracy and Loss normalization
        accuracy_vals = [r.get("best_score", r.get("accuracy", 0.0)) for r in successful]
        loss_vals = [r.get("loss", 0.0) for r in successful if "loss" in r]
        
        avg_accuracy = float(np.mean(accuracy_vals)) if accuracy_vals else 0.0
        avg_loss = float(np.mean(loss_vals)) if loss_vals else 0.0

        if n_failed == 0:
            status = "completed"
        elif successful:
            status = "partial"
        else:
            status = "failed"

        return {
            "batch_id": batch_id,
            "status": status,
            "tickers": batch,
            "difficulty": difficulty,
            "training_time": end_time - start_time,
            "accuracy": avg_accuracy,
            "loss": avg_loss,
            "memory_used": difficulty.get("estimated_memory_gb", 0.0),
            "models_trained": sum(len(r.get("trained_models", [])) for r in group_results.values()),
            "success_rate": len(successful) / len(batch) if batch else 0.0,
            "ticker_results": group_results,
        }

    def _update_state(self, batch_result: Dict[str, Any]):
        """Synchronizes batch completion results with the global training ledger."""
        tickers = batch_result["tickers"]
        status = batch_result["status"]
        
        self.state.processed_tickers.update(tickers)
        
        if status == "completed":
            self.state.successful_tickers.update(tickers)
        else:
            self.state.failed_tickers.update(tickers)
        
        self.state.total_batches_processed += 1
        self.state.last_checkpoint = time.time()
        
        # Buffer for persistent analytics
        self.analytics["batch_results"].append(batch_result)
        self.analytics["success_rate"].append(1.0 if status == "completed" else 0.0)
        self.analytics["accuracy"].append(batch_result.get("accuracy", 0.0))
        self.analytics["loss"].append(batch_result.get("loss", 1.0))
    
    def _analyze_batch_quality(self, batch_result: Dict[str, Any]):
        """Audits the statistical quality of the model outputs in a finished batch."""
        accuracy = batch_result.get("accuracy", 0.0)
        loss = batch_result.get("loss", 1.0)
        
        if accuracy < self.progressive_config.min_accuracy_threshold:
            self.logger.warning(f"Quality Alert (Accuracy) in Batch {batch_result['batch_id']}: {accuracy:.4f}")
        
        if loss > self.progressive_config.max_loss_threshold:
            self.logger.warning(f"Quality Alert (Loss) in Batch {batch_result['batch_id']}: {loss:.4f}")
        
        self.performance_history.append({
            "batch_id": batch_result["batch_id"],
            "timestamp": time.time(),
            "accuracy": accuracy,
            "loss": loss,
            "status": batch_result["status"]
        })
    
    def _adjust_training_strategy(self, batch_result: Dict[str, Any]):
        """Modifies future scaling factors based on current batch stability."""
        accuracy = batch_result.get("accuracy", 0.0)
        status = batch_result["status"]
        
        # Scale back in case of degradation
        if accuracy < self.progressive_config.min_accuracy_threshold and status == "completed":
            self.state.current_batch_size = max(
                self.progressive_config.initial_batch_size,
                int(self.state.current_batch_size * 0.8)
            )
            self.logger.info(f"Stability Control: Reducing next batch capacity to {self.state.current_batch_size}")
        
        # Aggressive scaling for high-confidence sectors
        elif accuracy > 0.9 and status == "completed":
            self.state.current_batch_size = min(
                self.progressive_config.max_batch_size,
                int(self.state.current_batch_size * 1.1)
            )
            self.logger.info(f"Optimization: Increasing next batch capacity to {self.state.current_batch_size}")
    
    def _check_resources(self) -> bool:
        """Hardware telemetry audit."""
        elapsed_time = time.time() - self.state.start_time
        if elapsed_time > self.progressive_config.max_time_hours * 3600:
            return False
        
        # Resource monitor integration (optional placeholder)
        return True
    
    def _save_checkpoint(self, batch_id: int):
        """Serializes current state to a JSON checkpoint for recovery."""
        checkpoint = {
            "batch_id": batch_id,
            "state": {
                "processed_tickers": list(self.state.processed_tickers),
                "successful_tickers": list(self.state.successful_tickers),
                "failed_tickers": list(self.state.failed_tickers),
                "current_batch_size": self.state.current_batch_size,
                "total_batches_processed": self.state.total_batches_processed
            },
            "analytics": dict(self.analytics),
            "timestamp": datetime.now().isoformat()
        }
        
        filepath = self.checkpoints_dir / f"checkpoint_batch_{batch_id}.json"
        try:
            with open(filepath, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            self.logger.debug(f"Checkpoint synchronized: {filepath.name}")
        except Exception as e:
            self.logger.error(f"Failed to synchronize state to disk: {e}")
    
    def _create_final_results(self, batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregates individual batch metrics into a finalized deployment report."""
        total_time = time.time() - self.state.start_time
        
        successful_batches = [r for r in batch_results if r.get("status") == "completed"]
        failed_batches = [r for r in batch_results if r.get("status") == "failed"]
        
        return {
            "training_summary": {
                "total_tickers": len(self.state.processed_tickers),
                "successful_tickers": len(self.state.successful_tickers),
                "failed_tickers": len(self.state.failed_tickers),
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
                "processed_tickers": list(self.state.processed_tickers),
                "successful_tickers": list(self.state.successful_tickers),
                "failed_tickers": list(self.state.failed_tickers),
                "current_batch_size": self.state.current_batch_size
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def _save_final_results(self, results: Dict[str, Any]):
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
        try:
            sanitized_checkpoint_file = sanitize_path_input(checkpoint_file)
            with open(sanitized_checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            
            state_data = checkpoint["state"]
            self.state.processed_tickers = set(state_data["processed_tickers"])
            self.state.successful_tickers = set(state_data["successful_tickers"])
            self.state.failed_tickers = set(state_data["failed_tickers"])
            self.state.current_batch_size = state_data["current_batch_size"]
            self.state.total_batches_processed = state_data["total_batches_processed"]
            
            self.analytics = defaultdict(list, checkpoint["analytics"])
            self.logger.info(f"Recovered from checkpoint: {checkpoint_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"State recovery failed for {checkpoint_file}: {e}")
            return False

if __name__ == "__main__":
    # Internal validation logic
    import argparse
    
    parser = argparse.ArgumentParser(description='Progressive Training Controller')
    parser.add_argument('--tickers', default='all', help='Target sectors or list')
    parser.add_argument('--initial-batch', type=int, default=5)
    parser.add_argument('--max-batch', type=int, default=20)
    parser.add_argument('--resume', help='Checkpoint URI for recovery')
    
    args = parser.parse_args()
    
    config = ProgressiveConfig(
        initial_batch_size=args.initial_batch,
        max_batch_size=args.max_batch
    )
    trainer = ProgressiveTrainer(config)
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    trainer.logger.info("Progressive training controller initialized via CLI.")
