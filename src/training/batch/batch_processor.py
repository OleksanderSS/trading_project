#!/usr/bin/env python3
"""
Batch Processor - Progressive Batch Processing Logic
Handles batch creation, prioritization, and processing logic.
"""

import time
from typing import Any

import numpy as np

from src.config.unified_config_manager import get_current_config
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("BatchProcessor")


class BatchProcessor:
    """
    Batch processor for progressive training.

    Handles:
    - Batch creation
    - Ticker prioritization
    - Batch difficulty estimation
    - Batch metrics aggregation
    """

    def __init__(self, enable_adaptive_batching: bool = True):
        """
        Initialize Batch Processor.

        Args:
            enable_adaptive_batching: Whether to enable adaptive batch sizing
        """
        self.logger = logger
        self.enable_adaptive_batching = enable_adaptive_batching
        self.logger.info("✅ BatchProcessor initialized")

    def create_progressive_batches(self,
                                   tickers: list[str],
                                   processed_tickers: set[str],
                                   initial_batch_size: int = 5,
                                   max_batch_size: int = 20,
                                   growth_factor: float = 1.5) -> list[list[str]]:
        """
        Segments the ticker universe into progressive batches.

        Args:
            tickers: Full list of assets to be processed
            processed_tickers: Set of already processed tickers
            initial_batch_size: Starting batch size
            max_batch_size: Maximum batch size
            growth_factor: Growth factor for batch size scaling

        Returns:
            Nested list of adaptive asset batches
        """
        if not self.enable_adaptive_batching:
            return self._create_fixed_batches(tickers, initial_batch_size)

        # Hierarchy-aware ticker prioritization
        prioritized_tickers = self._prioritize_tickers(tickers)

        batches = []
        current_batch = []
        current_batch_size = initial_batch_size

        for ticker in prioritized_tickers:
            # Skip assets already synchronized in previous cycles
            if ticker in processed_tickers:
                continue

            current_batch.append(ticker)

            # Finalize current batch and compute scaling for the next
            batch_ready = len(current_batch) >= current_batch_size
            if batch_ready:
                batches.append(current_batch)
                current_batch = []

                # Scale batch size progressively towards the ceiling
                current_batch_size = min(
                    int(current_batch_size * growth_factor),
                    max_batch_size
                )

        # Append remaining assets
        if current_batch:
            batches.append(current_batch)

        return batches

    def _create_fixed_batches(self, tickers: list[str], batch_size: int) -> list[list[str]]:
        """Fallback for fixed-size batch segmentation."""
        batches = []
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            if batch:
                batches.append(batch)
        return batches

    def _prioritize_tickers(self, tickers: list[str]) -> list[str]:
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

    def estimate_batch_difficulty(self, batch: list[str]) -> dict[str, float]:
        """
        Predicts computational overhead and expected stability of a proposed batch.

        Args:
            batch: Proposed asset group

        Returns:
            Resource demand and difficulty projections
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

        total_difficulty = 0.0
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

    def aggregate_batch_metrics(self,
                                 batch_info: dict[str, Any],
                                 group_results: dict[str, Any],
                                 start_time: float) -> dict[str, Any]:
        """Aggregates metrics from individual ticker results."""
        successful = [r for r in group_results.values() if r.get("status") == "success"]
        accuracy_vals = [r.get("best_score", r.get("accuracy", 0.0)) for r in successful]
        loss_vals = [r.get("loss", 0.0) for r in successful if "loss" in r]

        avg_accuracy = float(np.mean(accuracy_vals)) if accuracy_vals else 0.0
        avg_loss = float(np.mean(loss_vals)) if loss_vals else 0.0
        status = ("completed" if len(successful) == len(batch_info["tickers"])
                 else ("partial" if successful else "failed"))

        return {
            "batch_id": batch_info["id"],
            "status": status,
            "tickers": batch_info["tickers"],
            "difficulty": batch_info["diff"],
            "training_time": time.time() - start_time,
            "accuracy": avg_accuracy,
            "loss": avg_loss,
            "memory_used": batch_info["diff"].get("estimated_memory_gb", 0.0),
            "models_trained": sum(len(r.get("trained_models", [])) for r in group_results.values()),
            "success_rate": len(successful) / len(batch_info["tickers"]) if batch_info["tickers"] else 0.0,
            "ticker_results": group_results,
        }

    def create_empty_batch_result(self,
                                 batch_id: int,
                                 batch: list[str],
                                 difficulty: dict[str, float],
                                 error_msg: str) -> dict[str, Any]:
        """Creates a standardized empty result for skipped/errored batches."""
        return {
            "batch_id": batch_id,
            "status": "skipped",
            "tickers": batch,
            "difficulty": difficulty,
            "training_time": 0.0,
            "accuracy": 0.0,
            "loss": 1.0,
            "memory_used": 0.0,
            "models_trained": 0,
            "success_rate": 0.0,
            "error": error_msg,
        }
