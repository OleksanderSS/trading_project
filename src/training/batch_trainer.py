# src/training/batch_trainer.py

from typing import Any

from joblib import Parallel, delayed

from src.core.logging.logger import ProjectLogger
from src.training.base_trainer import BaseTrainer, TrainerConfig
from src.training.constants import BATCH_TRAINER_DEFAULT_BATCH_SIZE, BATCH_TRAINER_DEFAULT_MAX_MEMORY_GB

logger = ProjectLogger.get_logger("BatchTrainer")

class BatchTrainer(BaseTrainer):
    """
    Advanced parallelized batch trainer.

    Trains all tickers in a single batch using parallel execution.
    Extends BaseTrainer with batch-specific grouping and parallel training logic.
    """

    def __init__(self, config: TrainerConfig | None = None):
        super().__init__(config or TrainerConfig(
            batch_size=BATCH_TRAINER_DEFAULT_BATCH_SIZE,
            max_memory_gb=BATCH_TRAINER_DEFAULT_MAX_MEMORY_GB
        ))

    def _prepare_ticker_groups(self, plan: dict[str, Any]) -> list[list[str]]:
        """
        Batch trainer: All tickers in one group.

        The entire set of tickers is trained in parallel as a single batch.
        """
        tickers = plan.get('tickers', [])
        if not tickers:
            return []
        return [tickers]  # All tickers in one group

    def _train_ticker_group(self, ticker_group: list[str], data_context: dict[str, Any]) -> dict[str, Any]:
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

        return dict(zip(ticker_group, batch_results, strict=False))

    def create_batch_plan(self, tickers: list[str], strategy: str = 'batch') -> dict[str, Any]:
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
