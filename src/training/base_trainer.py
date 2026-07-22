"""
BaseTrainer: Abstract base class for training orchestration

This module provides the template method pattern for training execution,
eliminating duplication between BatchTrainer and ProgressiveTrainer.
All trainer implementations share common workflow:
1. Prepare ticker groups (different strategies for batch vs progressive)
2. Train each group (parallel vs sequential)
3. Generate results summary
"""
import logging
import shutil
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.config.unified_config_manager import get_current_config
from src.core.logging.logger import ProjectLogger
from src.factories.model_factory import ModelFactory
from src.meta_learning.memory.diary_engine import DiaryEngine
from src.metrics.model.ml_evaluator import MLEvaluator


class TrainingException(Exception):
    """Base exception for training errors"""
    pass


class TrainingConfigException(TrainingException):
    """Exception for training configuration issues"""
    pass


class TrainerConfig:
    """
    Base configuration for all trainers.

    This is the parent class for all training configurations.
    Subclasses can extend with additional parameters.
    """
    def __init__(
        self,
        batch_size: int = 10,
        max_memory_gb: float = 12.0,
        # Progressive-specific parameters (optional)
        initial_batch_size: int | None = None,
        max_batch_size: int | None = None,
        growth_factor: float | None = None,
        min_accuracy_threshold: float | None = None,
        max_loss_threshold: float | None = None,
        enable_adaptive_batching: bool = True,
        enable_quality_filtering: bool = True,
        enable_smart_scheduling: bool = True,
        save_intermediate_results: bool = True,
        checkpoint_interval: int = 5,
        max_time_hours: float = 10.0,
        # Adaptive-specific parameters (optional)
        mode: str | None = None,
        strategy: str | None = None,
        max_targets_per_ticker: int | None = None,
        target_diversity_threshold: float | None = None,
        intraday_data_limit_days: int | None = None,
        daily_data_limit_years: int | None = None,
        enable_target_validation: bool = True
    ):
        # Common parameters
        self.batch_size = batch_size
        self.max_memory_gb = max_memory_gb

        # Progressive parameters
        self.initial_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.growth_factor = growth_factor
        self.min_accuracy_threshold = min_accuracy_threshold
        self.max_loss_threshold = max_loss_threshold
        self.enable_adaptive_batching = enable_adaptive_batching
        self.enable_quality_filtering = enable_quality_filtering
        self.enable_smart_scheduling = enable_smart_scheduling
        self.save_intermediate_results = save_intermediate_results
        self.checkpoint_interval = checkpoint_interval
        self.max_time_hours = max_time_hours

        # Adaptive parameters
        self.mode = mode
        self.strategy = strategy
        self.max_targets_per_ticker = max_targets_per_ticker
        self.target_diversity_threshold = target_diversity_threshold
        self.intraday_data_limit_days = intraday_data_limit_days
        self.daily_data_limit_years = daily_data_limit_years
        self.enable_target_validation = enable_target_validation


class BaseTrainer(ABC):
    """
    Abstract base class for training orchestration.

    Implements template method pattern for common training workflow:
    - Prepare ticker groups
    - Train each group
    - Aggregate results and generate summary

    Subclasses must implement:
    - _prepare_ticker_groups(): Define how to group tickers
    - _train_ticker_group(): Define how to train a group
    """

    def __init__(self, config: TrainerConfig | None = None):
        """
        Initialize BaseTrainer.

        Args:
            config: TrainerConfig instance with batch_size and max_memory_gb
        """
        self.config = config or TrainerConfig()
        self.config_manager = get_current_config()
        self.logger = ProjectLogger.get_logger(self.__class__.__name__)

        self.model_factory = ModelFactory()
        self.diary = DiaryEngine()
        self.evaluator = MLEvaluator()

        # Initialize output directory
        try:
            models_path = self.config_manager.get_models_path()
            self.output_dir = Path(models_path)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Output directory: {self.output_dir}")
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Failed to initialize output directory: {e}")
            raise TrainingConfigException(f"Cannot initialize output directory: {e}") from e

    def execute_training(self, plan: dict[str, Any], data_context: dict[str, Any]) -> dict[str, Any]:
        """
        Execute complete training workflow (Template Method).

        This is the common orchestration logic shared by all trainers.
        Subclasses override _prepare_ticker_groups() and _train_ticker_group()
        to define their specific behavior.

        Args:
            plan: Training plan with tickers, strategy, etc.
            data_context: Prepared data for training

        Returns:
            Dictionary with status, results, and summary
        """
        if not plan or not isinstance(plan, dict):
            raise TrainingException("Invalid training plan")

        tickers = plan.get('tickers', [])
        if not tickers:
            self.logger.warning("No tickers provided in training plan")
            return {"status": "failed", "reason": "no_tickers"}

        try:
            self.logger.info(
                f"Starting {self.__class__.__name__} training for {len(tickers)} tickers. "
                f"Strategy: {plan.get('strategy', 'unknown')}"
            )

            # Step 1: Prepare ticker groups (batch vs progressive logic)
            ticker_groups = self._prepare_ticker_groups(plan)
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Created {len(ticker_groups)} ticker groups")

            # Step 2: Train each group
            results = {}
            for group_idx, ticker_group in enumerate(ticker_groups, 1):
                self.logger.info(f"Training group {group_idx}/{len(ticker_groups)} ({len(ticker_group)} tickers)")

                # Inject plan into data_context so trainers can access plan details
                data_with_plan = data_context.copy()
                data_with_plan['plan'] = plan

                group_results = self._train_ticker_group(ticker_group, data_with_plan)
                results.update(group_results)

            # Step 3: Generate summary
            summary = self._generate_summary(results)

            self.logger.info(
                f"✅ Training complete. Success rate: {summary['success_rate']:.1%} "
                f"({summary['successful_tickers']}/{summary['total_tickers']})"
            )

            return {
                "status": "success",
                "tickers_results": results,
                "training_summary": summary
            }

        except (ValueError, TypeError, Exception) as e:
            self.logger.error(f"❌ Training failed: {e}", exc_info=True)
            raise TrainingException(f"Training failed: {e}") from e

    def execute_batch_training(self, plan: dict[str, Any], data_context: dict[str, Any]) -> dict[str, Any]:
        """Alias for execute_training for batch strategy."""
        return self.execute_training(plan, data_context)

    def execute_progressive_training(self, tickers: list[str], data_context: dict[str, Any]) -> dict[str, Any]:
        """Wrapper for execute_training for progressive strategy."""
        plan = {"tickers": tickers, "strategy": "progressive"}
        return self.execute_training(plan, data_context)

    @abstractmethod
    def _prepare_ticker_groups(self, plan: dict[str, Any]) -> list[list[str]]:
        """
        Prepare ticker groups for training.

        Subclasses implement their specific grouping strategy:
        - BatchTrainer: All tickers in one group
        - ProgressiveTrainer: Adaptive batches with growth factor

        Args:
            plan: Training plan containing tickers

        Returns:
            List of ticker groups: [[ticker1, ticker2], [ticker3, ...], ...]
        """
        pass

    @abstractmethod
    def _train_ticker_group(self, ticker_group: list[str], data_context: dict[str, Any]) -> dict[str, Any]:
        """
        Train a group of tickers.

        Subclasses implement their specific training strategy:
        - BatchTrainer: Parallel training using Parallel/delayed
        - ProgressiveTrainer: Sequential training with adaptation

        Args:
            ticker_group: List of tickers to train
            data_context: Prepared data for training

        Returns:
            Dictionary: {ticker: training_result, ...}
        """
        pass

    # CodeScene: Complex Method (cc=10), Large Method (77 lines) - acceptable for training orchestration
    def _train_ticker_suite(self, ticker: str, data: dict[str, Any]) -> dict:
        """Train all configured models for a specific ticker."""
        results = {
            "status": "success",
            "models": [],
            "metrics": {},
            "ticker": ticker,
            "target_name": data.get("target_name", "unknown"),
            "selected_features": list(data.get("feature_names") or []),
        }

        # 1. Data validation and prep
        X_train, y_train = data.get('X_train'), data.get('y_train')
        if X_train is None or y_train is None:
            return {"status": "failed", "ticker": ticker, "reason": "incomplete_data"}

        try:
            # 2. Determine which models to train
            model_types = self._prepare_model_training_list(ticker, data)
            is_classification = 'classification' in data.get('target_type', '')

            # 3. Execute training loop
            best_score, winner_name = self._execute_model_training_cycle(
                ticker, model_types, data, is_classification, results
            )

            # 4. Finalize results
            return self._finalize_ticker_results(results, winner_name, best_score)

        except (ValueError, TypeError, Exception) as e:
            self.logger.error(f"Error during training for {ticker}: {e}", exc_info=True)
            raise TrainingException(f"Training failed for {ticker}: {e}") from e

    def _prepare_model_training_list(self, ticker: str, data: dict[str, Any]) -> list[str]:
        """Determines the set of model types to train for this ticker."""
        plan = data.get('plan', {})
        ticker_plan = plan.get('ticker_plans', {}).get(ticker, {})
        model_types = ticker_plan.get('models')

        if not model_types:
            from src.factories.model_factory import ModelFactory
            model_types = self.config_manager.get_config('models.enabled_types', ModelFactory.get_available_models())
        return list(model_types) if model_types else []

    def _execute_model_training_cycle(self, ticker: str, model_types: list[str],
                                   data: dict[str, Any], is_classif: bool,
                                   results: dict[str, Any]) -> tuple[float, str | None]:
        """Iterates through model types and trains each one."""
        best_score = -np.inf
        winner_name = None

        for m_type in model_types:
            try:
                score_val = self._train_individual_model(ticker, m_type, data, is_classif, results)

                if score_val > best_score:
                    best_score = score_val
                    winner_name = m_type
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                self.logger.error(f"Failed to train {m_type} for {ticker}: {e}")
                continue

        return best_score, winner_name

    def _train_individual_model(self, ticker: str, m_type: str, data: dict[str, Any],
                              is_classif: bool, results: dict[str, Any]) -> float:
        """Handles creation, training, and evaluation of a single model instance."""
        model = self.model_factory.create_model(
            model_name=m_type,
            config=self.config_manager.get_config(f"models.{m_type}", {}),
            task_type="classification" if is_classif else "regression",
            is_classification=is_classif
        )

        model.train(data['X_train'], data['y_train'])
        preds = model.predict(data['X_test'])

        score = self.evaluator.calculate(
            data['y_test'], preds,
            task_type="classification" if is_classif else "regression"
        )

        score_val = float(score.get('F1' if is_classif else 'R2', 0.0))
        results['metrics'][m_type] = {
            'score': score_val,
            'accuracy': float(score.get('Accuracy', -score.get('MSE', 0.0) if not is_classif else 0.0)),
            'mse': float(score.get('MSE', 0.0)) if not is_classif else None
        }

        # Persist every candidate separately; promote only the actual winner.
        model_path = self._save_model_candidate(
            model,
            ticker=ticker,
            target=data.get("target_name", "unknown"),
            model_type=m_type,
        )
        results["models"].append(
            {
                "model_type": m_type,
                "model_path": str(model_path),
            }
        )
        self.diary.log_event(
            ticker=ticker, model_name=m_type, target=data.get('target_name', 'unknown'),
            metrics=score_val,
            context_fingerprint=data.get('context_fingerprint', 'default'),
            context_pattern_seq=data.get('context_pattern_seq')
        )

        return score_val

    def _finalize_ticker_results(self, results: dict[str, Any], winner: str | None, best_score: float) -> dict:
        """Packages the final results dictionary."""
        results['winner'] = winner
        results['best_score'] = float(best_score) if best_score > -np.inf else None
        results['winner_metrics'] = results['metrics'].get(winner, {})
        winner_record = next(
            (
                item
                for item in results.get("models", [])
                if item.get("model_type") == winner
            ),
            None,
        )
        if winner_record:
            winner_path = Path(winner_record["model_path"])
            champion_path = self._promote_champion_file(
                winner_path,
                ticker=str(results.get("ticker") or "unknown"),
                target=str(results.get("target_name") or "unknown"),
            )
            results["winner_model_path"] = str(winner_path)
            results["model_path"] = str(champion_path)
        return results

    def _save_model_candidate(
        self,
        model: Any,
        *,
        ticker: str,
        target: str,
        model_type: str,
    ) -> Path:
        """Persist one trained candidate without overwriting other models."""
        filename = f"model_{ticker}_{target}_{model_type}.joblib"
        path = self.output_dir / filename
        try:
            joblib.dump(model, path)
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Model candidate saved: {path}")
            return path
        except (OSError, TypeError, Exception) as e:
            self.logger.error(
                f"Error saving model candidate {filename}: {e}",
                exc_info=True,
            )
            raise TrainingException(
                f"Failed to save model candidate {filename}: {e}"
            ) from e

    def _promote_champion_file(
        self,
        winner_path: Path,
        *,
        ticker: str,
        target: str,
    ) -> Path:
        """Copy the selected winner to the stable champion path."""
        champion_path = self.output_dir / f"CHAMP_{ticker}_{target}.joblib"
        shutil.copy2(winner_path, champion_path)
        return champion_path

    def _save_champion(self, model: Any, ticker: str, target: str):
        """Compatibility helper for callers that already selected a winner."""
        path = self.output_dir / f"CHAMP_{ticker}_{target}.joblib"
        try:
            joblib.dump(model, path)
            return path
        except (OSError, TypeError, Exception) as e:
            self.logger.error(
                f"Error saving champion {path.name}: {e}",
                exc_info=True,
            )
            raise TrainingException(
                f"Failed to save champion {path.name}: {e}"
            ) from e

    def _generate_summary(self, results: dict[str, Any]) -> dict[str, Any]:
        """
        Generate training summary statistics.

        Common summary generation logic shared by all trainers.

        Args:
            results: Dictionary of training results by ticker

        Returns:
            Summary dictionary with statistics
        """
        total_tickers = len(results)
        successful_tickers = sum(1 for r in results.values() if r.get('status') == 'success')
        failed_tickers = total_tickers - successful_tickers

        # Calculate average score if available
        scores = []
        for result in results.values():
            if 'best_score' in result:
                scores.append(result['best_score'])

        avg_score = np.mean(scores) if scores else None

        return {
            "total_tickers": total_tickers,
            "successful_tickers": successful_tickers,
            "failed_tickers": failed_tickers,
            "success_rate": successful_tickers / total_tickers if total_tickers > 0 else 0,
            "average_score": float(avg_score) if avg_score is not None else None,
            "timestamp": datetime.now().isoformat()
        }

    def _validate_data_context(self, data_context: dict[str, Any]) -> bool:
        """
        Validate that data_context has required fields.

        Args:
            data_context: Data context to validate

        Returns:
            True if valid, False otherwise
        """
        required_keys = ['X_train', 'y_train', 'X_test', 'y_test', 'target_name']
        for key in required_keys:
            if key not in data_context or data_context[key] is None:
                self.logger.warning(f"Data context missing required key: {key}")
                return False
        return True
