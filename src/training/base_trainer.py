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

import numpy as np
import pandas as pd

from src.pipeline.constants import (
    champion_filename,
    model_candidate_filename,
    preprocessor_filename,
)
from src.config.feature_budget import get_model_max_features
from src.config.unified_config_manager import get_current_config
from src.core.logging.logger import ProjectLogger
from src.factories.model_factory import ModelFactory
from src.meta_learning.memory.diary_engine import DiaryEngine
from src.metrics.model.ml_evaluator import MLEvaluator
from src.models.artifact_store import get_model_artifact_store


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
        self.artifact_store = get_model_artifact_store()

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
            "timeframe": data.get("timeframe", ""),
            "target_name": data.get("target_name", "unknown"),
            "selected_features": list(data.get("feature_names") or []),
            # Carried through so _finalize_ticker_results can save it beside a
            # promoted champion; popped there, so it never reaches metadata.
            "_preprocessor": data.get("preprocessor"),
        }

        # 1. Data validation and prep
        X_train, y_train = data.get('X_train'), data.get('y_train')
        if X_train is None or y_train is None:
            return {"status": "failed", "ticker": ticker, "reason": "incomplete_data"}

        try:
            # 2. Determine which models to train
            results["training_sanity"] = self._check_training_sanity(data)
            model_types = self._prepare_model_training_list(ticker, data)
            is_classification = 'classification' in data.get('target_type', '')

            # 3. Execute training loop
            best_score, winner_name = self._execute_model_training_cycle(
                ticker, model_types, data, is_classification, results
            )

            # 4. Finalize results
            return self._finalize_ticker_results(results, winner_name, best_score)

        except (ValueError, TypeError, Exception) as e:
            # Name the target too: without it a failure here says only which
            # ticker died, which is not enough to tell whether the cause is a
            # degenerate target, a bad feature set, or a broken model config.
            target_name = data.get("target_name", "unknown")
            self.logger.error(
                f"Error during training for {ticker} target={target_name}: {e}",
                exc_info=True,
            )
            raise TrainingException(
                f"Training failed for {ticker} target={target_name}: {e}"
            ) from e

    def _prepare_model_training_list(self, ticker: str, data: dict[str, Any]) -> list[str]:
        """Determines the set of model types to train for this ticker."""
        plan = data.get('plan', {})
        ticker_plan = plan.get('ticker_plans', {}).get(ticker, {})
        model_types = ticker_plan.get('models')

        if not model_types:
            from src.factories.model_factory import ModelFactory
            model_types = self.config_manager.get('models.enabled_types', ModelFactory.get_available_models())
        return list(model_types) if model_types else []

    def _execute_model_training_cycle(self, ticker: str, model_types: list[str],
                                   data: dict[str, Any], is_classif: bool,
                                   results: dict[str, Any]) -> tuple[float, str | None]:
        """Iterates through model types and trains each one.

        Selection uses each candidate's validation score, never the test
        score — the test set is reserved for a single, honest post-selection
        read of the winner (see _record_winner_test_score).
        """
        best_score = -np.inf
        winner_name = None
        winner_model = None
        # Each model is now fitted on ITS OWN feature budget, so the columns
        # it was trained on have to travel with it. Scoring the winner on the
        # holdout without them raises "The feature names should match those
        # that were passed during fit" and takes the whole stage down.
        winner_columns: list[str] | None = None

        for m_type in model_types:
            try:
                score_val, model, columns = self._train_individual_model(
                    ticker, m_type, data, is_classif, results
                )

                if score_val > best_score:
                    best_score = score_val
                    winner_name = m_type
                    winner_model = model
                    winner_columns = columns
            except Exception as e:
                # Deliberately broad. This loop exists precisely so that one
                # unusable (ticker, target, model) combination is skipped and
                # the remaining models still train. The previous tuple
                # (ValueError/TypeError/AttributeError/KeyError/ZeroDivisionError)
                # missed the most common real failure: CatBoostError inherits
                # straight from Exception, so "Target contains only one unique
                # value" -- a degenerate target, e.g. a rare-event class with
                # no positives in the window -- escaped this guard, hit the
                # caller's handler, and took down the ticker, the stage, and
                # the whole pipeline run with it.
                self.logger.error(
                    f"Failed to train {m_type} for {ticker} "
                    f"target={data.get('target_name', 'unknown')}: {e}"
                )
                results.setdefault("skipped_models", []).append({
                    "model": m_type,
                    "target": data.get("target_name", "unknown"),
                    "reason": str(e),
                })
                continue

        if winner_model is not None:
            # Travels into the champion metadata, which is what Stage 5 uses
            # to build the frame it feeds the model. Without it, prediction
            # hands a 35-column model all 388 columns and is refused.
            results['winner_selected_features'] = list(winner_columns or [])
            self._record_winner_test_score(
                winner_model, data, is_classif, results, columns=winner_columns
            )

        return best_score, winner_name

    def _train_individual_model(self, ticker: str, m_type: str, data: dict[str, Any],
                              is_classif: bool, results: dict[str, Any]
                              ) -> tuple[float, Any, list[str] | None]:
        """Train one model on its own feature budget; return score, model, columns.

        LEAKAGE FIX: model selection must score candidates on the validation
        split, not the test split — scoring on test here would let the
        "best model" choice itself be informed by the held-out data that's
        supposed to give an unbiased final read. Falls back to the test
        split only when no validation split is present in `data` (older
        callers that don't produce one), so this stays backward compatible.

        The selected columns are returned because a model fitted on 35 of 388
        columns cannot later be handed all 388: sklearn raises "The feature
        names should match those that were passed during fit". Every caller
        that predicts with this model must project the frame the same way.
        """
        model = self.model_factory.create_model(
            model_name=m_type,
            config=self.config_manager.get(f"models.per_model.{m_type}", {}),
            task_type="classification" if is_classif else "regression",
            is_classification=is_classif
        )

        columns = self._select_features_for_model(m_type, data, is_classif)

        model.train(self._project(data['X_train'], columns), data['y_train'])

        eval_X = data['X_val'] if data.get('X_val') is not None else data['X_test']
        eval_y = data['y_val'] if data.get('y_val') is not None else data['y_test']
        preds = model.predict(self._project(eval_X, columns))

        score = self.evaluator.calculate(
            eval_y, preds,
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
            timeframe=data.get("timeframe", ""),
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

        return score_val, model, columns

    @staticmethod
    def _project(frame: Any, columns: list[str] | None) -> Any:
        """Restrict a feature matrix to `columns`, if it is column-addressable."""
        if not columns or not hasattr(frame, "columns"):
            return frame
        usable = [column for column in columns if column in frame.columns]
        return frame[usable] if usable else frame

    def _select_features_for_model(self, model_type: str, data: dict[str, Any],
                                   is_classif: bool) -> list[str] | None:
        """Pick this model's feature budget, ranked on TRAIN ROWS ONLY.

        The light branch had no budget at all: every model was handed every
        numeric column — a measured median of 388 features against roughly 308
        training rows on a daily context. That is not a weak model, it is an
        unfalsifiable one, and it is why "the champion beat the others" has
        never meant anything here.

        Ranking is by absolute Pearson correlation with the target over the
        TRAINING split only, so neither the validation split that chooses the
        winner nor the holdout that judges it takes any part in deciding which
        columns exist (Codex §6.2). Cheap, order-stable, and deliberately not a
        new selection framework: the point is a budget, and a budget only helps
        if it is applied everywhere rather than perfectly anywhere.

        Returns None when there is nothing to do, in which case the caller
        passes the frame through untouched.
        """
        X_train = data.get('X_train')
        if X_train is None or not hasattr(X_train, 'columns'):
            return None

        budget = get_model_max_features(model_type, self.config_manager)
        if len(X_train.columns) <= budget:
            return list(X_train.columns)

        try:
            y = np.asarray(data.get('y_train')).ravel()
            numeric = X_train.select_dtypes(include=[np.number])
            if numeric.empty or len(y) != len(numeric):
                return list(X_train.columns[:budget])

            y_series = pd.Series(y, index=numeric.index).astype(float)
            correlations = numeric.corrwith(y_series).abs()
            # A column with no variance correlates to NaN; it carries no
            # information either way, so it sorts last rather than randomly.
            ranked = correlations.fillna(-1.0).sort_values(
                ascending=False, kind="mergesort"
            )
            chosen = list(ranked.index[:budget])
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"{model_type}: {len(X_train.columns)} -> {len(chosen)} features"
                )
            return chosen
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            self.logger.warning(
                f"Could not rank features for '{model_type}' ({e}); "
                f"falling back to the first {budget} columns."
            )
            return list(X_train.columns[:budget])

    def _record_winner_test_score(self, winner_model: Any, data: dict[str, Any],
                                is_classif: bool, results: dict[str, Any],
                                columns: list[str] | None = None) -> None:
        """Scores the already-selected winner on the untouched holdout.

        `columns` are the features the winner was FITTED on. They must be
        applied here too — a model trained on 35 of 388 columns rejects the
        full frame outright ("The feature names should match those that were
        passed during fit"), which took the whole modelling stage down on the
        first run after per-model budgets were introduced.

        It never influences which model wins — it's recorded so downstream
        consumers get one selection-independent number.

        The honesty of that number depends entirely on WHICH split arrives
        here, and for a long time the wrong one did. Stage 4's orchestrator
        passes the validation split under the key ``X_test`` (its own
        docstring says so), so this method re-scored the winner on exactly the
        rows that had just chosen it and published the result as
        ``winner_test_metrics``. Every "test" number Stage 4 ever produced was
        a second copy of the selection score.

        The real purged holdout now travels under ``X_holdout``/``y_holdout``,
        a key nothing in the selection path reads. When it is absent, no score
        is emitted at all: an absent measurement must not be representable as
        a passing one.
        """
        X_holdout = data.get('X_holdout')
        y_holdout = data.get('y_holdout')
        if X_holdout is None or y_holdout is None or len(X_holdout) == 0:
            results['winner_holdout_metrics'] = {
                'status': 'no_holdout_available',
                'reason': (
                    "Caller supplied no X_holdout/y_holdout. The winner was "
                    "not measured on any data outside model selection."
                ),
                'holdout_sample_count': 0,
            }
            return

        task_type = "classification" if is_classif else "regression"
        preds = winner_model.predict(self._project(X_holdout, columns))
        score = self.evaluator.calculate(y_holdout, preds, task_type=task_type)
        metric_key = 'F1' if is_classif else 'R2'

        results['winner_holdout_predictions'] = self._holdout_prediction_series(
            X_holdout, y_holdout, preds
        )
        results['winner_holdout_metrics'] = {
            'status': 'measured',
            'score': float(score.get(metric_key, 0.0)),
            'metric': metric_key,
            'accuracy': float(score.get('Accuracy', -score.get('MSE', 0.0) if not is_classif else 0.0)),
            'mse': float(score.get('MSE', 0.0)) if not is_classif else None,
            'holdout_sample_count': int(len(X_holdout)),
            **self._score_naive_baselines(data, is_classif, task_type, metric_key),
        }

    @staticmethod
    def _holdout_prediction_series(X_holdout: Any, y_holdout: Any, preds: Any) -> list[dict[str, Any]]:
        """Keep the winner's holdout predictions, with their timestamps.

        These were computed, reduced to a single R2, and thrown away — and
        they are the only genuinely out-of-sample, time-stamped predictions
        the pipeline produces.

        Stage 7's backtest currently builds its equity curve from Stage 5
        instead, which answers a different question: Stage 5 predicts the
        LATEST bar of each context, one point apiece, so 540 predictions
        pivoted to a `(3, 22)` table. Three time points. That is why the last
        run reported a Sharpe of -329.82 on a volatility of 8.46e-05 — not a
        flat curve, a three-point one. No financial number computed from it
        can mean anything, however good the models are.

        The holdout is ~100-220 purged bars per context that the model never
        saw and never selected on. Retaining these rows is what lets an equity
        curve be built from real out-of-sample forecasts rather than from
        three live signals.

        Returned as plain records so they survive JSON; the caller decides
        where to persist them.
        """
        try:
            index = getattr(X_holdout, 'index', None)
            y_true = np.asarray(y_holdout).ravel()
            y_pred = np.asarray(preds).ravel()
            n = min(len(y_true), len(y_pred))
            if n == 0:
                return []

            if index is not None and len(index) >= n:
                stamps = [
                    value.isoformat() if hasattr(value, 'isoformat') else str(value)
                    for value in list(index)[:n]
                ]
            else:
                stamps = [None] * n

            return [
                {
                    'datetime': stamps[i],
                    'prediction': float(y_pred[i]),
                    'actual': float(y_true[i]),
                }
                for i in range(n)
                if np.isfinite(y_pred[i]) and np.isfinite(y_true[i])
            ]
        except (ValueError, TypeError, AttributeError, IndexError) as e:
            logging.getLogger(__name__).warning(
                f"Could not retain holdout predictions: {e}"
            )
            return []

    def _score_naive_baselines(self, data: dict[str, Any], is_classif: bool,
                               task_type: str, metric_key: str) -> dict[str, Any]:
        """Score the naive predictors the winner has to beat.

        Two of them, and which one binds matters enormously:

        - CONSTANT: majority class, or the train mean. Fitted on TRAIN alone.
        - PERSISTENCE (regression only): "tomorrow equals today", i.e. the
          previous actual value. Zero work, no model, no features.

        The gate used to compare against the constant only, and for a slow,
        trending series that is a hopeless opponent — which is how seven
        indicator-prediction targets produced 138 of 354 champions. Measured on
        this batch, persistence ALONE explains:

            target_sma_20_f1        R2 0.9994
            target_ema_20_f1        R2 0.9994
            target_bb_upper_f1      R2 0.9984
            target_macd_hist_f1     R2 0.9264
            target_volume_ratio_f1  R2 0.8077
            target_atr_14_f5        R2 0.8043
            target_rsi_14_f1        R2 0.8010

        A model scoring 0.998 on tomorrow's SMA_20 has added nothing over
        doing nothing: 19 of the 20 closes in that average are already known.
        Against the train mean it looked like a triumph.

        So the bar is the STRONGER of the two. Nothing is removed and no target
        is disabled — the targets may still be useful, as diagnostics or as
        inputs elsewhere. What changes is that beating a constant no longer
        counts as skill on a series that barely moves.

        Persistence uses the previous ACTUAL value, which is genuinely known at
        forecast time, so this is a fair opponent rather than a leak. The first
        holdout row has no predecessor and is given its own value, which flatters
        the baseline very slightly — deliberately, since erring toward a harder
        gate is the safe direction.
        """
        out: dict[str, Any] = {'baseline_score': None}
        try:
            y_train = np.asarray(data.get('y_train')).ravel()
            y_holdout = data.get('y_holdout')
            if y_train.size == 0 or y_holdout is None:
                return out

            y_true = np.asarray(y_holdout).ravel()
            n = y_true.size
            if n == 0:
                return out

            if is_classif:
                values, counts = np.unique(y_train[~pd.isna(y_train)], return_counts=True)
                if values.size == 0:
                    return out
                constant = values[int(np.argmax(counts))]
            else:
                constant = float(np.nanmean(y_train))

            constant_score = float(
                self.evaluator.calculate(
                    y_holdout, np.full(n, constant), task_type=task_type
                ).get(metric_key, 0.0)
            )
            out['baseline_constant_score'] = constant_score
            out['baseline_score'] = constant_score
            out['baseline_kind'] = 'constant'

            if is_classif or n < 3:
                return out

            persistence = np.empty(n, dtype=float)
            persistence[0] = y_true[0]
            persistence[1:] = y_true[:-1]
            persistence_score = float(
                self.evaluator.calculate(
                    y_holdout, persistence, task_type=task_type
                ).get(metric_key, 0.0)
            )
            out['baseline_persistence_score'] = persistence_score

            if np.isfinite(persistence_score) and persistence_score > constant_score:
                out['baseline_score'] = persistence_score
                out['baseline_kind'] = 'persistence'
            return out
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.warning(f"Could not score naive baselines: {e}")
            return out

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
            results["winner_model_path"] = str(winner_path)

            gate = self._evaluate_promotion_gate(results)
            results["promotion_gate"] = gate

            if gate["passed"]:
                champion_path = self._promote_champion_file(
                    winner_path,
                    ticker=str(results.get("ticker") or "unknown"),
                    timeframe=str(results.get("timeframe") or ""),
                    target=str(results.get("target_name") or "unknown"),
                )
                results["model_path"] = str(champion_path)
                preprocessor_path = self._persist_preprocessor(
                    results.pop("_preprocessor", None),
                    ticker=str(results.get("ticker") or "unknown"),
                    timeframe=str(results.get("timeframe") or ""),
                    target=str(results.get("target_name") or "unknown"),
                )
                if preprocessor_path:
                    results["preprocessor_path"] = str(preprocessor_path)
            else:
                # No CHAMP_ write. Any champion already on disk for this
                # (ticker, timeframe, target) stays as it is -- a failed gate
                # withholds a promotion, it does not retract the incumbent.
                # Downstream keeps resolving the previous champion, which is
                # why blocking here cannot zero out Stage 5.
                self.logger.warning(
                    "Champion NOT promoted for %s/%s/%s: %s",
                    results.get("ticker"), results.get("timeframe"),
                    results.get("target_name"), "; ".join(gate["reasons"]),
                )
        return results

    #: A holdout score at or above this is treated as evidence of leakage or a
    #: degenerate split rather than of skill. Codex §8.7: an R2 near 0.99 or an
    #: accuracy near 100% on financial data should trigger an audit, never an
    #: automatic promotion.
    IMPLAUSIBLE_SCORE = 0.99

    def _check_training_sanity(self, data: dict[str, Any]) -> dict[str, Any]:
        """Shape checks that decide whether a result can mean anything.

        Measured on this project's own artifacts: a typical run fits roughly
        237 features on ~192 training rows with ~19 validation rows. More
        parameters than observations does not produce a weak model, it
        produces an unfalsifiable one -- any score it reports is a property of
        the split, not of the market.

        `blocking` entries stop promotion. `warnings` are recorded and do not.
        """
        X_train = data.get('X_train')
        train_rows = int(len(X_train)) if X_train is not None else 0
        feature_names = data.get('feature_names') or []
        n_features = len(feature_names)
        if not n_features and hasattr(X_train, 'shape') and len(getattr(X_train, 'shape', ())) > 1:
            n_features = int(X_train.shape[1])

        X_val = data.get('X_val')
        X_holdout = data.get('X_holdout')
        val_rows = int(len(X_val)) if X_val is not None else 0
        holdout_rows = int(len(X_holdout)) if X_holdout is not None else 0

        blocking: list[str] = []
        warnings: list[str] = []

        # Whether an over-parameterised fit blocks promotion outright is a
        # policy choice, and the measurement argues for making it opt-in.
        # On the 2026-08-06 batch EVERY context is over-parameterised:
        # AAPL 1d 327 usable numeric features on ~196 train rows, SPY 60m
        # 605 on ~151, AAPL 15m 870 on ~390. Blocking by default would stop
        # the light branch producing champions at all.
        #
        # The holdout-versus-baseline test below is the empirical answer to
        # "is this overfit": a model that memorised its training rows fails on
        # a split it has never been evaluated against. Refusing on the ratio
        # alone pre-emptively discards models that evidence might vindicate.
        # So the ratio is reported loudly and left to the operator.
        config_manager = getattr(self, 'config_manager', None)
        block_on_ratio = False
        if config_manager is not None:
            try:
                cfg = config_manager.get('training.promotion_gate', {}) or {}
                block_on_ratio = bool(cfg.get('block_when_features_exceed_rows', False))
            except (AttributeError, TypeError, KeyError):
                block_on_ratio = False

        if train_rows == 0:
            blocking.append("no training rows")
        elif n_features >= train_rows:
            message = (
                f"{n_features} features on {train_rows} training rows: "
                f"at least as many parameters as observations"
            )
            (blocking if block_on_ratio else warnings).append(message)
        elif n_features > train_rows / 3:
            warnings.append(
                f"{n_features} features on {train_rows} training rows "
                f"(ratio {n_features / train_rows:.2f}); Codex §2.5 suggests "
                f"20-40 train-only selected features for a first honest contour"
            )

        if 0 < val_rows < 20:
            warnings.append(f"validation split has only {val_rows} rows")
        if holdout_rows == 0:
            warnings.append("no holdout split supplied")

        return {
            'train_rows': train_rows,
            'validation_rows': val_rows,
            'holdout_rows': holdout_rows,
            'feature_count': n_features,
            'blocking': blocking,
            'warnings': warnings,
        }

    def _persist_preprocessor(self, preprocessor: dict[str, Any] | None, *,
                              ticker: str, timeframe: str, target: str) -> Path | None:
        """Save the imputer+scaler the champion was fitted behind.

        A model trained on standardised features is unusable without the
        transform that standardised them, and this pipeline kept the two
        apart: `prepare_data_for_models` fits a SimpleImputer and a
        StandardScaler on the training split, returns both in `light_data`,
        and the prediction path collected neither. Stage 5 sliced raw columns
        out of the feature frame, so a tree that learned "close > 0.3" in
        z-space was asked about a close of 120, and a linear model fitted
        against unit variance was handed a volume of 5e7.

        Measured on a real champion (35 features): z-scored input gave
        [0.033, -0.023, 0.156, ...]; the identical model on the raw values
        Stage 5 supplies gave [128288, 127314, 133867, ...]. Nothing crashed.

        The fit-time column ORDER travels with them. A StandardScaler applied
        to the same columns in a different order is a different transform, and
        the frame Stage 5 assembles has no reason to match by luck.
        """
        if not isinstance(preprocessor, dict):
            return None
        scaler = preprocessor.get('scaler')
        imputer = preprocessor.get('imputer')
        feature_names = list(preprocessor.get('feature_names') or [])
        if scaler is None and imputer is None:
            return None
        if not feature_names:
            self.logger.warning(
                "Refusing to save a preprocessor for %s/%s/%s without the "
                "fit-time column order: applying it later would be guesswork.",
                ticker, timeframe, target,
            )
            return None

        path = self.output_dir / preprocessor_filename(ticker, timeframe, target)
        payload = {
            'imputer': imputer,
            'scaler': scaler,
            'feature_names': feature_names,
        }
        if not self.artifact_store.save_joblib(payload, path):
            self.logger.warning(f"Failed to save preprocessor {path.name}")
            return None
        return path

    def _evaluate_promotion_gate(self, results: dict[str, Any]) -> dict[str, Any]:
        """Decide whether the winner may be written as CHAMP_.

        Promotion used to be an unconditional `shutil.copy2` executed for
        whichever model happened to score highest, however badly: a model that
        lost to predicting the training mean still became the champion the
        prediction stage would load, because nothing ever compared it to
        anything. "Champion" named the winner of the round, not a model shown
        to be useful.

        Three conditions, all cheap and all measurable from what training
        already produces:

        1. the winner was scored on a real holdout (not the selection split);
        2. that holdout was not vanishingly small;
        3. the winner beat the train-only naive baseline on it.

        Everything richer that a promotion gate should eventually check --
        walk-forward stability, cost stress, shadow outcomes -- needs evidence
        this pipeline does not yet produce. This is the floor, not the target.
        """
        # Not every BaseTrainer subclass carries a config_manager (BatchTrainer
        # does not), and a promotion gate that raises is worse than one that
        # falls back to its documented defaults.
        config_manager = getattr(self, 'config_manager', None)
        cfg = {}
        if config_manager is not None:
            try:
                cfg = config_manager.get('training.promotion_gate', {}) or {}
            except (AttributeError, TypeError, KeyError):
                cfg = {}
        enabled = bool(cfg.get('enabled', True))
        min_rows = int(cfg.get('min_holdout_rows', 20))
        min_margin = float(cfg.get('min_baseline_margin', 0.0))

        holdout = results.get('winner_holdout_metrics') or {}
        reasons: list[str] = []

        if not enabled:
            return {
                'passed': True,
                'reasons': ['gate_disabled_by_config'],
                'enabled': False,
            }

        if holdout.get('status') != 'measured':
            reasons.append(
                f"no holdout measurement ({holdout.get('status', 'missing')})"
            )
        else:
            n = int(holdout.get('holdout_sample_count', 0))
            if n < min_rows:
                reasons.append(f"holdout has {n} rows, minimum is {min_rows}")

            score = holdout.get('score')
            baseline = holdout.get('baseline_score')
            if score is None or not np.isfinite(score):
                reasons.append("holdout score is not finite")
            elif baseline is None:
                reasons.append("naive baseline could not be scored for comparison")
            elif float(score) <= float(baseline) + min_margin:
                reasons.append(
                    f"holdout score {float(score):.4f} does not beat the naive "
                    f"baseline {float(baseline):.4f} by {min_margin}"
                )
            elif float(score) >= self.IMPLAUSIBLE_SCORE:
                # A near-perfect score on market data is the signature of a
                # leak or a degenerate target, and the one case where a HIGHER
                # number must be treated worse than a middling one.
                reasons.append(
                    f"holdout score {float(score):.4f} is implausibly high "
                    f"(>= {self.IMPLAUSIBLE_SCORE}); audit for leakage before "
                    f"promoting"
                )

        # Shape problems make the score above unfalsifiable, so they block
        # regardless of how the comparison went.
        sanity = results.get('training_sanity') or {}
        reasons.extend(sanity.get('blocking') or [])

        return {
            'passed': not reasons,
            'reasons': reasons or ['holdout_measured_and_beats_baseline'],
            'enabled': True,
            'min_holdout_rows': min_rows,
            'min_baseline_margin': min_margin,
        }

    def _save_model_candidate(
        self,
        model: Any,
        *,
        ticker: str,
        timeframe: str,
        target: str,
        model_type: str,
    ) -> Path:
        """Persist one trained candidate without overwriting other models.

        "Other models" has to include the same model type fitted to a
        different timeframe. Stage 4 runs this suite once per (ticker,
        timeframe) and every run wrote to one directory under a name built
        from ticker and target alone -- so the 15m fit, the 60m fit and the
        1d fit were three writes to one path, and only the last survived.

        The champion metadata has always been keyed by
        {ticker}_{timeframe}_{target}_{pattern}, so it listed three distinct
        champions while the files behind them were one file. Whichever
        timeframe Stage 4 happened to process last answered for all three.
        """
        filename = model_candidate_filename(ticker, timeframe, target, model_type)
        path = self.output_dir / filename
        if not self.artifact_store.save_joblib(model, path):
            raise TrainingException(f"Failed to save model candidate {filename}")
        self._invalidate_model_pool_entry(path.stem)
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Model candidate saved: {path}")
        return path

    def _promote_champion_file(
        self,
        winner_path: Path,
        *,
        ticker: str,
        target: str,
        timeframe: str = "",
    ) -> Path:
        """Copy the selected winner to the stable champion path.

        Writes to a fixed name per (ticker, timeframe, target), so the
        ModelPool entry keyed by that stem must be invalidated after
        overwriting — see _invalidate_model_pool_entry(). The timeframe is
        part of the name for the same reason it is part of the candidate
        name: without it the three timeframes' champions are one file.
        """
        champion_path = self.output_dir / champion_filename(ticker, timeframe, target)
        shutil.copy2(winner_path, champion_path)
        self._invalidate_model_pool_entry(champion_path.stem)
        return champion_path

    @staticmethod
    def _invalidate_model_pool_entry(model_stem: str) -> None:
        """Evict `model_stem` from the global ModelPool if present.

        ModelPool.get_model() is a pure cache lookup by this same stem key
        (see model_resolver.py) — it never re-invokes loader_fn for a hit,
        so writing a new file under a name that's already cached would
        otherwise be invisible to a long-running process until the entry
        aged out via LRU eviction, which could be arbitrarily far in the
        future (or never, if the pool isn't full).
        """
        try:
            from src.models.model_pool import get_model_pool
            get_model_pool().remove_model(model_stem)
        except (ImportError, AttributeError) as e:
            logging.getLogger(__name__).warning(
                f"Could not invalidate model pool cache for {model_stem}: {e}"
            )

    def _save_champion(self, model: Any, ticker: str, target: str, timeframe: str = ""):
        """Compatibility helper for callers that already selected a winner."""
        path = self.output_dir / champion_filename(ticker, timeframe, target)
        if not self.artifact_store.save_joblib(model, path):
            raise TrainingException(f"Failed to save champion {path.name}")
        return path

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
