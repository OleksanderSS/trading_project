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
from src.targets.timeframe_contract import target_horizon_bars
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


#: Which metric decides a classification context — for the arena that picks
#: the winner AND for the gate that judges it. One name, one place: the two
#: read it separately and a third copy is how `max_features` ended up with
#: three answers and Sharpe with two.
#:
#: Owner's decision, 2026-08-31, taken on measured numbers. F1 with
#: average='binary' scores the positive class only, so it is maximised by
#: saying yes more often — and the champion that prompted this predicted 1 on
#: 86.5% of rows against a 26.2% event rate, beating "always yes" by 0.0046
#: while sitting 39 points below "always no" on plain accuracy. Balanced
#: accuracy gives BOTH constants exactly 0.5, so a score above it says
#: something about the model rather than about the class balance. On that
#: metric the same champion scores 0.5257.
CLASSIFICATION_METRIC = 'BalancedAccuracy'
REGRESSION_METRIC = 'R2'


def _governing_metric(is_classif: bool) -> str:
    return CLASSIFICATION_METRIC if is_classif else REGRESSION_METRIC


def _opponent(holdout: dict[str, Any], key: str) -> str:
    """One rung's score for the refusal message, or "n/a" if unmeasured.

    An unmeasured opponent must read as unmeasured. Printing 0.0000 for a
    rung that never ran is how a gate ends up looking stricter in its logs
    than it was in fact.
    """
    value = holdout.get(key)
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{float(value):.4f}"


def _sigma_multiplier(alpha: float, family_size: int) -> float:
    """How many standard errors a margin must clear, given how many were tried.

    One-sided, because the gate asks whether the model is BETTER, not whether
    it differs. Bonferroni over the family: each attempt is tested at
    `alpha / family_size`, so the whole run keeps its stated error rate
    instead of spending it once per context.

    Bonferroni assumes independent tests and the attempts here are not -- the
    eight model types see the same data -- so this is conservative. The
    alternative, a permutation null per context, costs one training run per
    permutation and is not affordable at 27 contexts; when it becomes
    affordable it should replace this.
    """
    from statistics import NormalDist

    size = max(int(family_size), 1)
    per_test = min(max(alpha / size, 1e-12), 0.5)
    return float(NormalDist().inv_cdf(1.0 - per_test))


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
            # How many promotion attempts this run makes. The gate turns it
            # into the number of standard errors a margin must clear: a bar
            # set for one test, applied 27 times, is not the bar it claims.
            "promotion_family_size": data.get("promotion_family_size"),
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
            results['winner_feature_importance'] = self._winner_feature_importance(
                winner_model, winner_columns
            )
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

        # The arena must select on the SAME metric the gate judges by. It did
        # not: selection took F1, so the winner of every classification
        # context was whichever model said "yes" most confidently, and the
        # gate then measured that winner on a metric it had not competed on.
        score_val = float(score.get(_governing_metric(is_classif), 0.0))
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

    @staticmethod
    def _winner_feature_importance(winner_model: Any,
                                   winner_columns: list[str] | None) -> dict[str, float]:
        """Which features the winner actually leaned on.

        `extract_native_feature_importance` unwraps the model wrapper and reads
        `feature_importances_`, `coef_` or `get_feature_importance` — it works,
        it is covered by tests, and the only caller it ever had lives in
        src/archive/dead_pipeline_code/modeling/training.py. When the live
        stage replaced that code it kept the artifact and dropped the call,
        passing a literal `{}` instead. All 3,207 feature-stability artifacts
        from the 2026-08-14 batch therefore read
        "feature_importance_status": "not_available_from_model" with a count of
        0 — from RandomForest, CatBoost, LightGBM and XGBoost winners, every
        one of which exposes importances.

        The columns matter as much as the model. A winner is fitted on its own
        budget (5 to 35 of 388), and `_importance_dict` returns {} on a length
        mismatch — silently, the same way it reads when the model genuinely has
        nothing to give. So the names passed here must be the ones the model
        was FITTED on, not the full prepared frame.
        """
        if winner_model is None or not winner_columns:
            return {}
        try:
            from src.pipeline.stages.modeling import pipeline_control_artifacts

            return pipeline_control_artifacts.extract_native_feature_importance(
                winner_model, list(winner_columns)
            )
        except Exception as e:  # noqa: BLE001 - any failure here means "unknown"
            # `{}` is what a model with nothing to offer returns, so a FAILURE
            # to read importances arrived at the artifact writer wearing the
            # same clothes as a genuine absence -- and the artifact then said
            # "not_available_from_model", which is a claim about the MODEL made
            # on the strength of an exception. The docstring above records what
            # that cost once already: 3,207 artifacts from the 2026-08-14 batch
            # reported no importances from RandomForest, CatBoost, LightGBM and
            # XGBoost winners, every one of which exposes them.
            #
            # The names are logged at error level with the model type and the
            # column count, because those two numbers are what tells a length
            # mismatch apart from an unsupported model.
            logging.getLogger(__name__).error(
                "Could not read the winner's feature importance from %s over "
                "%d column(s) (%s: %s). This is a failed READ, not a model "
                "without importances -- do not record it as the latter.",
                type(winner_model).__name__, len(winner_columns),
                type(e).__name__, e,
            )
            return {}

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
        projected = self._project(X_holdout, columns)
        preds = winner_model.predict(projected)
        score = self.evaluator.calculate(y_holdout, preds, task_type=task_type)
        metric_key = _governing_metric(is_classif)

        results['winner_holdout_predictions'] = self._holdout_prediction_series(
            X_holdout, y_holdout, preds,
            self._holdout_probabilities(winner_model, projected, is_classif),
        )
        results['winner_holdout_metrics'] = {
            'status': 'measured',
            'score': float(score.get(metric_key, 0.0)),
            'metric': metric_key,
            'accuracy': float(score.get('Accuracy', -score.get('MSE', 0.0) if not is_classif else 0.0)),
            'mse': float(score.get('MSE', 0.0)) if not is_classif else None,
            'holdout_sample_count': int(len(X_holdout)),
            # Rows are not evidence; EVENTS are. A binary target with 331
            # holdout rows and three positives passes every row-count check
            # trivially, and the precision measured on it means nothing —
            # measured on the 2026-08-15 batch, 36 of 58 promoted contexts
            # carried fewer than ten positive events.
            **self._holdout_event_count(y_holdout, is_classif),
            **self._score_naive_baselines(data, is_classif, task_type, metric_key),
            **self._score_passive_holding(y_holdout, preds, is_classif),
            **self._score_single_feature_baseline(
                data, is_classif, task_type, metric_key, self.evaluator),
        }

        # How big a difference this holdout can produce by luck. Measured
        # against the opponent that actually binds, and stored beside the two
        # scores so the gate compares like with like.
        holdout = results['winner_holdout_metrics']
        opponent = holdout.pop('_baseline_prediction', None)
        if opponent is not None:
            holdout['baseline_margin_sigma'] = self._block_bootstrap_sigma(
                np.asarray(y_holdout).ravel(),
                np.asarray(preds).ravel(),
                np.asarray(opponent).ravel(),
                task_type=task_type,
                metric_key=metric_key,
            )
            holdout['baseline_margin'] = (
                None if holdout.get('score') is None
                or holdout.get('baseline_score') is None
                else float(holdout['score']) - float(holdout['baseline_score'])
            )

    @staticmethod
    def _holdout_event_count(y_holdout: Any, is_classif: bool) -> dict[str, Any]:
        """How many times the thing being predicted actually happened.

        Reported separately from the row count because they answer different
        questions and only one of them bounds what can be concluded. A
        precision measured on five positives has a 95% interval roughly thirty
        points wide; the same figure on a hundred is worth acting on.

        Absent for a regression target, where "event" has no meaning — and
        absent rather than zero, so the gate can tell "not applicable" from
        "nothing happened".
        """
        if not is_classif or y_holdout is None:
            return {}
        try:
            values = pd.to_numeric(
                pd.Series(np.asarray(y_holdout).ravel()), errors='coerce'
            ).dropna()
            if values.empty:
                return {}
            if not set(np.unique(values)) <= {0.0, 1.0}:
                # Multiclass: the minority class is what limits the estimate.
                counts = values.value_counts()
                return {
                    'holdout_event_count': int(counts.min()),
                    'holdout_event_rate': float(counts.min() / len(values)),
                }
            return {
                'holdout_event_count': int(values.sum()),
                'holdout_event_rate': float(values.mean()),
            }
        except (ValueError, TypeError, AttributeError):
            return {}

    @staticmethod
    def _holdout_probabilities(winner_model: Any, X: Any,
                               is_classif: bool) -> "np.ndarray | None":
        """The confidence behind each holdout call, not only the call.

        `predict` collapses a probability into 0 or 1, and 0 or 1 is all this
        stage ever persisted: the 2026-08-14 batch stored exactly two distinct
        values across 4,844 out-of-sample rows. Everything downstream that
        could act on confidence — position size, a per-regime cut-off, any
        ranking of one signal against another — was handed a coin flip and a
        near-certainty as the same number.

        What that costs is measurable on that same batch. On
        target_hourly_breakout_1h the winner fired 22 times in NORMAL and was
        right 91% of the time, and fired 138 times in TRENDING_UP and was
        right 48% — at balanced accuracies of 0.841 and 0.837. It ranks
        equally well in both regimes; only the cut-off is wrong for one of
        them. Raising it is a one-line change that cannot even be attempted
        while the probability is discarded at this line.

        Every wrapper under src/models already implements `predict_proba`, and
        SVMModel passes `probability=True` to SVC specifically so that it can.
        Nothing in the training path had ever asked.

        Returns P(positive class) for a binary target — directly comparable
        against a threshold — and the probability of the predicted class for a
        multiclass one. None for regression, or for any model that cannot
        produce one: an absent confidence must stay absent rather than be
        invented as 1.0.
        """
        if not is_classif or not hasattr(winner_model, 'predict_proba'):
            return None
        try:
            proba = np.asarray(winner_model.predict_proba(X))
            if proba.ndim != 2 or proba.shape[1] < 2:
                return None
            classes = np.asarray(getattr(winner_model, 'classes_', None)
                                 if getattr(winner_model, 'classes_', None) is not None
                                 else getattr(getattr(winner_model, 'model', None),
                                              'classes_', []))
            if proba.shape[1] == 2:
                # Binary: the positive class, wherever sklearn ordered it.
                positive = 1
                if classes.size == 2:
                    match = np.flatnonzero(classes == classes.max())
                    positive = int(match[0]) if match.size else 1
                return proba[:, positive]
            return proba.max(axis=1)
        except (ValueError, TypeError, AttributeError, IndexError, NotImplementedError) as e:
            logging.getLogger(__name__).warning(
                f"Winner exposes predict_proba but it failed ({e}); "
                f"holdout confidence will be absent for this context."
            )
            return None

    @staticmethod
    def _holdout_prediction_series(X_holdout: Any, y_holdout: Any, preds: Any,
                                   proba: Any = None) -> list[dict[str, Any]]:
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

            confidence = None
            if proba is not None:
                flat = np.asarray(proba).ravel()
                if len(flat) >= n:
                    confidence = flat

            return [
                {
                    'datetime': stamps[i],
                    'prediction': float(y_pred[i]),
                    'actual': float(y_true[i]),
                    'probability': (
                        float(confidence[i])
                        if confidence is not None and np.isfinite(confidence[i])
                        else None
                    ),
                }
                for i in range(n)
                if np.isfinite(y_pred[i]) and np.isfinite(y_true[i])
            ]
        except (ValueError, TypeError, AttributeError, IndexError) as e:
            logging.getLogger(__name__).warning(
                f"Could not retain holdout predictions: {e}"
            )
            return []

    @staticmethod
    def _score_single_feature_baseline(data: dict, is_classif: bool,
                                       task_type: str, metric_key: str,
                                       evaluator: Any) -> dict[str, Any]:
        """Can ONE column and a straight line do what the model does?

        Added 2026-08-20 after measuring `target_hourly_breakout_1h`, the
        target that produced more champions than any other. It asks whether
        price crosses today's upper Bollinger band within four bars — and the
        distance from the close to that band, one arithmetic expression known
        at the time of the forecast, scores **AUC 0.9666** on it. Event rate
        goes from 8.1% overall to 64.0% in the decile nearest the band.

        On money, at matched selectivity, on the real holdout:

            model, top 30% by probability     3,494 trades   -0.00022
            closest 30% to the band, no model 3,495 trades   -0.00021

        Identical. Two thousand two hundred features, a model competition and a
        promotion gate reproduced a one-line geometric fact.

        The existing opponents cannot see this. A constant predictor and a
        persistence predictor are both about the SHAPE of the series; neither
        asks whether a single column already contains the answer. This is the
        same family as the seven indicator targets retired as tautologies
        ("tomorrow's SMA_20 is nineteen-twentieths known today") — breakout
        survived that purge wearing a less obvious disguise.

        The feature is chosen on TRAIN and scored on the holdout, so the
        baseline cannot cheat in the way the model is forbidden to.
        """
        out: dict[str, Any] = {'single_feature_score': None,
                               'single_feature_name': None}
        X_tr, y_tr = data.get('X_train'), data.get('y_train')
        X_ho, y_ho = data.get('X_holdout'), data.get('y_holdout')
        if any(v is None for v in (X_tr, y_tr, X_ho, y_ho)):
            out['single_feature_status'] = 'no_train_or_holdout'
            return out
        try:
            xt = pd.DataFrame(X_tr).select_dtypes('number')
            xh = pd.DataFrame(X_ho).select_dtypes('number')
            # The target must carry the FRAME'S index, not a fresh 0..n-1 one.
            #
            # `corrwith` aligns on the index. `prepare_data_for_models` gives
            # every split a `model_datetime` DatetimeIndex, so a target built
            # with the default RangeIndex shared not one label with it: every
            # correlation came out NaN, `dropna()` emptied the series, and
            # this baseline reported "no usable feature" for EVERY context it
            # was ever asked about.
            #
            # Measured 2026-08-31 on the run this was found in: the same data
            # scores 1.0 with a RangeIndex and returns nothing with the
            # DatetimeIndex the pipeline actually passes. So rung 5 of the
            # opponent ladder has been silently absent since the datetime
            # index was introduced -- the rung that caught
            # target_hourly_breakout_1h, where distance to the upper Bollinger
            # band scores AUC 0.9666 and earns the same money as the model.
            #
            # The failure was invisible because "no usable feature" is a
            # legitimate state, and the gate only binds when the score is not
            # None. An opponent that cannot be measured does not block, so a
            # dead rung and a passed rung look identical from outside.
            # Lengths checked BEFORE the Series is built: passing a mismatched
            # index to the constructor raises, and an exception here would be
            # caught below and filed as "failed: ValueError" -- a third way for
            # this rung to disappear quietly.
            train_values = np.asarray(y_tr).ravel()
            holdout_values = np.asarray(y_ho).ravel()
            shared = [c for c in xt.columns if c in xh.columns]
            if (not shared or len(xt) != len(train_values)
                    or len(xh) != len(holdout_values)):
                out['single_feature_status'] = 'shape_mismatch'
                return out
            yt = pd.Series(train_values, index=xt.index).astype(float)
            yh = pd.Series(holdout_values, index=xh.index).astype(float)
            corr = xt[shared].corrwith(yt).abs().dropna()
            if corr.empty:
                out['single_feature_status'] = 'no_usable_feature'
                return out
            best = str(corr.idxmax())
            a, b = np.polyfit(xt[best].fillna(0.0), yt, 1)   # one column, one line
            pred = a * xh[best].fillna(0.0).to_numpy() + b
            if is_classif:
                # Threshold where the train event rate says to, so the baseline
                # fires as often as the target actually occurs.
                cut = np.quantile(a * xt[best].fillna(0.0).to_numpy() + b,
                                  1.0 - float(yt.mean()))
                pred = (pred >= cut).astype(int)
            score = evaluator.calculate(yh.to_numpy(), pred, task_type=task_type)
            out['single_feature_score'] = float(score.get(metric_key, 0.0))
            out['single_feature_name'] = best
            out['single_feature_status'] = 'measured'
        except Exception as exc:  # noqa: BLE001 - a baseline must not kill training
            out['single_feature_status'] = f'failed: {type(exc).__name__}'
        return out

    #: Share of the holdout the model would actually act on. Matches the
    #: ranking decision the 2026-08-20 walk-forward used, so the number here
    #: means the same thing as the number that produced the finding.
    PASSIVE_TOP_QUANTILE = 0.30

    @staticmethod
    def _score_passive_holding(y_holdout: Any, preds: Any,
                               is_classif: bool) -> dict[str, Any]:
        """What the model adds over simply holding everything.

        Added 2026-08-20, because the absence of this one number is what made a
        worthless result read as a triumph. A walk-forward over eleven
        independent folds scored positive in ELEVEN OF ELEVEN and looked like
        success -- until the passive baseline was put beside it:

            absolute target   11/11 folds positive   excess over passive +0.00021, t=0.55
            relative target    9/11 folds positive   excess over passive +0.00132, t=2.78

        The first arm was earning the MARKET and adding noise. The gate's
        existing opponents cannot see this: a constant predictor and a
        persistence predictor are both about the SHAPE of the series, and
        neither asks the question an investor asks, which is whether acting on
        the model beats owning the same thing and doing nothing.

        Reported for every regression target whose values look like returns,
        whether or not it gates, so that no figure is ever read without it.

        The heuristic for "looks like a return" is deliberate and narrow: a
        finite, signed series whose 99th percentile of absolute value is under
        0.5. A target measured in prices, volumes or indicator levels is not
        comparable to holding anything, and says so rather than inventing a
        benchmark it cannot support.
        """
        if is_classif:
            # A class label carries no payoff. Making this comparison would
            # need the return column plumbed through training, which it is
            # not -- and reporting `0.0` here would be a fabricated benchmark.
            return {'excess_over_passive': None,
                    'passive_status': 'not_applicable_classification'}
        try:
            y = pd.Series(y_holdout).astype(float).to_numpy().ravel()
            p = pd.Series(preds).astype(float).to_numpy().ravel()
        except (TypeError, ValueError):
            return {'excess_over_passive': None,
                    'passive_status': 'unreadable'}
        if len(y) != len(p) or len(y) == 0:
            return {'excess_over_passive': None,
                    'passive_status': 'length_mismatch'}

        ok = np.isfinite(y) & np.isfinite(p)
        y, p = y[ok], p[ok]
        if len(y) < 10:
            return {'excess_over_passive': None,
                    'passive_status': 'too_few_rows'}
        if not (np.nanpercentile(np.abs(y), 99) < 0.5 and (y < 0).any() and (y > 0).any()):
            return {'excess_over_passive': None,
                    'passive_status': 'not_a_return_target'}
        if np.allclose(p, p[0]):
            # Every row ranks the same, so "the top 30%" is arbitrary and the
            # comparison would measure the sort order, not the model.
            return {'excess_over_passive': None,
                    'passive_status': 'predictions_are_constant'}

        cut = np.quantile(p, 1.0 - BaseTrainer.PASSIVE_TOP_QUANTILE)
        picked = p >= cut
        if picked.sum() < 5:
            return {'excess_over_passive': None,
                    'passive_status': 'too_few_selected'}
        passive = float(np.mean(y))
        selected = float(np.mean(y[picked]))

        # Beating passive on RAW return is not enough, and 2026-08-20 showed
        # exactly how it fools this gate. A cross-sectional model was carried
        # all the way to a real overlapping portfolio over 27 years:
        #
        #   passive equal weight   CAGR +18.06%  vol 22.77%  Sharpe 0.79
        #   the strategy           CAGR +23.62%  vol 30.72%  Sharpe 0.77
        #
        # It beat passive in 20 years of 28 with a median excess of +7.15%,
        # and every per-trade figure said "edge". It was LEVERAGE: volatility
        # 1.35x, return 1.31x. The same passive holding levered to the same
        # volatility returns +23.14%, so the model's contribution was +0.48% a
        # year -- with a drawdown of -68% against passive's -51%.
        #
        # Concentrating into a top share raises risk mechanically, so a gate
        # that reads only the mean promotes leverage and calls it skill. The
        # comparison has to be at MATCHED RISK: what would owning the same
        # thing return if it took the risk this selection takes?
        passive_std = float(np.std(y, ddof=1))
        selected_std = float(np.std(y[picked], ddof=1))
        matched = None
        if np.isfinite(passive_std) and passive_std > 1e-12:
            matched = passive * (selected_std / passive_std)
        return {
            'passive_mean': passive,
            'selected_mean': selected,
            'selected_count': int(picked.sum()),
            'excess_over_passive': selected - passive,
            'passive_std': passive_std,
            'selected_std': selected_std,
            'risk_matched_passive': matched,
            'excess_risk_adjusted': None if matched is None else selected - matched,
            'passive_status': 'measured',
        }

    def _score_naive_baselines(self, data: dict[str, Any], is_classif: bool,
                               task_type: str, metric_key: str) -> dict[str, Any]:
        """Score the naive predictors the winner has to beat.

        Three of them, and which one binds matters enormously:

        - CONSTANT: the best single answer -- every observed class, or the
          train mean. Fitted on TRAIN alone.
        - PERSISTENCE: "the same thing, h bars ago", lagged inside each
          series. Zero work, no model, no features.
        - CLOCK: what the weekday and hour alone say, fitted on TRAIN.

        The gate checked two of these five rungs for regression and one for
        classification, which is why seven 15m "champions" passed it on
        2026-08-30 and then lost to hand-run opponents on the same holdouts.
        The remaining rungs of the ladder in docs/WORKING_METHOD.md -- the
        base rate, which the best constant subsumes, and the source-column
        rule, which `_score_single_feature_baseline` covers separately -- are
        checked elsewhere in this same gate.

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
                # The BEST constant, not the most common one. F1 with
                # average='binary' scores the positive class, so "always the
                # majority class" is always-zero and scores exactly 0.0 — a
                # bar anything clears by predicting a single true positive.
                #
                # The negative control found this within minutes: models
                # trained on a SHUFFLED target passed the gate at the same
                # rate as models trained on the real one, 28% each. They
                # degenerate to predicting almost all ones, and always-ones
                # scores 2p/(1+p) — 0.61 on a holdout with a 44% positive
                # rate, against a majority-class bar of zero.
                #
                # Scoring every observed class and keeping the strongest makes
                # the opponent what it should be: the best a model can do
                # while learning nothing.
                #
                # Since 2026-08-31 the governing metric is balanced accuracy
                # (`CLASSIFICATION_METRIC`), on which EVERY constant scores
                # exactly 0.5 — so this loop now returns 0.5 whichever class
                # wins, and the bar stops depending on the class balance at
                # all. The loop is kept rather than replaced by the constant
                # 0.5: it is the metric's job to say what a constant is worth,
                # not this function's assumption, and the same code has to
                # keep working if the metric is ever changed back.
                observed = np.unique(y_train[~pd.isna(y_train)])
                if observed.size == 0:
                    return out
                candidates = list(observed)
            else:
                candidates = [float(np.nanmean(y_train))]

            # The PREDICTIONS of the winning constant are kept, not only its
            # score: the gate needs the binding opponent's own series to
            # measure how far apart the two really are (see
            # `_block_bootstrap_sigma`). Popped again before the metrics are
            # written out, so nothing array-shaped reaches the artifact.
            constants = [
                (
                    float(
                        self.evaluator.calculate(
                            y_holdout, np.full(n, candidate), task_type=task_type
                        ).get(metric_key, 0.0)
                    ),
                    np.full(n, float(candidate)),
                )
                for candidate in candidates
            ]
            constant_score, constant_prediction = max(constants, key=lambda p: p[0])
            out['baseline_constant_score'] = constant_score
            out['baseline_score'] = constant_score
            out['baseline_kind'] = 'constant'
            out['_baseline_prediction'] = constant_prediction

            if n < 3:
                return out

            # The persistence opponent has to be a FORECAST, and this one was
            # not. It predicted y[t] from y[t-1] -- but for a target with a
            # horizon of h bars, y[t-1] is the outcome measured from t-1 to
            # t+h-2, which nobody can know at t. For h=5 it needs four more
            # days of prices.
            #
            # Because such a target overlaps itself, that unknowable value is
            # also an excellent predictor. Measured on the 2026-08-22 batch:
            # lag-1 autocorrelation 0.778 on target_relative_return_5d, so the
            # baseline scored R^2 0.5564 -- and the gate, which takes the best
            # opponent, then demanded every model beat 0.56 instead of the
            # 0.0000 the constant baseline set. No return or direction target
            # has ever been promoted in this project, across 4,613 models.
            #
            # Lagging by the horizon fixes the arrow of time: y[t-h] is the
            # most recent value actually observable at t. Measured, the same
            # baseline then scores -1.0879 -- worse than the mean, so the
            # constant becomes the opponent, which is the standard bar.
            #
            # This does NOT hand out champions. The models on those targets
            # scored -0.09 and -0.01, still below the constant's 0.0. It only
            # stops them being compared against an oracle.
            #
            # h=1 targets are unaffected: there y[t-1] IS known at t, and the
            # measurement confirms it (lag-1 and lag-h agree exactly on
            # target_return_1d). Backward-window targets are also left alone --
            # their smoothness is real and known in time.
            # Unresolved means lag 1, deliberately. The first instinct was to
            # skip the opponent instead, and that was wrong: the targets whose
            # names carry no horizon are the INDICATOR ones, and for a
            # backward-looking window shifted forward by a single bar, y[t-1]
            # genuinely IS known at t. Persistence there is both legitimate and
            # the most valuable opponent in the set -- it scores R2 0.9994 on
            # target_sma_20_f1, which is how a model that merely retraces a
            # moving average gets caught. Skipping it would have weakened the
            # gate exactly where it works.
            #
            # So the only behaviour that changes is for targets proven to look
            # more than one bar forward.
            # The horizon has to count the FORWARD WINDOW, not just the name.
            #
            # `target_horizon_bars` reads the suffix: `_5d` is five bars, `_1d`
            # is one. That is right for targets whose name carries their whole
            # reach and wrong for every windowed one.
            # `target_daily_trend_strength_1d` is `shift: -1, window: 20` --
            # it reaches twenty bars ahead while its name says one.
            #
            # Measured 2026-09-03 on the batch, lag-by-name against
            # lag-by-window:
            #
            #     target_daily_trend_strength_1d   R2  0.8973  ->  -1.0213
            #     target_hourly_breakout_1h    BalAcc  0.8573  ->   0.5406
            #     target_daily_momentum_score_1d   R2  0.7579  ->  -1.0020
            #
            # The left column is what this gate was using. It is not a property
            # of those targets -- at their own horizon they are as
            # unpredictable as any return (-1.02 is what a lag scores on a
            # series it cannot predict, by construction). It is an ORACLE: the
            # opponent was handed a value nobody can know at forecast time, and
            # a model on `target_daily_trend_strength_1d` had to beat R2 0.897
            # to be promoted, which no honest forecast will ever do.
            #
            # So the direction of the damage is false REFUSALS, and the targets
            # it silenced never had a chance to be judged. Nine of eighteen
            # targets in the batch have the two horizons disagreeing;
            # `target_hourly_volume_spike_1h` is name 1 against window 23.
            #
            # This is #191 again, on the path #191 did not reach: that fix made
            # the lag the horizon instead of one bar, and this makes the
            # horizon the real one instead of the name's. `_get_target_horizon_rows`
            # is the policy manager's answer and already exists for the purge
            # gap, which needs the same number for the same reason.
            from src.pipeline.stages.modeling.walk_forward_validation import (
                _get_target_horizon_rows,
            )

            target_name = str(data.get('target_name') or '')
            named = target_horizon_bars(target_name, data.get('timeframe')) or 1
            windowed = _get_target_horizon_rows(target_name) or named
            horizon = max(int(named), int(windowed))
            if horizon != named:
                self.logger.info(
                    "Persistence lag for %s is %d bars, not the %d its name "
                    "implies: the target reaches past its own suffix.",
                    target_name, horizon, named,
                )

            persistence = self._persistence_prediction(
                y_true, y_train,
                horizon=horizon,
                groups=data.get('holdout_groups'),
                is_classif=is_classif,
            )
            if persistence is not None:
                out['baseline_persistence_score'] = float(
                    self.evaluator.calculate(
                        y_holdout, persistence, task_type=task_type
                    ).get(metric_key, 0.0)
                )
                out['baseline_persistence_lag_bars'] = horizon
                out['baseline_persistence_grouped'] = (
                    data.get('holdout_groups') is not None
                )

            # WHICH clock scheme is the opponent is decided on VALIDATION,
            # and only then scored on the holdout.
            #
            # It used to be decided on the holdout: all three schemes were
            # scored against y_holdout and the best kept. That was deliberate
            # and documented -- "flattering the opponent is the safe direction
            # for a gate, and the constant baseline already does the same
            # across its candidate classes" -- and the second half of that
            # reasoning does not hold. The constant's candidates are the
            # observed classes, and for a binary target under balanced
            # accuracy every constant scores exactly 0.5, so its maximum
            # selects nothing; measured on twelve null runs the constant came
            # back 0.5000 every time. The clock's three schemes are three
            # genuinely different predictors with real variance, so taking
            # their maximum on the holdout inflates the opponent by the
            # max-of-three selection effect.
            #
            # The cost is not that the gate is strict. It is that the gate's
            # own arithmetic stops meaning what it says: the margin's sigma is
            # computed as though the baseline were a fixed quantity, and a
            # maximum chosen on the same rows it is then compared against is
            # not one. This project spent R11 establishing what its bar means;
            # an opponent that peeks at the answer takes that back, in the
            # conservative direction and by an unmeasured amount.
            #
            # Validation is the honest split for this because it is where the
            # MODEL was chosen too, so opponent and champion are picked under
            # the same rules. Where validation cannot serve, the scheme is
            # fixed by a stated order rather than by the holdout, and the
            # record says which happened -- a silent fallback to the old
            # behaviour would be the same defect with better manners.
            clock_prediction = None
            holdout_schemes = self._clock_prediction(
                data, y_train, is_classif, split='X_holdout'
            )
            if holdout_schemes:
                chosen, chosen_on = self._choose_clock_scheme(
                    data, y_train, is_classif, task_type, metric_key,
                    list(holdout_schemes),
                )
                if chosen is not None:
                    prediction, buckets = holdout_schemes[chosen]
                    scored = float(
                        self.evaluator.calculate(
                            y_holdout, prediction, task_type=task_type
                        ).get(metric_key, 0.0)
                    )
                    if np.isfinite(scored):
                        out['baseline_clock_score'] = scored
                        out['baseline_clock_scheme'] = chosen
                        out['baseline_clock_buckets'] = buckets
                        out['baseline_clock_scheme_chosen_on'] = chosen_on
                        clock_prediction = prediction

            predictions = {
                'persistence': persistence,
                'clock': clock_prediction,
            }

            # The bar is the STRONGEST opponent measured, not the first one.
            # Every rung is kept in the record either way, because "which
            # opponent bound" is the most informative single number the gate
            # produces -- a champion that loses to the clock and one that
            # loses to the constant have failed for completely different
            # reasons.
            for kind, key in (
                ('persistence', 'baseline_persistence_score'),
                ('clock', 'baseline_clock_score'),
            ):
                value = out.get(key)
                if (value is not None and np.isfinite(value)
                        and value > float(out['baseline_score'])):
                    out['baseline_score'] = float(value)
                    out['baseline_kind'] = kind
                    if predictions.get(kind) is not None:
                        out['_baseline_prediction'] = predictions[kind]
            return out
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.warning(f"Could not score naive baselines: {e}")
            return out

    #: Resamples for the margin's standard error. Two hundred is enough for a
    #: standard deviation (the quantity used here); a confidence INTERVAL
    #: would need an order of magnitude more, and none is taken.
    MARGIN_BOOTSTRAP_RESAMPLES = 200

    def _block_bootstrap_sigma(self, y_true: np.ndarray, model: np.ndarray,
                               baseline: np.ndarray, *, task_type: str,
                               metric_key: str) -> float | None:
        """How far apart model and opponent could be by luck alone.

        The gate compared the two scores and promoted on any difference at
        all, `min_baseline_margin` being 0.0. On the first champion of the
        2026-08-31 run that difference was **0.0046** of F1 -- a model that
        beats "always predict the same class" in the fourth decimal.

        #175 had already established the principle on a different target: the
        edge there was 0.041 against an opponent whose own weekly figure
        varied by 0.07, and the honest conclusion was not "signal" or "no
        signal" but "the data ran out before the question". A margin is only
        evidence when it is larger than the noise in the measurement itself.

        Measured PAIRED, on the same resampled rows for both series, because
        model and opponent are scored on the identical holdout and their
        errors are correlated. The standard error of the difference is
        smaller than either score's own, so this is the tighter and fairer
        quantity -- not a way of raising the bar arbitrarily.

        A MOVING BLOCK bootstrap, not an i.i.d. one: holdout rows are a time
        series and adjacent rows are dependent, so resampling single rows
        would understate the spread and hand back a margin that looks
        significant because the resampling pretended the data were
        independent. Block length is the usual n^(1/3).

        Returns None when the holdout is too short to resample, which the
        caller must treat as "no margin could be measured" rather than as a
        margin of zero.
        """
        n = int(len(y_true))
        if n < 60:
            return None
        block = max(1, int(round(n ** (1.0 / 3.0))))
        blocks_needed = int(np.ceil(n / block))
        rng = np.random.default_rng(0)          # fixed: the gate must be reproducible
        offsets = np.arange(block)

        differences: list[float] = []
        for _ in range(self.MARGIN_BOOTSTRAP_RESAMPLES):
            starts = rng.integers(0, n - block + 1, size=blocks_needed)
            rows = (starts[:, None] + offsets).ravel()[:n]
            truth = y_true[rows]
            try:
                a = float(self.evaluator.calculate(
                    truth, model[rows], task_type=task_type).get(metric_key, np.nan))
                b = float(self.evaluator.calculate(
                    truth, baseline[rows], task_type=task_type).get(metric_key, np.nan))
            except (ValueError, TypeError, ZeroDivisionError):
                continue
            if np.isfinite(a) and np.isfinite(b):
                differences.append(a - b)

        if len(differences) < self.MARGIN_BOOTSTRAP_RESAMPLES // 2:
            return None
        return float(np.std(differences, ddof=1))

    @staticmethod
    def _persistence_prediction(y_true: np.ndarray, y_train: np.ndarray, *,
                                horizon: int, groups: Any,
                                is_classif: bool) -> np.ndarray | None:
        """"The same thing, h bars ago" -- lagged WITHIN each series.

        Two things were wrong with the previous version, and both made the
        opponent weaker than it should be:

        1. It was scored for regression only. For a binary target "did this
           already happen h bars ago" is just as legitimate an opponent, and
           on a persistent target it is a strong one. Classification simply
           never met it.
        2. It lagged by ROW. A pooled frame interleaves 22 tickers at the
           same timestamp, so `y[t-h]` is another company a few minutes
           earlier, not this company h bars ago -- an opponent measuring
           nothing in particular. With `groups` the lag is taken inside each
           ticker, which is what the sentence "tomorrow equals today" means.

        The first `horizon` rows of each series have no predecessor. They get
        the train answer -- the majority class, or the train mean -- because
        that is what is actually known before the holdout starts. Seeding
        them with the truth would hand the opponent free exact predictions.
        """
        n = int(np.asarray(y_true).size)
        if n < 3 or horizon < 1 or n <= horizon:
            return None
        train = np.asarray(y_train, dtype=float).ravel()
        train = train[np.isfinite(train)]
        if train.size == 0:
            return None
        if is_classif:
            values, counts = np.unique(train, return_counts=True)
            fill = float(values[int(np.argmax(counts))])
        else:
            fill = float(train.mean())

        prediction = np.full(n, fill, dtype=float)
        labels = None if groups is None else np.asarray(groups).ravel()
        if labels is None or labels.size != n:
            prediction[horizon:] = y_true[:-horizon]
            return prediction

        for label in pd.unique(labels):
            rows = np.flatnonzero(labels == label)
            if rows.size <= horizon:
                continue
            prediction[rows[horizon:]] = y_true[rows[:-horizon]]
        return prediction

    #: A calendar bucket needs this many training rows before it is allowed
    #: its own answer. Below it the bucket mean is noise wearing a schedule.
    #: Order used when validation cannot choose the clock scheme: most
    #: specific first, so the fallback is the strictest opponent available
    #: rather than the luckiest one. Stated here so the choice is a rule and
    #: not an accident of dictionary order.
    CLOCK_SCHEME_FALLBACK_ORDER: tuple[str, ...] = (
        "weekday_hour", "weekday", "hour",
    )

    def _choose_clock_scheme(self, data: dict[str, Any], y_train: np.ndarray,
                             is_classif: bool, task_type: str, metric_key: str,
                             available: list[str]) -> tuple[str | None, str]:
        """Pick the clock opponent's scheme WITHOUT looking at the holdout.

        Returns the scheme and how it was chosen, because "chosen on
        validation" and "chosen because validation was unusable" are different
        facts about the number that follows, and a gate that cannot say which
        one applies is back to reporting a score whose provenance nobody can
        check.
        """
        if not available:
            return None, "none_available"

        y_val = data.get('y_val')
        val_schemes = self._clock_prediction(
            data, y_train, is_classif, split='X_val'
        )
        if y_val is not None and val_schemes:
            observed = np.asarray(y_val, dtype=float).ravel()
            best_name, best_score = None, -np.inf
            for scheme in available:
                candidate = val_schemes.get(scheme)
                if candidate is None:
                    continue
                prediction = candidate[0]
                if len(prediction) != observed.size:
                    continue
                scored = float(
                    self.evaluator.calculate(
                        y_val, prediction, task_type=task_type
                    ).get(metric_key, 0.0)
                )
                if np.isfinite(scored) and scored > best_score:
                    best_name, best_score = scheme, scored
            if best_name is not None:
                return best_name, "validation"

        for scheme in self.CLOCK_SCHEME_FALLBACK_ORDER:
            if scheme in available:
                return scheme, "fallback_fixed_order"
        return available[0], "fallback_fixed_order"

    MIN_CLOCK_BUCKET_ROWS = 20

    #: How the calendar is cut. Three schemes rather than one, because the
    #: finest is not the strongest: (weekday, hour) is 168 buckets, and an
    #: hourly context with 3,000 training rows leaves 18 rows in each -- below
    #: the minimum, so the opponent goes unmeasured exactly where the effect
    #: it looks for is largest. Hour alone and weekday alone are coarse enough
    #: to be estimated on the histories this project actually has.
    CLOCK_SCHEMES: tuple[str, ...] = ("hour", "weekday", "weekday_hour")

    @classmethod
    def _clock_buckets(cls, index: pd.DatetimeIndex, scheme: str) -> np.ndarray:
        if scheme == "hour":
            return index.hour.to_numpy()
        if scheme == "weekday":
            return index.weekday.to_numpy()
        return (index.weekday * 100 + index.hour).to_numpy()

    @classmethod
    def _clock_prediction(cls, data: dict[str, Any], y_train: np.ndarray,
                          is_classif: bool, split: str = 'X_holdout',
                          ) -> dict[str, tuple[np.ndarray, int]]:
        """What the calendar alone predicts, fitted on train.

        The rung the gate was missing entirely. A model can beat the constant
        and beat persistence while reproducing nothing but "this hour of this
        weekday behaves like this" -- and on intraday targets that is a large
        effect: the first and last bars of a session carry most of the
        volatility and most of the volume, every day, for reasons that have
        nothing to do with any feature in this pipeline.

        Buckets come from the row's own timestamp, cut three ways (see
        `CLOCK_SCHEMES`). On a daily frame the hour is constant, so the hour
        scheme collapses to a constant and the other two to weekday -- the
        right behaviour rather than a special case.

        For regression the bucket predicts its own training mean. For
        classification it predicts the event whenever the bucket's training
        event rate exceeds the overall training rate -- "the clock says this
        slot is unusually likely" -- since predicting each bucket's majority
        class on an imbalanced target degenerates to predicting nothing at
        all and would make this opponent free to beat.

        Every scheme is returned; the CALLER decides which one is the
        opponent. `split` names the frame the predictions are aligned to, so
        the same buckets -- always fitted on TRAIN -- can be laid over the
        validation rows to CHOOSE a scheme and over the holdout rows to SCORE
        it. Choosing and scoring on the same rows is what this used to do, and
        why it no longer does, is argued at the call site.

        Returns an empty mapping when the frame carries no usable timestamps
        or no bucket has enough training rows -- an honest "not measured"
        rather than a passing score.
        """
        empty: dict[str, tuple[np.ndarray, int]] = {}
        train_index = getattr(data.get('X_train'), 'index', None)
        holdout_index = getattr(data.get(split), 'index', None)
        if not isinstance(train_index, pd.DatetimeIndex):
            return empty
        if not isinstance(holdout_index, pd.DatetimeIndex):
            return empty

        train = np.asarray(y_train, dtype=float).ravel()
        if train.size != len(train_index):
            return empty
        observed = np.isfinite(train)
        if not observed.any():
            return empty
        train_values = train[observed]
        overall = float(train_values.mean())

        out: dict[str, tuple[np.ndarray, int]] = {}
        for scheme in cls.CLOCK_SCHEMES:
            train_buckets = cls._clock_buckets(train_index, scheme)[observed]
            holdout_buckets = cls._clock_buckets(holdout_index, scheme)
            prediction = np.full(len(holdout_index),
                                 0.0 if is_classif else overall, dtype=float)
            used = 0
            for bucket in np.unique(holdout_buckets):
                rows = train_buckets == bucket
                if int(rows.sum()) < cls.MIN_CLOCK_BUCKET_ROWS:
                    continue
                bucket_value = float(train_values[rows].mean())
                answer = (
                    float(bucket_value > overall) if is_classif else bucket_value
                )
                prediction[holdout_buckets == bucket] = answer
                used += 1
            # One bucket is a constant wearing a schedule; the constant
            # opponent already covers it and it only clutters the record.
            if used > 1:
                out[scheme] = (prediction, used)
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

        Conditions, all cheap and all measurable from what training already
        produces:

        1. the winner was scored on a real holdout (not the selection split);
        2. that holdout was not vanishingly small, in rows OR in events;
        3. the winner beat the STRONGEST train-only naive opponent on it --
           best constant, lag-h persistence within its own series, and the
           weekday/hour clock (see `_score_naive_baselines`);
        4. it beat one column fitted with one straight line;
        5. it beat holding the same exposure, raw and at matched risk;
        6. its score is not implausibly high, which reads as a leak.

        Everything richer -- cost stress, shadow outcomes -- needs evidence
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
        # The effective bar is the LARGER of the configured margin and one
        # standard error of the model-minus-opponent difference, measured on
        # the holdout itself. Owner's decision of 2026-08-31, taken after the
        # gate promoted a model that led its opponent by 0.0046 of F1. Set
        # false to compare the two scores directly again.
        require_sigma = bool(cfg.get('require_baseline_margin_sigma', True))
        # How many standard errors the margin must clear. One was the rule
        # until 2026-09-01, and one standard error ONE-SIDED is p = 0.159 --
        # measured, not estimated: twenty panels of pure noise were run
        # through this gate and it promoted three of them (CLAIMS.md R11).
        # 15% of nothing becomes a champion.
        #
        # The multiplier is derived from the number of promotion attempts the
        # run makes, because that is what the old rule never looked at: run 7
        # took 27 verdicts over 8 model types, and a per-test 0.159 across 27
        # verdicts is about four false champions out of nine.
        #
        #     family_size = 1   ->  1.645 sigma  (a single test at 5%)
        #     family_size = 27  ->  2.90 sigma
        #     family_size = 216 ->  3.50 sigma
        #
        # `family_size` comes from the caller when it knows how many contexts
        # it will evaluate, and from config otherwise. Setting it to 1 does
        # NOT restore the old behaviour: the old behaviour was never a 5%
        # test, it was a 16% one.
        family_alpha = float(cfg.get('family_alpha', 0.05))
        family_size = int(
            results.get('promotion_family_size')
            or cfg.get('family_size', 1)
            or 1
        )
        sigma_multiplier = _sigma_multiplier(family_alpha, family_size)
        # Ten is a floor, not a target: below it a proportion carries an
        # interval about thirty points wide, so the comparison against the
        # baseline cannot separate skill from chance. Set
        # `min_holdout_events: 0` to restore the previous behaviour.
        min_events = int(cfg.get('min_holdout_events', 10))
        # A model that does not beat owning the same thing and doing
        # nothing has not earned a promotion, whatever its R2. Only
        # binds where the comparison is measurable -- see
        # _score_passive_holding. Set false to restore the old behaviour.
        require_excess = bool(cfg.get('require_excess_over_passive', True))
        # Measured 2026-08-20: a model can beat passive in 20 years of 28
        # while adding +0.48%/yr over the same holding levered to its own
        # volatility. Raw excess alone promotes leverage as skill.
        require_risk_adjusted = bool(cfg.get('require_risk_adjusted_excess', True))
        # A model that only reproduces one column has not earned a
        # promotion. Measured on target_hourly_breakout_1h, 2026-08-20.
        require_beat_single_feature = bool(
            cfg.get('require_beat_single_feature', True))
        min_excess = float(cfg.get('min_excess_over_passive', 0.0))

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

            # Rows are not evidence; events are. The gate asked "does it beat
            # the baseline" and never "are there enough events for the answer
            # to mean anything", so a context with three positives passed on
            # the strength of a number with a thirty-point interval around it.
            #
            # Measured on the 2026-08-15 batch: 36 of 58 promoted contexts
            # carried fewer than ten events, and their figures went into every
            # aggregate beside the ones that meant something. Extending the
            # hourly history fourfold moved rows per context 86 -> 331 and
            # events only 5 -> 7, which is why this has to be a gate rather
            # than something more data will fix.
            events = holdout.get('holdout_event_count')
            if events is not None and int(events) < min_events:
                reasons.append(
                    f"holdout carries {int(events)} events, minimum is "
                    f"{min_events}: a score on fewer cannot distinguish skill "
                    f"from chance"
                )

            # The opponent an investor actually has. The constant and
            # persistence baselines are both about the SHAPE of the series;
            # neither asks whether acting on the model beats owning the same
            # exposure. Measured 2026-08-20: without this column an arm that
            # scored positive in 11 of 11 walk-forward folds was earning the
            # market and adding +0.00021 of noise.
            # Raw excess first: a model that does not even out-return passive
            # has nothing to argue about.
            excess = holdout.get('excess_over_passive')
            # Then at matched risk, because raising risk raises raw return for
            # free and this gate cannot otherwise tell that apart from skill.
            risk_adj = holdout.get('excess_risk_adjusted')
            if require_risk_adjusted and risk_adj is not None and risk_adj <= min_excess:
                reasons.append(
                    f"at matched risk the model returns {holdout.get('selected_mean'):+.5f} "
                    f"against {holdout.get('risk_matched_passive'):+.5f} for the same "
                    f"exposure levered to the same volatility "
                    f"(selection std {holdout.get('selected_std'):.5f} vs "
                    f"{holdout.get('passive_std'):.5f}): excess {risk_adj:+.5f}. "
                    f"Raw out-performance here is leverage, not skill"
                )
            if require_excess and excess is not None and excess <= min_excess:
                reasons.append(
                    f"selecting on this model returns {holdout.get('selected_mean'):+.5f} "
                    f"against {holdout.get('passive_mean'):+.5f} for holding "
                    f"everything: excess {excess:+.5f} does not clear "
                    f"{min_excess:+.5f}"
                )

            score = holdout.get('score')
            # Can one column and a straight line already do this? Measured
            # 2026-08-20: distance to the upper Bollinger band scores AUC
            # 0.9666 on target_hourly_breakout_1h, and at matched selectivity
            # earns the same money as the model (-0.00021 vs -0.00022). The
            # constant and persistence opponents cannot see that, because both
            # are about the shape of the series rather than about whether a
            # single feature already contains the answer.
            single = holdout.get('single_feature_score')
            if (require_beat_single_feature and single is not None
                    and np.isfinite(single) and score is not None
                    and np.isfinite(score) and score <= single + min_margin):
                reasons.append(
                    f"holdout score {score:.4f} does not beat the SINGLE FEATURE "
                    f"'{holdout.get('single_feature_name')}' fitted with one "
                    f"straight line ({single:.4f}): the model reproduces what one "
                    f"column already says"
                )

            baseline = holdout.get('baseline_score')

            if score is None or not np.isfinite(score):
                reasons.append("holdout score is not finite")
            elif baseline is None:
                reasons.append("naive baseline could not be scored for comparison")
            else:
                # The bar is the opponent PLUS the noise in the comparison.
                #
                # `min_baseline_margin` was 0.0, so any difference promoted --
                # and on 2026-08-31 the first champion of the run cleared its
                # opponent by 0.0046 of F1. The standard error of that very
                # difference, measured by block bootstrap on the same holdout,
                # is what says whether 0.0046 is an edge or a rounding artefact.
                # The owner's decision, taken on the number: require one.
                sigma = holdout.get('baseline_margin_sigma')
                required = min_margin
                bound_by = 'the configured margin'
                if (require_sigma and sigma is not None and np.isfinite(sigma)
                        and float(sigma) * sigma_multiplier > required):
                    required = float(sigma) * sigma_multiplier
                    bound_by = (
                        f'{sigma_multiplier:.2f} standard errors of the '
                        f'difference, the bar for {family_size} attempt(s) '
                        f'at a family-wise {family_alpha:.0%}'
                    )

                if float(score) <= float(baseline) + required:
                    # Name the rung. "Does not beat the naive baseline" was
                    # true of every refusal and told the reader nothing:
                    # losing to a constant and losing to the clock are
                    # different diagnoses, and only the second one says the
                    # target is a schedule.
                    reasons.append(
                        f"holdout score {float(score):.4f} does not beat the "
                        f"{holdout.get('baseline_kind', 'naive')} baseline "
                        f"{float(baseline):.4f} by {required:.4f} "
                        f"({bound_by}; margin "
                        f"{float(score) - float(baseline):+.4f}, sigma "
                        f"{_opponent(holdout, 'baseline_margin_sigma')}) "
                        f"[constant {_opponent(holdout, 'baseline_constant_score')}, "
                        f"lag-{holdout.get('baseline_persistence_lag_bars', '?')} "
                        f"{_opponent(holdout, 'baseline_persistence_score')}, "
                        f"clock {_opponent(holdout, 'baseline_clock_score')}, "
                        f"one feature {_opponent(holdout, 'single_feature_score')}]"
                    )
                elif require_sigma and sigma is None:
                    # An unmeasurable margin must not pass as a measured one.
                    reasons.append(
                        "the margin over the naive baseline could not be "
                        "measured (holdout too short to resample), so it "
                        "cannot be distinguished from noise"
                    )
                elif float(score) >= self.IMPLAUSIBLE_SCORE:
                    # A near-perfect score on market data is the signature of
                    # a leak or a degenerate target, and the one case where a
                    # HIGHER number must be treated worse than a middling one.
                    reasons.append(
                        f"holdout score {float(score):.4f} is implausibly high "
                        f"(>= {self.IMPLAUSIBLE_SCORE}); audit for leakage "
                        f"before promoting"
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
