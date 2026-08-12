# audit-ignore: ARCHITECTURAL_USAGE
import datetime
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analytics.analyzers.model_comparison_analyzer import ModelComparisonAnalyzer
from src.config.unified_config_manager import UnifiedConfigManager
from src.core.logging.logger import ProjectLogger
from src.models.adapters.data_preparation import prepare_data_for_models
from src.pipeline.modeling_context import iter_model_contexts
from src.pipeline.stages.base_stage import BaseStage
from src.pipeline.stages.modeling import pipeline_control_artifacts
from src.pipeline.stages.modeling.walk_forward_validation import (
    PipelineWalkForwardValidationEvaluator,
    WalkForwardValidationConfig,
    build_purged_expanding_folds,
)
from src.pipeline.stages.prediction.lineage import (
    source_lineage_attrs,
    trusted_context_fingerprint,
)
from src.pipeline.target_column_utils import is_direct_target_column
from src.pipeline.timeframe_lineage import is_timeframe_token
from src.training.constants import (
    BATCH_TRAINER_DEFAULT_BATCH_SIZE,
    BATCH_TRAINER_DEFAULT_MAX_MEMORY_GB,
    DEFAULT_TEST_SIZE,
)
from src.training.unified_training_manager import TrainerConfig, TrainingStrategy, UnifiedTrainingManager

logger = ProjectLogger.get_logger('ModelingStage')


@dataclass
class TargetProcessingConfig:
    """Configuration for target processing."""
    ticker: str
    df: Any
    target_name: str
    timeframe: str
    champions: dict[str, Any]


class ModelingStage(BaseStage):
    """
    Stage 4: Advanced ML Arena with Pattern-Based Champions.

    🎯 REGIME-SPECIFIC CHAMPIONS:
    - Тренує та зберігає найкращі моделі для кожної пари (Ticker, Context Pattern).
    - Використовує Purged Validation для чесного оцінювання.
    """

    def __init__(self, config_manager: UnifiedConfigManager, brain: dict[str, Any] | None = None, error_handler=None, **kwargs):
        super().__init__(config_manager, error_handler, brain=brain, **kwargs)
        self.modeling_config = self.config_manager.get_config('modeling') or {}
        self.system_config = self.config_manager.get_config('system') or {}

        strategy_str = self.modeling_config.get('strategy', 'hybrid').upper()
        strategy = (TrainingStrategy[strategy_str] if strategy_str in
            TrainingStrategy.__members__ else TrainingStrategy.HYBRID)

        training_config = TrainerConfig(
            strategy=strategy,
            batch_size=self.modeling_config.get('batch_size', BATCH_TRAINER_DEFAULT_BATCH_SIZE),
            max_memory_gb=self.modeling_config.get('max_memory_gb', BATCH_TRAINER_DEFAULT_MAX_MEMORY_GB)
        )

        self.training_manager = UnifiedTrainingManager(training_config)
        self.comparison_analyzer = ModelComparisonAnalyzer()
        # These three were the tail of __init__ until a4ca9176 inserted
        # _resolve_test_size above them; they ended up AFTER that method's
        # `return` and became unreachable, so models_dir and diary_path were
        # never assigned and _init_infrastructure never ran. Restored here.
        self.models_dir = self.config_manager.get_models_path()
        self.diary_path = Path(self.system_config.get('diary_path', 'logs/experience_diary.csv'))
        self._init_infrastructure()

    def _resolve_purge_gap(self, configured: int) -> int:
        """Purge gap widened to cover the furthest target horizon.

        A fixed 10 was passed here while `target_daily_trend_strength_1d`
        looks 20 bars forward (shift -1 over a 20-bar window), so the tail of
        each training split carried targets computed from rows inside the
        following split.
        """
        try:
            from src.policy import get_policy_manager

            gap = get_policy_manager(self.config_manager).purge_gap(configured)
            if gap != configured:
                logger.info(f'Purge gap widened {configured} -> {gap} for target horizon.')
            return gap
        except Exception as e:
            logger.warning(f'Could not resolve purge gap ({e}); using {configured}.')
            return configured

    def _resolve_test_size(self) -> float:
        """Split ratio from PipelinePolicyManager, so there is one answer.

        `self.modeling_config.get('test_size', DEFAULT_TEST_SIZE)` always
        returned the constant: `get_config('modeling')` was None because no
        config file declared that key, while `test_size: 0.2` sat unread under
        the top-level `data_preparation` key. Routing through the policy
        manager makes the configured value actually govern -- and it reports
        which source it came from, so a fallback is visible in the log rather
        than silent.
        """
        try:
            from src.policy import get_policy_manager

            split = get_policy_manager(self.config_manager).split_policy()
            logger.info(
                f'Using test_size={split.test_size} (source: {split.source})')
            return split.test_size
        except Exception as e:
            logger.warning(
                f'Policy manager unavailable ({e}); '
                f'falling back to test_size={DEFAULT_TEST_SIZE}.')
            return self.modeling_config.get('test_size', DEFAULT_TEST_SIZE)

    def _init_infrastructure(self):
        """Initializes the environment."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        if not self.diary_path.exists():
            self.diary_path.parent.mkdir(parents=True, exist_ok=True)
            columns = ['timestamp', 'ticker', 'tf', 'target', 'pattern_id', 'model_name', 'score', 'is_champion']
            pd.DataFrame(columns=columns).to_csv(self.diary_path, index=False)

    async def run(self, **kwargs) -> dict[str, Any]:
        """Runs the full training cycle with Pattern-Aware logic."""
        enriched_data = kwargs.get('enriched_data')
        if enriched_data is None or (isinstance(enriched_data, pd.DataFrame) and enriched_data.empty):
            logger.error('Enriched data not found. Skipping Modeling Stage.')
            return {}

        if kwargs.get("walk_forward_review_only"):
            return self._run_walk_forward_review_only(
                enriched_data,
                **kwargs,
            )

        champions = {}
        metric_artifacts: list[dict[str, Any]] = []
        metric_artifact_dir = Path(
            kwargs.get("pipeline_control_artifact_dir")
            or self.system_config.get(
                "pipeline_control_artifact_dir",
                "data/results/pipeline_control_stage4_training",
            )
        )
        logger.info('--- [Modeling Stage] Starting Regime-Aware Training Arena ---')

        for ticker, timeframe, df in self._iter_model_contexts(enriched_data):
            # The dominant pattern for this ticker's window.
            #
            # This was `df['context_pattern_id'].iloc[-1] if
            # 'context_pattern_id' in df.columns else 'normal'` -- the BARE
            # column name. ContextMapEnricher runs per timeframe and emits
            # context_pattern_id_1d / _60m, so the condition was never true
            # and current_pattern was the literal 'normal' on every pass.
            # Confirmed on the 2026-08-04 run: all 506 champions carry
            # pattern_id='normal'. The pattern is the axis this entire
            # "Regime-Aware Training Arena" is built around, and it had never
            # varied once.
            #
            # Same defect as the context lookup fixed in ce566d0f; this call
            # site was missed because it tests a column directly instead of
            # going through _latest_context_value.
            #
            # THEN: context_pattern_id turned out to be unusable as a key,
            # for a reason independent of which column was read. It is a
            # SHA-256 of a five-fingerprint sequence, each fingerprint ~185
            # tri-state drivers, so the space is astronomically large and
            # every row gets its own value. Measured on the export:
            #
            #     15m  14,209 rows -> 14,201 distinct patterns
            #     60m   5,652 rows ->  5,647 distinct
            #     1d    7,128 rows ->  7,112 distinct
            #
            # One observation per pattern. Keying champions by it gives one
            # champion per row, which is the same amount of information as
            # the constant 'normal' it replaced -- both extremes, neither a
            # regime. The fix in 9fa3a84a read the right column; the column
            # cannot do this job.
            #
            # MARKET_REGIME can: 6 to 8 values per timeframe, well spread
            # (15m: RANGING 5131, TRENDING_UP 4136, TRENDING_DOWN 3363,
            # NORMAL 1012, MEAN_REVERSION 550). That is what a champion
            # keyed by regime needs -- enough repetition for the diary to
            # accumulate evidence per regime.
            #
            # Read from THIS timeframe, not from the daily bar. A 15m model
            # sees 15m features and its edge is a 15m phenomenon; keying it
            # by a regime that changes once a day would hold nearly constant
            # across a session and erase the variation the model exists to
            # trade. The higher-timeframe view is not lost -- the assembler
            # already puts ctx_1d_MARKET_REGIME_1d on the finer rows, so the
            # model can use the daily regime as a FEATURE without it being
            # the key.
            current_pattern = self._latest_context_value(
                df,
                ("MARKET_REGIME", "market_regime", "regime"),
                default='normal',
                timeframe=str(timeframe),
            ) or 'normal'
            logger.info(
                "Ticker %s/%s is currently in pattern: %s",
                ticker,
                timeframe,
                current_pattern,
            )

            await self._process_ticker_with_async(
                ticker,
                df,
                champions,
                current_pattern,
                timeframe=timeframe,
                metric_artifacts=metric_artifacts,
                metric_artifact_dir=metric_artifact_dir,
            )

        logger.info(f'Modeling Stage complete. Trained {len(champions)} expert models.')
        manifests = sorted(
            {
                str(item["manifest"])
                for item in metric_artifacts
                if item.get("manifest")
            }
        )
        holdout_path = self._write_holdout_predictions(champions)
        return {
            'models_metadata': champions,
            'processed_data': enriched_data,
            'pipeline_control_metric_artifacts': metric_artifacts,
            'pipeline_control_metric_artifact_manifests': manifests,
            'holdout_predictions_path': str(holdout_path) if holdout_path else None,
        }

    @staticmethod
    def _write_holdout_predictions(champions: dict[str, Any]) -> "Path | None":
        """Collect every champion's out-of-sample series into one artifact.

        Stage 7 builds its equity curve from Stage 5, which answers a
        different question: Stage 5 predicts the LATEST bar of each context,
        one point apiece, so 540 predictions pivoted to a (3, 22) table. Three
        time points, from which it reported a Sharpe of -329.82 on a
        volatility of 8.46e-05 — a three-point curve, not a nearly flat one.

        The holdout is ~100-220 purged bars per context that the model never
        saw and was never selected on, and each row already carries its
        timestamp, its prediction and the realised value. For a return target
        that realised value IS the return, so an honest out-of-sample curve
        needs no price data at all: position * actual, summed across contexts.
        """
        rows: list[dict[str, Any]] = []
        for context_key, champion in champions.items():
            series = champion.get('holdout_predictions') or []
            for record in series:
                rows.append({
                    'context': context_key,
                    'ticker': champion.get('ticker'),
                    'timeframe': champion.get('timeframe'),
                    'target': champion.get('target_name') or champion.get('target'),
                    'model_type': champion.get('model_type'),
                    'datetime': record.get('datetime'),
                    'prediction': record.get('prediction'),
                    'actual': record.get('actual'),
                })
        if not rows:
            logger.info('No holdout predictions to persist.')
            return None

        frame = pd.DataFrame(rows)
        directory = Path('data/results')
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (
            f"holdout_predictions_{datetime.datetime.now():%Y%m%d_%H%M%S}.parquet"
        )
        frame.to_parquet(path, index=False)
        logger.info(
            'Wrote %d out-of-sample holdout predictions across %d contexts to %s',
            len(frame), frame['context'].nunique(), path.name,
        )
        return path

    def _run_walk_forward_review_only(
        self,
        stage_enriched_data,
        **kwargs,
    ) -> dict[str, Any]:
        """Evaluate development folds without entering normal model training."""
        if not kwargs.get("acknowledge_no_test"):
            raise ValueError(
                "walk_forward_review_only requires acknowledge_no_test=True."
            )
        target_name = kwargs.get("target_column")
        if not target_name:
            raise ValueError(
                "walk_forward_review_only requires one explicit target_column."
            )
        config = WalkForwardValidationConfig(
            min_train_rows=int(kwargs.get("min_train_rows", 360)),
            validation_rows=int(kwargs.get("validation_rows", 120)),
            step_rows=int(kwargs.get("step_rows", 120)),
            purge_rows=int(kwargs.get("purge_rows", 5)),
            max_folds=int(kwargs.get("max_folds", 4)),
            max_features=int(kwargs.get("max_features", 40)),
        )
        evaluator = PipelineWalkForwardValidationEvaluator(config)
        candidates: dict[str, Any] = {}
        for ticker, timeframe, frame in self._iter_model_contexts(
            stage_enriched_data
        ):
            if target_name not in frame.columns or not frame[target_name].notna().any():
                continue
            candidate = evaluator.evaluate(
                frame,
                ticker=ticker,
                timeframe=timeframe,
                target_name=target_name,
                timeframe_context_report=kwargs.get(
                    "timeframe_context_report"
                ),
                source_lineage=kwargs.get("source_lineage"),
            )
            candidates[f"{ticker}_{timeframe}_{target_name}"] = candidate
        if not candidates:
            raise ValueError(
                f"No model context contains target {target_name}."
            )
        return {
            "status": "walk_forward_review_only_complete",
            "review_only": True,
            "walk_forward_validation_candidates": candidates,
            "models_metadata": {},
            "processed_data": stage_enriched_data,
            "timeframe_context_report": kwargs.get(
                "timeframe_context_report"
            ),
            "can_promote_model": False,
            "can_trade": False,
        }

    def _iter_model_contexts(self, enriched_data):
        """Yield isolated ticker/timeframe frames for model preparation."""
        yield from iter_model_contexts(enriched_data)

    async def _process_ticker_with_async(
        self,
        ticker,
        df,
        champions,
        current_pattern,
        *,
        timeframe=None,
        metric_artifacts: list[dict[str, Any]] | None = None,
        metric_artifact_dir: Path | None = None,
    ):
        """Process data for a single ticker."""
        try:
            logger.info(f"Ticker {ticker}: DataFrame columns: {list(df.columns[:20])}... (total {len(df.columns)} columns)")
            target_cols = [
                column
                for column in df.columns
                if is_direct_target_column(column) and df[column].notna().any()
            ]
            logger.info(f"Ticker {ticker}: Found {len(target_cols)} target columns: {target_cols}")
            if not timeframe:
                timeframe = source_lineage_attrs(df).get(
                    "prediction_timeframe"
                )
            if not timeframe:
                raise ValueError(
                    f"Ticker {ticker} has no cadence-validated timeframe"
                )

            for target_name in target_cols:
                # Готуємо дані з PURGED GAP
                prepared_data = prepare_data_for_models(
                    df=df, ticker=ticker, timeframe=timeframe,
                    target_cols=[target_name],
                    gap_size=self._resolve_purge_gap(10),
                    test_size=self._resolve_test_size()
                )

                if not prepared_data:
                    continue

                # Запускаємо уніфіковане тренування
                context_fingerprint = self._build_context_fingerprint(
                    frame=df,
                    prepared_data=prepared_data,
                    ticker=str(ticker),
                    timeframe=str(timeframe),
                    target_name=str(target_name),
                    current_pattern=str(current_pattern),
                )
                training_context = self._build_unified_training_context(
                    prepared_data,
                    target_name=target_name,
                    context_fingerprint=context_fingerprint,
                    context_pattern_seq=self._latest_context_value(
                        df, ("context_pattern_seq",), default=None,
                        timeframe=str(timeframe),
                    ),
                    timeframe=str(timeframe),
                )
                training_results = self.training_manager.execute_unified_training(
                    tickers=[ticker], data_context=training_context
                )

                # Вибираємо переможця для конкретного ПАТЕРНА
                ticker_result = training_results.get('tickers_results', {}).get(ticker, {})
                if ticker_result.get('status') == 'success':
                    winner_name = ticker_result.get('winner')
                    metrics = ticker_result.get('winner_metrics', {})

                    context_key = (
                        f"{ticker}_{timeframe}_{target_name}_{current_pattern}"
                    )
                    artifact_paths = self._write_active_stage4_candidates(
                        ticker=ticker,
                        timeframe=timeframe,
                        target_name=target_name,
                        current_pattern=str(current_pattern),
                        context_fingerprint=context_fingerprint,
                        df=df,
                        prepared_data=prepared_data,
                        ticker_result=ticker_result,
                        output_dir=metric_artifact_dir,
                    )
                    if artifact_paths and metric_artifacts is not None:
                        metric_artifacts.append(artifact_paths)

                    # A blocked promotion must not be announced as a champion.
                    # BaseTrainer withholds the CHAMP_ file when the winner
                    # fails the holdout-versus-baseline gate, but this metadata
                    # was written regardless -- so one run logged
                    # "Champion NOT promoted for AAPL/15m/target_intraday_return_15m"
                    # and then "Pattern Champion ... catboost" for that very
                    # context, seven seconds apart. Stage 5 reads THIS dict, so
                    # the refusal would have been cosmetic: it would resolve the
                    # context anyway and fall back to whatever CHAMP_ file was
                    # already on disk -- which today means a model trained on
                    # the corrupted batch.
                    if not self._champion_is_allowed(ticker_result, context_key):
                        continue

                    stability = self._walk_forward_stability(
                        df, ticker=ticker, timeframe=str(timeframe),
                        target_name=target_name, context_key=context_key,
                    )
                    if stability and not stability.get('passed', True):
                        logger.info(
                            "No champion recorded for %s: %s",
                            context_key, stability.get('reason'),
                        )
                        continue

                    champions[context_key] = {
                        'ticker': ticker,
                        'timeframe': timeframe,
                        'target': target_name,
                        'target_name': target_name,
                        'target_type': training_context.get('target_type'),
                        'pattern_id': current_pattern,
                        'winner': winner_name,
                        'model_type': winner_name,
                        # 'light', not 'unified'. Stage 5 counts categories
                        # with m.get('model_category') == 'light', so every
                        # model this stage produced was reported as neither
                        # light nor heavy: the 2026-08-04 run logged
                        # "Models: 0 light, 0 heavy, 506 total". The models
                        # were there; the label was not the one the counter
                        # looks for. These ARE the light half of the hybrid
                        # split -- models.yaml categories.light.
                        'model_category': 'light',
                        'metrics': metrics,
                        'model_path': ticker_result.get('model_path'),
                        # The WINNER'S OWN columns, not every column the
                        # context offered. Each model is now fitted on its own
                        # feature budget, so a champion trained on 35 of 388
                        # columns must be described by those 35: Stage 5 builds
                        # its input frame from this list
                        # (data_preparation_service: ticker_df_clean[
                        # selected_features]), and handing the model the full
                        # set makes it reject the frame outright.
                        'selected_features': list(
                            ticker_result.get("winner_selected_features")
                            or training_context.get("feature_names")
                            or []
                        ),
                        'context_fingerprint': context_fingerprint,
                        'pipeline_control_metric_artifacts': artifact_paths,
                        # The winner's out-of-sample series, kept so Stage 7
                        # can build an equity curve from real forecasts rather
                        # than from Stage 5's three live signals.
                        'holdout_predictions': ticker_result.get(
                            'winner_holdout_predictions'
                        ) or [],
                        'walk_forward_stability': stability,
                        'timestamp': datetime.datetime.now().isoformat()
                    }

                    self._log_expert_to_diary(champions[context_key], timeframe)
                    logger.info(f"🏆 Pattern Champion for {context_key}: {winner_name} (Score: {metrics.get('score', 0):.4f})")

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f"Error modeling {ticker}: {e}")

    #: A context must show signal on more than one fold before anything from
    #: it is promoted. One fold is a coin toss with a decimal point.
    _MIN_STABLE_FOLDS = 2

    #: Share of folds that must beat their majority baseline. Chosen from the
    #: arithmetic, not from taste: with no signal at all, each fold clears its
    #: baseline about half the time, so
    #:     >= 2 of 4 happens by chance 69% of the time
    #:     >= 3 of 4                   31%
    #:        4 of 4                    6%
    #: A threshold of two would have passed noise more often than not — a
    #: decoration rather than a filter. Three of four still admits a coin-flip
    #: context roughly once in three, which is honest to state and far better
    #: than the alternative of pretending one split proves anything.
    _STABLE_FOLD_SHARE = 0.75

    #: No fold may come out worse than a coin. Not a tuned threshold — 0.5
    #: balanced accuracy IS chance, so this asks only that the context never
    #: collapsed below it on any window. It is independent of the count above,
    #: which measures how OFTEN signal held but not whether it ever fell
    #: apart: AAPL/1d cleared 2 of 4 folds with a worst fold of 0.388, and a
    #: model that is materially worse than guessing on a quarter of the
    #: history has not shown stability, it has shown two lucky windows.
    _MIN_WORST_FOLD_BALANCED_ACCURACY = 0.5

    #: Floors for a shrunken fold geometry. Below these a fold stops being a
    #: measurement: a validation window of a dozen rows and a training window
    #: of eighty tell you about the split, not the market.
    _MIN_FOLD_VALIDATION_ROWS = 40
    _MIN_FOLD_TRAIN_ROWS = 150

    @classmethod
    def _walk_forward_config_for(cls, row_count: int) -> WalkForwardValidationConfig:
        """Fold geometry that fits the context, instead of one fixed size.

        The defaults (min_train 360, validation 120) need ~485 rows for a
        single fold. A daily context has about 511, so it produced exactly ONE
        fold — and one fold cannot show stability, it is a single split with a
        decimal point. That is 396 of 660 contexts, the majority of the
        pipeline, silently exempt from the check.

        Shrinking the windows trades "no measurement" for "noisier
        measurement", which is the better trade: the criterion is only that
        signal held on at least two folds, and a threshold that coarse
        tolerates noise far better than it tolerates absence.

        Intraday contexts keep the defaults, because they can afford them.
        """
        default = WalkForwardValidationConfig()
        if len(build_purged_expanding_folds(row_count, config=default)) >= cls._MIN_STABLE_FOLDS:
            return default

        validation_rows = max(cls._MIN_FOLD_VALIDATION_ROWS, row_count // 8)
        min_train_rows = max(cls._MIN_FOLD_TRAIN_ROWS, row_count // 2)
        return WalkForwardValidationConfig(
            min_train_rows=min_train_rows,
            validation_rows=validation_rows,
            step_rows=validation_rows,
            purge_rows=default.purge_rows,
            max_folds=default.max_folds,
            max_features=default.max_features,
        )

    def _regression_fold_stability(self, frame: "pd.DataFrame", *, target_name: str,
                                   context_key: str) -> dict[str, Any]:
        """Fold stability for a continuous target, in its own currency.

        PipelineWalkForwardValidationEvaluator scores with classification
        metrics, and on a continuous target those are not imprecise but
        undefined: a shuffled target_return_1d with 511 distinct values
        returned a balanced accuracy of 1.0 on every fold. Return targets —
        the ones that matter most — were therefore passing a check that had
        measured nothing.

        The idea was sound, only the metric was wrong. This reuses the parts
        that were right: `build_purged_expanding_folds` for the geometry
        (expanding, purged, horizon-aware) and the same naive opponents the
        holdout gate uses. Per fold, a reference regressor must beat the
        better of "predict the training mean" and "tomorrow equals today".

        Deliberately a REFERENCE model rather than the champion, matching the
        classification path: this asks whether the context holds stable signal
        at all, not whether one particular winner does.
        """
        try:
            from sklearn.ensemble import RandomForestRegressor

            config = self._walk_forward_config_for(len(frame))
            folds = build_purged_expanding_folds(len(frame), config=config)
            if len(folds) < self._MIN_STABLE_FOLDS:
                return {'passed': True, 'measured': False,
                        'fold_count': len(folds),
                        'reason': 'too few folds to measure stability'}

            ordered = frame.sort_index()
            y_all = pd.to_numeric(ordered[target_name], errors='coerce')
            numeric = ordered.select_dtypes(include=[np.number])
            numeric = numeric.drop(columns=[
                column for column in numeric.columns
                if is_direct_target_column(column) or column == target_name
            ], errors='ignore')

            beat_baseline = 0
            margins: list[float] = []
            for fold in folds:
                train = slice(0, fold['train_end'])
                validate = slice(fold['validation_start'], fold['validation_end'])
                x_train, y_train = numeric.iloc[train], y_all.iloc[train]
                x_validate, y_validate = numeric.iloc[validate], y_all.iloc[validate]

                # Rank on rows where the TARGET exists, then demand complete
                # values only from the columns actually chosen. Requiring
                # every numeric column to be present first is what made this
                # find zero usable folds: with ~1,900 columns, of which many
                # are sparse news and macro series, essentially no row is
                # complete across all of them.
                labelled = y_train.notna()
                if labelled.sum() < 30 or y_validate.notna().sum() < 10:
                    continue
                columns = self._top_correlated(
                    x_train[labelled], y_train[labelled], config.max_features
                )
                usable = labelled & x_train[columns].notna().all(axis=1)
                if usable.sum() < 30:
                    continue
                model = RandomForestRegressor(
                    n_estimators=config.n_estimators,
                    max_depth=config.max_depth,
                    min_samples_leaf=config.min_samples_leaf,
                    random_state=config.random_state,
                    n_jobs=1,
                )
                model.fit(x_train.loc[usable, columns], y_train[usable])

                valid = y_validate.notna() & x_validate[columns].notna().all(axis=1)
                if valid.sum() < 10:
                    continue
                predicted = model.predict(x_validate.loc[valid, columns])
                actual = y_validate[valid].to_numpy(dtype=float)

                model_r2 = self._r_squared(actual, predicted)
                baseline_r2 = max(
                    self._r_squared(actual, np.full(actual.size, float(y_train[usable].mean()))),
                    self._persistence_r_squared(actual),
                )
                margins.append(model_r2 - baseline_r2)
                if model_r2 > baseline_r2:
                    beat_baseline += 1

            measured_folds = len(margins)
            if measured_folds < self._MIN_STABLE_FOLDS:
                return {'passed': True, 'measured': False,
                        'fold_count': measured_folds,
                        'reason': 'too few usable folds to measure stability'}

            required = max(self._MIN_STABLE_FOLDS,
                           math.ceil(self._STABLE_FOLD_SHARE * measured_folds))
            return {
                'passed': beat_baseline >= required,
                'measured': True,
                'fold_count': measured_folds,
                'folds_above_majority': beat_baseline,
                'folds_required': required,
                'worst_fold_margin': round(min(margins), 6),
                'reason': (
                    f"beat the naive baseline on {beat_baseline} of "
                    f"{measured_folds} folds; {required} required"
                ),
            }
        except (ImportError, ValueError, TypeError, KeyError, IndexError) as e:
            logger.debug(f"{context_key}: regression stability not evaluable ({e})")
            return {'passed': True, 'measured': False, 'reason': str(e)}

    @staticmethod
    def _top_correlated(x: "pd.DataFrame", y: "pd.Series", budget: int) -> list[str]:
        """Train-only ranking, so no fold's validation rows pick its features."""
        if len(x.columns) <= budget:
            return list(x.columns)
        ranked = x.corrwith(y).abs().fillna(-1.0).sort_values(
            ascending=False, kind='mergesort')
        return list(ranked.index[:budget])

    @staticmethod
    def _r_squared(actual: "np.ndarray", predicted: "np.ndarray") -> float:
        residual = float(((actual - predicted) ** 2).sum())
        total = float(((actual - actual.mean()) ** 2).sum())
        return 1.0 - residual / total if total > 0 else 0.0

    @classmethod
    def _persistence_r_squared(cls, actual: "np.ndarray") -> float:
        """"Tomorrow equals today", the opponent a slow series really has."""
        if actual.size < 3:
            return -np.inf
        persistence = np.empty_like(actual)
        persistence[0] = actual[0]
        persistence[1:] = actual[:-1]
        return cls._r_squared(actual, persistence)

    @staticmethod
    def _is_binary_target(frame: "pd.DataFrame", target_name: str) -> bool:
        """Is this a 0/1 label, or a continuous value wearing a target's name?"""
        if target_name not in frame.columns:
            return False
        values = pd.to_numeric(frame[target_name], errors='coerce').dropna()
        if values.empty:
            return False
        return bool(values.isin([0, 1]).all())

    @staticmethod
    def _chance_margin(fold: dict[str, Any]) -> float:
        """How far above 0.5 a fold must land before it means anything.

        "Better than chance by any amount" is a bar noise clears about half
        the time. Measured on a SHUFFLED target with 63-row validation
        windows, folds scored 0.500, 0.554, 0.597 and 0.523 balanced accuracy
        — three of four above 0.5, purely from sampling.

        The margin is one standard error of balanced accuracy, computed from
        the fold's own size and class balance rather than picked:

            se = sqrt( p(1-p)/n_pos + p(1-p)/n_neg ) / 2, with p = 0.5

        which for a 63-row window at a 35% positive rate is about 0.066. So a
        fold has to reach ~0.57, not 0.501. Small folds demand more, large
        folds less, and no constant has to be invented.
        """
        metrics = fold.get('validation_metrics', {})
        window = fold.get('validation_window', {})
        n = int(window.get('sample_count') or 0)
        positive_rate = float(metrics.get('actual_positive_rate') or 0.0)
        n_pos = max(1, int(round(n * positive_rate)))
        n_neg = max(1, n - n_pos)
        # Recall of each class is a proportion; balanced accuracy averages
        # two of them, so its variance is a quarter of their sum.
        return float(np.sqrt(0.25 / n_pos + 0.25 / n_neg) / 2.0)

    def _walk_forward_stability(self, frame: "pd.DataFrame", *, ticker: str,
                                timeframe: str, target_name: str,
                                context_key: str) -> dict[str, Any] | None:
        """Does a reference model find signal across folds, not just one split?

        The holdout-versus-baseline check the champion has already passed says
        "better than nothing, once". It cannot distinguish an edge from a
        lucky split, which is exactly what four hundred contexts competing for
        promotion will produce by chance.

        PipelineWalkForwardValidationEvaluator was already in this file,
        reachable only through `walk_forward_review_only` — a branch that
        returns before training and was never part of promotion. It builds
        purged expanding folds with the purge raised to the target's own
        horizon, and reports how many folds beat their majority baseline.

        Note what it measures: a FIXED reference model on the context, not the
        champion. So this is a statement about the context — "is there stable
        signal here at all" — and it is used as a precondition rather than as
        a verdict on the winner. Run only for champions that already passed
        the cheap gate, because it fits a model per fold and there is no point
        paying that for a candidate already refused.

        Returns None when the context is too short to build folds; a context
        that cannot be measured is not failed for it, it is passed through
        with that stated.
        """
        # The evaluator scores with CLASSIFICATION metrics. On a continuous
        # target they are not merely imprecise, they are undefined: measured
        # on a shuffled target_return_1d with 511 distinct values, all four
        # folds returned a balanced accuracy of 1.0, so every regression
        # context would sail through a gate that had measured nothing.
        # Classification targets on the same data behaved correctly
        # (target_up_1d, shuffled: 1 of 4 folds, worst 0.5, refused).
        #
        # So stability is reported as UNMEASURED here rather than passed on a
        # number that does not exist. That leaves return targets without this
        # check, which is a real gap and better stated than papered over.
        if not self._is_binary_target(frame, target_name):
            return self._regression_fold_stability(
                frame, target_name=target_name, context_key=context_key,
            )

        try:
            evaluator = PipelineWalkForwardValidationEvaluator(
                self._walk_forward_config_for(len(frame))
            )
            summary = evaluator.evaluate(
                frame, ticker=ticker, timeframe=timeframe, target_name=target_name,
            )
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.debug(f"{context_key}: walk-forward not evaluable ({e})")
            return None

        payload = summary.get('metrics', {}) if isinstance(summary, dict) else {}
        fold_count = int(payload.get('fold_count') or 0)
        if fold_count < self._MIN_STABLE_FOLDS:
            logger.debug(
                f"{context_key}: only {fold_count} walk-forward fold(s); "
                f"stability not measurable, not held against it"
            )
            return {'passed': True, 'fold_count': fold_count,
                    'reason': 'too few folds to measure stability'}

        # Folds counted HERE, from balanced accuracy, rather than taking the
        # evaluator's `validation_above_majority_fold_count`. That counter is
        #     sum(accuracy >= majority_baseline)
        # — plain accuracy against the majority-class rate, with `>=`. A model
        # that predicts the majority class everywhere scores EXACTLY the
        # baseline and is therefore counted as "above" on every fold, so the
        # most useless possible predictor passes it perfectly.
        #
        # The negative control caught it: shuffled targets cleared the
        # stability gate MORE often than real ones, 86% against 71%, because
        # a model given noise settles on the majority class and scores the
        # baseline exactly, while a real model sometimes deviates and lands
        # just under it.
        #
        # Balanced accuracy has no such degeneracy: a constant predictor
        # scores exactly 0.5 whatever the class balance, so 0.5 is a real
        # floor rather than a moving target.
        folds = summary.get('folds') or []
        fold_scores = [
            float(fold.get('validation_metrics', {}).get('balanced_accuracy', 0.0))
            for fold in folds
        ]
        above = sum(
            1 for fold, score in zip(folds, fold_scores)
            if score > 0.5 + self._chance_margin(fold)
        )
        worst = min(fold_scores) if fold_scores else payload.get(
            'minimum_validation_balanced_accuracy'
        )
        required = max(
            self._MIN_STABLE_FOLDS,
            math.ceil(self._STABLE_FOLD_SHARE * fold_count),
        )
        held_often_enough = above >= required
        never_collapsed = (
            worst is None
            or float(worst) >= self._MIN_WORST_FOLD_BALANCED_ACCURACY
        )

        reasons = []
        if not held_often_enough:
            reasons.append(
                f"balanced accuracy beat chance on {above} of {fold_count} "
                f"walk-forward folds; {required} required"
            )
        if not never_collapsed:
            reasons.append(
                f"worst fold scored {float(worst):.3f} balanced accuracy, "
                f"below chance ({self._MIN_WORST_FOLD_BALANCED_ACCURACY})"
            )

        return {
            'passed': held_often_enough and never_collapsed,
            'fold_count': fold_count,
            'folds_above_majority': above,
            'folds_required': required,
            'worst_fold_balanced_accuracy': worst,
            'reason': '; '.join(reasons) or (
                f"signal held on {above} of {fold_count} folds, worst fold "
                f"{worst}"
            ),
        }

    @staticmethod
    def _champion_is_allowed(ticker_result: dict[str, Any], context_key: str) -> bool:
        """A refused promotion must not be recorded as a champion.

        BaseTrainer withholds the CHAMP_ file when the winner fails the
        holdout-versus-baseline gate, but this stage wrote the champion
        METADATA regardless -- and Stage 5 reads the metadata. One run logged

            Champion NOT promoted for AAPL/15m/target_intraday_return_15m
            Pattern Champion ... target_intraday_return_15m ... catboost

        seven seconds apart, for the same context. The refusal was therefore
        cosmetic: Stage 5 would resolve that context anyway and fall back to
        whichever CHAMP_ file already sat on disk -- which today means a model
        trained on the corrupted batch. A gate that blocks a file but not the
        record of it blocks nothing.
        """
        gate = ticker_result.get('promotion_gate') or {}
        if not gate or gate.get('passed', True):
            return True
        logger.info(
            "No champion recorded for %s: %s",
            context_key, "; ".join(gate.get('reasons') or ['promotion gate failed']),
        )
        return False

    def _build_unified_training_context(
        self,
        prepared_data: dict[str, Any],
        *,
        target_name: str,
        context_fingerprint: str,
        context_pattern_seq: str | None = None,
        timeframe: str = "",
    ) -> dict[str, Any]:
        """Adapt nested preparation output and forward the holdout separately.

        UnifiedTrainingManager calls its selection split ``X_test``. Stage 4
        supplies validation there and does not expose the prepared holdout to
        model selection.

        The prepared holdout used to stop here: reserving it kept it out of
        selection, but it also meant BaseTrainer._record_winner_test_score --
        which reads ``X_test`` -- re-scored the winner on the very rows that
        chose it and filed the result as a test metric. It now travels under
        ``X_holdout``/``y_holdout``, keys no selection code reads, so the
        number is measured on data the model has never been evaluated against
        and the promotion gate has something real to check.
        """
        light = prepared_data.get("light_models")
        if not isinstance(light, dict):
            raise ValueError("Prepared data has no light-model split.")
        required = ("X_train", "y_train", "X_val", "y_val")
        missing = [key for key in required if light.get(key) is None]
        if missing:
            raise ValueError(
                f"Prepared light-model split is incomplete: {missing}."
            )
        y_train = light["y_train"]
        return {
            "X_train": light["X_train"],
            "y_train": y_train,
            "X_test": light["X_val"],
            "y_test": light["y_val"],
            # Selection reads X_val when present and falls back to X_test
            # otherwise; passing it explicitly means the fallback is never
            # what decides, and the two keys can no longer drift apart.
            "X_val": light["X_val"],
            "y_val": light["y_val"],
            # The purged holdout, produced with a gap by
            # prepare_data_for_models. Deliberately NOT named *_test: the test
            # keys are the selection split here, and one confusable name is
            # what caused the original defect.
            "X_holdout": light.get("X_test"),
            "y_holdout": light.get("y_test"),
            # The transformers the models are fitted BEHIND. prepare_data_for_models
            # fits these on the training split and hands the models z-scores;
            # they were returned in `light_data` and collected by nobody, so
            # Stage 5 fed raw columns to models trained on standardised ones.
            # `feature_names` is the fit-time COLUMN ORDER and is as essential
            # as the objects themselves -- a StandardScaler applied to columns
            # in a different order is a different transform.
            "preprocessor": {
                "imputer": light.get("imputer"),
                "scaler": light.get("scaler"),
                "feature_names": list(light.get("feature_names") or []),
            },
            "feature_names": list(light.get("feature_names") or []),
            "target_name": target_name,
            # Reaches BaseTrainer, which names the model file with it. Without
            # it the three timeframes' candidates and champions collapse onto
            # one filename each -- see base_trainer._save_model_candidate.
            "timeframe": timeframe,
            "target_type": self._infer_target_type(y_train),
            "context_fingerprint": context_fingerprint,
            # BaseTrainer already forwards this to the diary
            # (base_trainer.py: log_event(..., context_pattern_seq=
            # data.get('context_pattern_seq'))), and ContextMapEnricher
            # already produces the column, keeping the RAW sequence
            # specifically so KNN can measure distance between contexts --
            # its own comment says so. The key was simply never put in this
            # dict, so data.get() returned None and all 19,305 diary rows
            # were written with a NULL sequence, leaving the KNN expansion
            # with nothing to search.
            "context_pattern_seq": context_pattern_seq,
            "selection_split_role": "validation",
            "prepared_holdout_reserved": True,
        }

    def _write_active_stage4_candidates(
        self,
        *,
        ticker: str,
        timeframe: str,
        target_name: str,
        current_pattern: str,
        context_fingerprint: str,
        df: pd.DataFrame,
        prepared_data: dict[str, Any],
        ticker_result: dict[str, Any],
        output_dir: Path | None,
    ) -> dict[str, Any]:
        """Write partial/measured evidence without inventing unavailable data."""
        if output_dir is None:
            return {}
        try:
            light = prepared_data["light_models"]
            feature_names = list(light.get("feature_names") or [])
            validation_metrics = dict(
                ticker_result.get("winner_metrics") or {}
            )
            stability_analysis = (
                pipeline_control_artifacts
                .build_feature_distribution_stability_analysis(
                    light.get("X_train"),
                    light.get("X_val"),
                    feature_names,
                )
            )
            evaluation_window = (
                pipeline_control_artifacts.build_split_evaluation_window(
                    light.get("X_val"),
                    source="stage4_validation_feature_index",
                )
            )
            # Suffixed here as well: the export carries volatility_regime_1d
            # and volatility_regime_60m. Without the timeframe this artifact
            # is being written for, the lookup would take whichever came
            # first in column order and label a 60m model with the daily
            # regime.
            #
            # MARKET_REGIME first, because that is the name the producer
            # uses: technical_analysis_enricher._add_market_regime_features
            # writes df_enriched['MARKET_REGIME']. The lookup asked for
            # lower-case market_regime/regime, which exists under no
            # spelling, so every artifact recorded "unknown" while
            # MARKET_REGIME_1d sat right there in the same frame with 7,185
            # real values (TRENDING_UP, TRENDING_DOWN, MOMENTUM, NORMAL,
            # RANGING, MEAN_REVERSION, VOLATILE). The lower-case forms are
            # kept as aliases rather than dropped -- other producers may use
            # them, and asking costs nothing.
            market_regime = self._latest_context_value(
                df,
                ("MARKET_REGIME", "market_regime", "regime"),
                default="unknown",
                timeframe=str(timeframe),
            )
            volatility_regime = self._latest_context_value(
                df,
                ("volatility_regime",),
                default="unknown",
                timeframe=str(timeframe),
            )
            winner = str(ticker_result.get("winner") or "unknown")
            model_candidate = (
                pipeline_control_artifacts
                .build_model_evaluation_candidate(
                    ticker=ticker,
                    target_name=target_name,
                    model_type=winner,
                    timeframe=timeframe,
                    context_fingerprint=context_fingerprint,
                    market_regime=market_regime,
                    volatility_regime=volatility_regime,
                    train_metrics=dict(
                        ticker_result.get("train_metrics") or {}
                    ),
                    validation_metrics=validation_metrics,
                    train_sample_count=len(light.get("y_train", [])),
                    validation_sample_count=len(light.get("y_val", [])),
                    test_metrics=None,
                    test_sample_count=0,
                    max_drawdown=None,
                    evaluation_window=evaluation_window,
                )
            )
            feature_candidate = (
                pipeline_control_artifacts
                .build_feature_stability_candidate(
                    ticker=ticker,
                    target_name=target_name,
                    model_type=winner,
                    timeframe=timeframe,
                    context_fingerprint=context_fingerprint,
                    market_regime=market_regime,
                    volatility_regime=volatility_regime,
                    feature_importance={},
                    stability_analysis=stability_analysis,
                )
            )
            return (
                pipeline_control_artifacts
                .write_pipeline_control_metric_artifact_candidates(
                    batch_dir=output_dir,
                    context_key=(
                        f"{ticker}_{timeframe}_{target_name}_"
                        f"{current_pattern}_{winner}"
                    ),
                    model_evaluation=model_candidate,
                    feature_stability=feature_candidate,
                )
            )
        except (
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
            ZeroDivisionError,
            OSError,
        ) as exc:
            logger.warning(
                "Could not write active Stage 4 pipeline-control candidates: %s",
                exc,
            )
            return {}

    @staticmethod
    def _latest_context_value(
        frame: pd.DataFrame,
        columns: tuple[str, ...],
        *,
        default: str | None,
        timeframe: str | None = None,
    ) -> str | None:
        """Last non-null value of the first matching context column.

        Accepts the timeframe-suffixed forms the enrichers actually emit.
        ContextMapEnricher runs per timeframe and its output arrives as
        context_pattern_seq_1d / context_pattern_seq_60m -- checked against
        the exported features.parquet, where the BARE names
        (context_pattern_seq, context_fingerprint, context_pattern_id) are
        absent entirely.

        Looking only for the bare name meant this always returned the
        default. For context_fingerprint that quietly sent
        _build_context_fingerprint down its fallback branch, which is why
        every fingerprint in experience_diary is a SHA-256 rather than the
        tri-state form the KNN similarity search needs. For
        context_pattern_seq it meant the sequence never reached the diary at
        all.
        """
        candidates: list[str] = []
        for column in columns:
            if timeframe:
                candidates.append(f"{column}_{timeframe}")
            candidates.append(column)
            # Any timeframe-suffixed variant, in column order, as a last
            # resort: better a context from a neighbouring timeframe than a
            # hash standing in for one.
            #
            # The suffix must actually BE a timeframe. A bare startswith
            # would let MARKET_REGIME_ENCODED_1d answer a request for
            # MARKET_REGIME, and that column holds detector confidence
            # (0.72, 0.6, 0.9) -- a float would be stringified and filed as
            # a regime label.
            prefix = f"{column}_"
            candidates.extend(
                name for name in frame.columns
                if isinstance(name, str)
                and name.startswith(prefix)
                and is_timeframe_token(name[len(prefix):])
                and name not in candidates
            )

        for column in candidates:
            if column not in frame.columns:
                continue
            values = frame[column].dropna()
            if not values.empty:
                return str(values.iloc[-1])
        return default

    @classmethod
    def _build_context_fingerprint(
        cls,
        *,
        frame: pd.DataFrame,
        prepared_data: dict[str, Any],
        ticker: str,
        timeframe: str,
        target_name: str,
        current_pattern: str,
    ) -> str:
        existing = trusted_context_fingerprint(
            cls._latest_context_value(
                frame,
                ("context_fingerprint",),
                default=None,
                # Same reason as the sequence: the enricher emits
                # context_fingerprint_1d / _60m, so the bare lookup never
                # matched and every fingerprint fell through to the SHA-256
                # branch below -- which cannot be vectorised, so the KNN
                # similarity search had nothing to measure.
                timeframe=str(timeframe),
            )
        )
        if existing:
            return existing

        feature_names = sorted(
            str(value)
            for value in (
                prepared_data.get("light_models", {}).get(
                    "feature_names", []
                )
                or []
            )
        )
        last_values: dict[str, Any] = {}
        if not frame.empty:
            row = frame.iloc[-1]
            for name in feature_names:
                if name not in frame.columns:
                    continue
                value = row[name]
                if pd.isna(value):
                    last_values[name] = None
                elif hasattr(value, "item"):
                    try:
                        last_values[name] = value.item()
                    except (TypeError, ValueError):
                        last_values[name] = str(value)
                else:
                    last_values[name] = value
        lineage = source_lineage_attrs(frame)
        payload = {
            "schema_version": "pipeline_model_context_fingerprint_v1",
            "ticker": ticker.upper(),
            "timeframe": timeframe.lower(),
            "target_name": target_name,
            "context_pattern_id": current_pattern,
            "observed_at": lineage.get("prediction_observed_at"),
            "feature_names": feature_names,
            "last_feature_values": last_values,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _infer_target_type(values: Any) -> str:
        flattened = pd.Series(
            values.reshape(-1)
            if hasattr(values, "reshape")
            else list(values)
        ).dropna()
        unique = set(flattened.unique().tolist())
        if unique and unique.issubset({-1, 0, 1}):
            return "classification"
        return "regression"

    def _log_expert_to_diary(self, info: dict[str, Any], tf: str):
        """Зберігає інформацію про експертну модель у щоденник досвіду."""
        entry = {
            'timestamp': info['timestamp'], 'ticker': info['ticker'],
            'tf': tf, 'target': info['target'], 'pattern_id': info['pattern_id'],
            'model_name': info['winner'], 'score': info['metrics'].get('score', 0),
            'is_champion': True
        }
        pd.DataFrame([entry]).to_csv(self.diary_path, mode='a', header=False, index=False)
