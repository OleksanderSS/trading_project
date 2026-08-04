# audit-ignore: ARCHITECTURAL_USAGE
import datetime
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
            # ✅ ELITE FIX: Визначаємо домінуючий патерн для цього тікера у вибірці
            current_pattern = df['context_pattern_id'].iloc[-1] if 'context_pattern_id' in df.columns else 'normal'
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
        return {
            'models_metadata': champions,
            'processed_data': enriched_data,
            'pipeline_control_metric_artifacts': metric_artifacts,
            'pipeline_control_metric_artifact_manifests': manifests,
        }

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
                        'selected_features': list(
                            training_context.get("feature_names") or []
                        ),
                        'context_fingerprint': context_fingerprint,
                        'pipeline_control_metric_artifacts': artifact_paths,
                        'timestamp': datetime.datetime.now().isoformat()
                    }

                    self._log_expert_to_diary(champions[context_key], timeframe)
                    logger.info(f"🏆 Pattern Champion for {context_key}: {winner_name} (Score: {metrics.get('score', 0):.4f})")

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f"Error modeling {ticker}: {e}")

    def _build_unified_training_context(
        self,
        prepared_data: dict[str, Any],
        *,
        target_name: str,
        context_fingerprint: str,
        context_pattern_seq: str | None = None,
    ) -> dict[str, Any]:
        """Adapt nested preparation output and reserve the holdout.

        UnifiedTrainingManager calls its selection split ``X_test``. Stage 4
        supplies validation there and does not expose the prepared holdout to
        model selection.
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
            "feature_names": list(light.get("feature_names") or []),
            "target_name": target_name,
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
