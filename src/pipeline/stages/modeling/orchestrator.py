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
from src.pipeline.modeling_context import is_pooled, iter_model_contexts
from src.pipeline.stages.base_stage import BaseStage
from src.pipeline.stages.modeling import pipeline_control_artifacts
from src.pipeline.stages.modeling.context_ledger import ContextLedger
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


def _shown(value: Any) -> str:
    """A number, or "n/a" when the rung was never measured.

    Never 0.0000 for an unmeasured opponent: that reads as "the model beat
    it", which is the opposite of what an absent measurement means.
    """
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.4f}" if np.isfinite(number) else "n/a"


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

        # Per-run state, given safe defaults HERE and not only in `run()`.
        # `_process_ticker_with_async` is reachable from the stage-4 contract
        # tests and from the walk-forward review branch without `run()` ever
        # executing; an attribute that exists only inside `run()` turns any
        # such entry into an AttributeError swallowed by the stage's own
        # `except`, which reads in the log as "Error modeling NVDA" and looks
        # like a data problem. A null ledger also means a caller that did not
        # start a run cannot write to the real one.
        self._gate_refusals: list[dict[str, Any]] = []
        self._ledger: ContextLedger | None = None
        self._resume_contexts = False
        self._replayed_contexts = 0
        # Contexts whose pipeline-control artifacts could not be written. The
        # writer returned {} on failure and the caller appends only truthy
        # results, so the context simply vanished from the artifact set with
        # nothing saying it had. Counted here and reported at the stage
        # boundary: an artifact set that is quietly short is the same problem
        # as a check that quietly did not run (REGISTER #201).
        self._artifact_write_failures: list[str] = []
        #: How many (context, target) verdicts this run actually attempted.
        #
        #: `models.yaml` states `family_size: 27` and says of it: "it cannot be
        #: known before the run starts -- contexts are materialised lazily --
        #: so it is stated here and the stage compares it against the actual
        #: count at the end, and says so loudly if they differ. A number that
        #: drifts silently is the thing this whole gate exists to prevent."
        #
        #: That comparison did not exist. `family_size` appeared nowhere in
        #: `src/pipeline` except `_promotion_family_size`, which was set to
        #: None here and never assigned anywhere, so the training context
        #: always carried None and the gate fell back to the configured 27 --
        #: correct by luck as long as a run happens to make 27 verdicts, and
        #: silently wrong the moment it does not. A run of 216 verdicts would
        #: be judged at 2.90 sigma where its own stated 5% needs 3.50.
        #
        #: This counter is the missing half. It changes no bar: the multiplier
        #: still comes from the config, because the count is only complete
        #: when the run is over and the verdicts are long since made. What it
        #: does is make the discrepancy VISIBLE at the stage boundary, which is
        #: what the config promised and CRITIQUE section 11 asks for.
        self._promotion_attempts: int = 0
        # Set at the top of run() once the contexts are known; the gate reads
        # it to size its bar. None means "unknown", and the gate then falls
        # back to config rather than pretending the run is a single test.
        self._promotion_family_size: int | None = None

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

    def _resolve_resume_contexts(self) -> bool:
        """May this run replay contexts an earlier run already finished?

        Default false. A context replayed from the ledger is a number this
        run did not compute, and every previous version of "reuse what is on
        disk" in this project ended as a stale artifact being read as a fresh
        one. Turning it on is a decision an operator makes after a crash, in
        `src/config/processing.yaml` under `modeling.resume_completed_contexts`.
        """
        try:
            enabled = bool(self.modeling_config.get(
                'resume_completed_contexts', False))
        except (AttributeError, TypeError, KeyError) as e:
            # Assign and fall through rather than log-and-return: a handler
            # that returns a falsy literal is the shape the silent-failure
            # ratchet counts, and it earns that reputation here -- "resume is
            # off" and "the config could not be read" would look identical.
            logger.warning(
                "Could not read modeling.resume_completed_contexts (%s); "
                "resume stays off.", e,
            )
            enabled = False
        if enabled:
            logger.warning(
                "Resume is ON: contexts already finished on identical data "
                "will be REPLAYED from the ledger rather than trained. Their "
                "numbers come from an earlier run.",
            )
        return enabled

    def _replay_context(self, key: str, entry: dict[str, Any],
                        champions: dict[str, Any]) -> None:
        """Put a finished context's outcome back without training it."""
        champion = entry.get("champion")
        if champion:
            champions[key] = champion
            logger.info(
                "Replayed champion for %s from the ledger (%s, recorded %s).",
                key, champion.get("winner"), entry.get("recorded_at"),
            )
        else:
            refusal = entry.get("refusal")
            if refusal:
                self._gate_refusals.append(dict(refusal))
            logger.info(
                "Replayed refusal for %s from the ledger (recorded %s).",
                key, entry.get("recorded_at"),
            )
        self._replayed_contexts += 1

    def _init_infrastructure(self):
        """Initializes the environment."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        if not self.diary_path.exists():
            self.diary_path.parent.mkdir(parents=True, exist_ok=True)
            columns = ['timestamp', 'ticker', 'tf', 'target', 'pattern_id', 'model_name', 'score', 'is_champion']
            pd.DataFrame(columns=columns).to_csv(self.diary_path, index=False)

    async def run(self, **kwargs) -> dict[str, Any]:
        """Runs the full training cycle with Pattern-Aware logic."""
        # Why a context produced no champion, kept rather than logged away.
        self._gate_refusals = []
        # What has already been trained, and on exactly which data. Written
        # always, read only when the operator asks -- a replayed context is a
        # result the reporting run did not compute.
        self._ledger = ContextLedger()
        self._resume_contexts = self._resolve_resume_contexts()
        self._replayed_contexts = 0
        self._artifact_write_failures = []
        self._promotion_attempts = 0
        # `--tickers` reaches this stage and used to stop here.
        #
        # The caller passes the resolved list all the way down, and the
        # context iterator never saw it: a run asked for AAPL and MSFT on
        # 2026-08-29 trained AAPL, then went on to ABBV. There is no cheap
        # end-to-end check of a seven-stage pipeline if the smallest possible
        # run is the full universe -- which is why this counts as tooling
        # rather than a nicety.
        requested = kwargs.get('tickers') or None
        self._requested_tickers = (
            {str(x) for x in requested} if requested else None
        )
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
            # ...and on 2026-08-28 MARKET_REGIME was switched OFF at the
            # source. It cost 5.4 hours of a twelve-hour rebuild and failed
            # its own leading-feature test, so `_add_market_regime_features`
            # is now opt-in behind MARKET_REGIME_FEATURES. That decision was
            # taken in the enricher and never carried here.
            #
            # The consequence is silent and total: every context falls back
            # to the literal 'normal', so the "Regime-Aware Training Arena"
            # has ONE value on the axis it is named after. Measured on the
            # run of 2026-08-31 -- all ten champion keys end in `_normal`,
            # which is the exact state a fix in 9fa3a84a was written to end.
            #
            # Repointing this at `volatility_regime_*`, which DOES exist in
            # the batch, was considered and measured away on 2026-08-31.
            #
            # First correcting a wrong reason: it would not "split the data
            # three ways". This value never filters `df`. It is a LABEL on
            # the champion -- it reaches the context key, the champion
            # filename and the diary, and the model is trained on every row
            # regardless of it. So the cost of changing it is zero, and the
            # question is only whether the label discriminates.
            #
            # Measured on the export, value at each ticker's last bar:
            #
            #     15m   low 110 of 110          -> ONE value
            #     60m   low 98, normal 8, high 3, extreme 1
            #     1d    extreme 50, normal 26, high 20, low 14
            #
            # The daily spread looks usable until pooling is accounted for.
            # With `pool_tickers` on there is one context per (timeframe,
            # target), so this reads the last row of the POOLED frame -- and
            # at that timestamp 108 tickers hold four different regimes. The
            # label would then be decided by whichever ticker sorts last:
            # change the row order and the champion's regime changes with no
            # change to the data or the model. That is not an axis, it is a
            # coin flip that gets recorded.
            #
            # So the key stays constant, deliberately, and the promise of
            # regime-awareness is not kept by pretending otherwise. Training
            # one model PER regime -- an actual split, an actual experiment --
            # is register #182 and roadmap §26, to be run when there is a
            # surviving champion worth conditioning.
            #
            # What changes here is only that the fallback stops being silent.
            current_pattern = self._latest_context_value(
                df,
                ("MARKET_REGIME", "market_regime", "regime"),
                default='normal',
                timeframe=str(timeframe),
            ) or 'normal'
            if current_pattern == 'normal':
                logger.warning(
                    "Ticker %s/%s has no MARKET_REGIME column, so its "
                    "champions are keyed by the literal 'normal'. The regime "
                    "axis of the arena carries one value. Set "
                    "MARKET_REGIME_FEATURES=1 to compute it (~5.4h on the "
                    "daily frame), or see register #182.",
                    ticker, timeframe,
                )
            else:
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

        if self._replayed_contexts:
            # Say it in the completion line, not only where it happened. A
            # summary that reads "trained N models" when some of them were
            # replayed is the kind of sentence that gets quoted later.
            logger.warning(
                'Modeling Stage complete. %d champion(s), of which %d '
                'context(s) were REPLAYED from the ledger and not trained in '
                'this run.', len(champions), self._replayed_contexts,
            )
        else:
            logger.info(f'Modeling Stage complete. Trained {len(champions)} expert models.')
        manifests = sorted(
            {
                str(item["manifest"])
                for item in metric_artifacts
                if item.get("manifest")
            }
        )
        holdout_path = self._write_holdout_predictions(champions)
        refusals_path = self._write_gate_refusals(self._gate_refusals)
        # Said at the stage boundary, where a verdict can still be formed. A
        # per-context write failure logged in passing is a line nobody reads;
        # a count next to the champion count is a discrepancy anyone reading
        # the summary trips over.
        if self._artifact_write_failures:
            logger.error(
                'Pipeline-control artifacts could not be written for %d '
                'context(s), so the artifact set is short by that many and '
                'does not say so on its own: %s',
                len(self._artifact_write_failures),
                ', '.join(self._artifact_write_failures[:10]),
            )
        self._reconcile_promotion_family()
        return {
            'models_metadata': champions,
            'processed_data': enriched_data,
            'pipeline_control_metric_artifacts': metric_artifacts,
            'pipeline_control_metric_artifact_manifests': manifests,
            'holdout_predictions_path': str(holdout_path) if holdout_path else None,
            'gate_refusals_path': str(refusals_path) if refusals_path else None,
            'artifact_write_failures': list(self._artifact_write_failures),
        }

    def _reconcile_promotion_family(self) -> None:
        """Say whether the bar the gate used matched the run the gate judged.

        The promotion bar is a multiple-comparison correction, so it is only
        meaningful relative to the number of comparisons. `models.yaml` states
        that number in advance and promises this check; the check was never
        written, so a stated 27 and an actual 216 would have looked identical
        from outside -- and the difference between them is 2.90 sigma against
        3.50, which is the difference between a 5% run-wide error rate and
        something several times that.

        Nothing is corrected here. The verdicts are already made and a bar
        cannot be applied backwards; what this does is refuse to let the
        discrepancy pass unsaid, which is the whole content of CRITIQUE
        section 11 ("we do not count our own attempts").
        """
        try:
            cfg = (self.config_manager.get_config('models') or {}).get(
                'promotion_gate', {}
            ) or {}
        except (AttributeError, KeyError, TypeError):
            cfg = {}
        declared = cfg.get('family_size')
        actual = int(self._promotion_attempts)
        if declared is None:
            logger.error(
                'The run made %d promotion attempt(s) and no family_size is '
                'configured, so the gate judged them at the single-test bar. '
                'Every verdict in this run carries a higher error rate than '
                'it states.', actual,
            )
            return
        declared = int(declared)
        if declared == actual:
            logger.info(
                'Promotion family reconciled: %d attempt(s), matching the '
                'configured family_size.', actual,
            )
            return
        logger.error(
            'PROMOTION BAR DOES NOT MATCH THIS RUN: the gate judged %d '
            'attempt(s) at the bar for %d. %s Set models.yaml '
            'promotion_gate.family_size to %d and re-read this run\'s '
            'champions as provisional.',
            actual, declared,
            (
                'The bar was too LOOSE, so the run-wide error rate is higher '
                'than the stated one.'
                if actual > declared else
                'The bar was too STRICT, so real edges may have been refused.'
            ),
            actual,
        )

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
                    # The confidence behind the call, not just the call. A
                    # hard 0/1 makes a coin flip and a near-certainty
                    # indistinguishable downstream, and this column is
                    # enumerated by name -- so BaseTrainer producing it is not
                    # enough on its own for it to reach the artifact.
                    'probability': record.get('probability'),
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
        """Yield isolated ticker/timeframe frames for model preparation.

        Honours the ticker list the caller asked for. A pooled context is
        never filtered: it is one frame carrying every name by design, and
        dropping it because its label is not a ticker symbol would silently
        disable pooling.
        """
        # Pooling is a switch, not a rewrite.
        #
        # `iter_model_contexts` has carried `pool_tickers` all along, with the
        # measurement in its own docstring: one pooled model beats 22
        # per-ticker models at every cost ratio from 0.5 to 3.0, widening as
        # false signals get more expensive. The call here never passed it, so
        # the better-measured mode has been off by default.
        #
        # Two numbers from 2026-08-29 say why it matters beyond that
        # measurement. Per ticker, a context holds ~900 training rows, which
        # is what forced the feature budget down to five; 47 contexts across
        # AAPL and MSFT produced ZERO champions. And it took 31 minutes for
        # two names -- about thirty hours for 110, so the per-ticker shape is
        # not merely worse, it is the one that cannot be run in full.
        #
        # Off by default all the same: this changes what the stage produces,
        # and the comparison between the two modes is the point. Set
        # `modeling.pool_tickers: true` to turn it on.
        pooled = bool(self.modeling_config.get('pool_tickers', False))
        if pooled:
            logger.info(
                "Pooling tickers: one model per (timeframe, target) across "
                "every name, instead of one per ticker."
            )
        wanted = getattr(self, '_requested_tickers', None)
        skipped = 0
        for ticker, timeframe, frame in iter_model_contexts(
            enriched_data, pool_tickers=pooled
        ):
            if wanted and is_pooled(ticker):
                # A pooled context is not dropped, it is NARROWED.
                #
                # Skipping it because `__POOLED__` is not a ticker symbol
                # would silently switch pooling off; ignoring the list
                # entirely makes `--tickers` meaningless in pooled mode,
                # which is how the run of 2026-08-30 became a 25-hour job
                # while being discussed as a two-name smoke test. Narrowing
                # the ROWS keeps both promises.
                if 'ticker' in frame.columns:
                    narrowed = frame[frame['ticker'].isin(wanted)]
                    if narrowed.empty:
                        skipped += 1
                        continue
                    if len(narrowed) != len(frame):
                        logger.info(
                            "Pooled %s narrowed to %d of %d name(s): "
                            "%s rows instead of %s.",
                            timeframe, narrowed['ticker'].nunique(),
                            frame['ticker'].nunique(),
                            f"{len(narrowed):,}", f"{len(frame):,}",
                        )
                    yield ticker, timeframe, narrowed
                    continue
            if wanted and not is_pooled(ticker) and str(ticker) not in wanted:
                skipped += 1
                continue
            yield ticker, timeframe, frame
        if skipped:
            logger.info(
                "Modeling skipped %d context(s) outside the requested "
                "ticker list (%d name(s) asked for).", skipped, len(wanted),
            )

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
                # One promotion attempt, counted where it is made rather than
                # inferred afterwards from champions -- a refused verdict is an
                # attempt too, and counting only the survivors is exactly the
                # arithmetic that makes a multiple-comparison correction wrong.
                self._promotion_attempts += 1
                # What this context is, and what its data is, BEFORE paying
                # for either. A run that dies in the eighth hour otherwise
                # loses everything it had already finished: the 15m frame was
                # recomputed six times across runs 1-6 for byte-identical
                # numbers.
                context_key = (
                    f"{ticker}_{timeframe}_{target_name}_{current_pattern}"
                )
                fingerprint = ContextLedger.fingerprint(df, str(target_name))
                if self._resume_contexts:
                    entry = self._ledger.lookup(context_key, fingerprint)
                    if entry is not None:
                        self._replay_context(context_key, entry, champions)
                        continue

                # Готуємо дані з PURGED GAP
                prepared_data = prepare_data_for_models(
                    df=df, ticker=ticker, timeframe=timeframe,
                    target_cols=[target_name],
                    gap_size=self._resolve_purge_gap(10),
                    test_size=self._resolve_test_size()
                )

                if not prepared_data:
                    # The most invisible way to produce no champion: the split
                    # never happened, so there was no model to refuse and the
                    # context simply vanished from the artifact. Counting it
                    # matters because it is the one category that means "not
                    # enough data to tell" rather than "no edge" -- the
                    # distinction the whole artifact exists to draw.
                    self._record_unprepared_context(
                        ticker=str(ticker),
                        timeframe=str(timeframe),
                        target_name=str(target_name),
                    )
                    self._remember_refusal(context_key, fingerprint)
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
                        self._collect_gate_refusal(ticker_result, context_key)
                        self._remember_refusal(context_key, fingerprint)
                        continue

                    if self._is_indicator_prediction(target_name):
                        logger.info(
                            "No champion recorded for %s: indicator_prediction "
                            "targets are measured but not promoted", context_key,
                        )
                        self._collect_gate_refusal(
                            ticker_result, context_key,
                            note="indicator_prediction targets are measured but not promoted",
                        )
                        self._remember_refusal(context_key, fingerprint)
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
                        # This path logged and persisted nothing while the two
                        # above it recorded, so the artifact built to answer
                        # "why did nothing get promoted" was silently missing a
                        # whole category -- the one that reads "balanced
                        # accuracy beats chance on only some folds", 66 of the
                        # 446 refusals in the run that was parsed by hand.
                        self._collect_gate_refusal(
                            ticker_result, context_key,
                            note=stability.get('reason'),
                        )
                        self._remember_refusal(context_key, fingerprint)
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
                        # What this champion actually BEAT, kept with it.
                        #
                        # The gate recorded its evidence only when it refused
                        # (`_collect_gate_refusal`), so a promoted model left
                        # no trace of which opponents it faced or by how much.
                        # Noticed on 2026-08-31, on the first champion of the
                        # run that added the missing ladder rungs: there was
                        # no way to tell from any artifact whether the clock
                        # opponent had been measured at all. A gate whose
                        # passes are unauditable is a gate you have to trust.
                        'ladder': self._ladder_evidence(ticker_result),
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

                    self._remember_champion(
                        context_key, fingerprint, champions[context_key])
                    self._log_expert_to_diary(champions[context_key], timeframe)
                    ladder = champions[context_key]['ladder']
                    logger.info(
                        "🏆 Pattern Champion for %s: %s (Score: %.4f) — holdout "
                        "%s beat %s %s by %s (sigma %s) [constant %s, lag-%s %s, "
                        "clock %s (%s), one feature %s]",
                        context_key, winner_name, metrics.get('score', 0),
                        _shown(ladder.get('score')),
                        ladder.get('binding_opponent') or 'nothing measured',
                        _shown(ladder.get('baseline_score')),
                        # The two numbers that decide promotion since #186.
                        # Printing the verdict without them is how the first
                        # version of this line failed to answer the question
                        # it was added to answer.
                        _shown(ladder.get('baseline_margin')),
                        _shown(ladder.get('baseline_margin_sigma')),
                        _shown(ladder.get('baseline_constant_score')),
                        ladder.get('baseline_persistence_lag_bars', '?'),
                        _shown(ladder.get('baseline_persistence_score')),
                        _shown(ladder.get('baseline_clock_score')),
                        ladder.get('baseline_clock_scheme') or 'not measured',
                        _shown(ladder.get('single_feature_score')),
                    )

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

        The window scales in BOTH directions, which it used not to.

        The early return below fired whenever the defaults produced enough
        folds -- that is, whenever data was plentiful -- so a bigger context
        was validated on the same fixed 120 rows as a small one. Measured on
        the pooled run of 2026-08-30: 104,267 training rows, four folds, and
        all four validation windows inside the LAST 480 rows. 0.46% of the
        data, at the very end of the timeline, deciding whether a model
        becomes a champion.

        That was invisible while every context held ~900 rows, where 120 is a
        sensible eighth. Pooling multiplied the rows by a hundred and left the
        window where it was, so the comment "intraday contexts can afford the
        defaults" quietly inverted: the more data a context has, the smaller
        the fraction it was checked on.

        `row_count // 8` now applies always. On 900 rows it yields 112, the
        `max` keeps 120, and nothing about the per-ticker world changes. On
        104,267 it yields 13,033 per fold -- about 52,000 rows actually
        validated instead of 480.
        """
        default = WalkForwardValidationConfig()
        validation_rows = max(
            cls._MIN_FOLD_VALIDATION_ROWS, default.validation_rows, row_count // 8
        )
        min_train_rows = max(cls._MIN_FOLD_TRAIN_ROWS, row_count // 2)
        if (validation_rows <= default.validation_rows
                and min_train_rows <= default.min_train_rows
                and len(build_purged_expanding_folds(row_count, config=default))
                >= cls._MIN_STABLE_FOLDS):
            return default
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
        except Exception as e:  # noqa: BLE001 - see below; breadth is the point
            # This handler is where #189 actually lived. The pooled filter was
            # the cause that time, but the SHAPE is what let it run for weeks:
            # catch, write the reason to `logger.debug`, return None -- and
            # the caller's `if stability and not stability.get('passed', True)`
            # reads None as "no objection" and promotes.
            #
            # Two cases were being merged. "Too short to build folds" is a
            # known limitation, and passing through with that stated is a
            # decision this project has taken deliberately. "The evaluator
            # raised" is not that: it is an unknown, and an unknown that
            # cannot be told apart from a pass is the whole defect family.
            #
            # So the known case still returns None below, and the unknown case
            # refuses -- loudly, and with the reason carried into the refusal
            # record. If that starts refusing every champion, an evaluator is
            # broken and we will find out in the first run instead of in the
            # audit that follows the fourth.
            #
            # The catch is broad on purpose: the previous tuple was
            # (ValueError, TypeError, KeyError, AttributeError), which is the
            # narrow tuple the silent-failure scanner already counts 629 of,
            # and the exception that produced #189 was not in it.
            logger.error(
                "%s: walk-forward stability could not be evaluated (%s: %s). "
                "Refusing rather than promoting: a check that failed is not a "
                "check that passed.",
                context_key, type(e).__name__, e, exc_info=True,
            )
            return {
                'passed': False,
                'measured': False,
                'reason': (
                    f"walk-forward stability could not be evaluated "
                    f"({type(e).__name__}: {e})"
                ),
            }

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
    def _is_indicator_prediction(target_name: str) -> bool:
        """Indicator targets are measured, never promoted to trade on.

        `target_sma_20_f1` asks for tomorrow's 20-period moving average: 19 of
        its 20 terms are already known today, which is why persistence alone
        scores R2 0.998-0.9994 on this family. A model that beats that is
        doing arithmetic, not forecasting, and the score it earns is ~0.999 --
        sitting in the same champion table as a directional model at 0.55
        balanced accuracy. Any downstream ranking that compares champions
        across targets picks the arithmetic.

        On the 2026-08-12 run 12 of 65 champions were here
        (volume_ratio_f1 10, macd_hist_f1 2), and none of them names a price
        move anyone can trade. They still train, still score, still write
        holdout predictions -- so the family stays available as evidence and
        as a feature source. It just stops reaching Stage 5.
        """
        if not target_name:
            return False
        try:
            from src.config.target_type_registry import load_target_types
            return load_target_types().get(target_name) == 'indicator_prediction'
        except Exception as e:  # noqa: BLE001 - any failure here means "unknown"
            # This used to return False -- "not an indicator" -- and promote.
            # The registry is the ONLY thing that distinguishes a tradeable
            # target from arithmetic on data already known, so when it cannot
            # be read the honest answer is not "no" but "cannot tell", and the
            # question being asked is whether to PROMOTE. Answering "no" under
            # uncertainty is how `target_sma_20_f1` at R2 0.998 lands in the
            # champion table beside a directional model at 0.55.
            #
            # Returning True refuses every target for the run. That is loud,
            # it is caught by the run verdict (zero champions is a failure),
            # and it costs one rerun after fixing the registry. The old
            # behaviour cost a champion table nobody could trust, which is the
            # more expensive of the two by a distance.
            logger.error(
                "Could not read the target registry (%s: %s). Refusing every "
                "target this run rather than promoting under a check that "
                "could not be made -- fix the registry and rerun.",
                type(e).__name__, e,
            )
            return True

    def _collect_gate_refusal(
        self,
        ticker_result: dict[str, Any],
        context_key: str,
        note: str | None = None,
    ) -> None:
        """Keep the gate's reason, not only its verdict.

        The gate logs why it refused and persists nothing. Answering "why did
        no return target ever produce a champion" therefore meant finding the
        run's log and parsing 446 lines out of it -- which worked only because
        that log happened to still exist.

        What it found is worth having as an artifact rather than an
        excavation: 342 of the 446 were "does not beat the naive baseline",
        24 were "too few events", and the numbers in each line are what
        separate "no edge" from "not enough data to tell".
        """
        # Read from where the numbers ARE. The first version of this asked the
        # gate for holdout_score, holdout_rows and holdout_events; the gate
        # carries only `passed` and `reasons`, so all four numeric columns came
        # out null in the 2026-08-23 artifact -- 497 rows of text and not one
        # figure, in the file built specifically to keep the figures. The same
        # for the winner, which is `winner` and not `model_type`.
        gate = ticker_result.get("promotion_gate") or {}
        holdout = ticker_result.get("winner_holdout_metrics") or {}

        # The note has to be ADDED, not used as a fallback. Written as
        # `reasons or [note]`, a refusal that came from the walk-forward
        # stability check was labelled with whatever the promotion gate happened
        # to say -- and for 99 of 497 rows that was
        # "holdout_measured_and_beats_baseline", a refusal explaining that the
        # model beat its baseline.
        reasons = list(gate.get("reasons") or [])
        if note:
            reasons.append(note)

        self._gate_refusals.append({
            "context": context_key,
            "ticker": ticker_result.get("ticker"),
            "timeframe": ticker_result.get("timeframe"),
            "target": ticker_result.get("target_name") or ticker_result.get("target"),
            "model_type": ticker_result.get("winner"),
            "reasons": "; ".join(reasons) or "no reason given",
            "holdout_score": holdout.get("score"),
            "baseline_score": holdout.get("baseline_score"),
            # Which opponent actually won, and by how much each. Worth keeping
            # separately since the persistence baseline was found on
            # 2026-08-22 to have been reading the future on every multi-bar
            # target: a refusal that lost to `persistence` means something
            # different from one that lost to `constant`.
            "baseline_kind": holdout.get("baseline_kind"),
            "baseline_constant_score": holdout.get("baseline_constant_score"),
            "baseline_persistence_score": holdout.get("baseline_persistence_score"),
            # The clock rung, added 2026-08-31. "Lost to the weekday and the
            # hour" is a different finding from "lost to a constant": it says
            # the target is a schedule, which no feature in this pipeline can
            # improve on.
            "baseline_clock_score": holdout.get("baseline_clock_score"),
            "baseline_clock_scheme": holdout.get("baseline_clock_scheme"),
            "single_feature_score": holdout.get("single_feature_score"),
            "single_feature_name": holdout.get("single_feature_name"),
            "holdout_rows": holdout.get("holdout_sample_count"),
            "holdout_events": holdout.get("holdout_event_count"),
        })

    @staticmethod
    def _ladder_evidence(ticker_result: dict[str, Any]) -> dict[str, Any]:
        """Every rung the winner was measured against, and its own score.

        `None` for a rung means NOT MEASURED, and it has to stay
        distinguishable from a rung the model beat. A promoted champion whose
        `baseline_clock_score` is null did not beat the clock -- nobody asked
        the clock.
        """
        holdout = ticker_result.get('winner_holdout_metrics') or {}

        # EVERY number the gate weighed, copied wholesale rather than listed.
        #
        # The first version of this method enumerated fields, and within the
        # hour it was already out of date: `baseline_margin_sigma` -- the
        # value that decides promotion since #186 -- was added to the gate
        # and not to this list, so the champion record still could not say
        # what had cleared it. That is the SAME defect this method was
        # written to fix, reappearing inside the fix. A list of fields has to
        # be maintained; a copy cannot fall behind.
        #
        # Safe to copy: `_record_winner_test_score` pops the one array-shaped
        # entry (`_baseline_prediction`) before this is reached, so what
        # remains is scalars and short strings.
        evidence = {
            key: value for key, value in holdout.items()
            if not key.startswith('_')
        }
        evidence['binding_opponent'] = holdout.get('baseline_kind')
        # The verdict beside the numbers, so "why was this promoted" is
        # answerable from the record alone.
        evidence['gate'] = ticker_result.get('promotion_gate') or {}
        return evidence

    def _remember_refusal(self, context_key: str, fingerprint: str) -> None:
        """Record the refusal just appended, so a restart need not redo it.

        Reads the last row rather than taking an argument, because every
        refusal path in this loop appends exactly one and the alternative is
        four call sites each passing a slightly different dict.
        """
        if self._ledger is None:
            return
        self._ledger.record(
            context_key, fingerprint,
            refusal=self._gate_refusals[-1] if self._gate_refusals else None,
        )

    def _remember_champion(self, context_key: str, fingerprint: str,
                           champion: dict[str, Any]) -> None:
        """Record a promoted champion and the data it was promoted on."""
        if self._ledger is None:
            return
        self._ledger.record(context_key, fingerprint, champion=champion)

    def _record_unprepared_context(
        self,
        ticker: str,
        timeframe: str,
        target_name: str,
    ) -> None:
        """A context that never reached training still produced no champion.

        `prepare_data_for_models` returns nothing when the split cannot be
        built -- too few rows, no usable target values, the purge gap eating
        what was left. The loop simply moved on, so these contexts appeared
        nowhere: not among the champions, not among the refusals.

        That is the reverse of the mistake the artifact was built to fix. Every
        other row here says "a model was trained and judged not good enough";
        this one says "there was never enough to train on", and reading the
        first as the second is exactly how "no edge" and "no data" get
        confused. The pattern is absent from the key because training, which
        selects it, never ran.
        """
        self._gate_refusals.append({
            "context": f"{ticker}_{timeframe}_{target_name}",
            "ticker": ticker,
            "timeframe": timeframe,
            "target": target_name,
            "model_type": None,
            "reasons": "no usable train/test split was built; training never ran",
            "holdout_score": None,
            "baseline_score": None,
            # Same columns as a real refusal, so the two kinds of row stack into
            # one frame instead of parquet inventing nulls for a ragged schema.
            "baseline_kind": None,
            "baseline_constant_score": None,
            "baseline_persistence_score": None,
            "baseline_clock_score": None,
            "baseline_clock_scheme": None,
            "single_feature_score": None,
            "single_feature_name": None,
            "holdout_rows": None,
            "holdout_events": None,
        })

    @staticmethod
    def _write_gate_refusals(refusals: "list[dict[str, Any]]") -> "Path | None":
        """One row per context that produced no champion, and why."""
        if not refusals:
            logger.info("Every context produced a champion; no refusals to record.")
            return None
        frame = pd.DataFrame(refusals)
        directory = Path("data/results")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (
            f"gate_refusals_{datetime.datetime.now():%Y%m%d_%H%M%S}.parquet"
        )
        frame.to_parquet(path, index=False)
        by_target = frame.groupby("target").size().sort_values(ascending=False)
        logger.info(
            "Recorded %d promotion-gate refusals across %d targets to %s. "
            "Most refused: %s",
            len(frame), frame["target"].nunique(), path.name,
            ", ".join(f"{name} x{count}" for name, count in by_target.head(3).items()),
        )
        return path

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
            # Which series each holdout row belongs to. The naive opponents in
            # the promotion gate lag WITHIN a series; on a pooled frame, "the
            # row h positions back" is another ticker at nearly the same
            # timestamp, which makes the opponent measure nothing.
            "holdout_groups": light.get("holdout_groups"),
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
            # How many promotion attempts this run will make. The gate turns
            # it into the number of standard errors a margin must clear.
            # Before 2026-09-01 the bar was one standard error regardless,
            # which is a 16% false-positive rate per test -- measured, by
            # running twenty panels of pure noise through the gate and
            # watching it promote three (CLAIMS.md R11). Applied 27 times in
            # run 7, that is about four false champions out of nine.
            "promotion_family_size": getattr(self, "_promotion_family_size", None),
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
                    # Was a literal {}. `extract_native_feature_importance`
                    # sits in the module this call already imports and does
                    # exactly this job; the live stage simply never called it,
                    # so every artifact reported the winner as having no
                    # importances to give. BaseTrainer now reads them where the
                    # model and the columns it was FITTED on are both in hand —
                    # a length mismatch there returns {} indistinguishably from
                    # a genuine absence.
                    feature_importance=(
                        ticker_result.get('winner_feature_importance') or {}
                    ),
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
            # A write failure is not an absence of candidates. Returning {}
            # here means the caller's `if artifact_paths` skips the append and
            # the context leaves no trace in the artifact set at all -- so a
            # short artifact set and a complete one look identical.
            logger.error(
                "Could not write active Stage 4 pipeline-control candidates "
                "for %s/%s/%s (%s: %s).",
                ticker, timeframe, target_name, type(exc).__name__, exc,
            )
            self._artifact_write_failures.append(
                f"{ticker}/{timeframe}/{target_name}"
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
