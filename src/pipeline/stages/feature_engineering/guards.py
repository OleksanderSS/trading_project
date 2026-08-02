import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.features.validation.feature_leakage_guard import get_leakage_guard
from src.pipeline.target_column_utils import split_model_features_and_targets
from src.pipeline.guards.macro_release_timing_guard import get_macro_release_timing_guard
from src.pipeline.guards.safe_feature_combiner import get_safe_feature_combiner
from src.pipeline.guards.temporal_leakage_guard import get_temporal_leakage_guard
from src.pipeline.guards.temporal_target_guard import get_temporal_target_guard
from src.pipeline.guards.timeframe_alignment_guard import get_timeframe_alignment_guard
from src.pipeline.timeframe_lineage import normalize_timeframe

logger = ProjectLogger.get_logger('FeatureGuards')

class FeatureGuards:
    """Manages temporal safety guards for feature engineering."""

    def __init__(self, mode: str = 'full'):
        self.logger = logger
        self.mode = mode
        self.strict_mode = mode != 'prepare'
        self._initialize_guards()

    def _initialize_guards(self):
        """Initialize all safety guards."""
        # Wired into the main Stage 3 path on 2026-08-01. It was previously
        # reachable only from the Colab/hybrid branch (colab_manager.py), so
        # a normal training run got no runtime leakage check at all -- the
        # pattern-based TemporalLeakageGuard matches 0 of the 713 real feature
        # names, since its patterns describe a naming convention this project
        # does not use.
        #
        # This one is different: it detects by CORRELATION against the actual
        # targets, which does not care what anything is called, and by the
        # project's own is_target_like_column rule. block_on_forbidden matches
        # the Colab branch rather than inventing a second convention -- and
        # the blocking condition is unambiguous (a target column sitting among
        # the features), while the fuzzy half, correlation, only ever warns.
        self.leakage_guard = get_leakage_guard(block_on_forbidden=self.strict_mode)
        self.timeframe_guard = get_timeframe_alignment_guard(strict_mode=self.strict_mode)
        self.safe_combiner = get_safe_feature_combiner(self.timeframe_guard)
        self.temporal_target_guard = get_temporal_target_guard()
        self.temporal_leakage_guard = get_temporal_leakage_guard()
        self.macro_guard = get_macro_release_timing_guard()
        self.logger.info(f'✅ Temporal safety guards initialized (mode: {self.mode}, strict: {self.strict_mode})')

    def apply_guards(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply various safety checks to the feature set."""
        if df is None or df.empty:
            return df

        guarded = df.loc[:, ~df.columns.duplicated()].copy()
        datetime_col = next((col for col in ('datetime', 'timestamp', 'date') if col in guarded.columns), None)
        if datetime_col:
            sort_cols = [
                col
                for col in ('ticker', datetime_col)
                if col in guarded.columns
            ]
            guarded = guarded.sort_values(
                sort_cols,
                kind='mergesort',
            ).reset_index(drop=True)

        if datetime_col:
            current_time = pd.to_datetime(guarded[datetime_col], errors='coerce').max()
            timeframe = self._infer_timeframe(guarded)
            validation = self.temporal_leakage_guard.validate_rolling_windows(
                guarded,
                current_time=current_time,
                timeframe=timeframe,
            )
            if validation.get('status') == 'invalid':
                issues = validation.get('issues', [])
                # 'Rolling window too large' was removed from this list, and
                # from the guard, on 2026-08-02. It is not a leakage
                # condition: a long trailing window reads only past bars. The
                # guard's pattern never matched this project's naming, so the
                # check had never fired -- and repairing the pattern without
                # this change would have aborted Stage 3 on SMA_200_60m and
                # EMA_200_60m, four clean columns in the current export whose
                # only sin is a 200-period lookback against a 168 budget. It
                # is a warning now.
                #
                # 'Negative shift detected' and 'Backfill operation detected'
                # stay listed but can no longer be produced here: they were
                # matched against feature NAMES, and a column is never called
                # "close.shift(-1)". That check moved to a source scanner,
                # tests/contracts/test_lookahead_operations.py, where the
                # expression it looks for actually exists. Kept in this tuple
                # so that a future runtime detector emitting the same wording
                # is fatal by default, which is the right default for
                # lookahead.
                actionable_issues = [
                    issue for issue in issues
                    if any(marker in issue for marker in (
                        'Feature name indicates',
                        'Negative shift detected',
                        'Backfill operation detected',
                    ))
                ]
                message = f"Temporal leakage guard found {len(issues)} issues"
                # Temporal leakage is ALWAYS fail-closed: lookahead bias
                # silently inflates backtests and destroys live performance.
                # Mode only downgrades non-leakage warnings, never leaks.
                if actionable_issues:
                    raise ValueError(f"{message}: {' | '.join(actionable_issues)}")
                self.logger.warning(message)

        # Feature-vs-target leakage. Skips itself when the frame carries no
        # target columns, so it is a no-op on intermediate frames.
        ticker = str(guarded['ticker'].iloc[0]) if (
            'ticker' in guarded.columns and not guarded.empty
        ) else 'all'

        # Check the columns that will ACTUALLY become features, not every
        # column in the frame. At this point the frame still holds
        # target-derived columns such as state_TARGET_RETURN_1P, which
        # split_model_features_and_targets drops before training -- verified
        # on the 2026-08-02 prepare run: the guard flagged that column 14
        # times (AAPL 5, AMD 9) while the exported features.parquet contains
        # ZERO target-derived columns out of 1,189.
        #
        # Left as it was, this would have been worse than noise: Stage 3
        # builds FeatureGuards with mode defaulting to 'full', so
        # block_on_forbidden is True for every mode except 'prepare', and the
        # next --mode continue run would have raised ValueError and killed
        # the stage over a column the pipeline itself removes. A
        # target-derived column that survives THIS split is a real defect;
        # one that does not is the pipeline working.
        feature_columns, target_columns, dropped = split_model_features_and_targets(
            guarded.columns
        )
        if dropped:
            # The forbidden-column half of the guard becomes tautological once
            # it is handed the split's output -- both use
            # is_target_like_column, so nothing forbidden can survive. The
            # information is not thrown away: what the split had to remove is
            # worth knowing, because it means an enricher produced a
            # target-derived column. It is reported rather than raised,
            # because removing it is the pipeline behaving correctly. The
            # correlation half, which does not depend on naming at all,
            # remains the real detector.
            self.logger.warning(
                "Dropped %d target-derived column(s) before feature checks: "
                "%s. These are excluded from training, but an enricher is "
                "producing them.",
                len(dropped), [str(column) for column in dropped][:5],
            )
        self.leakage_guard.check(
            guarded,
            feature_cols=[str(column) for column in feature_columns],
            target_cols=[str(column) for column in target_columns],
            ticker=ticker,
        )

        return guarded

    def _infer_timeframe(self, df: pd.DataFrame) -> str | None:
        """The timeframe key handed to the leakage guard.

        Normalised, because the two sides spell the same timeframe
        differently: market_data_raw stores interval='1h' (9,549 rows) while
        TemporalLeakageGuard.SAFE_ROLLING_CONFIGS is keyed '60m'. Passing the
        raw value made that lookup miss, so max_periods silently fell back to
        100 instead of 168 -- and "rolling window too large" is one of the
        conditions this stage treats as fatal. price_filter.py already worked
        around the same split by accepting both spellings locally.
        """
        for column in ('interval', 'timeframe'):
            if column in df.columns and df[column].nunique() == 1:
                return normalize_timeframe(df[column].iloc[0])

        # More than one timeframe in the frame is the normal case for a
        # combined feature set, and it means no single rolling-window budget
        # applies. Said out loud rather than returned as a bare None, because
        # the guard skips its entire window check when the timeframe is
        # unknown.
        present = [c for c in ('interval', 'timeframe') if c in df.columns]
        if present:
            self.logger.debug(
                "Rolling-window leakage check skipped: %s holds %d distinct "
                "timeframes, so no single window budget applies.",
                present[0], int(df[present[0]].nunique()),
            )
        return None


