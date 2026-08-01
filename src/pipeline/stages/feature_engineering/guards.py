import pandas as pd

from src.core.logging.logger import ProjectLogger
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
                actionable_issues = [
                    issue for issue in issues
                    if any(marker in issue for marker in (
                        'Feature name indicates',
                        'Negative shift detected',
                        'Backfill operation detected',
                        'Rolling window too large',
                    ))
                ]
                message = f"Temporal leakage guard found {len(issues)} issues"
                # Temporal leakage is ALWAYS fail-closed: lookahead bias
                # silently inflates backtests and destroys live performance.
                # Mode only downgrades non-leakage warnings, never leaks.
                if actionable_issues:
                    raise ValueError(f"{message}: {' | '.join(actionable_issues)}")
                self.logger.warning(message)

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


