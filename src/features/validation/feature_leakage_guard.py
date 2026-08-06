"""
FeatureLeakageGuard — Data leakage detector between features and targets.

Checks if features "see" the future via:
1. Too high feature<->target correlation (> threshold)
2. Presence of forbidden columns (future_price, next_close, etc.) in X
3. Stops transmission of contaminated data to Colab for training heavy models.

Integration: called from ColabManager._check_feature_leakage()
(src/pipeline/hybrid/colab_manager.py) before saving a batch's
features/targets to Parquet, with block_on_forbidden=True - a forbidden
column stops the save. NOT currently called from Stage 3
(FeatureEngineeringStage) itself, despite an earlier version of this
docstring claiming otherwise.
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.pipeline.target_column_utils import is_direct_target_column, is_target_like_column

logger = ProjectLogger.get_logger('FeatureLeakageGuard')
_FORBIDDEN_PATTERNS = ['future_', 'next_close', 'next_open', 'next_high',
    'next_low', 'next_price', 'forward_', 'fwd_', 't+1', 't_plus',
    'tomorrow_', 'lead_', 'ahead_']


class LeakageReport:
    """Data leakage check result."""

    def __init__(self):
        self.forbidden_cols: list[str] = []
        self.high_corr_cols: dict[str, dict[str, float]] = {}
        self.timestamp = datetime.now().isoformat()
        # 'not_checked' until check() decides otherwise. It used to default to
        # 'clean', so a report from the no-targets early return -- where
        # nothing is compared at all -- was indistinguishable from one that
        # looked and found nothing.
        self.status: str = 'not_checked'

    @property
    def has_issues(self) ->bool:
        return bool(self.forbidden_cols or self.high_corr_cols)

    def to_dict(self) ->dict:
        return {'status': self.status, 'timestamp': self.timestamp,
            'forbidden_columns': self.forbidden_cols,
            'high_correlation_columns': self.high_corr_cols, 'total_issues':
            len(self.forbidden_cols) + len(self.high_corr_cols)}

    def __repr__(self) ->str:
        return (
            f'LeakageReport(status={self.status}, forbidden={len(self.forbidden_cols)}, high_corr={len(self.high_corr_cols)})'
            )


class FeatureLeakageGuard:
    """
    Data leakage detector between features and targets.

    Use before passing data to Colab or model training.
    Prevents "good-but-wrong" models that actually see the future.
    """

    def __init__(self, corr_threshold: float=0.95, block_on_forbidden: bool
        =True, report_dir: str | None='reports/leakage',
        min_overlap: int=100):
        """
        Args:
            corr_threshold: Correlation threshold above which a feature is suspicious.
            block_on_forbidden: If True — raises ValueError if forbidden column found (stops pipeline).
            report_dir: Where to save leakage_report.json after each check.
            min_overlap: Minimum rows where a feature and a target both exist
                before their correlation counts as evidence. Sparse features
                (news, macro) overlap a target on few rows, and a correlation
                from a handful of points is noise, not leakage.
        """
        self.corr_threshold = corr_threshold
        self.min_overlap = min_overlap
        self.block_on_forbidden = block_on_forbidden
        self.report_dir = Path(report_dir) if report_dir else None
        if self.report_dir:
            self.report_dir.mkdir(parents=True, exist_ok=True)

    def check(self, df: pd.DataFrame, feature_cols: list[str] | None=
        None, target_cols: list[str] | None=None, ticker: str='unknown'
        ) ->LeakageReport:
        """
        Perform full data leakage check.

        Args:
            df: DataFrame with features and targets.
            feature_cols: List of feature columns. If None — all non-target columns.
            target_cols: List of target columns. If None — columns with prefix 'target_'.
            ticker: Ticker identifier (for logs).

        Returns:
            LeakageReport with check results.
        """
        report = LeakageReport()
        if target_cols is None:
            target_cols = [c for c in df.columns if is_direct_target_column(c)]
        if feature_cols is None:
            meta_cols = {'datetime', 'ticker', 'published_at', 'date'} | set(
                target_cols)
            feature_cols = [c for c in df.columns if c not in meta_cols]
        if not target_cols:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f'[{ticker}] No target columns found — skipping leakage check.'
                    )
            return report
        report.forbidden_cols = self._check_forbidden_cols(feature_cols, ticker
            )
        report.high_corr_cols = self._check_correlation(df, feature_cols,
            target_cols, ticker)
        if report.forbidden_cols:
            report.status = 'blocked'
            msg = (
                f'[{ticker}] ⛔ FeatureLeakageGuard: {len(report.forbidden_cols)} FORBIDDEN column(s) detected → {report.forbidden_cols}. These likely contain future information!'
                )
            logger.error(msg)
            if self.block_on_forbidden:
                self._save_report(report, ticker)
                raise ValueError(msg)
        elif report.high_corr_cols:
            report.status = 'warning'
            logger.warning(
                f'[{ticker}] ⚠️ FeatureLeakageGuard: {len(report.high_corr_cols)} high-correlation feature(s) detected (threshold={self.corr_threshold}). Review before training.'
                )
        else:
            report.status = 'clean'
            logger.info(
                f'[{ticker}] ✅ FeatureLeakageGuard: No leakage detected.')
        self._save_report(report, ticker)
        return report

    def _check_forbidden_cols(self, feature_cols: list[str], ticker: str
        ) ->list[str]:
        """Searches for forbidden column names (future patterns)."""
        forbidden = []
        for col in feature_cols:
            col_lower = str(col).lower()
            if is_target_like_column(col) or any(pattern in col_lower for pattern in _FORBIDDEN_PATTERNS):
                forbidden.append(col)
                logger.warning(
                    f"[{ticker}] ⛔ Forbidden pattern found in column: '{col}'")
        return forbidden

    def _check_correlation_per_ticker(
        self,
        working: pd.DataFrame,
        numeric_features: list[str],
        numeric_targets: list[str],
        ticker: str,
    ) -> dict[str, dict[str, float]]:
        """Highest within-ticker correlation for each feature/target pair.

        Leakage is a property of one instrument's own history, so it is
        measured there and the worst instrument decides. Averaging would let
        a feature that leaks badly in one ticker hide behind a hundred where
        it does not.

        A ticker whose rows are too few to say anything is skipped rather
        than counted as clean -- min_overlap applies per ticker for the same
        reason it applied to the pooled frame.
        """
        high_corr: dict[str, dict[str, float]] = {}
        worst: dict[tuple[str, str], float] = {}

        for name, group in working.groupby('ticker', sort=False):
            if len(group) < self.min_overlap:
                continue
            for tgt in numeric_targets:
                target_values = group[tgt]
                if target_values.notna().sum() < self.min_overlap:
                    continue
                # nunique() <= 1: a constant series has no correlation with
                # anything, and pandas returns NaN rather than 0.
                if target_values.nunique(dropna=True) <= 1:
                    continue
                corr_vec = group[numeric_features].corrwith(target_values).abs()
                overlap = group.loc[
                    target_values.notna(), numeric_features
                ].notna().sum()
                corr_vec = corr_vec[overlap >= self.min_overlap].dropna()
                for feat, value in corr_vec.items():
                    key = (str(feat), str(tgt))
                    if value > worst.get(key, 0.0):
                        worst[key] = float(value)

        for (feat, tgt), value in worst.items():
            if value < self.corr_threshold:
                continue
            high_corr.setdefault(feat, {})[tgt] = round(value, 4)
            logger.warning(
                f"[{ticker}] ⚠️ HIGH CORRELATION within a single ticker: "
                f"'{feat}' ↔ '{tgt}' = {value:.3f} "
                f"(threshold={self.corr_threshold}) — possible data leakage!"
            )
        return high_corr

    def _check_correlation(self, df: pd.DataFrame, feature_cols: list[str],
        target_cols: list[str], ticker: str) ->dict[str, dict[str, float]]:
        """
        Checks correlation between features and targets.
        Returns dict: {feature_col: {target_col: correlation}}.
        """
        high_corr: dict[str, dict[str, float]] = {}
        numeric_features = [c for c in feature_cols if c in df.columns and
            pd.api.types.is_numeric_dtype(df[c])]
        numeric_targets = [c for c in target_cols if c in df.columns and pd
            .api.types.is_numeric_dtype(df[c])]
        if not numeric_features or not numeric_targets:
            return high_corr
        # No global dropna. It used to be df[...].dropna(), which drops a row
        # if ANY of the ~700 columns is null there. Measured on a frame shaped
        # like ours, a SINGLE sparse column -- a news or macro feature present
        # in 3% of rows, which we have -- cut 5,000 rows to 150. Two such
        # columns empty the frame outright, and the check then returns "clean"
        # having compared nothing. corrwith aligns pairwise on its own, so
        # each feature is scored on every row where it and the target both
        # exist.
        # Correlate WITHIN a ticker, not across the panel.
        #
        # Pooling every ticker's rows makes any two price-level series
        # correlate at ~0.95, because both are "the price level of this
        # instrument" and the instruments differ. Measured on the
        # 2026-08-05 export, 110 tickers with median prices from $5 to $955,
        # against target_ema_20_f1:
        #
        #     pooled          within-ticker median
        #     close        0.9394        0.0000
        #     EMA_100_1d   0.9521       -0.0008
        #     EMA_200_1d   0.9588        0.0012
        #
        # The guard reported EMA_100_1d and EMA_200_1d as possible leakage on
        # that run. `close` scores the same, which is the giveaway: if the
        # 100-day average leaks, so does the price, and then the warning
        # means nothing.
        #
        # It fails in the other direction too. A feature that genuinely leaks
        # WITHIN a ticker can be diluted below the threshold once 109 other
        # instruments are mixed in, so pooling both invents leakage and hides
        # it. Neither had been visible before because the panel was 22
        # tickers of similar price; growing it to 110 exposed both.
        #
        # The maximum across tickers is the number that matters: leakage in
        # one instrument is leakage.
        working = df[numeric_features + numeric_targets + (
            ['ticker'] if 'ticker' in df.columns else []
        )]
        if len(working) > 50000:
            working = working.sample(50000, random_state=42)
        if working.empty:
            return high_corr

        if 'ticker' in working.columns and working['ticker'].nunique() > 1:
            return self._check_correlation_per_ticker(
                working, numeric_features, numeric_targets, ticker
            )
        sample_df = working[numeric_features + numeric_targets]
        try:
            # Build suspicious set against ALL numeric targets, not just [0].
            # A feature that leaks target_weekly_return_1w but not target_up_1d
            # would be missed if we only checked the first target column.
            suspicious: set[str] = set()
            for tgt in numeric_targets:
                corr_vec = sample_df[numeric_features].corrwith(sample_df[tgt]).abs()
                # A correlation computed from a handful of overlapping points
                # is not evidence of anything. Without this, a feature present
                # in three rows that happens to track the target there would
                # be reported as leakage.
                overlap = sample_df.loc[
                    sample_df[tgt].notna(), numeric_features
                ].notna().sum()
                corr_vec = corr_vec[overlap >= self.min_overlap]
                suspicious.update(corr_vec[corr_vec >= self.corr_threshold].index.tolist())

            for feat in suspicious:
                feat_corrs = {}
                for tgt in numeric_targets:
                    corr_val = sample_df[feat].corr(sample_df[tgt])
                    if abs(corr_val) >= self.corr_threshold:
                        feat_corrs[tgt] = round(float(corr_val), 4)
                        logger.warning(
                            f"[{ticker}] ⚠️ HIGH CORRELATION: '{feat}' ↔ '{tgt}' = {corr_val:.3f} (threshold={self.corr_threshold}) — possible data leakage!"
                            )
                if feat_corrs:
                    high_corr[feat] = feat_corrs
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f'Виникла помилка: {e}', exc_info=True)
            logger.warning(f'[{ticker}] Correlation check failed: {e}')
            raise
        return high_corr

    def _save_report(self, report: LeakageReport, ticker: str) ->None:
        """Saves leakage report to JSON file."""
        if not self.report_dir:
            return
        try:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = (self.report_dir /
                f'leakage_report_{ticker}_{ts}.json')
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report.to_dict(), f, indent=2, ensure_ascii=False, default=str)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'Leakage report saved: {report_path}')
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f'Виникла помилка: {e}', exc_info=True)
            logger.warning(f'Could not save leakage report: {e}')
            raise


_guard_instance: FeatureLeakageGuard | None = None


def get_leakage_guard(corr_threshold: float=0.95, block_on_forbidden: bool=
    True, report_dir: str='reports/leakage') ->FeatureLeakageGuard:
    """
    Returns singleton instance of FeatureLeakageGuard for pipeline use.
    Currently unused - ColabManager._check_feature_leakage() constructs
    its own FeatureLeakageGuard instance directly instead of using this
    singleton factory. Kept for any future caller that wants a shared
    instance.
    """
    global _guard_instance
    if _guard_instance is None:
        _guard_instance = FeatureLeakageGuard(corr_threshold=corr_threshold,
            block_on_forbidden=block_on_forbidden, report_dir=report_dir)
    return _guard_instance
