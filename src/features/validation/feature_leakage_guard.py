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
        self.status: str = 'clean'

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
        =True, report_dir: str | None='reports/leakage'):
        """
        Args:
            corr_threshold: Correlation threshold above which a feature is suspicious.
            block_on_forbidden: If True — raises ValueError if forbidden column found (stops pipeline).
            report_dir: Where to save leakage_report.json after each check.
        """
        self.corr_threshold = corr_threshold
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
        sample_df = df[numeric_features + numeric_targets].dropna()
        if len(sample_df) > 50000:
            sample_df = sample_df.sample(50000, random_state=42)
        if sample_df.empty:
            return high_corr
        try:
            # Build suspicious set against ALL numeric targets, not just [0].
            # A feature that leaks target_weekly_return_1w but not target_up_1d
            # would be missed if we only checked the first target column.
            suspicious: set[str] = set()
            for tgt in numeric_targets:
                corr_vec = sample_df[numeric_features].corrwith(sample_df[tgt]).abs()
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
