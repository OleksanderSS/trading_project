from typing import Any

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger('EventDatasetValidator')


class EventDatasetValidationError(Exception):
    """Exception raised for invalid event dataset payloads."""
    pass


class EventDatasetValidator:
    """Validates event-centric datasets produced by NewsEventDatasetBuilder."""

    def __init__(self, nan_threshold: float=0.1):
        self.nan_threshold = nan_threshold
        self.required_columns = ['published_at', 'datetime', 'ticker',
            'news_title', 'news_sentiment']
        self.target_prefix = 'target_'

    def validate(self, df: pd.DataFrame) ->dict[str, Any]:
        """Validates the event dataset and returns a report."""
        issues: list[str] = []
        if df is None:
            issues.append('Event dataset is None.')
            return self._make_report(False, issues, df)
        if df.empty:
            issues.append('Event dataset is empty.')
            return self._make_report(False, issues, df)
        issues.extend(self._check_required_columns(df))
        issues.extend(self._check_datetime_columns(df))
        issues.extend(self._check_news_sentiment(df))
        issues.extend(self._check_target_columns(df))
        issues.extend(self._check_ticker_column(df))
        issues.extend(self._check_nan_inf(df))
        issues.extend(self._check_duplicates(df))
        is_valid = len([i for i in issues if i.startswith('CRITICAL:') or i
            .startswith('Missing') or i.startswith('No target_')]) == 0
        return self._make_report(is_valid, issues, df)

    def _make_report(self, is_valid: bool, issues: list[str], df: pd.DataFrame
        ) ->dict[str, Any]:
        return {'is_valid': is_valid, 'issues': issues, 'summary': {'rows':
            int(len(df)), 'columns': int(len(df.columns)), 'target_columns':
            len([c for c in df.columns if c.lower().startswith(self.
            target_prefix)]), 'missing_required': len([c for c in self.
            required_columns if c not in df.columns]), 'duplicate_rows':
            int(df.duplicated(subset=['ticker', 'datetime']).sum()) if all(
            c in df.columns for c in ['ticker', 'datetime']) else 0}}

    def _check_required_columns(self, df: pd.DataFrame) ->list[str]:
        missing = [col for col in self.required_columns if col not in df.
            columns]
        return [f'Missing required column: {col}' for col in missing]

    def _check_datetime_columns(self, df: pd.DataFrame) ->list[str]:
        issues: list[str] = []
        for column in ['published_at', 'datetime']:
            if column in df.columns:
                try:
                    pd.to_datetime(df[column], errors='raise')
                except Exception as e:
                    self.logger.error(f'Виникла помилка під час парсингу {column}: {e}', exc_info=True)
                    issues.append(
                        f"CRITICAL: Column '{column}' contains unparseable datetime values."
                        )
                    raise
        return issues

    def _check_news_sentiment(self, df: pd.DataFrame) ->list[str]:
        if 'news_sentiment' not in df.columns:
            return ['Missing required column: news_sentiment']
        if not pd.api.types.is_numeric_dtype(df['news_sentiment']):
            return ['CRITICAL: news_sentiment is not numeric.']
        invalid = df['news_sentiment'].dropna().apply(lambda x: x < -1 or x > 1
            )
        if invalid.any():
            return [
                'CRITICAL: news_sentiment contains values outside the range [-1, 1].'
                ]
        return []

    def _check_target_columns(self, df: pd.DataFrame) ->list[str]:
        target_cols = [col for col in df.columns if col.lower().startswith(
            self.target_prefix)]
        if not target_cols:
            return ['CRITICAL: No target_* columns found in event dataset.']
        if df[target_cols].isna().all(axis=None):
            return ['CRITICAL: All target_* values are missing.']
        return []

    def _check_ticker_column(self, df: pd.DataFrame) ->list[str]:
        if 'ticker' not in df.columns:
            return ['CRITICAL: Missing ticker column.']
        if df['ticker'].isna().all():
            return [
                'CRITICAL: ticker column exists but contains only null values.'
                ]
        return []

    def _check_nan_inf(self, df: pd.DataFrame) ->list[str]:
        issues: list[str] = []
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return issues
        nan_ratio = numeric_df.isna().mean().max()
        if nan_ratio > self.nan_threshold:
            issues.append(
                f'CRITICAL: Numeric data contains too many NaNs ({nan_ratio:.2%}).'
                )
        inf_count = np.isinf(numeric_df).sum().sum()
        if inf_count > 0:
            issues.append(
                f'CRITICAL: Detected {int(inf_count)} infinite values.')
        return issues

    def _check_duplicates(self, df: pd.DataFrame) ->list[str]:
        issues: list[str] = []
        if 'news_id' in df.columns:
            duplicated_news_id = int(df['news_id'].duplicated().sum())
            if duplicated_news_id > 0:
                issues.append(
                    f'CRITICAL: Found {duplicated_news_id} duplicated news_id values.'
                    )
        if all(col in df.columns for col in ['ticker', 'datetime']):
            duplicated_rows = int(df.duplicated(subset=['ticker',
                'datetime']).sum())
            if duplicated_rows > 0:
                issues.append(
                    f'CRITICAL: Found {duplicated_rows} duplicated ticker/datetime rows.'
                    )
        return issues
