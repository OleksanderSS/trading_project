import logging

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger('DataCleaner')


class DataCleaner:
    """
    A utility class providing static methods for data sanitization and cleaning
    of market and news data before feature engineering.
    """

    @staticmethod
    def remove_outliers_zscore(df: pd.DataFrame, columns: str | list[str]='close', threshold: float=3.0) ->pd.DataFrame:
        """
        Removes outliers from specified columns based on Z-score calculated on rolling log returns.

        Args:
            df: Input DataFrame.
            columns: Column name or list of columns to analyze.
            threshold: Z-score threshold for outlier detection (default 3.0).

        Returns:
            DataFrame with rows containing outliers in specified columns removed.
        """
        if df is None or df.empty:
            return df
        if isinstance(columns, str):
            columns = [columns]
        df_out = df.copy()
        try:
            total_mask = pd.Series([False] * len(df_out), index=df_out.index)
            for col in columns:
                if col not in df_out.columns:
                    continue
                if 'ticker' in df_out.columns:
                    group_columns = ['ticker']
                    if 'interval' in df_out.columns:
                        group_columns.append('interval')
                    sort_columns = list(group_columns)
                    if 'datetime' in df_out.columns:
                        sort_columns.append('datetime')
                    ordered = df_out.sort_values(
                        sort_columns,
                        kind='mergesort',
                    )
                    log_returns = ordered.groupby(
                        group_columns,
                        group_keys=False,
                    )[col].transform(
                        lambda series: np.log(
                            series / series.shift(1)
                        )
                    )
                    grouping = [
                        ordered[column]
                        for column in group_columns
                    ]
                    rolling_mean = log_returns.groupby(
                        grouping,
                        group_keys=False,
                    ).transform(
                        lambda series: series.rolling(
                            20,
                            min_periods=1,
                        ).mean()
                    )
                    rolling_std = log_returns.groupby(
                        grouping,
                        group_keys=False,
                    ).transform(
                        lambda series: series.rolling(
                            20,
                            min_periods=1,
                        ).std()
                    )
                else:
                    log_returns = np.log(df_out[col] / df_out[col].shift(1))
                    rolling_mean = log_returns.rolling(20, min_periods=1).mean()
                    rolling_std = log_returns.rolling(20, min_periods=1).std()
                z_scores = (log_returns - rolling_mean) / rolling_std
                col_mask = (z_scores.abs() > threshold).where(z_scores.notna(), False)
                total_mask |= col_mask.reindex(
                    df_out.index,
                    fill_value=False,
                )
            outlier_count = total_mask.sum()
            if outlier_count > 0:
                df_out = df_out[~total_mask]
                logger.info(
                    f'Removed {outlier_count} rows containing outliers in columns {columns} (Threshold: {threshold})'
                    )
            return df_out
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(
                f'Error during outlier removal for columns {columns}: {e}')
            return df

    @staticmethod
    def clean_text_data(df: pd.DataFrame, text_columns: list[str]) -> pd.DataFrame:
        """
        Cleans text columns by removing HTML tags, special characters, and lowercasing.
        """
        if df is None or df.empty:
            return df
        df_out = df.copy()

        for col in text_columns:
            if col in df_out.columns:
                df_out[col] = df_out[col].fillna('')
                df_out[col] = df_out[col].astype(str).str.replace(r'<[^>]*>', '', regex=True)
                df_out[col] = df_out[col].str.replace(r'[^\w\s\.,!?]', '', regex=True)
                df_out[col] = df_out[col].str.replace(r'\s+', ' ', regex=True).str.strip()
                df_out[col] = df_out[col].str.lower()

        logger.info(f"Cleaned text data for columns: {text_columns}")
        return df_out

    @staticmethod
    def clean_macro_data(df: pd.DataFrame, numeric_columns: list[str], threshold: float = 3.0) -> pd.DataFrame:
        """
        Cleans macro-economic data by clipping outliers instead of dropping them, and interpolating missing values.

        NOTE: this method has no callers as of 2026-08-02. It carried two
        lookahead defects (a trailing .bfill() and centred rolling windows),
        both fixed here rather than left waiting for whoever wires it up.
        Macro data reaches the model through MacroFeaturesEnricher, which
        does not route through this.
        """
        if df is None or df.empty:
            return df
        df_out = df.copy()

        for col in numeric_columns:
            if col not in df_out.columns:
                continue

            df_out[col] = pd.to_numeric(df_out[col], errors='coerce')
            # ffill only. The .bfill() that used to follow filled a LEADING
            # gap with the first value that came later -- so rows dated
            # before a series began publishing were handed a number nobody
            # could have known at the time. A leading NaN is the truth: the
            # series had no value yet.
            df_out[col] = df_out[col].ffill()

            # center=False. A centred window at row i spans i-10..i+9, so the
            # clipping bounds applied to row i were computed partly from bars
            # up to nine steps in its future -- and an outlier was therefore
            # judged against data that had not happened. Trailing windows are
            # the only causal choice here.
            window = df_out[col].rolling(window=20, min_periods=1)
            rolling_median = window.median()
            rolling_mad = window.apply(lambda x: np.median(np.abs(x - np.median(x))))
            rolling_std = rolling_mad * 1.4826

            lower_bound = rolling_median - (threshold * rolling_std)
            upper_bound = rolling_median + (threshold * rolling_std)

            df_out[col] = df_out[col].clip(lower=lower_bound, upper=upper_bound)

        logger.info(f"Cleaned macro data for columns: {numeric_columns}")
        return df_out

    @staticmethod
    def handle_missing_values(df: pd.DataFrame, method: str='ffill'
        ) ->pd.DataFrame:
        """
        Handles missing values in the DataFrame using specified method.
        """
        if df is None or df.empty:
            return df
        df_out = df.copy()
        nan_count = df_out.isna().sum().sum()
        if nan_count > 0:
            group_columns = [
                column
                for column in ('ticker', 'interval')
                if column in df_out.columns
            ]
            service_columns = {
                'ticker',
                'interval',
                'datetime',
                'timestamp',
                'date',
            }
            data_cols = [
                col
                for col in df_out.columns
                if col not in service_columns
            ]
            if method == 'ffill':
                if group_columns:
                    sort_columns = list(group_columns)
                    datetime_column = next(
                        (
                            column
                            for column in (
                                'datetime',
                                'timestamp',
                                'date',
                            )
                            if column in df_out.columns
                        ),
                        None,
                    )
                    if datetime_column:
                        sort_columns.append(datetime_column)
                    ordered = df_out.sort_values(
                        sort_columns,
                        kind='mergesort',
                    )
                    ordered[data_cols] = ordered.groupby(
                        group_columns,
                        dropna=False,
                    )[data_cols].ffill()
                    df_out.loc[ordered.index, data_cols] = ordered[
                        data_cols
                    ]
                else:
                    df_out = df_out.ffill()
            elif method == 'bfill':
                logger.warning(
                    "Backfill is disabled for causal time series cleaning; using forward-fill instead."
                    )
                if group_columns:
                    sort_columns = list(group_columns)
                    datetime_column = next(
                        (
                            column
                            for column in (
                                'datetime',
                                'timestamp',
                                'date',
                            )
                            if column in df_out.columns
                        ),
                        None,
                    )
                    if datetime_column:
                        sort_columns.append(datetime_column)
                    ordered = df_out.sort_values(
                        sort_columns,
                        kind='mergesort',
                    )
                    ordered[data_cols] = ordered.groupby(
                        group_columns,
                        dropna=False,
                    )[data_cols].ffill()
                    df_out.loc[ordered.index, data_cols] = ordered[
                        data_cols
                    ]
                else:
                    df_out = df_out.ffill()
            logger.info(f'Handled {nan_count} missing values using {method}.')
        return df_out

    @staticmethod
    def validate_schema(df: pd.DataFrame, required_cols: list[str]) ->bool:
        """
        Validates if the DataFrame contains all required columns.
        """
        if df is None:
            logger.error('Schema validation failed: DataFrame is None')
            return False
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            logger.warning(
                f'Schema validation: Missing required columns: {missing}')
            return False
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f'Schema validation successful for columns: {required_cols}')
        return True
