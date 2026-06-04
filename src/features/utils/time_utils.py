# src/features/utils/time_utils.py

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_time_features(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    enabled_features: list[str] = None
) -> pd.DataFrame:
    """
    Adds specified time-based features to the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        timestamp_col (str): The name of the column containing datetime stamps.
        enabled_features (List[str]): A list of feature names to generate.

    Returns:
        pd.DataFrame: DataFrame with added time features.
    """
    if timestamp_col not in df.columns:
        logger.error(f"Timestamp column '{timestamp_col}' not found.")
        return df

    if enabled_features is None:
        logger.warning("No time features were specified in the config.")
        return df

    df_out = df.copy()
    timestamps = pd.to_datetime(df_out[timestamp_col])

    # Feature generation mapping
    feature_generators = {
        'hour': lambda ts: ts.dt.hour,
        'day_of_week': lambda ts: ts.dt.dayofweek,
        'day_of_month': lambda ts: ts.dt.day,
        'day_of_year': lambda ts: ts.dt.dayofyear,
        'week_of_year': lambda ts: ts.dt.isocalendar().week.astype(int),
        'month_of_year': lambda ts: ts.dt.month,
        'quarter': lambda ts: ts.dt.quarter,
        'is_weekend': lambda ts: (ts.dt.dayofweek >= 5).astype(int),
        'is_month_start': lambda ts: (ts.dt.is_month_start).astype(int),
        'is_month_end': lambda ts: (ts.dt.is_month_end).astype(int),
        'is_quarter_start': lambda ts: (ts.dt.is_quarter_start).astype(int),
        'is_quarter_end': lambda ts: (ts.dt.is_quarter_end).astype(int),
        'is_year_start': lambda ts: (ts.dt.is_year_start).astype(int),
        'is_year_end': lambda ts: (ts.dt.is_year_end).astype(int),
        'market_session': lambda ts: np.select(
            [
                (ts.dt.hour >= 9) & (ts.dt.hour < 14), # Pre-market (UTC)
                (ts.dt.hour >= 14) & (ts.dt.hour < 21)  # Market (UTC)
            ],
            [1, 2], # 1: Pre, 2: Market
            default=0 # 0: Post
        ),
        # Cyclical features
        'hour_sin': lambda ts: np.sin(2 * np.pi * ts.dt.hour / 24.0),
        'hour_cos': lambda ts: np.cos(2 * np.pi * ts.dt.hour / 24.0),
        'day_of_week_sin': lambda ts: np.sin(2 * np.pi * ts.dt.dayofweek / 7.0),
        'day_of_week_cos': lambda ts: np.cos(2 * np.pi * ts.dt.dayofweek / 7.0),
    }

    added_cols = []
    for feature_name in enabled_features:
        if feature_name in feature_generators:
            df_out[feature_name] = feature_generators[feature_name](timestamps)
            added_cols.append(feature_name)
        else:
            logger.warning(f"Time feature '{feature_name}' is not a known generator.")

    if added_cols:
        logger.info(f"Added {len(added_cols)} time features: {added_cols}")

    return df_out
