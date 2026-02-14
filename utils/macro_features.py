# utils/macro_features.py

import pandas as pd
import numpy as np
from typing import Optional
from config.macro_config import (
    FRED_ALIAS,
    DATA_INTERVALS,
    NORMALIZATION_SCALES,
    DECAY_LAMBDAS_BY_FREQ,
    MACRO_WINDOWS  # новий словник у конфігу: {"trend":30, "zscore":180}
)
from utils.logger import ProjectLogger

logger = ProjectLogger.get_logger("TradingProjectLogger")

# ---------------- HELPERS ----------------
def _alias_for_column(col: str) -> str:
    key = col.replace("fred_", "")
    return FRED_ALIAS.get(key, key)

def _lambda_for_column(col: str) -> float:
    key = col.replace("fred_", "")
    freq = DATA_INTERVALS.get(key, "monthly")
    return DECAY_LAMBDAS_BY_FREQ.get(freq, 0.01)

def _norm_scale_for_alias(alias: str) -> float:
    return NORMALIZATION_SCALES.get(alias, 1.0)

# ---------------- MAIN FUNCTION ----------------
def enrich_macro_features(df_macro: pd.DataFrame, end_date: Optional[str] = None) -> pd.DataFrame:
    if df_macro is None or df_macro.empty:
        logger.warning("[macro_features] ❌ Порожній DataFrame")
        return df_macro

    df = df_macro.copy()
    if "date" not in df.columns:
        raise ValueError("[macro_features] df_macro має містити колонку 'date'")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).sort_values("date")

    # Перейменування колонок до alias
    renamed = {}
    for col in df.columns:
        if col == "date":
            continue
        alias = _alias_for_column(col)
        if alias != col:
            renamed[col] = alias
    if renamed:
        df = df.rename(columns=renamed)

    # enrichment: delta / trend / rolling zscore
    base_cols = [c for c in df.columns if c != "date"]
    new_features = {}
    for col in base_cols:
        series = df[col]

        delta = series.diff()
        trend = series.rolling(MACRO_WINDOWS["trend"], min_periods=1).mean()
        roll_mean = series.rolling(MACRO_WINDOWS["zscore"], min_periods=20).mean()
        roll_std = series.rolling(MACRO_WINDOWS["zscore"], min_periods=20).std()
        zscore = (series - roll_mean) / roll_std.replace(0, np.nan)

        new_features[f"{col}_delta"] = delta
        new_features[f"{col}_trend"] = trend
        new_features[f"{col}_zscore"] = zscore

    df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]

    # yield curve inversion
    if "T10Y2Y" in df.columns:
        df["yield_curve_inversion"] = df["T10Y2Y"] < 0

    # decay signals
    if end_date is not None:
        full_range = pd.date_range(start=df["date"].min(),
                                   end=pd.to_datetime(end_date).normalize(),
                                   freq="D")
        df_daily = pd.DataFrame({"date": full_range})
        df = df_daily.merge(df, on="date", how="left")

        enriched_suffixes = ("_delta", "_trend", "_zscore")
        decay_targets = [
            c for c in df.columns
            if c != "date" and not c.endswith(enriched_suffixes) and c not in ("yield_curve_inversion",)
        ]

        decay_features = {}
        for col in decay_targets:
            lam = _lambda_for_column(col)
            alias = _alias_for_column(col)
            norm = _norm_scale_for_alias(alias)

            series = df[col].astype(float)

            if series.notna().sum() == 0:
                logger.warning(f"[macro_features] ⚠️ {alias} не має значень")
                decay_features[f"{col}_DAYS_SINCE_RELEASE"] = 0
                decay_features[f"{col}_SIGNAL"] = 0.0
                decay_features[f"{col}_WEIGHTED"] = 0.0
                continue

            release_dates = df["date"].where(series.notna()).ffill()
            carried = series.ffill().bfill().fillna(0.0)

            days_since = (df["date"] - release_dates).dt.days.clip(lower=0)
            signal = carried / norm
            weighted = signal * np.exp(-lam * days_since)

            decay_features[f"{col}_DAYS_SINCE_RELEASE"] = days_since
            decay_features[f"{col}_SIGNAL"] = signal
            decay_features[f"{col}_WEIGHTED"] = weighted

        df = pd.concat([df, pd.DataFrame(decay_features, index=df.index)], axis=1)
        df = df.loc[:, ~df.columns.duplicated()]

    logger.info(f"[macro_features] ✅ Збагачено макро-ознаки: {len(df.columns)} колонок")
    return df
