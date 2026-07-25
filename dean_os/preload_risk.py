"""Preload risk data: features dataframe + returns from close prices."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

FEATURES_PATH = Path("data/colab/accumulated/main_database/features.parquet")


def preload_risk_data(ctx: Any, features_path: str | Path | None = None) -> None:
    path = Path(features_path) if features_path else FEATURES_PATH
    if not path.exists():
        return
    df = pd.read_parquet(path)

    tickers = getattr(ctx, "tickers", [])
    tf = getattr(ctx, "timeframe", "1d")

    ticker_col = _resolve_col(df, "ticker")
    if ticker_col and tickers:
        df = df[df[ticker_col].astype(str).str.upper().isin({t.upper() for t in tickers})]

    # The real time column in features.parquet is "datetime", not "timestamp"
    # -- resolve it here so it survives the column filter below and can be
    # used for a real chronological sort, instead of silently being dropped
    # and falling back to an arbitrary column order.
    datetime_col = _resolve_col(df, "datetime")
    base_cols = {"close", "high", "low", "open", "volume", "ticker", "interval", "hash"}
    if datetime_col:
        base_cols = base_cols | {datetime_col}
    tf_cols = [c for c in df.columns if c in base_cols or str(c).endswith(f"_{tf}")]
    df = df[tf_cols]

    ctx.dataframes = dict(ctx.dataframes or {})
    ctx.dataframes["features"] = df

    ctx.positions = dict(ctx.positions or {})
    for ticker in tickers:
        ctx.positions[ticker] = 0.0

    close_col = _resolve_col(df, "close")
    if ticker_col and close_col:
        series_list = []
        for ticker in sorted(df[ticker_col].unique()):
            sort_col = datetime_col if datetime_col and datetime_col in df.columns else df.columns[0]
            tdf = df[df[ticker_col] == ticker].sort_values(sort_col)
            close = tdf[close_col].astype(float)
            ret = close.pct_change().dropna()
            series_list.append(pd.Series(ret.values, name=ticker))
        if series_list:
            ctx.returns = pd.concat(series_list, axis=1)


def _resolve_col(df: pd.DataFrame, name: str) -> str | None:
    if name in df.columns:
        return name
    lowered = {str(c).lower(): c for c in df.columns}
    return lowered.get(name.lower())
