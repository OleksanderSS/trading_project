"""Precompute regime from features.parquet for --preload-regime flag."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from dean_os.regime_context import RegimeContextBuilder
from dean_os.schemas import MarketContext

FEATURES_PATH = Path("data/colab/accumulated/main_database/features.parquet")


def _detect_timeframe(df: pd.DataFrame) -> str:
    """Detect the predominant interval from data, fall back to '15m'."""
    interval_col = _resolve_col(df, "interval")
    if interval_col:
        intervals = df[interval_col].dropna()
        if not intervals.empty:
            mode_val = intervals.mode().iloc[0]
            return str(mode_val).strip().lower()
    datetime_col = _resolve_col(df, "datetime")
    if datetime_col and len(df) > 1:
        import numpy as np
        dts = pd.to_datetime(df[datetime_col]).dropna().sort_values()
        if len(dts) > 5:
            gaps = dts.diff().dt.total_seconds().dropna()
            median_gap = float(np.median(gaps.values))
            if median_gap < 120:
                return "1m"
            elif median_gap < 600:
                return "5m"
            elif median_gap < 1800:
                return "15m"
            elif median_gap < 3600:
                return "30m"
            elif median_gap < 7200:
                return "1h"
            elif median_gap < 86400:
                return "4h"
            else:
                return "1d"
    return "15m"


def preload_regime(
    ctx: MarketContext,
    features_path: str | Path | None = None,
    timeframe: str | None = None,
) -> None:
    path = Path(features_path) if features_path else FEATURES_PATH
    if not path.exists():
        return
    df = pd.read_parquet(path)
    ticker_col = _resolve_col(df, "ticker")
    if ticker_col is None:
        return

    effective_tf = timeframe or str(ctx.timeframe or "").lower() or _detect_timeframe(df)
    builder = RegimeContextBuilder()
    contexts: list[dict[str, Any]] = []

    for ticker in df[ticker_col].unique():
        tdf = df[df[ticker_col] == ticker].copy()
        regime_snap = builder.from_price_frame(
            tdf,
            close_col=_resolve_col(tdf, "close"),
            volume_col=_resolve_col(tdf, "volume"),
        )
        contexts.append({
            "ticker": str(ticker).upper(),
            "timeframe": effective_tf,
            "regime": regime_snap.regime,
            "confidence": regime_snap.confidence,
            "as_of": ctx.as_of,
            "context_tags": regime_snap.context_tags,
            "metrics": regime_snap.metrics,
            "context_key": f"{ticker}_{effective_tf}",
        })

    ctx.metadata["stage7_regime_review"] = {
        "schema_version": "dean_stage7_regime_review_v1",
        "status": "stage7_regime_contexts_recorded",
        "contexts": contexts,
        "as_of": ctx.as_of,
        "evidence_class": "preloaded_regime",
        "supporting_review_only": True,
    }


def _resolve_col(df: pd.DataFrame, name: str) -> str | None:
    if name in df.columns:
        return name
    lowered = {str(c).lower(): c for c in df.columns}
    return lowered.get(name.lower())
