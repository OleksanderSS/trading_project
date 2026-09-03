from typing import Any

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger

from .base import BaseEnricher

logger = ProjectLogger.get_logger("VolatilityEnricher")


class VolatilityEnricher(BaseEnricher):
    """
    Volatility analysis enricher.
    Adds volatility-based indicators.
    """

    @property
    def name(self) -> str:
        return "volatility"

    @property
    def priority(self) -> int:
        return 30

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__()
        self.config = config or {}

    #: Trading days in a year. Correct for a DAILY bar and for nothing else.
    TRADING_DAYS = 252

    @classmethod
    def _bars_per_year(cls, df: pd.DataFrame) -> float:
        """Annualisation base measured from the frame's own cadence.

        `* np.sqrt(252)` was applied to every timeframe. 252 is the number of
        trading DAYS in a year, so it is right for a daily bar and wrong by
        the square root of the intraday bar count for anything finer.

        Measured on the export of 2026-08-31: 25 bars per trading day on the
        15m frame and 7 on the 60m one, so `volatility_5/10/20` came out
        5.00x and 2.65x too small there. The damage is visible in the one
        place that compares those numbers to an absolute threshold --
        `volatility_regime` binned 95.3% of 15m rows and 82.3% of 60m rows as
        "low", against a well-spread 33/25/24/18 on the daily frame. Rescaled
        by the measured cadence the three frames agree: 37/29/21/14 and
        33/29/22/16 against the daily 33/25/24/18. Three timeframes agreeing
        once one factor is corrected is what says the bins were never the
        problem.

        Measured from the data rather than taken from a parameter: the frame
        knows its own spacing, and a caller that forgets to pass a timeframe
        would silently reintroduce exactly this defect. A daily frame yields
        one bar per day and comes out at 252 unchanged.

        The cadence table itself is NOT written here. `infer_periods_per_year`
        already holds it, for the same reason and in almost the same words --
        "a fixed 252 assumes daily bars regardless of what's actually being
        measured". The first version of this method counted bars per calendar
        day itself and got 25 and 7 where the canonical table says 26 and 7;
        two answers to one question is the shape this project has already paid
        for twice (two Sharpe ratios, three copies of `max_features`).
        """
        from src.metrics.financial.financial_metrics_library import (
            infer_periods_per_year,
        )

        stamps = None
        for column in ("datetime", "timestamp", "date"):
            if column in df.columns:
                stamps = pd.to_datetime(df[column], errors="coerce", utc=True)
                break
        if stamps is None and isinstance(df.index, pd.DatetimeIndex):
            stamps = pd.Series(df.index)
        if stamps is None or stamps.notna().sum() < 2:
            logger.warning(
                "No usable timestamps, so volatility is annualised as if the "
                "bars were daily. If they are not, the scale is wrong."
            )
            return float(cls.TRADING_DAYS)

        # One series only. A pooled frame interleaves tickers at the same
        # timestamp, so consecutive gaps there are zero and the cadence would
        # read as "sub-minute" for every timeframe alike.
        if "ticker" in df.columns:
            first = df["ticker"].iloc[0]
            stamps = stamps[df["ticker"].to_numpy() == first]

        stamps = stamps.dropna().sort_values()
        if len(stamps) < 2:
            logger.warning(
                "Fewer than two timestamps for one ticker; volatility is "
                "annualised as if the bars were daily."
            )
            return float(cls.TRADING_DAYS)

        probe = pd.Series(0.0, index=pd.DatetimeIndex(stamps))
        return float(infer_periods_per_year(probe))

    def _enrich_impl(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Enrich DataFrame with volatility indicators."""
        df_enriched = df.copy()

        if "close" in df_enriched.columns:
            bars_per_year = self._bars_per_year(df_enriched)
            logger.info(
                "Annualising volatility on sqrt(%.0f) bars per year "
                "(%.1f bars per trading day).",
                bars_per_year, bars_per_year / self.TRADING_DAYS,
            )
            if "ticker" in df_enriched.columns:
                groups = [
                    self._compute_volatility_indicators(group.copy(), bars_per_year)
                    for _ticker, group in df_enriched.groupby("ticker")
                ]
                df_enriched = pd.concat(groups).sort_index()
            else:
                df_enriched = self._compute_volatility_indicators(
                    df_enriched, bars_per_year)

            # Volatility Regime: stateless row-wise binning of an already
            # correctly (per-ticker) computed column - safe to apply globally.
            volatility_regime = pd.cut(
                df_enriched["volatility_10"],
                bins=[0, 0.15, 0.25, 0.35, float("inf")],
                labels=["low", "normal", "high", "extreme"],
            )
            df_enriched["volatility_regime"] = (
                volatility_regime.cat.add_categories(["unknown"]).fillna("unknown")
            )

            logger.info(f"✅ Added {8} volatility indicators")

        return df_enriched

    def _compute_volatility_indicators(
        self, df_enriched: pd.DataFrame, bars_per_year: float | None = None,
    ) -> pd.DataFrame:
        """Compute volatility indicators for a single ticker's rows (chronological)."""
        annualise = np.sqrt(
            self.TRADING_DAYS if bars_per_year is None else bars_per_year
        )
        # Returns
        df_enriched["returns"] = (
            df_enriched["close"]
            .pct_change(fill_method=None)
            .replace([np.inf, -np.inf], np.nan)
        )

        # Historical Volatility
        df_enriched["volatility_5"] = df_enriched["returns"].rolling(window=5, min_periods=1).std().shift(1) * annualise
        df_enriched["volatility_10"] = df_enriched["returns"].rolling(window=10, min_periods=1).std().shift(1) * annualise
        df_enriched["volatility_20"] = df_enriched["returns"].rolling(window=20, min_periods=1).std().shift(1) * annualise

        # Average True Range (ATR)
        if all(col in df_enriched.columns for col in ["high", "low", "close"]):
            tr1 = df_enriched["high"] - df_enriched["low"]
            tr2 = abs(df_enriched["high"] - df_enriched["close"].shift())
            tr3 = abs(df_enriched["low"] - df_enriched["close"].shift())
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df_enriched["atr_14"] = true_range.rolling(window=14, min_periods=1).mean().shift(1)

        # Garman-Klass Volatility
        if "high" in df_enriched.columns and "low" in df_enriched.columns:
            gk = (
                0.5 * (np.log(df_enriched["high"]) - np.log(df_enriched["low"])) ** 2
                - (2 * np.log(2) - 1)
                * (np.log(df_enriched["close"]) - np.log(df_enriched["close"].shift())) ** 2
            )
            df_enriched["gk_volatility"] = gk.rolling(window=20, min_periods=1).sum().shift(1)

        return df_enriched
