"""Point-in-time market context feature enrichment."""

from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd

from src.core.logging.logger import ProjectLogger

from .base import BaseEnricher

logger = ProjectLogger.get_logger("MarketContextEnricher")


class MarketContextEnricher(BaseEnricher):
    """Add causal market, macro, technical, and temporal context features."""

    @property
    def name(self) -> str:
        return "market_context"

    @property
    def priority(self) -> int:
        return 85

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__()
        self.config = config or {}
        self.context_features = self.config.get(
            "context_features",
            [
                "volatility_5d",
                "volatility_20d",
                "volatility_ratio",
                "trend_5d",
                "trend_20d",
                "trend_alignment",
                "rsi_current",
                "volume_ratio",
                "price_to_ma20",
                "hour_of_day",
                "day_of_week",
                "yield_curve_slope",
                "yield_curve_inverted",
                "fed_funds_trend",
                "fed_funds_velocity",
                "market_breadth",
                "dollar_strength",
                "put_call_ratio",
            ],
        )
        # `self.analyzer = MarketContextAnalyzer(...)` stood here, described as
        # "kept available to callers that need a latest snapshot". No caller
        # ever asked: the attribute was set and never read again, and this
        # enricher computes its own columns in _build_single_series_context --
        # a causal, vectorised implementation that supersedes the analyzer's
        # point-in-time one. Construction cost plus a false impression that
        # the analyzer participates in feature building.
        #
        # The class stays where it is: it still has tests, and tests/
        # smoke_test_system.py exercises it directly. It simply is not part of
        # this path.
        logger.info(
            "MarketContextEnricher initialized with %s features",
            len(self.context_features),
        )

    def _enrich_impl(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if df.empty:
            logger.warning("Empty DataFrame provided, skipping enrichment")
            return df

        result = df.copy()
        context = self._build_causal_context_frame(df)
        for feature in self.context_features:
            result[f"market_context_{feature}"] = context[feature].to_numpy(copy=True)
        if not context.empty and logger.isEnabledFor(10):
            logger.debug("Latest causal market context: %s", context.iloc[-1].to_dict())
        return result

    def _build_causal_context_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build each row from information available at or before that row."""
        result = pd.DataFrame(
            np.nan,
            index=range(len(df)),
            columns=self.context_features,
            dtype=float,
        )
        if "ticker" in df.columns:
            position_groups = df.groupby("ticker", sort=False).indices.values()
        else:
            position_groups = [np.arange(len(df))]

        for positions in position_groups:
            position_array = np.asarray(positions, dtype=int)
            group = self._build_single_series_context(df.iloc[position_array])
            result.iloc[position_array, :] = group[self.context_features].to_numpy()
        return result

    def _build_single_series_context(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(index=range(len(df)))
        close = self._numeric_series(df, ("close",))
        volume = self._numeric_series(df, ("volume",))
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)

        volatility_5 = returns.rolling(5, min_periods=2).std()
        volatility_20 = returns.rolling(20, min_periods=2).std()
        frame["volatility_5d"] = volatility_5
        frame["volatility_20d"] = volatility_20
        frame["volatility_ratio"] = volatility_5 / volatility_20.where(
            volatility_20.abs() > 1e-12
        )

        trend_5 = self._rolling_slope(close, 5)
        trend_20 = self._rolling_slope(close, 20)
        frame["trend_5d"] = trend_5
        frame["trend_20d"] = trend_20
        frame["trend_alignment"] = np.sign(trend_5 * trend_20)

        rsi_column = next(
            (
                column
                for column in df.columns
                if "rsi" in str(column).lower()
                and "velocity" not in str(column).lower()
                and not str(column).lower().startswith("state_")
            ),
            None,
        )
        frame["rsi_current"] = (
            pd.to_numeric(df[rsi_column], errors="coerce").reset_index(drop=True)
            if rsi_column
            else 50.0
        )
        volume_5 = volume.rolling(5, min_periods=1).mean()
        volume_20 = volume.rolling(20, min_periods=1).mean()
        frame["volume_ratio"] = volume_5 / volume_20.where(volume_20.abs() > 1e-12)
        ma20 = close.rolling(20, min_periods=1).mean()
        frame["price_to_ma20"] = close / ma20.where(ma20.abs() > 1e-12) - 1.0

        timestamps = self._timestamps(df)
        frame["hour_of_day"] = timestamps.dt.hour
        frame["day_of_week"] = timestamps.dt.dayofweek

        dgs10 = self._numeric_series(df, ("DGS10", "FRED_DGS10"))
        dgs2 = self._numeric_series(df, ("DGS2", "FRED_DGS2", "FRED_GS2"))
        frame["yield_curve_slope"] = dgs10 - dgs2
        frame["yield_curve_inverted"] = (frame["yield_curve_slope"] < 0).astype(float)

        fed_funds = self._numeric_series(df, ("FEDFUNDS", "FRED_FEDFUNDS"))
        frame["fed_funds_trend"] = fed_funds - fed_funds.shift(60)
        frame["fed_funds_velocity"] = fed_funds - fed_funds.shift(20)

        advances = self._numeric_series(df, ("advances",))
        declines = self._numeric_series(df, ("declines",))
        if advances.notna().any() and declines.notna().any():
            frame["market_breadth"] = advances / declines.where(declines.abs() > 1e-12)
        else:
            sma50 = close.rolling(50, min_periods=2).mean()
            frame["market_breadth"] = pd.Series(
                np.where(close > sma50, 1.0, 0.5),
                index=frame.index,
            ).where(sma50.notna())

        frame["dollar_strength"] = self._numeric_series(
            df,
            ("DXY", "FRED_DTWEXBGS"),
        )
        frame["put_call_ratio"] = self._numeric_series(df, ("put_call_ratio",))

        defaults = {
            "rsi_current": 50.0,
            "volume_ratio": 1.0,
            "volatility_ratio": 1.0,
            "trend_alignment": 0.0,
            "price_to_ma20": 0.0,
            "yield_curve_slope": 0.0,
            "yield_curve_inverted": 0.0,
            "fed_funds_trend": 0.0,
            "fed_funds_velocity": 0.0,
            "market_breadth": 1.0,
            "dollar_strength": 100.0,
            "put_call_ratio": 1.0,
            "hour_of_day": 0.0,
            "day_of_week": 0.0,
        }
        fully_defaulted = []
        for feature in self.context_features:
            if feature not in frame.columns:
                frame[feature] = defaults.get(feature, 0.0)
            computed = (
                pd.to_numeric(frame[feature], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
            )
            if len(computed) and not computed.notna().any():
                fully_defaulted.append(feature)
            frame[feature] = computed.fillna(defaults.get(feature, 0.0))

        if fully_defaulted:
            # A feature filled entirely by its neutral default is a constant
            # column: inert for a model, and indistinguishable in the batch
            # from one that was measured and happened to be neutral. Measured
            # on 2026-08-13, market_context_put_call_ratio is the constant 1.0
            # and market_context_dollar_strength the constant 100.0 on all
            # 25,172 15m rows -- put_call_ratio because CBOE serves 403 and
            # the collector is off, dollar_strength because no DXY or
            # FRED_DTWEXBGS column reaches here.
            #
            # The default stays: NaN would only move the same problem into the
            # imputer. What was missing is that nobody said so. The cost of
            # silence is not today, when the column is constant, but the first
            # run where the feed starts arriving -- the column then acquires a
            # discontinuity at the boundary with nothing marking it.
            logger.warning(
                "Market context features filled entirely by their default "
                "(no source column reached this enricher): %s",
                ", ".join(sorted(fully_defaulted)),
            )
        return frame[self.context_features]

    @staticmethod
    def _numeric_series(df: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series:
        column = next((candidate for candidate in candidates if candidate in df.columns), None)
        if column is None:
            return pd.Series(np.nan, index=range(len(df)), dtype=float)
        return pd.to_numeric(df[column], errors="coerce").reset_index(drop=True)

    @staticmethod
    def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
        """Trailing least-squares slope, in closed form rather than per window.

        This was `rolling(window).apply(np.polyfit(...))`, which calls into
        Python once per window. Measured on 2026-08-28 over 156,372 rows:
        **21.18 seconds against 0.11**, and it is called twice (windows 5 and
        20), so about forty seconds of every frame.

        The slope of a least-squares line against evenly spaced x is
        `sum((x - x̄)(y - ȳ)) / sum((x - x̄)²)`, and for a fixed window the
        denominator is a constant. Verified against the previous
        implementation: identical on every full window, largest difference
        6.3e-15, which is float noise.

        Partial windows at the start are computed the same way rather than
        dropped -- `min_periods=2` produced 18 values in the first 19 rows and
        losing them would be a silent change to the feature's beginning.
        """
        values = series.to_numpy(dtype=float)
        count = len(values)
        result = np.full(count, np.nan, dtype=float)
        if count == 0:
            return pd.Series(result, index=series.index)

        # Full windows, all at once.
        if count >= window:
            offsets = np.arange(window, dtype=float)
            centred = offsets - offsets.mean()
            denominator = float((centred ** 2).sum())
            if denominator > 0:
                windows = sliding_window_view(values, window)
                numerator = (
                    windows - windows.mean(axis=1, keepdims=True)
                ) @ centred
                result[window - 1:] = numerator / denominator

        # The first `window - 1` rows, where pandas had min_periods=2.
        for end in range(1, min(window - 1, count)):
            piece = values[: end + 1]
            piece = piece[~np.isnan(piece)]
            if piece.size < 2:
                continue
            offsets = np.arange(piece.size, dtype=float)
            centred = offsets - offsets.mean()
            denominator = float((centred ** 2).sum())
            if denominator > 0:
                result[end] = float(
                    ((piece - piece.mean()) @ centred) / denominator
                )

        return pd.Series(result, index=series.index)

    @staticmethod
    def _timestamps(df: pd.DataFrame) -> pd.Series:
        """The bar times, positionally — or all-NaT when there are none to be had.

        Two defects lived in the single expression this replaces, and the
        second one hid the first.

        `pd.Series(existing_series, index=range(len(df)))` does NOT relabel a
        series positionally. It REINDEXES: pandas looks up each requested label
        in the series it was handed. The frame reaching this enricher is the
        twenty-first in the chain and its index is whatever survived that
        journey, so the labels 0..n-1 often matched nothing. Measured:

            index 0..n-1        every hour correct
            index 500..500+n    every hour NaT -> the default 0.0
            index 0,2,4,...     SOME hours correct and the rest defaulted,
                                interleaved, with nothing marking which

        That third row is the dangerous one. A wholly constant column is at
        least inert; a column that is half real and half filler is a feature
        the model will happily learn from.

        The fallback to `df.index` then made a wrong answer out of a missing
        one. `to_datetime` on a RangeIndex reads the integers as nanoseconds
        since the epoch, so a frame with no timestamp at all came out as
        1970-01-01 -- hour 0 and weekday 3 on every row, non-null, and
        therefore silently past the check that reports fully-defaulted
        features. Only a real DatetimeIndex is accepted now, and a column only
        counts when it actually yields a time: `''` is not NaN and satisfies
        every existence test in this codebase, which is a trap this pipeline
        has fallen into six times.

        Returning NaT when there is nothing is deliberate. The caller fills
        defaults and reports what it filled, which is the honest outcome; the
        old code denied it that chance.
        """
        for candidate in ("datetime", "timestamp", "date"):
            if candidate not in df.columns:
                continue
            stamps = pd.to_datetime(df[candidate], errors="coerce", utc=True)
            if stamps.notna().any():
                return stamps.reset_index(drop=True)
        if isinstance(df.index, pd.DatetimeIndex):
            return pd.Series(
                pd.to_datetime(df.index, errors="coerce", utc=True)
            ).reset_index(drop=True)
        return pd.Series(
            pd.to_datetime(pd.Series([None] * len(df)), errors="coerce", utc=True)
        )
