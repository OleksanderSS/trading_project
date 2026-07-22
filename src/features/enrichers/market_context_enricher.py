"""Point-in-time market context feature enrichment."""

from typing import Any

import numpy as np
import pandas as pd

from src.analytics.context.market_context_analyzer import MarketContextAnalyzer
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
        # Keep the point-in-time analyzer available to callers that need a latest snapshot.
        self.analyzer = MarketContextAnalyzer(context_features=self.context_features)
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
        for feature in self.context_features:
            if feature not in frame.columns:
                frame[feature] = defaults.get(feature, 0.0)
            frame[feature] = (
                pd.to_numeric(frame[feature], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(defaults.get(feature, 0.0))
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
        def slope(values: np.ndarray) -> float:
            if len(values) < 2 or not np.isfinite(values).all():
                return np.nan
            return float(np.polyfit(np.arange(len(values)), values, 1)[0])

        return series.rolling(window, min_periods=2).apply(slope, raw=True)

    @staticmethod
    def _timestamps(df: pd.DataFrame) -> pd.Series:
        column = next(
            (
                candidate
                for candidate in ("datetime", "timestamp", "date")
                if candidate in df.columns
            ),
            None,
        )
        values = df[column] if column else df.index
        return pd.Series(
            pd.to_datetime(values, errors="coerce", utc=True),
            index=range(len(df)),
        )
