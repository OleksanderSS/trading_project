"""Per-ticker signals that come from outside the price and news feeds.

Two sources, both collected for months and neither ever reaching a model,
because both are keyed by (ticker, date) and nothing merged them:

  wikipedia_attention   11,417 rows, 309 articles, daily since 2026-06-30.
                        The `article` column IS the ticker symbol — AAPL,
                        ABBV, ABT — so no name mapping is needed, which is
                        the part everyone assumes will be the hard bit.
  insider_trades         1,395 rows, 778 tickers, since 2026-07-29.

Each carries a timing trap of its own, and both are the same trap the CFTC
report has: the day a fact HAPPENED is not the day it became knowable.

  Wikipedia's pageview API publishes a day's counts the following day, so a
  bar inside day D cannot read D's views.

  An insider trade is private until the Form 4 is filed. The stored rows show
  a median gap of 2 days between `trade_date` and `filing_date`, and joining
  on trade_date would hand every bar two days of the future. `filing_date` is
  the only honest key, and it is the one used here.

Scale, stated plainly rather than discovered later: insider covers 9 records
across our 22 tickers, so its columns will be near-empty until the collector
accumulates. Wikipedia covers 44 days, which is most of the 15-minute window
and 6% of the daily one. Both are wired correctly so they become useful as
history builds, not because they are useful today.
"""
from typing import Any

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.features.enrichers.base import BaseEnricher

logger = ProjectLogger.get_logger("TickerExternalEnricher")

_PURCHASE = "P"
_SALE = "S"


class TickerExternalEnricher(BaseEnricher):
    """Wikipedia attention and insider filings, joined per ticker."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__()
        self.config = config or {}
        self.wiki_publication_lag_days = int(
            self.config.get("wiki_publication_lag_days", 1)
        )
        self.attention_window = int(self.config.get("attention_window", 20))
        self.insider_window_days = int(self.config.get("insider_window_days", 30))

    @property
    def name(self) -> str:
        return "ticker_external"

    @property
    def priority(self) -> int:
        return 33  # after the market-wide pair, before the news enrichers

    def _enrich_impl(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if df.empty:
            return df
        if "ticker" not in df.columns:
            logger.error(
                "No 'ticker' column; per-ticker sources cannot be joined."
            )
            return df

        frame = df.copy()
        restore_index = False
        if "datetime" not in frame.columns:
            if isinstance(frame.index, pd.DatetimeIndex):
                frame = frame.reset_index()
                if "index" in frame.columns and "datetime" not in frame.columns:
                    frame = frame.rename(columns={"index": "datetime"})
                restore_index = True
            else:
                logger.error(
                    "No 'datetime' column or DatetimeIndex; bars unchanged."
                )
                return df

        bar_time = pd.to_datetime(frame["datetime"], errors="coerce")
        if getattr(bar_time.dt, "tz", None) is not None:
            bar_time = bar_time.dt.tz_localize(None)
        bar_ticker = frame["ticker"].astype(str).str.strip().str.upper()

        attached = 0
        attached += self._attach_attention(
            frame, bar_time, bar_ticker,
            self._source(kwargs, "wikipedia_attention", "wikipedia"),
        )
        attached += self._attach_insider(
            frame, bar_time, bar_ticker,
            self._source(kwargs, "insider_trades", "insider"),
        )
        if not attached:
            logger.warning(
                "Neither wikipedia_attention nor insider_trades reached this "
                "enricher; no per-ticker external columns were added."
            )

        if restore_index:
            frame = frame.set_index("datetime")
        return frame

    # ------------------------------------------------------------------ #

    def _attach_attention(self, frame: pd.DataFrame, bar_time: pd.Series,
                          bar_ticker: pd.Series, source: Any) -> int:
        if not isinstance(source, pd.DataFrame) or source.empty:
            return 0
        if not {"date", "article", "views"}.issubset(source.columns):
            logger.error(
                "wikipedia_attention lacks date/article/views (has %s).",
                list(source.columns)[:8],
            )
            return 0

        table = source.copy()
        table["_key"] = table["article"].astype(str).str.strip().str.upper()
        table["_at"] = self._shift(table["date"], self.wiki_publication_lag_days)
        table["_views"] = pd.to_numeric(table["views"], errors="coerce")
        table = (table.dropna(subset=["_at", "_views"])
                      .sort_values("_at")
                      .drop_duplicates(subset=["_key", "_at"], keep="last"))
        if table.empty:
            logger.error("wikipedia_attention has no usable date/views pairs.")
            return 0

        # Attention only means something against a ticker's own normal level:
        # AAPL's floor is another company's spike. Expanding, so the baseline
        # a bar uses could have been computed at that bar.
        grouped = table.groupby("_key")["_views"]
        mean = grouped.transform(lambda s: s.expanding(3).mean())
        std = grouped.transform(lambda s: s.expanding(3).std())
        table["_z"] = (table["_views"] - mean) / (std + 1e-6)

        merged = self._asof_per_ticker(
            bar_time, bar_ticker, table, ["_views", "_z"]
        )
        frame["wiki_views"] = merged["_views"].to_numpy()
        frame["wiki_attention_z"] = merged["_z"].to_numpy()
        frame["wiki_attention_available"] = (
            merged["_views"].notna().astype(int).to_numpy()
        )
        covered = int(frame["wiki_attention_available"].sum())
        logger.info(
            "wikipedia attention on %d of %d bars (%.0f%%), %d tickers matched.",
            covered, len(frame), 100 * covered / max(1, len(frame)),
            int(table["_key"].isin(bar_ticker.unique()).sum() > 0
                and table.loc[table["_key"].isin(bar_ticker.unique()),
                              "_key"].nunique()),
        )
        return 1

    def _attach_insider(self, frame: pd.DataFrame, bar_time: pd.Series,
                        bar_ticker: pd.Series, source: Any) -> int:
        if not isinstance(source, pd.DataFrame) or source.empty:
            return 0
        if not {"filing_date", "ticker", "trade_type"}.issubset(source.columns):
            logger.error(
                "insider_trades lacks filing_date/ticker/trade_type (has %s).",
                list(source.columns)[:8],
            )
            return 0

        table = source.copy()
        table["_key"] = table["ticker"].astype(str).str.strip().str.upper()
        # filing_date, never trade_date: the trade is private until it is
        # filed, a median of two days later.
        table["_at"] = self._shift(table["filing_date"], 0)
        side = table["trade_type"].astype(str).str.strip().str.upper().str[:1]
        value = pd.to_numeric(table.get("value"), errors="coerce")
        if value.isna().all():
            price = pd.to_numeric(table.get("price"), errors="coerce")
            quantity = pd.to_numeric(table.get("quantity"), errors="coerce")
            value = price * quantity
        signed = value.where(side == _PURCHASE, -value.where(side == _SALE, 0.0))
        table["_signed"] = pd.to_numeric(signed, errors="coerce").fillna(0.0)
        table = table.dropna(subset=["_at"]).sort_values("_at")
        if table.empty:
            logger.error("insider_trades has no usable filing dates.")
            return 0

        # Net dollar value filed in a trailing window, per ticker. Rolling on
        # time rather than on row count, because filings arrive in clusters.
        window = f"{self.insider_window_days}D"
        rolled = (table.set_index("_at")
                       .groupby("_key")["_signed"]
                       .rolling(window).sum()
                       .reset_index())
        rolled.columns = ["_key", "_at", "_net"]
        rolled = rolled.sort_values("_at").drop_duplicates(
            subset=["_key", "_at"], keep="last"
        )

        merged = self._asof_per_ticker(bar_time, bar_ticker, rolled, ["_net"])
        frame["insider_net_value_30d"] = merged["_net"].to_numpy()
        frame["insider_available"] = merged["_net"].notna().astype(int).to_numpy()
        covered = int(frame["insider_available"].sum())
        logger.info(
            "insider filings on %d of %d bars (%.0f%%); %d of our tickers "
            "appear in the source.",
            covered, len(frame), 100 * covered / max(1, len(frame)),
            int(rolled["_key"].isin(bar_ticker.unique()).sum() > 0
                and rolled.loc[rolled["_key"].isin(bar_ticker.unique()),
                               "_key"].nunique()),
        )
        return 1

    # ------------------------------------------------------------------ #

    @staticmethod
    def _source(kwargs: dict, *names: str) -> Any:
        for stem in names:
            for key in (stem, f"{stem}_data"):
                value = kwargs.get(key)
                if isinstance(value, pd.DataFrame) and not value.empty:
                    return value
        return None

    @staticmethod
    def _shift(dates: pd.Series, lag_days: int) -> pd.Series:
        parsed = pd.to_datetime(dates, errors="coerce")
        if getattr(parsed.dt, "tz", None) is not None:
            parsed = parsed.dt.tz_localize(None)
        return parsed + pd.Timedelta(days=lag_days)

    @staticmethod
    def _asof_per_ticker(bar_time: pd.Series, bar_ticker: pd.Series,
                         table: pd.DataFrame,
                         value_cols: list[str]) -> pd.DataFrame:
        """Backward as-of join within each ticker, in the caller's row order.

        merge_asof cannot take a `by` key and preserve arbitrary row order at
        once, and it returns a fresh RangeIndex either way, so the result is
        written back onto the caller's original positions. A positional copy
        here is what once put 54,000 bars on other bars' dates.
        """
        out = pd.DataFrame(index=range(len(bar_time)), columns=value_cols,
                           dtype=float)
        left = pd.DataFrame({
            "_bar": bar_time.to_numpy().astype("datetime64[ns]"),
            "_key": bar_ticker.to_numpy(),
            "_pos": range(len(bar_time)),
        }).dropna(subset=["_bar"])
        right = table.copy()
        right["_at"] = right["_at"].to_numpy().astype("datetime64[ns]")

        for key, group in left.groupby("_key", sort=False):
            per = right[right["_key"] == key]
            if per.empty:
                continue
            merged = pd.merge_asof(
                group.sort_values("_bar"),
                per[["_at"] + value_cols].sort_values("_at"),
                left_on="_bar", right_on="_at", direction="backward",
            )
            out.loc[merged["_pos"].to_numpy(), value_cols] = (
                merged[value_cols].to_numpy()
            )
        return out

    def get_feature_names(self) -> list[str]:
        return ["wiki_views", "wiki_attention_z", "wiki_attention_available",
                "insider_net_value_30d", "insider_available"]
