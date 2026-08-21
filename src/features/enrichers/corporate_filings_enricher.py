"""Corporate filings as events, not as sentiment.

`sec_filings` was classified as a news source and fed into the news frame,
where it died: the table names its date column `filingDate` while the news
path's alias list said `filing_date`, so 24,365 dated, ticker-tagged filings
were discarded every run over the capital D. They were counted into a single
warning about 762,436 "lost news records", which hid which source they came
from.

Renaming the column would have been the wrong fix. What a filing carries is
`form`, `primaryDocDescription`, `accessionNumber` -- codes like "10-Q" and
"8-K", not prose. Handing those to the sentiment model would manufacture a
reading rather than recover one: FinBERT will return a number for the string
"10-Q" and it will be noise wearing the label of sentiment.

What a filing actually is: a company told the regulator something on a date.
That is an event, and events are measured by when they happen, how often, and
what kind -- which is what this builds.

Point in time: `filingDate` only. The table also carries `reportDate`, the
period the filing covers, which is always earlier and is never what the market
knew. Using it would date a June disclosure to March.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseEnricher

logger = logging.getLogger(__name__)

#: Forms that announce something unscheduled -- an acquisition, a resignation,
#: a material agreement. These are the ones with news value.
MATERIAL_FORMS = ("8-K",)

#: Scheduled periodic reports. Their timing is known in advance, so their
#: information is in the content rather than the arrival.
PERIODIC_FORMS = ("10-K", "10-Q", "20-F", "40-F")


class CorporateFilingsEnricher(BaseEnricher):
    """Counts and recency of regulatory filings, per ticker, as of each bar."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__()
        self.config = config or {}
        self.window_days = int(self.config.get("window_days", 30))

    @property
    def name(self) -> str:
        return "corporate_filings"

    @property
    def priority(self) -> int:
        return 30  # after economic_calendar (29): both are date-keyed events

    def get_feature_names(self) -> list[str]:
        return [
            "filing_days_since_last",
            f"filing_count_{self.window_days}d",
            f"filing_material_{self.window_days}d",
            f"filing_periodic_{self.window_days}d",
            "filing_data_available",
        ]

    # ------------------------------------------------------------------ #

    def _enrich_impl(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if df.empty:
            return df

        filings = self._filings(kwargs)
        if filings is None or filings.empty:
            logger.warning(
                "No corporate filings available; filing event features not added."
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
                    "No 'datetime' column or DatetimeIndex; filings cannot be "
                    "placed in time. Bars returned unchanged."
                )
                return df

        if "ticker" not in frame.columns:
            logger.error(
                "No 'ticker' column; a filing belongs to one company and "
                "cannot be attached market-wide. Bars returned unchanged."
            )
            return df

        try:
            daily = self._cumulative_by_ticker(filings)
        except (ValueError, TypeError, KeyError) as exc:
            logger.error("Could not aggregate filings: %s", exc)
            return df
        if daily.empty:
            logger.warning("No filing carried both a ticker and a filing date.")
            return df

        bar_time = pd.to_datetime(frame["datetime"], errors="coerce")
        if getattr(bar_time.dt, "tz", None) is not None:
            bar_time = bar_time.dt.tz_localize(None)

        counted = self._as_of_counts(bar_time, frame["ticker"], daily)

        window = self.window_days
        frame["filing_days_since_last"] = counted["days_since"].to_numpy()
        frame[f"filing_count_{window}d"] = counted["count"].to_numpy()
        frame[f"filing_material_{window}d"] = counted["material"].to_numpy()
        frame[f"filing_periodic_{window}d"] = counted["periodic"].to_numpy()
        # Absent is absent. A ticker with no filing on record is not a ticker
        # that has been quiet for zero days, and the two must not share a
        # number -- that is the mistake that put a neutral 0.0 in front of
        # every gate in this system.
        frame["filing_data_available"] = (
            counted["days_since"].notna().astype(int).to_numpy()
        )

        covered = int(frame["filing_data_available"].sum())
        logger.info(
            "Filing events attached to %d of %d bars (%.1f%%) across %d "
            "tickers; %d filings in the source spanning %s to %s.",
            covered, len(frame), 100 * covered / max(1, len(frame)),
            int(daily["ticker"].nunique()), int(daily["filings"].sum()),
            daily["_at"].min().date(), daily["_at"].max().date(),
        )
        if restore_index:
            frame = frame.set_index("datetime")
        return frame

    # ------------------------------------------------------------------ #

    def _filings(self, kwargs: dict) -> pd.DataFrame | None:
        """The stage hands its sources in; nothing is fetched here."""
        for key in ("sec_filings", "corporate_filings", "filings"):
            value = kwargs.get(key)
            if isinstance(value, pd.DataFrame) and not value.empty:
                return value
        return None

    def _cumulative_by_ticker(self, filings: pd.DataFrame) -> pd.DataFrame:
        """One row per (ticker, filing day) with running totals.

        Running totals make a window a subtraction: filings in the last N days
        is the total as of the bar minus the total as of N days earlier, and
        both are one backward as-of lookup.
        """
        date_col = self._date_column(filings)
        if date_col is None:
            raise KeyError(
                f"filings carry no filing date; columns are {sorted(filings.columns)}"
            )
        if "ticker" not in filings.columns:
            raise KeyError("filings carry no ticker")

        table = filings.copy()
        stamps = pd.to_datetime(table[date_col], errors="coerce", utc=True)
        table["_at"] = stamps.dt.tz_localize(None).dt.floor("D")
        table["ticker"] = table["ticker"].astype(str).str.strip().str.upper()
        table = table.dropna(subset=["_at"])
        table = table[table["ticker"].ne("") & table["ticker"].ne("NAN")]
        if table.empty:
            return pd.DataFrame(columns=["ticker", "_at", "filings"])

        form = table.get("form", pd.Series("", index=table.index)).astype(str).str.upper()
        table["_material"] = form.str.startswith(MATERIAL_FORMS).astype(int)
        table["_periodic"] = form.str.startswith(PERIODIC_FORMS).astype(int)

        daily = (
            table.groupby(["ticker", "_at"])
            .agg(filings=("_at", "size"),
                 material=("_material", "sum"),
                 periodic=("_periodic", "sum"))
            .reset_index()
            .sort_values(["ticker", "_at"])
        )
        for column in ("filings", "material", "periodic"):
            daily[f"cum_{column}"] = daily.groupby("ticker")[column].cumsum()
        return daily

    @staticmethod
    def _date_column(filings: pd.DataFrame) -> str | None:
        """`filingDate` is when it became public. `reportDate` is not.

        The period a filing covers is always earlier than the day it was filed,
        so reading it as the event time backdates every disclosure and lets a
        model see a June statement in March.
        """
        for candidate in ("filingDate", "filing_date", "filed_at", "published_at"):
            if candidate in filings.columns:
                return candidate
        return None

    def _as_of_counts(
        self,
        bar_time: pd.Series,
        tickers: pd.Series,
        daily: pd.DataFrame,
    ) -> pd.DataFrame:
        """Totals known at each bar, and the same totals a window earlier."""
        left = pd.DataFrame({
            "_bar": bar_time.to_numpy().astype("datetime64[ns]"),
            "ticker": tickers.astype(str).str.strip().str.upper().to_numpy(),
        })
        left["_pos"] = np.arange(len(left))
        left = left.dropna(subset=["_bar"])

        right = daily.copy()
        right["_at"] = right["_at"].to_numpy().astype("datetime64[ns]")
        right = right.sort_values("_at")

        cum_cols = ["cum_filings", "cum_material", "cum_periodic"]
        now = pd.merge_asof(
            left.sort_values("_bar"), right[["ticker", "_at", *cum_cols]],
            left_on="_bar", right_on="_at", by="ticker", direction="backward",
        )

        earlier_left = left.copy()
        earlier_left["_bar"] = earlier_left["_bar"] - pd.Timedelta(days=self.window_days)
        earlier = pd.merge_asof(
            earlier_left.sort_values("_bar"), right[["ticker", "_at", *cum_cols]],
            left_on="_bar", right_on="_at", by="ticker", direction="backward",
        )
        earlier = earlier.set_index("_pos").reindex(now["_pos"].to_numpy())

        out = pd.DataFrame(
            index=range(len(bar_time)),
            columns=["days_since", "count", "material", "periodic"],
            dtype=float,
        )
        positions = now["_pos"].to_numpy()

        # A ticker with no filing at or before the bar has nothing to report;
        # NaN here becomes filing_data_available = 0 rather than a zero count.
        seen = now["cum_filings"].notna().to_numpy()
        days_since = (now["_bar"] - now["_at"]).dt.days.to_numpy(dtype="float64")

        # No filing a window ago means nothing had happened yet: zero is the
        # right baseline there, unlike in the "never filed" case above.
        before = {c: np.nan_to_num(earlier[c].to_numpy(dtype="float64"), nan=0.0)
                  for c in cum_cols}

        out.loc[positions, "days_since"] = np.where(seen, days_since, np.nan)
        for name, column in (("count", "cum_filings"),
                             ("material", "cum_material"),
                             ("periodic", "cum_periodic")):
            delta = now[column].to_numpy(dtype="float64") - before[column]
            out.loc[positions, name] = np.where(seen, delta, np.nan)
        return out
