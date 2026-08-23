"""Value ratios, each built only from numbers already filed at the bar.

The first features in this system that are not derived from price. Measured on
2026-08-23, all 430 existing ones are: the strongest reaches an out-of-sample
IC of 0.046, no model built from all of them beats it, and 171 of 448 cannot
even vary between names on a date. New information has to come from outside the
price series, and a company's own accounts are the oldest source there is.

Three things decide whether this is honest.

**Availability is `filed`.** A June quarter reaches the SEC in August. Joining
on the period it describes would put it into June's bars -- measured on Apple's
filings, the shortest gap between a period ending and the filing that first
reported it is 25 days, and the median across all repeats is 216. Every join
here is `merge_asof(..., direction="backward")` on the filing date.

**The latest view AS OF the bar, not the latest view.** The same quarter is
restated by later filings, and 1,972 of Apple's 2,939 facts are such repeats.
Taking today's number for a bar in 2019 would be reading a correction written
years afterwards. The backward join takes what was on the record then.

**A flow concept is picked by its SPAN.** `NetIncomeLoss` arrives for the
quarter AND for the year to date, both ending the same day, from the same
filing. Mixing them silently multiplies earnings by up to four. Only entries
whose span looks quarterly are used, and the rule is explicit rather than
implied by whichever row happened to sort first.

Ratios needing a price use the bar's own close, which is known at the bar by
definition. Ratios that do not are computed from the accounts alone, so they
survive on a frame with no price at all.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseEnricher

logger = logging.getLogger(__name__)

#: A quarterly span, generously bounded. 10-Q periods run 84 to 98 days; the
#: year-to-date entries that share their end date run 175 to 370, so the two
#: groups do not overlap and the rule needs no tie-breaking.
QUARTER_DAYS = (60, 130)

#: Balance-sheet concepts are instantaneous -- they have no span, so there is
#: nothing to disambiguate.
_INSTANT = (
    "Assets", "AssetsCurrent", "Liabilities", "LiabilitiesCurrent",
    "StockholdersEquity", "LongTermDebtNoncurrent",
    "CashAndCashEquivalentsAtCarryingValue",
)

#: Flow concepts cover a period, and must be filtered by span.
_FLOW = ("NetIncomeLoss", "Revenues", "OperatingIncomeLoss")

#: Share counts, needed to put a balance-sheet figure next to a share price.
_SHARES = ("CommonStockSharesOutstanding",
           "WeightedAverageNumberOfSharesOutstandingBasic")


class FundamentalsEnricher(BaseEnricher):
    """Point-in-time value ratios from reported SEC figures."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__()
        self.config = config or {}
        # Beyond this the accounts are too old to describe the company. A
        # filing every quarter means a gap over ~200 days is a company that
        # stopped reporting, not one that is merely between filings.
        self.max_staleness_days = int(self.config.get("max_staleness_days", 200))

    @property
    def name(self) -> str:
        return "fundamentals"

    @property
    def priority(self) -> int:
        # After corporate_filings (30): both key off SEC filing dates, and the
        # counts there are a cheaper answer to a related question.
        return 31

    def get_feature_names(self) -> list[str]:
        return [
            "fund_price_to_book",
            "fund_earnings_yield",
            "fund_current_ratio",
            "fund_debt_to_equity",
            "fund_return_on_equity",
            "fund_days_since_report",
            "fund_data_available",
        ]

    # ------------------------------------------------------------- shaping

    @staticmethod
    def _pick_by_span(facts: pd.DataFrame) -> pd.DataFrame:
        """Keep quarterly spans for flow concepts, everything for instants.

        Without this, `NetIncomeLoss` for the quarter and for the nine months
        both survive -- same end date, same accession, different numbers -- and
        whichever sorts last wins. Earnings would be silently up to four times
        too large, in a direction nothing downstream could detect.
        """
        span = (facts["period_end"] - facts["period_start"]).dt.days
        is_flow = facts["concept"].isin(_FLOW)
        quarterly = span.between(*QUARTER_DAYS)
        # An instantaneous fact has no start; `span` is NaT there, which is
        # exactly the rows that should pass untouched.
        return facts[~is_flow | quarterly.fillna(False)]

    def _as_of(self, bars: pd.DataFrame, facts: pd.DataFrame,
               concept: str) -> pd.Series:
        """The value on the record for each bar, by filing date."""
        right = facts[facts["concept"].eq(concept)]
        if right.empty:
            return pd.Series(np.nan, index=bars.index)

        # Latest filing wins for a given period; latest period wins overall.
        # Sorting by `filed` and taking the backward match does both, because
        # a restatement is filed after the original by definition.
        right = (right.sort_values("filed")
                      .loc[:, ["ticker", "filed", "value", "period_end"]]
                      .rename(columns={"value": concept,
                                       "period_end": f"{concept}__end"}))
        merged = pd.merge_asof(
            bars.sort_values("_bar"),
            right,
            left_on="_bar", right_on="filed", by="ticker", direction="backward",
        )
        return merged.set_index("_pos")[concept].reindex(bars["_pos"].to_numpy())

    # -------------------------------------------------------------- enrich

    def _enrich_impl(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        frame = df.copy(deep=False)
        for name in self.get_feature_names():
            frame[name] = np.nan
        frame["fund_data_available"] = 0

        facts = kwargs.get("sec_fundamentals")
        if facts is None or getattr(facts, "empty", True):
            logger.warning(
                "FundamentalsEnricher: no sec_fundamentals supplied; the value "
                "ratios will be absent rather than wrong."
            )
            return frame

        required = {"ticker", "concept", "value", "filed", "period_end"}
        missing = required - set(facts.columns)
        if missing:
            logger.error(
                "FundamentalsEnricher: sec_fundamentals is missing %s. This is "
                "a schema break, not empty data.", sorted(missing),
            )
            return frame

        time_column = next(
            (name for name in ("datetime", "timestamp", "date") if name in frame.columns),
            None,
        )
        if time_column is None or "ticker" not in frame.columns:
            logger.error("FundamentalsEnricher: frame has no datetime or ticker.")
            return frame

        facts = facts.copy(deep=False)
        facts["filed"] = pd.to_datetime(facts["filed"], errors="coerce")
        facts["period_end"] = pd.to_datetime(facts["period_end"], errors="coerce")
        facts["period_start"] = pd.to_datetime(
            facts.get("period_start"), errors="coerce"
        )
        facts = facts.dropna(subset=["filed", "value"])
        facts = self._pick_by_span(facts)
        if facts.empty:
            logger.warning("FundamentalsEnricher: no facts survived the span filter.")
            return frame

        bars = pd.DataFrame({
            "_pos": np.arange(len(frame)),
            "ticker": frame["ticker"].astype(str).to_numpy(),
            "_bar": pd.to_datetime(frame[time_column], errors="coerce").to_numpy(),
        })
        if bars["_bar"].isna().all():
            logger.error("FundamentalsEnricher: no usable bar timestamps.")
            return frame
        # merge_asof refuses NaT keys, and a bar with no time cannot be placed
        # against a filing date anyway.
        bars = bars.dropna(subset=["_bar"])
        facts["filed"] = facts["filed"].to_numpy().astype("datetime64[ns]")
        bars["_bar"] = bars["_bar"].astype("datetime64[ns]")

        values = {name: self._as_of(bars, facts, name)
                  for name in (*_INSTANT, *_FLOW, *_SHARES)}
        latest_filed = self._latest_filed(bars, facts)

        equity = values["StockholdersEquity"]
        shares = values["CommonStockSharesOutstanding"].fillna(
            values["WeightedAverageNumberOfSharesOutstandingBasic"]
        )
        income = values["NetIncomeLoss"]

        # A ratio with a non-positive denominator is not a large number, it is
        # a different situation -- negative equity is distress, and dividing by
        # it produces a figure that ranks as if it were cheap.
        safe_equity = equity.where(equity > 0)
        safe_shares = shares.where(shares > 0)

        assigned = {
            "fund_current_ratio": values["AssetsCurrent"]
                / values["LiabilitiesCurrent"].where(values["LiabilitiesCurrent"] > 0),
            "fund_debt_to_equity": values["Liabilities"] / safe_equity,
            "fund_return_on_equity": income / safe_equity,
        }

        close = None
        for candidate in ("close", "Close", "close_1d"):
            if candidate in frame.columns:
                close = pd.to_numeric(frame[candidate], errors="coerce")
                break
        if close is not None:
            market_cap = close * safe_shares
            assigned["fund_price_to_book"] = market_cap / safe_equity
            assigned["fund_earnings_yield"] = income / market_cap.where(market_cap > 0)
        else:
            logger.info(
                "FundamentalsEnricher: no close column, so the two ratios that "
                "need a price are left absent; the rest are unaffected."
            )

        staleness = (
            pd.to_datetime(frame[time_column], errors="coerce")
            - pd.to_datetime(latest_filed)
        ).dt.days
        fresh = staleness.between(0, self.max_staleness_days)

        for name, series in assigned.items():
            series = pd.to_numeric(series, errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            frame[name] = series.where(fresh).to_numpy()
        frame["fund_days_since_report"] = staleness.where(fresh).to_numpy()
        frame["fund_data_available"] = fresh.fillna(False).astype(int).to_numpy()

        covered = int(frame["fund_data_available"].sum())
        logger.info(
            "Fundamentals on %d of %d bars (%.1f%%), %d tickers with filings.",
            covered, len(frame), 100 * covered / max(1, len(frame)),
            facts["ticker"].nunique(),
        )
        return frame

    def _latest_filed(self, bars: pd.DataFrame, facts: pd.DataFrame) -> pd.Series:
        """When the newest figure available at each bar was filed."""
        right = (facts[["ticker", "filed"]]
                 .sort_values("filed")
                 .assign(_filed_at=lambda part: part["filed"]))
        merged = pd.merge_asof(
            bars.sort_values("_bar"), right,
            left_on="_bar", right_on="filed", by="ticker", direction="backward",
        )
        return merged.set_index("_pos")["_filed_at"].reindex(bars["_pos"].to_numpy())
