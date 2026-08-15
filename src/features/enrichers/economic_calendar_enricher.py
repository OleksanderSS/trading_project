"""How far scheduled economic releases came in from what was expected.

A calendar entry carries `actual`, `forecast` and `previous`, so the number
that matters is the surprise: actual minus forecast, standardised per event so
that a 0.2pp miss on CPI is comparable to a 40k miss on payrolls.

This enricher existed and had never run once. Four reasons, and only the first
was obvious:

  * it declared neither `name` nor `priority`, so the class is abstract and
    FeatureOrchestrator could not instantiate it at all -- which is why it is
    absent from features.yaml rather than merely disabled;
  * `__init__` forwarded **kwargs to BaseEnricher, which takes none;
  * it read the calendar from a `db_manager` the orchestrator never passes,
    so with the first two fixed it would still have logged "No db_manager" and
    returned the bars untouched;
  * and the surprise was standardised across the WHOLE series --
    `transform(lambda x: (x - x.mean()) / x.std())` takes a mean and a
    deviation computed from every row including the future. A leak of that
    shape is identical in every fold, so walk-forward validation cannot see
    it; it just raises the score everywhere.

The standardisation is now expanding: at row i it uses rows 0..i of that
event's history and nothing later, so the figure a bar carries could have been
computed at that bar. Below `min_history` observations of an event there is no
baseline to be surprised against, and the enricher says so with NaN rather
than inventing a zero.

Honest about scale: the calendar holds 147 rows over 11 days, so this covers
about 2% of daily bars today. It is wired correctly so that it becomes useful
as the collector accumulates, not because it is useful yet.
"""
import logging
from typing import Any

import numpy as np
import pandas as pd

from src.features.enrichers.base import BaseEnricher

logger = logging.getLogger(__name__)

_IMPACT_RANK = {"low": 1, "medium": 2, "high": 3, "holiday": 0}


class EconomicCalendarEnricher(BaseEnricher):
    """Adds the surprise of scheduled releases to every bar."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__()
        self.config = config or {}
        self.min_history = int(self.config.get("min_history", 3))
        # A release is public the moment it prints, so no lag is added. The
        # calendar's own timestamp IS the publication time, unlike the CFTC
        # report which describes an earlier date.
        self.db_manager = self.config.get("db_manager")

    @property
    def name(self) -> str:
        return "economic_calendar"

    @property
    def priority(self) -> int:
        return 29  # beside market_wide (28): both are market-wide, date-keyed

    def _enrich_impl(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if df.empty:
            return df

        calendar = self._calendar(kwargs)
        if calendar is None or calendar.empty:
            logger.warning(
                "No economic calendar available; surprise features not added."
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
                    "No 'datetime' column or DatetimeIndex; calendar surprises "
                    "cannot be placed in time. Bars returned unchanged."
                )
                return df

        try:
            daily = self._daily_surprise(calendar)
        except (ValueError, TypeError, KeyError) as exc:
            logger.error("Could not compute calendar surprises: %s", exc)
            return df
        if daily.empty:
            logger.warning(
                "The calendar carried no event with both an actual and a "
                "forecast; nothing to be surprised about."
            )
            return df

        bar_time = pd.to_datetime(frame["datetime"], errors="coerce")
        if getattr(bar_time.dt, "tz", None) is not None:
            bar_time = bar_time.dt.tz_localize(None)

        merged = self._asof(bar_time, daily)
        frame["econ_surprise_index"] = merged["surprise_index"].to_numpy()
        frame["econ_event_impact"] = merged["impact_rank"].to_numpy()
        # Absent is absent. The previous version forward-filled without limit
        # and then filled 0, which made "no release has happened" and "the
        # release landed exactly on forecast" the same number.
        frame["econ_calendar_available"] = (
            merged["surprise_index"].notna().astype(int).to_numpy()
        )

        covered = int(frame["econ_calendar_available"].sum())
        logger.info(
            "Calendar surprises attached to %d of %d bars (%.0f%%); %d distinct "
            "values.", covered, len(frame), 100 * covered / max(1, len(frame)),
            int(merged["surprise_index"].nunique()),
        )
        if restore_index:
            frame = frame.set_index("datetime")
        return frame

    # ------------------------------------------------------------------ #

    def _calendar(self, kwargs: dict) -> pd.DataFrame | None:
        """From the stage's inputs first, the database only as a fallback.

        Every other enricher is handed its sources; this one reached into the
        database on its own and got nothing, because the orchestrator builds
        enrichers from a config dict and never supplies a db_manager.
        """
        for key in ("economic_calendar", "economic_calendar_data", "calendar"):
            value = kwargs.get(key)
            if isinstance(value, pd.DataFrame) and not value.empty:
                return value
        if self.db_manager is not None:
            try:
                return self.db_manager.execute_query(
                    "SELECT timestamp, event, actual, forecast, previous, impact "
                    "FROM economic_calendar"
                )
            except Exception as exc:  # noqa: BLE001 - fallback must not kill the run
                logger.warning("Calendar query failed: %s", exc)
        return None

    def _daily_surprise(self, calendar: pd.DataFrame) -> pd.DataFrame:
        needed = {"timestamp", "event", "actual", "forecast"}
        missing = needed - set(calendar.columns)
        if missing:
            raise KeyError(f"calendar lacks {sorted(missing)}")

        table = calendar.copy()
        stamps = pd.to_datetime(table["timestamp"], errors="coerce", utc=True)
        table["_at"] = stamps.dt.tz_localize(None)
        table["_actual"] = table["actual"].map(self._to_number)
        table["_forecast"] = table["forecast"].map(self._to_number)
        table = table.dropna(subset=["_at", "_actual", "_forecast"])
        if table.empty:
            return pd.DataFrame(columns=["_at", "surprise_index", "impact_rank"])

        table = table.sort_values("_at")
        table["_surprise"] = table["_actual"] - table["_forecast"]

        # Expanding, not whole-series: row i sees rows 0..i of its own event.
        grouped = table.groupby("event")["_surprise"]
        mean = grouped.transform(lambda s: s.expanding(self.min_history).mean())
        std = grouped.transform(lambda s: s.expanding(self.min_history).std())
        table["surprise_index"] = (table["_surprise"] - mean) / (std + 1e-6)

        table["impact_rank"] = (
            table.get("impact", pd.Series(index=table.index, dtype=object))
            .astype(str).str.strip().str.lower().map(_IMPACT_RANK)
        )

        daily = (table.dropna(subset=["surprise_index"])
                      .groupby(table["_at"].dt.floor("D"))
                      .agg(surprise_index=("surprise_index", "mean"),
                           impact_rank=("impact_rank", "max"))
                      .reset_index()
                      .rename(columns={"_at": "_at"}))
        daily.columns = ["_at", "surprise_index", "impact_rank"]
        return daily

    @staticmethod
    def _to_number(value: Any) -> float:
        """'150K', '2.5M', '1.2%' and '-3B' are all numbers here."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return np.nan
        text = str(value).strip().replace(",", "")
        if not text:
            return np.nan
        multiplier = 1.0
        if text.endswith("%"):
            text = text[:-1]
        elif text[-1:].upper() in {"K", "M", "B"}:
            multiplier = {"K": 1e3, "M": 1e6, "B": 1e9}[text[-1].upper()]
            text = text[:-1]
        try:
            return float(text) * multiplier
        except ValueError:
            return np.nan

    @staticmethod
    def _asof(bar_time: pd.Series, daily: pd.DataFrame) -> pd.DataFrame:
        cols = ["surprise_index", "impact_rank"]
        left = pd.DataFrame({"_bar": bar_time.to_numpy().astype("datetime64[ns]")})
        left["_pos"] = range(len(left))
        left = left.dropna(subset=["_bar"]).sort_values("_bar")
        right = daily.copy()
        right["_at"] = right["_at"].to_numpy().astype("datetime64[ns]")
        merged = pd.merge_asof(
            left, right.sort_values("_at"), left_on="_bar", right_on="_at",
            direction="backward",
        )
        out = pd.DataFrame(index=range(len(bar_time)), columns=cols, dtype=float)
        out.loc[merged["_pos"].to_numpy(), cols] = merged[cols].to_numpy()
        return out

    def get_feature_names(self) -> list[str]:
        return ["econ_surprise_index", "econ_event_impact",
                "econ_calendar_available"]
