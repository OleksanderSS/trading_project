"""Market-wide series that belong to every ticker at once.

Two sources were collected for a long time and reached no model. Neither
carries a ticker, because neither is about one company: the CNN Fear & Greed
index is a reading of the whole market's mood, and the CFTC Commitments of
Traders report is how futures traders are positioned in the S&P, Nasdaq, Dow,
gold and crude. Both belong on every bar of every ticker, which is exactly why
no per-ticker merge ever picked them up.

Coverage measured on the 2026-08-15 batch, against 11,390 daily bars:

    cftc         2,610 rows, weekly since 2016-08  -> 100% of bars
    fear_greed     267 rows, daily  since 2025-08  ->  49% of bars

CFTC is the deepest history of anything unused in this project.

PUBLICATION LAG IS THE WHOLE DIFFICULTY. A COT report is stamped with the
Tuesday it describes and released the following Friday at 15:30 ET, so joining
on its own date hands every bar three days of the future. The stored rows show
this plainly: `date` 2026-08-11 (a Tuesday) with `collected_at` 2026-08-14 (the
Friday). Fear & Greed updates through the trading day, so a bar inside day D
must not read D's closing reading either.

Both are therefore shifted to when they became knowable, and the shift is
configurable rather than hidden, because it is a property of the publisher and
publishers change their schedules.
"""
from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.features.enrichers.base import BaseEnricher

logger = ProjectLogger.get_logger("MarketWideEnricher")

# Instrument name in the CFTC table -> the stem used in column names.
_CFTC_INSTRUMENTS = {
    "S&P": "sp500",
    "NASDAQ": "nasdaq",
    "DOW": "dow",
    "GOLD": "gold",
    "CRUDE OIL": "crude",
}


class MarketWideEnricher(BaseEnricher):
    """Attaches market-wide sentiment and futures positioning to every bar."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__()
        self.config = config or {}
        # A COT report describes Tuesday and is released the following Friday.
        self.cftc_publication_lag_days = int(
            self.config.get("cftc_publication_lag_days", 3)
        )
        # Fear & Greed moves during the session, so the day's reading is only
        # safe to use once the day is over.
        self.fear_greed_publication_lag_days = int(
            self.config.get("fear_greed_publication_lag_days", 1)
        )

    @property
    def name(self) -> str:
        return "market_wide"

    @property
    def priority(self) -> int:
        # After macro_features (27), before the news enrichers, so its columns
        # exist when context_map builds state columns at 80.
        return 28

    def _enrich_impl(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if df.empty:
            logger.warning("Input frame is empty; market-wide enrichment skipped.")
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
                    "No 'datetime' column or DatetimeIndex; market-wide series "
                    "cannot be placed in time. Bars returned unchanged."
                )
                return df

        bar_time = pd.to_datetime(frame["datetime"], errors="coerce")
        if getattr(bar_time.dt, "tz", None) is not None:
            bar_time = bar_time.dt.tz_localize(None)

        attached = 0
        attached += self._attach_fear_greed(
            frame, bar_time, self._source(kwargs, "fear_greed")
        )
        attached += self._attach_cftc(frame, bar_time, self._source(kwargs, "cftc"))

        if not attached:
            logger.warning(
                "Neither fear_greed nor cftc reached this enricher; no "
                "market-wide columns were added."
            )

        if restore_index:
            frame = frame.set_index("datetime")
        return frame

    # ------------------------------------------------------------------ #

    def _attach_fear_greed(self, frame: pd.DataFrame, bar_time: pd.Series,
                           source: Any) -> int:
        if not isinstance(source, pd.DataFrame) or source.empty:
            return 0
        if "date" not in source.columns or "value" not in source.columns:
            logger.error(
                "fear_greed lacks 'date' or 'value' (has %s); not attached.",
                list(source.columns)[:8],
            )
            return 0

        table = source.copy()
        table["_at"] = self._available_at(
            table["date"], self.fear_greed_publication_lag_days
        )
        table["_value"] = pd.to_numeric(table["value"], errors="coerce")
        # Several readings are stored per day; the last one is the day's close.
        table = (table.dropna(subset=["_at", "_value"])
                      .sort_values("_at")
                      .drop_duplicates(subset=["_at"], keep="last"))
        if table.empty:
            logger.error("fear_greed has no usable date/value pairs.")
            return 0

        merged = self._asof(bar_time, table[["_at", "_value"]], ["_value"])
        frame["fear_greed_index"] = merged["_value"].to_numpy()
        frame["fear_greed_available"] = merged["_value"].notna().astype(int).to_numpy()
        covered = int(frame["fear_greed_available"].sum())
        logger.info(
            "fear_greed attached to %d of %d bars (%.0f%%), %d distinct readings.",
            covered, len(frame), 100 * covered / max(1, len(frame)),
            int(merged["_value"].nunique()),
        )
        return 1

    def _attach_cftc(self, frame: pd.DataFrame, bar_time: pd.Series,
                     source: Any) -> int:
        if not isinstance(source, pd.DataFrame) or source.empty:
            return 0
        needed = {"date", "instrument"}
        if not needed.issubset(source.columns):
            logger.error(
                "cftc lacks %s (has %s); not attached.",
                sorted(needed - set(source.columns)), list(source.columns)[:8],
            )
            return 0

        table = source.copy()
        table["_at"] = self._available_at(
            table["date"], self.cftc_publication_lag_days
        )
        table = table.dropna(subset=["_at"])
        added = 0
        for raw_name, stem in _CFTC_INSTRUMENTS.items():
            rows = table[table["instrument"].astype(str).str.upper() == raw_name]
            if rows.empty:
                continue
            wanted = {
                f"cftc_{stem}_net_pct": "net_position_pct",
                f"cftc_{stem}_ls_ratio": "long_short_ratio",
            }
            keep = {out: src for out, src in wanted.items() if src in rows.columns}
            if not keep:
                continue
            slim = rows[["_at"] + list(keep.values())].copy()
            for src in keep.values():
                slim[src] = pd.to_numeric(slim[src], errors="coerce")
            slim = (slim.sort_values("_at")
                        .drop_duplicates(subset=["_at"], keep="last"))
            merged = self._asof(bar_time, slim, list(keep.values()))
            for out, src in keep.items():
                frame[out] = merged[src].to_numpy()
                added += 1
        if added:
            first = next(iter(_CFTC_INSTRUMENTS.values()))
            flag_col = f"cftc_{first}_net_pct"
            covered = int(frame[flag_col].notna().sum()) if flag_col in frame else 0
            frame["cftc_available"] = (
                frame[flag_col].notna().astype(int) if flag_col in frame else 0
            )
            logger.info(
                "cftc attached %d columns; %d of %d bars covered (%.0f%%).",
                added, covered, len(frame), 100 * covered / max(1, len(frame)),
            )
        else:
            logger.error(
                "cftc carried none of the expected instruments %s; nothing attached.",
                sorted(_CFTC_INSTRUMENTS),
            )
        return 1 if added else 0

    # ------------------------------------------------------------------ #

    @staticmethod
    def _source(kwargs: dict, stem: str) -> Any:
        """Find a source under either spelling the pipeline uses.

        Stage 3 keys its inputs by TABLE name -- the run log lists
        `cftc_data` and `fear_greed_data` -- while a caller writing a test
        naturally passes `cftc` and `fear_greed`. Accepting one and silently
        ignoring the other is precisely how a working enricher reports that it
        found nothing, which this project has now paid for five times.
        """
        for key in (stem, f"{stem}_data"):
            value = kwargs.get(key)
            if isinstance(value, pd.DataFrame) and not value.empty:
                return value
        return None

    @staticmethod
    def _available_at(dates: pd.Series, lag_days: int) -> pd.Series:
        """When a figure stamped `date` could first have been read.

        The lag is added rather than assumed away: a COT report describes a
        Tuesday and appears on Friday, so merging on the stamp itself would
        put three days of the future into every bar.
        """
        parsed = pd.to_datetime(dates, errors="coerce")
        if getattr(parsed.dt, "tz", None) is not None:
            parsed = parsed.dt.tz_localize(None)
        return parsed + pd.Timedelta(days=lag_days)

    @staticmethod
    def _asof(bar_time: pd.Series, table: pd.DataFrame,
              value_cols: list[str]) -> pd.DataFrame:
        """Backward as-of join that returns rows in the CALLER's order.

        merge_asof needs both sides sorted and returns a fresh RangeIndex, so
        the result is re-indexed onto the caller's original positions. Getting
        this wrong is how 54,000 bars once ended up on other bars' dates.
        """
        # Parquet hands these back at microsecond resolution while the source
        # tables parse to nanoseconds, and merge_asof refuses to join
        # datetime64[us] to datetime64[ns] at all. Both sides are pinned to ns
        # rather than left to chance.
        left = pd.DataFrame({"_bar": bar_time.to_numpy().astype("datetime64[ns]")})
        left["_pos"] = range(len(left))
        left = left.dropna(subset=["_bar"]).sort_values("_bar")
        right = table.copy()
        right["_at"] = right["_at"].to_numpy().astype("datetime64[ns]")
        right = right.sort_values("_at")
        merged = pd.merge_asof(
            left, right, left_on="_bar", right_on="_at", direction="backward"
        )
        out = pd.DataFrame(index=range(len(bar_time)), columns=value_cols, dtype=float)
        out.loc[merged["_pos"].to_numpy(), value_cols] = merged[value_cols].to_numpy()
        return out
