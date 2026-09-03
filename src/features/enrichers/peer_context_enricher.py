"""What the neighbours did, which nothing in this pipeline could see.

A model predicting NVDA did not know that MSFT had just spiked. Measured on
the 2026-08-15 export: zero sector features, zero cross-ticker features, and
the only "market" columns are computed from the ticker's own history. Every
model looked at one company through a keyhole.

This is a DIFFERENT mechanism from pooling tickers into one model, and the
two were being conflated:

    pooling shares the LEARNING     — one model sees every ticker's history
                                      and learns what breakouts have in common
    peer features share the INFORMATION — each row knows what its sector did
                                      in that same hour

A pooled model still cannot see, at prediction time, that a peer moved an hour
ago. "Microsoft ships a strong model, so chips and power rise" needs the
second kind, and no amount of the first substitutes for it.

Four features, and the fourth is the point:

    peer_return         the sector's own move, EXCLUDING this ticker
    peer_volatility     dispersion across the sector
    peer_breadth        share of the sector that rose
    peer_divergence     this ticker's move minus its sector's

`peer_divergence` is the case the user described: the whole sector rises and
one name does not, or the sector holds and one name collapses. Google's
servers burn down; the index shrugs; GOOGL does not.

EXCLUDING THE TICKER IS THE WHOLE DESIGN. A sector average that contains the
row it is describing leaks that row's own return back into its features — for
a two-name sector, half of it. Every aggregate here is computed as
(sum over sector - this ticker) / (count - 1), so a bar never sees itself.
Aggregates are formed on each bar's own timestamp, so nothing arrives from the
future either.
"""
from typing import Any

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.features.enrichers.base import BaseEnricher

logger = ProjectLogger.get_logger("PeerContextEnricher")

# Fallback only. These seven groups name 31 tickers, and the universe is 110,
# so every other name fell into a single bucket called "other" --
# `ticker.map(self.sector_of).fillna("other")`.
#
# Measured on the 2026-08-29 batch: 79 of 110 tickers (72% of the universe,
# 69% of daily rows) had that bucket as their "sector", with a median of 65
# neighbours. For those rows `peer_return` is not the sector's move, it is the
# market's; `peer_breadth` is market breadth; `peer_divergence` is
# ticker-minus-market. Useful quantities, all of them -- but the columns are
# named as though they were sector context, and nothing said otherwise.
#
# No coverage metric showed it: 110 of 110 tickers had values, on 99.9% of
# rows. That is the shape this audit keeps finding -- a number that is present,
# plausible, and about something else.
#
# The real membership now comes from `assets.sector_partition`: 14 sectors,
# every one of the 110 tickers in exactly one, minimum three members
# (REGISTER #225, enforced by tests/contracts/test_sector_partition_is_a_partition.py).
_DEFAULT_SECTORS: dict[str, list[str]] = {
    "semis": ["NVDA", "AMD", "INTC", "TSM"],
    "big_tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META"],
    "finance": ["JPM", "BAC", "GS", "WFC"],
    "staples": ["KO", "WMT", "PG", "COST"],
    "energy": ["XOM", "CVX", "COP"],
    "etf_broad": ["SPY", "QQQ", "IWM", "DIA"],
    "etf_sector": ["XLK", "XLE", "XLF", "XLV", "XLI", "XLU"],
}

#: What an unmapped ticker is called. Kept as a named constant so the state is
#: countable rather than incidental -- see `_unmapped_share` below.
UNMAPPED_SECTOR = "other"


def _partition_from_assets() -> dict[str, list[str]] | None:
    """Sector membership from `assets.sector_partition`, or None if absent.

    Read at construction rather than hardcoded so the taxonomy has one home
    and a contract test can check it is a partition.
    """
    try:
        from pathlib import Path

        import yaml

        config = Path(__file__).resolve().parents[3] / "src" / "config" / "assets.yaml"
        block = (
            yaml.safe_load(config.read_text(encoding="utf-8"))
            .get("assets", {})
            .get("sector_partition")
        )
    except Exception as error:  # noqa: BLE001 - fall back to the constant above
        logger.warning(
            "Could not read assets.sector_partition (%s: %s); falling back to "
            "the built-in groups, which name 31 of 110 tickers.",
            type(error).__name__, error,
        )
        return None
    if not block:
        return None
    return {name: list(body.get("assets") or []) for name, body in block.items()}


class PeerContextEnricher(BaseEnricher):
    """Sector aggregates computed from a ticker's neighbours, never itself."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__()
        self.config = config or {}
        groups = (
            self.config.get("sectors")
            or _partition_from_assets()
            or _DEFAULT_SECTORS
        )
        self.sector_of: dict[str, str] = {}
        for sector, members in groups.items():
            for ticker in members:
                self.sector_of.setdefault(str(ticker).strip().upper(), sector)
        logger.info(
            "Peer context membership: %d sectors over %d tickers.",
            len(groups), len(self.sector_of),
        )

    @property
    def name(self) -> str:
        return "peer_context"

    @property
    def priority(self) -> int:
        # After derived_features (25) so `returns` exists, before the news
        # enrichers and well before context_map builds its state columns.
        return 26

    def _enrich_impl(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if df.empty:
            return df
        if "ticker" not in df.columns:
            logger.error("No 'ticker' column; peer context cannot be built.")
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
                logger.error("No 'datetime' column or DatetimeIndex; bars unchanged.")
                return df

        ret = self._returns(frame)
        if ret is None:
            logger.error(
                "No 'close' or 'returns' column; peer aggregates need a move "
                "to aggregate. Bars returned unchanged."
            )
            return df.copy() if not restore_index else df

        ticker = frame["ticker"].astype(str).str.strip().str.upper()
        sector = ticker.map(self.sector_of).fillna("other")
        stamp = pd.to_datetime(frame["datetime"], errors="coerce")
        if getattr(stamp.dt, "tz", None) is not None:
            stamp = stamp.dt.tz_localize(None)

        work = pd.DataFrame({
            "_sector": sector.to_numpy(),
            "_stamp": stamp.to_numpy(),
            "_ret": pd.to_numeric(ret, errors="coerce").to_numpy(),
        })
        work["_up"] = (work["_ret"] > 0).astype(float).where(work["_ret"].notna())

        grouped = work.groupby(["_sector", "_stamp"], dropna=False)["_ret"]
        # Leave-one-out: the sum and count of the group minus this row, so a
        # bar is never part of the average that describes it.
        total = grouped.transform("sum")
        count = grouped.transform("count")
        own = work["_ret"].fillna(0.0)
        has_own = work["_ret"].notna().astype(float)
        peers = (count - has_own).replace(0, np.nan)
        peer_mean = (total - own) / peers

        # Dispersion, also leave-one-out, via the sum of squares.
        sq = work.assign(_sq=work["_ret"] ** 2).groupby(
            ["_sector", "_stamp"], dropna=False
        )["_sq"].transform("sum")
        peer_var = ((sq - own ** 2) / peers) - peer_mean ** 2
        peer_std = np.sqrt(peer_var.clip(lower=0))

        up_total = work.groupby(["_sector", "_stamp"], dropna=False)["_up"].transform("sum")
        own_up = work["_up"].fillna(0.0)
        peer_breadth = (up_total - own_up) / peers

        # The market, on the same leave-one-out basis, so a bar is never part
        # of either average that describes it.
        by_stamp = work.groupby("_stamp", dropna=False)["_ret"]
        market_total = by_stamp.transform("sum")
        market_count = by_stamp.transform("count")
        market_peers = (market_count - has_own).replace(0, np.nan)
        market_mean = (market_total - own) / market_peers
        market_up_total = work.groupby("_stamp", dropna=False)["_up"].transform("sum")
        market_breadth = (market_up_total - own_up) / market_peers

        frame["peer_return"] = peer_mean.to_numpy()
        frame["peer_volatility"] = peer_std.to_numpy()
        frame["peer_breadth"] = peer_breadth.to_numpy()
        frame["peer_divergence"] = (work["_ret"] - peer_mean).to_numpy()
        frame["peer_count"] = peers.fillna(0).to_numpy()

        # The market factor, named rather than left implicit, and the two
        # sector aggregates with it removed.
        #
        # WHY THIS MATTERS MORE THAN IT LOOKS. Measured 2026-09-02 on daily
        # bars 2010-2026 over the 14-sector partition: the raw sector returns
        # carry 2.97 independent dimensions out of 14 -- their first principal
        # component is 56% of the variance and their mean pairwise correlation
        # is 0.505. Fourteen columns saying one thing, and that thing is the
        # market. Subtracting the market leaves 8.63 dimensions with a mean
        # correlation of -0.042 (CLAIMS.md R12).
        #
        # A DIFFERENCE, NOT A REGRESSION RESIDUAL. The measurement above used
        # OLS residuals and scored slightly better (9.35 dimensions), and that
        # is exactly why it must not be the feature: fitting beta over the
        # whole sample uses the future to describe the past. The plain
        # difference keeps 92% of the benefit and each bar is computed from
        # that bar alone.
        #
        # Dispersion is deliberately NOT residualised: measured on the same
        # data it is already orthogonal to the market -- 8.09 dimensions
        # before and 8.09 after, mean correlation 0.213 against 0.214.
        # Subtracting a factor it does not contain would only add noise.
        frame["market_return"] = market_mean.to_numpy()
        frame["market_breadth"] = market_breadth.to_numpy()
        frame["peer_return_excess"] = (peer_mean - market_mean).to_numpy()
        frame["peer_breadth_excess"] = (peer_breadth - market_breadth).to_numpy()

        alone = int((peers.isna() | (peers < 1)).sum())
        logger.info(
            "Peer context over %d sectors; %d of %d bars had at least one "
            "peer at their timestamp (%d alone).",
            sector.nunique(), len(frame) - alone, len(frame), alone,
        )
        if alone == len(frame):
            logger.warning(
                "No bar had a peer at its own timestamp. Either the batch holds "
                "one ticker, or the sector map covers none of these: %s",
                sorted(ticker.unique())[:8],
            )

        if restore_index:
            frame = frame.set_index("datetime")
        return frame

    @staticmethod
    def _returns(frame: pd.DataFrame) -> pd.Series | None:
        """A bar's own move, reused if a previous enricher already made it."""
        for name in ("returns", "return_1", "pct_change"):
            if name in frame.columns:
                return frame[name]
        if "close" not in frame.columns:
            return None
        close = pd.to_numeric(frame["close"], errors="coerce")
        # Per ticker, or the first bar of each name inherits the last bar of
        # the one before it in row order.
        return close.groupby(frame["ticker"].astype(str)).pct_change(fill_method=None)

    def get_feature_names(self) -> list[str]:
        return ["peer_return", "peer_volatility", "peer_breadth",
                "peer_divergence", "peer_count"]
