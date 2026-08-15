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

# One ticker may sit in several groups; the first match wins, so the most
# specific grouping is listed first.
_DEFAULT_SECTORS: dict[str, list[str]] = {
    "semis": ["NVDA", "AMD", "INTC", "TSM"],
    "big_tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META"],
    "finance": ["JPM", "BAC", "GS", "WFC"],
    "staples": ["KO", "WMT", "PG", "COST"],
    "energy": ["XOM", "CVX", "COP"],
    "etf_broad": ["SPY", "QQQ", "IWM", "DIA"],
    "etf_sector": ["XLK", "XLE", "XLF", "XLV", "XLI", "XLU"],
}


class PeerContextEnricher(BaseEnricher):
    """Sector aggregates computed from a ticker's neighbours, never itself."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__()
        self.config = config or {}
        groups = self.config.get("sectors") or _DEFAULT_SECTORS
        self.sector_of: dict[str, str] = {}
        for sector, members in groups.items():
            for ticker in members:
                self.sector_of.setdefault(str(ticker).strip().upper(), sector)

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

        frame["peer_return"] = peer_mean.to_numpy()
        frame["peer_volatility"] = peer_std.to_numpy()
        frame["peer_breadth"] = peer_breadth.to_numpy()
        frame["peer_divergence"] = (work["_ret"] - peer_mean).to_numpy()
        frame["peer_count"] = peers.fillna(0).to_numpy()

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
