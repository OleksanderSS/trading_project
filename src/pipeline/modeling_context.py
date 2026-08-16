from __future__ import annotations

import hashlib
from collections.abc import Iterator

import pandas as pd

from src.pipeline.timeframe_lineage import (
    partition_market_frame_by_timeframe,
)

# The ticker slot of a pooled context. Not a real ticker, and deliberately
# unlike one, so a pooled artifact can never be mistaken for AAPL's.
POOLED_TICKER = "__POOLED__"

# The name matters more than it looks. Feature selection drops any column whose
# name is or begins with an identity stem -- 'ticker', 'symbol', 'interval' and
# the rest -- because label-encoding an identifier hands a tree a row index to
# memorise. That rule is right and stays. But a pooled model DOES need to know
# which instrument a row belongs to, so the encoding has to be a genuine
# feature rather than a re-encoded identity, and it must carry a name that
# neither the identity rule nor `filter_data_by_ticker_timeframe` (which picks
# its filter column by searching for 'ticker'/'symbol' anywhere in the name)
# will claim.
INSTRUMENT_CODE_COLUMN = "instrument_code"


def instrument_code(ticker: object) -> int:
    """A stable integer for a ticker, computed rather than remembered.

    The obvious encoding is `factorize` or a sorted index, and both are wrong
    here for the same reason the label encoders elsewhere in this pipeline were
    wrong: the mapping lives only inside the call that made it. Training would
    encode AAPL as 0 because it happened to be first, prediction would encode it
    as whatever position it held that day, and nothing would report a problem.
    Adding one ticker to the config would silently renumber every other one.

    A hash of the name has none of that. It needs no persistence, it is
    identical in every process and every run, and a new ticker gets its own
    value without moving anybody else's. The ordering it induces is arbitrary,
    which costs a tree a few extra splits to isolate one instrument and costs
    nothing else -- the ordering of a factorised index is equally arbitrary.
    """
    text = str(ticker).strip().upper()
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def iter_model_contexts(
    enriched_data: pd.DataFrame | dict[str, pd.DataFrame],
    *,
    pool_tickers: bool = False,
) -> Iterator[tuple[str, str, pd.DataFrame]]:
    """Yield frames isolated by ticker and timeframe.

    With ``pool_tickers`` set, one frame per timeframe carrying every ticker is
    yielded instead, under :data:`POOLED_TICKER`. Measured on
    ``target_hourly_breakout_1h``: one pooled model beats 22 per-ticker models
    at every cost ratio tested from 0.5 to 3.0, and by a widening margin as
    false signals get more expensive. The timeframe is NOT pooled -- a 15m edge
    and a 1d edge are different phenomena, and the features carry a timeframe
    suffix, so a pooled-across-timeframes frame would be mostly missing values.
    """
    if not isinstance(enriched_data, pd.DataFrame):
        frames = []
        for identity, frame in enriched_data.items():
            if not isinstance(frame, pd.DataFrame) or frame.empty:
                continue
            candidate = frame.copy()
            if "ticker" not in candidate.columns:
                candidate["ticker"] = str(identity)
            if not pool_tickers:
                yield from iter_model_contexts(candidate)
                continue
            frames.append(candidate)
        if frames:
            # Concatenated first, so pooling actually pools. Recursing per
            # dict entry would have yielded one "pooled" frame per ticker,
            # i.e. the per-ticker split under a new name.
            yield from iter_model_contexts(
                pd.concat(frames, ignore_index=True), pool_tickers=True
            )
        return

    if "ticker" not in enriched_data.columns:
        raise ValueError("Enriched data must contain ticker.")
    timeframe_frames = partition_market_frame_by_timeframe(
        enriched_data
    )
    for timeframe, timeframe_frame in timeframe_frames.items():
        if pool_tickers:
            frame = timeframe_frame.copy()
            frame[INSTRUMENT_CODE_COLUMN] = (
                frame["ticker"].map(instrument_code).astype("int64")
            )
            yield POOLED_TICKER, timeframe, frame
            continue
        for ticker, frame in timeframe_frame.groupby(
            "ticker",
            sort=False,
            dropna=False,
        ):
            yield str(ticker), timeframe, frame.copy()
