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


def is_pooled(ticker: object) -> bool:
    """True when this context name means "every ticker", not one of them.

    One place decides this, because three separate places got it wrong in
    three different ways and each looked correct on its own:

    - `walk_forward_validation` filtered `ticker == "__POOLED__"` against
      159,149 real rows, matched none, raised, logged at DEBUG and returned
      None -- read downstream as "could not measure, skipping", so the
      stability check was off for every pooled context (#189);
    - `prediction/data_preparation_service` does the same filter and gets an
      empty frame, so Stage 5 produced 0 predictions from 7 champions while
      the pipeline reported success (#210);
    - `prediction/prediction_context_manager` carries a second copy of that
      same line.

    None of those is a reasoning error. Each is correct for a real ticker and
    silently wrong for the sentinel, which is exactly the shape that survives
    code review. A predicate in one place can be tested once.

    Comparison is case-insensitive on the stripped name: the sentinel travels
    through file names, JSON keys and context ids, and gets re-cased on the
    way.
    """
    return str(ticker).strip().upper() == POOLED_TICKER.upper()


#: Where a prediction request records the instrument it is about, when that
#: differs from the identity of the model serving it.
INSTRUMENT_META_KEY = "_predict_for"


def artifact_ticker(meta: dict) -> str:
    """The ticker in the names of this context's saved files.

    A pooled champion is stored as `CHAMP___POOLED___60m_target_x.joblib` with
    `PREP___POOLED___60m_target_x.joblib` beside it, and those names do not
    change when the model is applied to BA.
    """
    return str(meta.get("ticker") or "")


def instrument_ticker(meta: dict) -> str:
    """The instrument whose rows this request is about.

    `meta['ticker']` carried two identities at once - which rows to take AND
    which files to open - and they are the same thing only for a per-ticker
    model. Overriding it to fan a pooled model out across instruments
    therefore also redirected the artifact lookup: Stage 5 went looking for
    `PREP_BA_15m_target_volatility_spike_1h`, did not find it, and served the
    model RAW features instead of the z-scores it was trained on. It warned
    and predicted anyway - the exact defect `preprocessor_filename` was
    written to fix, where the same champion returns 0.033 on z-scores and
    128288 on raw values.

    One field, two meanings, and the fan-out silently picked the wrong one.
    Splitting them is the fix; the split has to be named, or the next caller
    makes the same choice.
    """
    return str(meta.get(INSTRUMENT_META_KEY) or meta.get("ticker") or "")


def rows_for_ticker(frame: pd.DataFrame, ticker: object,
                    column: str = "ticker") -> pd.DataFrame:
    """The rows belonging to `ticker`, or the whole frame for a pooled name.

    Returns the frame itself when pooled -- not a copy. The filtering path in
    continue mode has already died of memory three times, twice on duplicates
    of a frame that needed no filtering at all.
    """
    if is_pooled(ticker):
        return frame
    if column not in getattr(frame, "columns", ()):
        return frame
    return frame[frame[column] == ticker]


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
            # Copy only when a column has to be added.
            #
            # Every slice was duplicated here whether or not it needed the
            # `ticker` column, and the pooled branch then copied it again to
            # attach the instrument code. On the three slices of 2026-08-30
            # that is roughly 7 GiB of copies before a single model is fitted,
            # and the stage died holding about 9 GiB. A frame that already
            # carries its tickers is passed through untouched.
            complete = "ticker" in frame.columns
            if complete:
                candidate = frame
            else:
                candidate = frame.copy()
                candidate["ticker"] = str(identity)
            if not pool_tickers:
                yield from iter_model_contexts(candidate)
                continue
            frames.append((candidate, complete))
        if frames:
            # Concatenate ONLY when the dict is keyed by ticker.
            #
            # Two shapes arrive here. A dict keyed by TICKER has frames with
            # no `ticker` column of their own -- pooling those does require
            # concatenation, and recursing per entry would yield one "pooled"
            # frame per ticker, i.e. the per-ticker split under a new name.
            #
            # A dict keyed by TIMEFRAME is the shape `_load_prepared_batch`
            # returns, and each frame already carries every ticker. Merging
            # those rebuilds the union this project spent a day removing:
            # measured 2026-08-30, `pd.concat` of the three slices asked for
            # 1.95 GiB on a (210, 1_243_783) float64 block and killed the
            # stage before a single model was fitted. The docstring says the
            # timeframe is not pooled; concatenating first and partitioning
            # after made that true of the OUTPUT while paying the union's
            # price on the way.
            #
            # So: pool inside each entry that is already ticker-complete, and
            # concatenate only the rest.
            ready = [f for f, is_complete in frames if is_complete]
            partial = [f for f, is_complete in frames if not is_complete]
            for frame in ready:
                yield from iter_model_contexts(frame, pool_tickers=True)
            if partial:
                yield from iter_model_contexts(
                    pd.concat(partial, ignore_index=True), pool_tickers=True
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
