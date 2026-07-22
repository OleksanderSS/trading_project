from __future__ import annotations

from collections.abc import Iterator

import pandas as pd

from src.pipeline.timeframe_lineage import (
    partition_market_frame_by_timeframe,
)


def iter_model_contexts(
    enriched_data: pd.DataFrame | dict[str, pd.DataFrame],
) -> Iterator[tuple[str, str, pd.DataFrame]]:
    """Yield frames isolated by ticker and timeframe."""
    if not isinstance(enriched_data, pd.DataFrame):
        for identity, frame in enriched_data.items():
            if not isinstance(frame, pd.DataFrame) or frame.empty:
                continue
            candidate = frame.copy()
            if "ticker" not in candidate.columns:
                candidate["ticker"] = str(identity)
            yield from iter_model_contexts(candidate)
        return

    if "ticker" not in enriched_data.columns:
        raise ValueError("Enriched data must contain ticker.")
    timeframe_frames = partition_market_frame_by_timeframe(
        enriched_data
    )
    for timeframe, timeframe_frame in timeframe_frames.items():
        for ticker, frame in timeframe_frame.groupby(
            "ticker",
            sort=False,
            dropna=False,
        ):
            yield str(ticker), timeframe, frame.copy()
