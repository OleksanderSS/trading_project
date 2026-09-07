"""
Ticker-aware future shift for target calculation.

Every future target is a lookahead shift of some column. Performed with a plain
``Series.shift`` on a frame that holds more than one ticker, the shift walks off
the end of one ticker and picks up the first rows of the next one, so the last
rows of every ticker get another asset's price as their "future". That is not a
noisy label, it is a fabricated one.

``TargetOrchestrator`` already splits by ticker before calling a calculator, but
``TemporalTargetGuard`` calls the same calculators on the whole enriched frame.
Grouping inside the shift itself makes both paths correct and keeps the
orchestrator path a no-op (a single-ticker group groups into itself).
"""
from __future__ import annotations

import pandas as pd

TICKER_COLUMN = "ticker"
TIME_COLUMNS = ("datetime", "timestamp", "date")


def _time_column(df: pd.DataFrame) -> str | None:
    """Return the first recognised chronological column, if any."""
    for col in TIME_COLUMNS:
        if col in df.columns:
            return col
    return None


def future_shift(df: pd.DataFrame, column: str, shift: int) -> pd.Series:
    """
    Shift ``column`` by ``shift`` periods without crossing a ticker boundary.

    Rows are ordered chronologically inside each ticker before shifting when the
    frame carries a time column, so an unsorted frame yields real future values
    rather than whatever happened to sit in the next row. The result is always
    realigned to ``df.index``.

    Args:
        df: Input frame. May hold one ticker, many, or no ticker column at all.
        column: Column to look ahead on.
        shift: Lookahead offset; negative means "into the future".

    Returns:
        The shifted series, aligned to ``df.index``.
    """
    if TICKER_COLUMN not in df.columns:
        return df[column].shift(shift)

    time_col = _time_column(df)
    ordered = df.sort_values([TICKER_COLUMN, time_col]) if time_col else df

    shifted = ordered.groupby(TICKER_COLUMN, sort=False)[column].shift(shift)
    return shifted.reindex(df.index)
