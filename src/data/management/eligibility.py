"""
Training eligibility of stored rows.

Some collectors mark rows they could not vouch for: fabricated stand-in rows
(``put_call_ratio``, ``cftc``) and rows with no usable time axis
(``huggingface``). They set ``eligible_for_training = False``.

Until this module existed, nothing outside those collectors ever read the flag,
so the gate was decorative: the rows were labelled and then loaded anyway.

Three states, deliberately distinguished:

* column absent        -> the source has no opinion; keep everything
* value null           -> unknown, not ineligible; keep the row
* value explicitly False -> the source said it cannot vouch for this; drop it

Treating "unknown" as ineligible would drop every row from the six collectors
that never set the flag, which would silently destroy the dataset. Absence of a
verdict is not a negative verdict.
"""
from __future__ import annotations

import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("Eligibility")

ELIGIBILITY_COLUMN = "eligible_for_training"


def has_eligibility_opinion(df: pd.DataFrame) -> bool:
    """True when the frame carries an eligibility verdict at all."""
    return ELIGIBILITY_COLUMN in df.columns


def filter_eligible_rows(df: pd.DataFrame, source: str = "unknown") -> pd.DataFrame:
    """
    Drop rows a collector explicitly marked ineligible for training.

    Args:
        df: Frame as read from storage.
        source: Table or collector name, for the log line.

    Returns:
        The frame without explicitly-ineligible rows. Returned unchanged when
        the frame carries no eligibility column.
    """
    if df is None or df.empty or not has_eligibility_opinion(df):
        return df

    # Only an explicit False drops a row; null means "not stated".
    ineligible = df[ELIGIBILITY_COLUMN].fillna(True).astype(bool).eq(False)
    dropped = int(ineligible.sum())
    if not dropped:
        return df

    logger.warning(
        f"[{source}] Dropping {dropped}/{len(df)} rows marked "
        f"{ELIGIBILITY_COLUMN}=False (fabricated stand-in rows, or rows with no "
        "usable time axis). Pass eligible_only=False to read them anyway."
    )
    return df.loc[~ineligible].copy()
