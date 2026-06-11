"""
Utility functions for deduplication across the pipeline.
"""


import pandas as pd


def deduplicate_dataframe(df: pd.DataFrame, subset_cols: list[str]) -> tuple[pd.DataFrame, int]:
    """
    Drop duplicate rows based on the given subset of columns.

    Parameters
    ----------
    df : pd.DataFrame        DataFrame to deduplicate.
    subset_cols : List[str]
        Column names to use for identifying duplicates.

    Returns
    -------
    Tuple[pd.DataFrame, int]
        A tuple containing the deduplicated DataFrame and the number of
        duplicate rows that were removed.
    """
    if not subset_cols:
        return df, 0
    duplicates = int(df.duplicated(subset=subset_cols).sum())
    if duplicates > 0:
        df = df.drop_duplicates(subset=subset_cols)
    return df, duplicates
