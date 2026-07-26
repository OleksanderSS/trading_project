import numpy as np
import pandas as pd


class DataCleaner:
    """Not the live DataCleaner. src/processing/cleaners.py::DataCleaner
    is the one used in production (imported by
    src/pipeline/stages/processing/data_handler.py for
    remove_outliers_zscore/handle_missing_values). This class has no
    production callers of its own - it's kept only because
    clean_numeric_data() is exercised directly by
    tests/unit/test_p1_missing_policy_math.py, a regression test locking
    in a specific historical missing-data-imputation fix (infinities ->
    NaN, not silently dropped). Don't edit this file expecting it to
    change pipeline behavior - edit src/processing/cleaners.py instead.
    """

    @staticmethod
    def clean_numeric_data(df: pd.DataFrame) -> pd.DataFrame:
        """Replace infinite numeric values with missing values."""
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            return df
        df = df.copy()
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        return df
