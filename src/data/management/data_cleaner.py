import numpy as np
import pandas as pd


class DataCleaner:
    @staticmethod
    def clean_numeric_data(df: pd.DataFrame) -> pd.DataFrame:
        """Replace infinite numeric values with missing values."""
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            return df
        df = df.copy()
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        return df
