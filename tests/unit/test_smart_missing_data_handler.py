import numpy as np
import pandas as pd

from src.utils.smart_missing_data_handler import SmartMissingDataHandler


def test_indicator_fill_does_not_backfill_future_values():
    handler = SmartMissingDataHandler()
    series = pd.Series([np.nan, np.nan, 10.0, np.nan, 12.0], name="rsi")

    filled = handler._fill_indicator_data(series, "rsi")

    assert filled.iloc[0] == 50.0
    assert filled.iloc[1] == 50.0
    assert filled.iloc[2] == 10.0
    assert filled.iloc[3] == 10.0
    assert filled.iloc[4] == 12.0
