from __future__ import annotations

import numpy as np
import pandas as pd

from src.processing.cleaners import DataCleaner


def test_data_cleaner_forward_fill_never_crosses_interval():
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2025-01-01 10:00Z",
                    "2025-01-01 10:15Z",
                    "2025-01-01 10:00Z",
                ]
            ),
            "ticker": ["AAA", "AAA", "AAA"],
            "interval": ["15m", "15m", "60m"],
            "close": [100.0, np.nan, np.nan],
        }
    )

    result = DataCleaner.handle_missing_values(frame)

    assert result.loc[1, "close"] == 100.0
    assert pd.isna(result.loc[2, "close"])


def test_data_cleaner_outlier_returns_do_not_cross_interval():
    fifteen = pd.DataFrame(
        {
            "datetime": pd.date_range(
                "2025-01-01 10:00",
                periods=40,
                freq="15min",
                tz="UTC",
            ),
            "ticker": "AAA",
            "interval": "15m",
            "close": np.linspace(100.0, 104.0, 40),
        }
    )
    hourly = pd.DataFrame(
        {
            "datetime": pd.date_range(
                "2025-01-01 10:00",
                periods=40,
                freq="60min",
                tz="UTC",
            ),
            "ticker": "AAA",
            "interval": "60m",
            "close": np.linspace(1000.0, 1040.0, 40),
        }
    )
    frame = pd.concat([fifteen, hourly], ignore_index=True)

    result = DataCleaner.remove_outliers_zscore(
        frame,
        columns=["close"],
        threshold=3.0,
    )

    assert len(result) == len(frame)
