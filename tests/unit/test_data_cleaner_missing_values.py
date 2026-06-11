import pandas as pd

from src.processing.cleaners import DataCleaner


def test_bfill_method_does_not_backfill_future_values():
    df = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT"],
            "value": [None, 10.0, None, None, 20.0],
        }
    )

    cleaned = DataCleaner.handle_missing_values(df, method="bfill")

    assert pd.isna(cleaned.loc[0, "value"])
    assert cleaned.loc[1, "value"] == 10.0
    assert cleaned.loc[2, "value"] == 10.0
    assert pd.isna(cleaned.loc[3, "value"])
    assert cleaned.loc[4, "value"] == 20.0
