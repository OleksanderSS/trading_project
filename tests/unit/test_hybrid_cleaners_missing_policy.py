from pathlib import Path

import pandas as pd

from src.pipeline.hybrid.data_manager import HybridDataManager
from src.pipeline.hybrid.data_utils import DataUtils


def test_data_utils_clean_dataframe_preserves_numeric_missing_values():
    df = pd.DataFrame(
        {
            "price": [1.0, float("inf"), None, float("-inf")],
            "label": ["a", None, "c", "d"],
        }
    )

    cleaned = DataUtils().clean_dataframe(df)

    assert cleaned["price"].isna().tolist() == [False, True, True, True]
    assert cleaned["label"].tolist() == ["a", "unknown", "c", "d"]


def test_hybrid_data_manager_clean_dataframe_preserves_numeric_missing_values(tmp_path: Path):
    df = pd.DataFrame({"price": [1.0, float("inf"), None, float("-inf")]})

    cleaned = HybridDataManager(tmp_path).clean_dataframe(df)

    assert cleaned["price"].isna().tolist() == [False, True, True, True]
