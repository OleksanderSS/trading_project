import pandas as pd
import pytest

from src.analytics.analyzers.causal_event_finder import CausalEventFinder
from src.core.exceptions import DataProcessingError


def test_causal_event_finder_default_outcome_is_target_prefixed():
    finder = CausalEventFinder()
    df = pd.DataFrame(
        {
            "event_detected": [0, 1, 0, 1],
            "target_future_return": [0.01, 0.03, -0.01, 0.02],
        }
    )

    finder._validate_data_columns(df)


def test_causal_event_finder_rejects_future_outcome_without_target_prefix():
    finder = CausalEventFinder(outcome="future_return")
    df = pd.DataFrame(
        {
            "event_detected": [0, 1, 0, 1],
            "future_return": [0.01, 0.03, -0.01, 0.02],
        }
    )

    with pytest.raises(DataProcessingError, match="Leakage hazard"):
        finder._validate_data_columns(df)

