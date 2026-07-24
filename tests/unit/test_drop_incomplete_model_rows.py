"""Tests for DataPreparationService._drop_incomplete_model_rows.

Context: the function's own name and docstring said "drop rows with
unavailable model inputs instead of fabricating zeros", but the
implementation always did the opposite -- filled NaN feature values with
0.0 and never dropped a row (the log message even said "filling ... instead
of dropping rows", contradicting the docstring one line above it). Feeding a
trained model a zero-filled RSI/SMA/etc. value is indistinguishable from a
real 0 reading to the model -- it produces a confident, silently wrong
prediction instead of correctly skipping a row it can't honestly predict
for.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.stages.prediction.data_preparation_service import DataPreparationService


class TestDropIncompleteModelRows:
    def test_complete_rows_pass_through_unchanged(self):
        service = DataPreparationService()
        df = pd.DataFrame({"rsi": [50.0, 60.0, 70.0], "sma": [1.0, 2.0, 3.0]})
        result = service._drop_incomplete_model_rows(df, ["rsi", "sma"], "ctx")
        pd.testing.assert_frame_equal(result, df)

    def test_incomplete_rows_are_dropped_not_zero_filled(self):
        service = DataPreparationService()
        df = pd.DataFrame({"rsi": [50.0, np.nan, 70.0], "sma": [1.0, 2.0, 3.0]})
        result = service._drop_incomplete_model_rows(df, ["rsi", "sma"], "ctx")
        assert len(result) == 2
        assert list(result.index) == [0, 2]
        # The point of the fix: no 0.0 anywhere that used to be NaN.
        assert not (result[["rsi", "sma"]] == 0.0).any().any()
        assert not result[["rsi", "sma"]].isna().any().any()

    def test_all_rows_incomplete_returns_none(self):
        service = DataPreparationService()
        df = pd.DataFrame({"rsi": [np.nan, np.nan], "sma": [1.0, 2.0]})
        result = service._drop_incomplete_model_rows(df, ["rsi", "sma"], "ctx")
        assert result is None

    def test_no_model_feature_cols_returns_input_unchanged(self):
        service = DataPreparationService()
        df = pd.DataFrame({"rsi": [np.nan, 60.0]})
        result = service._drop_incomplete_model_rows(df, [], "ctx")
        pd.testing.assert_frame_equal(result, df)

    def test_missing_values_outside_model_feature_cols_are_ignored(self):
        """Only model_feature_cols matter -- NaN in an unrelated column
        must not cause a row to be dropped."""
        service = DataPreparationService()
        df = pd.DataFrame({"rsi": [50.0, 60.0], "unrelated_context_col": [np.nan, 1.0]})
        result = service._drop_incomplete_model_rows(df, ["rsi"], "ctx")
        assert len(result) == 2
