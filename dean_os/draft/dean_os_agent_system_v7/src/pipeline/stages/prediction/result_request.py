from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class PredictionResultRequest:
    """Request object describing a completed prediction for result building."""

    context_id: str
    ticker: str
    adjusted_prediction: float
    raw_prediction: float
    model_contributions: dict[str, float]
    best_model_name: str
    ticker_df_clean: pd.DataFrame
    meta: dict[str, Any]
    shap_explanations: dict[str, Any] | None = None
    model_output_contract: dict[str, Any] | None = None
