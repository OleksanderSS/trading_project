"""
Pipeline Configuration Data Classes.
Groups related parameters to reduce function argument count.
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class PipelineParams:
    """Parameters for pipeline execution."""
    tickers: list[str] | None = None
    timeframes: list[str] | None = None
    accumulate: bool = True
    force_training: bool = False
    skip_colab: bool = False
    force_feature_selection: bool = False


@dataclass
class FinalStagesParams:
    """Parameters for final stages execution."""
    features_df: pd.DataFrame | None = None
    targets_df: pd.DataFrame | None = None
    colab_results: dict[str, Any] | None = None
    light_results: dict[str, Any] | None = None
    tickers: list[str] | None = None
    timeframes: list[str] | None = None
    batch_name: str | None = None


@dataclass
class ColabBatchParams:
    """Parameters for Colab batch preparation."""
    features_df: pd.DataFrame
    targets_df: pd.DataFrame
    tickers: list[str]
    timeframes: list[str]
    batch_name: str | None = None
    accumulate: bool = True
    check_feature_selection: bool = True
    force_feature_selection: bool = False
