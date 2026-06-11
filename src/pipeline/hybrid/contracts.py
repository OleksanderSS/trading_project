from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class ModelTrainingContext:
    """Context object for training models."""

    features_df: pd.DataFrame
    targets_df: pd.DataFrame
    ticker_col: str | None
    batch_dir: Path
    light_trainer: Any


@dataclass
class ColabBatchRequest:
    """Data package request for Colab training."""

    features_df: pd.DataFrame
    targets_df: pd.DataFrame
    tickers: list[str]
    timeframes: list[str]
    batch_name: str | None = None
    accumulate: bool = True
    check_feature_selection: bool = True
    force_feature_selection: bool = False


@dataclass
class HybridPipelineRequest:
    """Request for running full hybrid pipeline."""

    tickers: list[str] | None = None
    timeframes: list[str] | None = None
    accumulate: bool = True
    force_training: bool = False
    skip_colab: bool = False
    force_feature_selection: bool = False


@dataclass
class HybridPipelineConfig:
    """Configuration for hybrid pipeline execution."""

    output_dir: Path
    models_dir: Path
    light_models: list[str]
    heavy_models: list[str]
    gdrive_enabled: bool = False


@dataclass
class HybridFinalStagesRequest:
    """Request for running final stages."""

    features_df: pd.DataFrame | None
    targets_df: pd.DataFrame | None
    colab_results: dict[str, Any] | None = None
    light_results: dict[str, Any] | None = None
    tickers: list[str] | None = None
    timeframes: list[str] | None = None
    batch_name: str | None = None
    news_data: pd.DataFrame | None = None
    economic_data: pd.DataFrame | None = None
    market_indicators: pd.DataFrame | None = None
    stages_to_run: list[int] | None = None


@dataclass
class HybridMockFeaturesRequest:
    """Request for creating mock selected features for testing."""

    batch_dir: Path
    test_ticker: str
    test_target: str
    light_models: list[str]
    features_df: pd.DataFrame
