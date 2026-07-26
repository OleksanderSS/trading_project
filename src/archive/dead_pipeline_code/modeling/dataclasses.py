from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TargetProcessingConfig:
    """Configuration for target processing."""

    ticker: str
    df: Any
    target_name: str
    timeframe: str
    champions: dict[str, Any]


@dataclass
class FeatureLoadingConfig:
    """Configuration for feature loading."""

    model_type: str
    ticker: str
    target_name: str
    batch_dir: Path
    x_train: Any


@dataclass
class TrainingDebugInfo:
    """Training debug information."""

    context_key: str
    winner_name: str
    winner_metrics: dict[str, Any]
    all_metrics: dict[str, Any]
    selected_features: list[str]


@dataclass
class SyncFeatureLoadingConfig:
    """Configuration for synchronous feature loading."""

    model_type: str
    ticker: str
    target_name: str
    batch_dir: Path
    x_train: Any


@dataclass
class SuccessfulTrainingConfig:
    """Configuration for successful training processing."""

    ticker: str
    target_name: str
    timeframe: str
    prepared_data: dict[str, Any]
    ticker_result: dict[str, Any]
    comparison_report: dict[str, Any]
    champions: dict[str, Any]


@dataclass
class ChampionInfoConfig:
    """Configuration for champion info creation."""

    ticker: str
    target_name: str
    winner_name: str
    comparison_report: dict[str, Any]
    context_fingerprint: str
    market_regime: str
    winner_metrics: dict[str, Any]
    all_metrics: dict[str, Any]
    ticker_result: dict[str, Any]
    selected_features: list[str]


@dataclass
class SingleModelTrainingConfig:
    """Configuration for single light model training."""

    model_type: str
    ticker: str
    target_name: str
    timeframe: str
    batch_dir: Path
    x_train: Any
    y_train: Any
    x_test: Any
    y_test: Any
    task_type: str
    light_trainer: Any
    context_fingerprint: str
    market_regime: str
    volatility_regime: str


@dataclass
class LightModelChampionConfig:
    """Configuration for light model champion info creation."""

    ticker: str
    target_name: str
    model_type: str
    model_key: str
    selected_features: list[str]
    metrics: dict[str, Any]
    model_path: Path
    context_fingerprint: str
    market_regime: str
    volatility_regime: str
