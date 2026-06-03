"""
Pipeline Configuration Data Classes.
Groups related parameters to reduce function argument count.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import pandas as pd


@dataclass
class PipelineParams:
    """Parameters for pipeline execution."""
    tickers: Optional[List[str]] = None
    timeframes: Optional[List[str]] = None
    accumulate: bool = True
    force_training: bool = False
    skip_colab: bool = False
    force_feature_selection: bool = False


@dataclass
class FinalStagesParams:
    """Parameters for final stages execution."""
    features_df: Optional[pd.DataFrame] = None
    targets_df: Optional[pd.DataFrame] = None
    colab_results: Optional[Dict[str, Any]] = None
    light_results: Optional[Dict[str, Any]] = None
    tickers: Optional[List[str]] = None
    timeframes: Optional[List[str]] = None
    batch_name: Optional[str] = None


@dataclass
class ColabBatchParams:
    """Parameters for Colab batch preparation."""
    features_df: pd.DataFrame
    targets_df: pd.DataFrame
    tickers: List[str]
    timeframes: List[str]
    batch_name: Optional[str] = None
    accumulate: bool = True
    check_feature_selection: bool = True
    force_feature_selection: bool = False
