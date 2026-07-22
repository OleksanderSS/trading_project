from typing import Any

import pandas as pd

from src.analytics.engines.causal_engine import CausalEngine
from src.core.exceptions import DataProcessingError
from src.core.logging.logger import ProjectLogger

from ..interfaces import IAnalyzer

logger = ProjectLogger.get_logger(__name__)

class CausalEventFinder(IAnalyzer):
    """
    Wrapper for CausalEngine that implements IAnalyzer interface.
    Detects causal events and estimates their effects on outcomes.
    """

    def __init__(self, treatment: str = "event_detected", outcome: str = "target_future_return",
                 common_causes: list = None):
        """
        Initialize the CausalEventFinder.
        """
        self.treatment = treatment
        self.outcome = outcome
        self.common_causes = common_causes or []
        logger.info(f"CausalEventFinder initialized: treatment={treatment}, outcome={outcome}")

    def analyze(self, data: Any, **kwargs) -> dict[str, Any]:
        """
        Analyze data to find causal effects.
        """
        df = self._prepare_analysis_data(data)

        # Validation checks
        self._validate_data_columns(df)

        available_causes = self._filter_available_causes(df)
        self._validate_treatment_outcome_variance(df)

        # Use CausalEngine for analysis
        engine = CausalEngine(df, self.treatment, self.outcome, available_causes)
        engine.identify_effect()
        effect = engine.estimate_effect()

        return {
            "causal_effect": float(effect),
            "treatment": self.treatment,
            "outcome": self.outcome,
            "common_causes": available_causes,
            "status": "success"
        }

    def _filter_available_causes(self, df: pd.DataFrame) -> list[str]:
        """Filter common causes to only existing columns."""
        available_causes = [c for c in self.common_causes if c in df.columns]

        if not available_causes:
            logger.warning("No common causes available, skipping causal analysis")

        return available_causes

    def _validate_treatment_outcome_variance(self, df: pd.DataFrame) -> None:
        """Validate treatment and outcome have sufficient variance."""
        if df[self.treatment].nunique() < 2:
            raise DataProcessingError(f"Treatment '{self.treatment}' has no variance")

        if df[self.outcome].nunique() < 2:
            raise DataProcessingError(f"Outcome '{self.outcome}' has no variance")

    def _prepare_analysis_data(self, data: Any) -> pd.DataFrame:
        """Prepare and validate input data for analysis."""
        if isinstance(data, dict):
            return self._prepare_dict_data(data)
        elif isinstance(data, pd.DataFrame):
            return self._validate_dataframe(data)
        else:
            raise DataProcessingError("Unsupported data format")

    def _prepare_dict_data(self, data: dict) -> pd.DataFrame:
        """Prepare data from dictionary input."""
        price_data = data.get('price_data')
        macro_data = data.get('macro_data')

        if price_data is None and macro_data is None:
            raise DataProcessingError("Both price_data and macro_data are None")

        df = self._merge_data_sources(price_data, macro_data)
        return self._validate_dataframe(df)

    def _merge_data_sources(self, price_data: Any, macro_data: Any) -> pd.DataFrame:
        """Merge price and macro data sources."""
        if price_data is not None and macro_data is not None:
            return pd.merge(price_data, macro_data,
                          left_index=True, right_index=True, how='left', on=None,
                          validate='one_to_one')
        elif price_data is not None:
            return price_data
        else:
            return macro_data

    def _validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate dataframe has sufficient data."""
        if df.empty or len(df) < 10:
            raise DataProcessingError("Insufficient data for causal analysis")

        return df

    def _validate_data_columns(self, df: pd.DataFrame) -> None:
        """Validate that required columns exist in the data and follow naming conventions."""

        # Check leakage naming convention
        forbidden_patterns = ['future_', 'next_', 'tomorrow_', 'forward_', 'fwd_', 'ahead_']  # audit-ignore: validation pattern list

        for col_name in [self.treatment, self.outcome]:
            if any(p in col_name.lower() for p in forbidden_patterns) and not col_name.startswith('target_'):
                raise DataProcessingError(
                    f"Leakage hazard: '{col_name}' contains future data patterns but lacks 'target_' prefix. "
                    "Please rename to 'target_<name>'."
                )

        if self.treatment not in df.columns:
            raise DataProcessingError(f"Treatment column '{self.treatment}' not found")

        if self.outcome not in df.columns:
            raise DataProcessingError(f"Outcome column '{self.outcome}' not found")
