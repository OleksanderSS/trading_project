
from typing import Any, Protocol, Type

import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.targets.calculators.classification_calculator import ClassificationCalculator
from src.targets.calculators.indicator_prediction_calculator import IndicatorPredictionCalculator
from src.targets.calculators.regression_calculator import RegressionCalculator

logger = ProjectLogger.get_logger("TargetOrchestrator")


class TargetCalculator(Protocol):
    def calculate(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        ...

class TargetOrchestrator:
    """
    Unified Orchestrator for target variable generation.
    """
    CALCULATOR_MAPPING: dict[str, Type[TargetCalculator]] = {
        "regression": RegressionCalculator,
        "classification_binary": ClassificationCalculator,
        "indicator_prediction": IndicatorPredictionCalculator,
    }

    def __init__(self, targets_list: Any):
        # Handle both dict and list input formats
        if isinstance(targets_list, dict):
            self.targets = [{'name': k, **v} for k, v in targets_list.items()]
        else:
            self.targets = targets_list

        logger.info(f"TargetOrchestrator initialized with {len(self.targets)} configurations.")

    @property
    def target_configs(self) -> dict[str, list[dict]]:
        """Returns targets grouped by timeframe keywords."""
        grouped: dict[str, list[dict]] = {'15m': [], '60m': [], '1d': [], 'mixed': []}
        for t in self.targets:
            name = t['name'].lower()
            if '15m' in name: grouped['15m'].append(t)
            elif '1h' in name or '60m' in name: grouped['60m'].append(t)
            elif '1d' in name: grouped['1d'].append(t)
            else: grouped['mixed'].append(t)
        return grouped

    def generate_targets(self, df: pd.DataFrame, timeframe: str | None = None, **kwargs) -> pd.DataFrame:
        """Main entry point for generating targets for a DataFrame."""
        if df.empty: return pd.DataFrame()

        # ✅ ENHANCED: Validate critical columns
        required = {'datetime', 'ticker'}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            logger.error(f"Missing required columns for target generation: {missing}")
            raise KeyError(f"TargetOrchestrator requires {missing} to be present in input DataFrame")

        # Filter targets if timeframe provided
        targets_to_run = self.target_configs.get(timeframe, self.targets) if timeframe else self.targets

        results = {'datetime': df['datetime'], 'ticker': df['ticker']}
        if 'interval' in df.columns: results['interval'] = df['interval']

        for config in targets_to_run:
            try:
                calc_type = config.get('type', 'regression')
                calc_class = self.CALCULATOR_MAPPING.get(calc_type, RegressionCalculator)
                calc: TargetCalculator = calc_class()

                # Perform calculation (assumes calculators have a 'calculate' method)
                # and handle ticker grouping internally if needed
                res = calc.calculate(df, **config.get('params', {}))
                results[config['name']] = res
            except Exception as e:
                logger.error(f"Failed to generate target {config['name']}: {e}")

        return pd.DataFrame(results)
