
import pandas as pd
import yaml
from typing import List, Dict, Any
from src.core.logging.logger import ProjectLogger
from src.targets.calculators.regression_calculator import RegressionCalculator
from src.targets.calculators.classification_calculator import ClassificationCalculator
from src.targets.calculators.indicator_prediction_calculator import IndicatorPredictionCalculator

logger = ProjectLogger.get_logger("TargetOrchestrator")

class TargetOrchestrator:
    """
    Orchestrates the generation of target variables based on a YAML configuration.
    It dynamically loads and applies the required calculators.
    """

    CALCULATOR_MAPPING = {
        "regression": RegressionCalculator,
        "classification_binary": ClassificationCalculator,
        "classification_multiclass": ClassificationCalculator,
        "indicator_prediction": IndicatorPredictionCalculator,
    }

    METHOD_MAPPING = {
        "classification_binary": "calculate_binary",
        "classification_multiclass": "calculate_multiclass",
    }

    def __init__(self, targets_list: List[Dict[str, Any]]):
        self.targets = targets_list
        logger.info(f"TargetOrchestrator initialized with {len(self.targets)} target configurations.")

    def generate_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates all configured targets for the given DataFrame.
        """
        if 'ticker' not in df.columns:
            logger.error("DataFrame must contain a 'ticker' column for target generation.")
            raise ValueError("Missing 'ticker' column.")

        df_with_targets = df.copy()
        # Use a list to collect results for each target configuration
        all_targets_df_list = [df_with_targets]

        for target_config in self.targets:
            name = target_config['name']
            target_type = target_config['type']
            params = target_config.get('params', {})

            logger.debug(f"Generating target: {name} (Type: {target_type})")

            calculator_class = self.CALCULATOR_MAPPING.get(target_type)
            if not calculator_class:
                logger.warning(f"No calculator found for target type '{target_type}'. Skipping target '{name}'.")
                continue

            try:
                calculator_instance = calculator_class()
                method_name = self.METHOD_MAPPING.get(target_type, 'calculate')
                calculation_method = getattr(calculator_instance, method_name)

                # This approach ensures that we handle single and multi-ticker data correctly
                # without relying on groupby().apply() which can have tricky return types.
                if 'ticker' in df.columns:
                    # Process by group and concatenate
                    target_series_list = []
                    for ticker, group in df.groupby('ticker'):
                        target_series_list.append(calculation_method(group.copy(), **params))
                    target_series = pd.concat(target_series_list)
                else:
                    # Process the whole dataframe if no ticker is present
                    target_series = calculation_method(df, **params)

                df_with_targets[name] = target_series
                logger.info(f"Successfully generated target '{name}'.")

            except Exception as e:
                logger.error(f"Failed to generate target '{name}'. Error: {e}", exc_info=True)

        return df_with_targets
