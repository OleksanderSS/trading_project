
import pandas as pd
import yaml
from typing import List, Dict, Any
from pathlib import Path
from src.core.logging.logger import ProjectLogger
from src.config.unified_config_manager import get_current_config
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

    def __init__(self, targets_list, timeframe=None):
        """
        Initialize with targets in either dict or list format.
        
        Args:
            targets_list: Either a dict {target_name: config} or list [{name: ..., type: ..., params: ...}]
            timeframe: Optional timeframe to filter targets (e.g., '15m', '60m', '1d')
        """
        # Convert dict format to list format if needed
        if isinstance(targets_list, dict):
            self.targets = [
                {'name': name, **config}
                for name, config in targets_list.items()
            ]
        else:
            self.targets = targets_list
        
        # ✅ TARGET FILTERING BY TIMEFRAME: Filter targets based on timeframe
        if timeframe:
            original_count = len(self.targets)
            self.targets = self._filter_targets_by_timeframe(timeframe)
            if self.targets:
                logger.info(f"🎯 TIMEFRAME FILTERING: {timeframe} (was {original_count}, remaining {len(self.targets)})")
            else:
                logger.warning(f"⚠️ No targets found for timeframe '{timeframe}'. Using all targets.")
        
        # ✅ TARGET FILTERING: If test_target is specified, only that target is used
        import json
        runtime_params = {}
        config_manager = get_current_config()
        params_path = config_manager.get_runtime_params_path()
        if params_path.exists():
            try:
                with open(params_path, 'r') as f:
                    runtime_params = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load runtime_params.json: {e}")
        
        test_mode = runtime_params.get('test_mode', {})
        test_target = test_mode.get('test_target') or runtime_params.get('test_target')
        if test_target:
            # Filter: leave ONLY the selected target
            original_count = len(self.targets)
            self.targets = [t for t in self.targets if t['name'] == test_target]
            if self.targets:
                logger.info(f"🎯 TARGET FILTERING: {test_target} (was {original_count}, remaining {len(self.targets)})")
            else:
                logger.warning(f"⚠️ test_target '{test_target}' not found in configuration! Using all targets.")
                # Restore all targets if the selected one is not found
                if isinstance(targets_list, dict):
                    self.targets = [
                        {'name': name, **config}
                        for name, config in targets_list.items()
                    ]
                else:
                    self.targets = targets_list
        
        logger.info(f"TargetOrchestrator initialized with {len(self.targets)} target configurations.")
    
    def _filter_targets_by_timeframe(self, timeframe: str) -> list:
        """
        Filter targets based on timeframe.
        
        Rules:
        - 15m timeframe: Only intraday_15m targets
        - 60m timeframe: Only hourly_1h targets + general targets that make sense
        - 1d timeframe: Only daily/weekly targets + general targets
        
        Args:
            timeframe: The timeframe to filter for
            
        Returns:
            Filtered list of target configurations
        """
        filtered_targets = []
        
        for target in self.targets:
            name = target['name']
            
            # Check if target name indicates specific timeframe
            if timeframe == '15m':
                # 15m timeframe: Only intraday_15m targets
                if 'intraday_15m' in name or 'intraday' in name:
                    filtered_targets.append(target)
                # Also include general targets that don't have timeframe-specific names
                elif not any(timeframe_indicator in name for timeframe_indicator in ['hourly', 'weekly', 'daily']):
                    filtered_targets.append(target)
                    
            elif timeframe == '60m':
                # 60m timeframe: Only hourly_1h targets
                if 'hourly_1h' in name or 'hourly' in name:
                    filtered_targets.append(target)
                # Also include general targets that don't have timeframe-specific names
                elif not any(timeframe_indicator in name for timeframe_indicator in ['weekly', 'daily', 'intraday']):
                    filtered_targets.append(target)
                    
            elif timeframe == '1d':
                # 1d timeframe: Only daily/weekly targets
                if 'weekly' in name or 'daily' in name:
                    filtered_targets.append(target)
                # Also include general targets that don't have timeframe-specific names
                elif not any(timeframe_indicator in name for timeframe_indicator in ['hourly', 'intraday']):
                    filtered_targets.append(target)
                    
            else:
                # Unknown timeframe, keep all targets
                filtered_targets.append(target)
        
        return filtered_targets

    def generate_targets(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Generates all configured targets for the given DataFrame.
        
        Returns ONLY target columns + minimal metadata (datetime, ticker, interval).
        This prevents data leakage and keeps targets DataFrame clean.
        """
        # Validate input
        self._validate_input_dataframe(df)
        
        # Prepare metadata and targets container
        targets_dict = self._prepare_metadata_container(df)
        
        logger.info(f"🎯 Generating {len(self.targets)} targets (clean mode - no feature leakage)")

        # Generate each target
        for target_config in self.targets:
            self._generate_single_target(df, target_config, targets_dict, **kwargs)

        # Create final targets DataFrame
        return self._create_targets_dataframe(targets_dict)
    
    def _validate_input_dataframe(self, df: pd.DataFrame) -> None:
        """Validate that DataFrame has required columns."""
        if 'ticker' not in df.columns:
            logger.error("DataFrame must contain a 'ticker' column for target generation.")
            raise ValueError("Missing 'ticker' column.")
    
    def _prepare_metadata_container(self, df: pd.DataFrame) -> dict:
        """Prepare metadata container with only essential columns."""
        metadata_columns = ['datetime', 'ticker', 'interval']
        available_metadata = [col for col in metadata_columns if col in df.columns]
        return {col: df[col] for col in available_metadata}
    
    def _generate_single_target(self, df: pd.DataFrame, target_config: dict, targets_dict: dict, **kwargs) -> None:
        """Generate a single target and add it to the targets dictionary."""
        name = target_config['name']
        target_type = target_config['type']
        params = target_config.get('params', {})

        logger.debug(f"Generating target: {name} (Type: {target_type})")

        calculator_class = self.CALCULATOR_MAPPING.get(target_type)
        if not calculator_class:
            logger.warning(f"No calculator found for target type '{target_type}'. Skipping target '{name}'.")
            return

        try:
            self._handle_standard_target(df, name, target_type, params, targets_dict)
        except Exception as e:
            logger.error(f"Failed to generate target '{name}'. Error: {e}", exc_info=True)
    
    def _handle_standard_target(self, df: pd.DataFrame, name: str, target_type: str, params: dict, targets_dict: dict) -> None:
        """Handle standard target generation."""
        calculator_class = self.CALCULATOR_MAPPING[target_type]
        calculator_instance = calculator_class()
        
        method_name = self.METHOD_MAPPING.get(target_type, 'calculate')
        calculation_method = getattr(calculator_instance, method_name)

        # Process by ticker groups if ticker column exists
        if 'ticker' in df.columns:
            target_series = self._process_by_ticker_groups(df, calculation_method, params)
        else:
            target_series = calculation_method(df, **params)

        targets_dict[name] = target_series
        logger.info(f"Successfully generated target '{name}'.")
    
    def _process_by_ticker_groups(self, df: pd.DataFrame, calculation_method, params: dict) -> pd.Series:
        """Process target calculation by ticker groups."""
        target_series_list = []
        for ticker, group in df.groupby('ticker'):
            target_series_list.append(calculation_method(self._sort_group_for_targets(group), **params))
        return pd.concat(target_series_list)

    def _sort_group_for_targets(self, group: pd.DataFrame) -> pd.DataFrame:
        """Sort each ticker group chronologically before future-shift target generation."""
        for col in ("datetime", "timestamp", "date"):
            if col in group.columns:
                return group.sort_values(col).copy()
        return group.sort_index().copy()
    
    def _create_targets_dataframe(self, targets_dict: dict) -> pd.DataFrame:
        """Create final targets DataFrame and log summary."""
        targets_df = pd.DataFrame(targets_dict)
        
        target_cols = [col for col in targets_df.columns if col.startswith('target_')]
        logger.info(f"✅ Generated {len(target_cols)} target columns (total {len(targets_df.columns)} with metadata)")
        
        return targets_df
