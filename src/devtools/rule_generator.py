# src/devtools/rule_generator.py

import logging
import os
from typing import Any

import pandas as pd
import yaml

# Assuming these imports are correct relative to the project structure
from src.config.unified_config_manager import UnifiedConfigManager
from src.data.management.data_manager import DataManager

logger = logging.getLogger(__name__)

class ContextRuleGenerator:
    """
    Analyzes historical data to identify statistically significant market regimes
    and generates rules for their application.
    """

    def __init__(self, config_manager: UnifiedConfigManager, data_manager: DataManager):
        """
        Initializes the generator with config and data managers.

        Args:
            config_manager (UnifiedConfigManager): The application's config manager.
            data_manager (DataManager): The application's data manager.
        """
        self.config_manager = config_manager
        self.data_manager = data_manager
        self.analysis_config = self.config_manager.get_config('context_rule_generation')

        if not self.analysis_config:
            raise ValueError("'context_rule_generation' section not found in configuration.")

        self.target_asset = self.analysis_config.get('target_asset', 'SPY')
        self.indicators_to_analyze = self.analysis_config.get('indicators_to_analyze', [])

    def run_analysis(self) -> None:
        """
        Main method to run the entire rule generation process.
        """
        logger.info("Starting context rule generation process...")

        # 1. Load data
        all_tickers = [self.target_asset] + [ind['name'] for ind in self.indicators_to_analyze]
        # This assumes DataManager has a method to fetch and align data for multiple tickers.
        try:
            historical_data = self.data_manager.load_data_for_tickers(all_tickers)
            if historical_data.empty:
                logger.error("No historical data loaded. Aborting.")
                return
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f"Failed to load historical data: {e}", exc_info=True)
            raise RuntimeError("Failed to load historical data for rule generation") from e

        # 2. Generate rules
        generated_rules = self._generate_rules(historical_data)

        # 3. Save rules
        if generated_rules:
            # The runner script will be responsible for providing a full, absolute path.
            output_path = self.analysis_config.get('output_path', 'src/config/generated_context_rules.yaml')
            self._save_rules_to_yaml(generated_rules, output_path)
        else:
            logger.warning("No rules were generated.")

    def _generate_rules(self, historical_data: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Analyzes indicators and generates rules based on the provided data.
        """
        all_rules = []
        for indicator_config in self.indicators_to_analyze:
            indicator = indicator_config['name']
            thresholds = indicator_config.get('thresholds', [])

            if indicator not in historical_data.columns:
                logger.warning(f"Indicator '{indicator}' not found in historical data. Skipping.")
                continue

            logger.info(f"Analyzing indicator: {indicator}")

            for threshold in thresholds:
                rule = self._analyze_single_indicator(historical_data.copy(), indicator, threshold)
                if rule:
                    all_rules.append(rule)

        return all_rules

    def _analyze_single_indicator(self, data: pd.DataFrame, indicator: str, threshold: dict[str, Any]) -> dict[str, Any] | None:
        """
        Analyzes the impact of a single indicator crossing a specific threshold.
        """
        condition = threshold.get('condition')
        value = threshold.get('value')
        effect_windows = threshold.get('effect_windows', [1, 5, 20])

        if not all([condition, value is not None]):
            logger.warning(f"Invalid threshold config for {indicator}: {threshold}")
            return None

        # Prepare future returns for the target asset
        target_returns = data[self.target_asset].pct_change(fill_method=None)
        for window in effect_windows:
            data[f'target_return_{window}d'] = target_returns.shift(-window)  # audit-ignore: NEGATIVE_SHIFT_INTENTIONAL target generation

        # Identify event occurrences
        if condition == '>':
            event_mask = data[indicator] > value
        elif condition == '<':
            event_mask = data[indicator] < value
        else:
            logger.warning(f"Unknown condition '{condition}' for indicator '{indicator}'")
            return None

        event_data = data[event_mask].dropna(subset=[f'target_return_{w}d' for w in effect_windows])

        if event_data.empty:
            logger.info(f"No events found for {indicator} {condition} {value}")
            return None

        effects = {}
        for window in effect_windows:
            returns_col = f'target_return_{window}d'
            effects[f'{window}d'] = {
                'mean_return': round(event_data[returns_col].mean(), 5),
                'median_return': round(event_data[returns_col].median(), 5),
                'win_rate': round((event_data[returns_col] > 0).mean(), 3)
            }

        rule = {
            'indicator': indicator,
            'condition': condition,
            'value': value,
            'event_count': int(event_mask.sum()),
            'effects_on_target': effects
        }

        logger.info(f"Generated rule: {rule}")
        return rule

    def _save_rules_to_yaml(self, rules: list[dict[str, Any]], path: str):
        """
        Saves the generated rules to a YAML file. Expects a full path.
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                yaml.dump({'generated_context_rules': rules}, f, allow_unicode=True, sort_keys=False)
            logger.info(f"Rules successfully saved to {path}")
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f"Failed to save rules to {path}: {e}")
            raise RuntimeError(f"Failed to save rules to {path}") from e
