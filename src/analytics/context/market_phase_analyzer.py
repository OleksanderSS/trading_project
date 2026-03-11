import pandas as pd
import logging
from typing import Dict, Any, List, Optional

from ..interfaces import IAnalyzer

logger = logging.getLogger(__name__)

class MarketPhaseAnalyzer(IAnalyzer):
    """
    Determines the current market phase based on a declarative set of rules
    from a configuration dictionary.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the analyzer with market phase definition rules.

        Args:
            config (Dict[str, Any]): A dictionary containing the market phase definition,
                                     including 'indicators' and 'rules'.
        """
        phase_config = config or {}
        self.indicators = phase_config.get('indicators', {})
        self.rules = phase_config.get('rules', [])
        logger.info(f"MarketPhaseAnalyzer initialized with {len(self.rules)} rules.")

    def analyze(self, data: Dict[str, pd.DataFrame], **kwargs) -> Dict[str, Any]:
        """
        Analyzes the latest market data to determine the current market phase.

        Args:
            data (Dict[str, pd.DataFrame]): Expects a key 'market_data' with a DataFrame
                                             containing the required indicator columns.

        Returns:
            Dict[str, Any]: A dictionary containing the determined 'market_phase'.
        """
        market_data = data.get('market_data')
        if not isinstance(market_data, pd.DataFrame) or market_data.empty:
            logger.error("Input 'market_data' is missing or empty.")
            return {"market_phase": "error", "reason": "Invalid input data"}

        market_phase = "unknown" # Default phase if no rules match
        try:
            # Get the very last row of data for evaluation
            latest_data_point = market_data.iloc[-1]
            
            # Map the required indicator values from the data
            latest_values = {key: latest_data_point[val] for key, val in self.indicators.items() if val in latest_data_point}

            # Check if all required indicators were found
            if len(latest_values) != len(self.indicators):
                missing = set(self.indicators.keys()) - set(latest_values.keys())
                logger.warning(f"Missing indicators in data: {missing}. Phase evaluation might be incorrect.")

            for rule in self.rules:
                if self._evaluate_condition(rule.get('condition', 'False'), latest_values):
                    market_phase = rule.get('phase', 'unknown')
                    break # First matching rule wins
            
            logger.info(f"Determinated market phase: {market_phase}")

        except KeyError as e:
            logger.error(f"A required indicator column was not found in the market data: {e}", exc_info=True)
            market_phase = "error"
        except Exception as e:
            logger.error(f"Error evaluating market phase: {e}", exc_info=True)
            market_phase = "error"

        return {"market_phase": market_phase}

    def _evaluate_condition(self, condition_str: str, values: Dict[str, float]) -> bool:
        """
        Safely evaluates a condition string using the provided values.
        Example: "gdp > 0.5 and vix < 20"
        """
        if not condition_str:
            return False
        try:
            # Using pd.eval is a safer way to evaluate these expressions
            return bool(pd.eval(condition_str, engine='python', local_dict=values))
        except Exception as e:
            logger.error(f"Could not safely evaluate condition '{condition_str}' with values {values}: {e}")
            return False
