import pandas as pd
import logging
from typing import Dict, Any, List, Optional

from ..interfaces import IAnalyzer

logger = logging.getLogger(__name__)

class PredictionAdjuster(IAnalyzer):
    """
    Adjusts model predictions based on a generic, declarative rule engine that uses
    market context.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the PredictionAdjuster with a list of adjustment rules.

        Args:
            config (Dict[str, Any]): A dictionary containing a 'rules' list.
        """
        self.rules: List[Dict[str, Any]] = (config or {}).get('rules', [])
        if not self.rules:
            logger.warning("PredictionAdjuster initialized with no rules. No adjustments will be made.")
        else:
            logger.info(f"PredictionAdjuster initialized with {len(self.rules)} adjustment rules.")

    def analyze(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Enhances model predictions by applying the configured rules to the context data.

        Args:
            data (Dict[str, Any]): A dictionary containing:
                - 'predictions' (Dict[str, float]): Model names and their raw predictions.
                - Other context features (e.g., 'market_phase', 'avg_sentiment') used by the rules.

        Returns:
            Dict[str, Any]: A dictionary containing the 'enhanced_predictions'.
        """
        model_predictions = data.get('predictions', {})
        if not model_predictions:
            logger.warning("No 'predictions' found in data to adjust.")
            return {'enhanced_predictions': {}}

        context_info = {key: val for key, val in data.items() if key != 'predictions'}
        enhanced_predictions = {}

        for model_name, prediction_value in model_predictions.items():
            adjustment_factor = 1.0

            for rule in self.rules:
                try:
                    if self._evaluate_rule_conditions(rule.get('if', {}), context_info):
                        adjustment_factor = self._apply_rule_action(rule.get('then', {}), adjustment_factor)
                except Exception as e:
                    logger.error(f"Error processing rule '{rule.get('name', 'Unnamed')}': {e}", exc_info=True)
            
            enhanced_value = prediction_value * adjustment_factor
            enhanced_predictions[model_name] = enhanced_value
            logger.debug(f"Adjusting '{model_name}': Original={prediction_value:.4f}, Factor={adjustment_factor:.4f}, Enhanced={enhanced_value:.4f}")

        logger.info(f"Completed prediction adjustments for {len(model_predictions)} model(s).")
        return {'enhanced_predictions': enhanced_predictions}

    def _evaluate_rule_conditions(self, conditions: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """
        Evaluates the 'if' block of a rule. Currently supports an 'all' (AND) block.
        """
        # If there are no conditions, the rule is not applicable.
        if 'all' not in conditions or not isinstance(conditions['all'], list):
            return False

        for condition in conditions['all']:
            feature = condition.get('context_feature')
            if feature not in context:
                return False # One of the required context features is missing.

            context_value = context[feature]

            # Categorical check (e.g., market_phase == 'Growth')
            if 'is' in condition and context_value != condition['is']:
                return False
            
            # Numerical checks
            if 'greater_than' in condition and not context_value > condition['greater_than']:
                return False
            if 'less_than' in condition and not context_value < condition['less_than']:
                return False
        
        # If we get here, all conditions in the 'all' block were met.
        return True

    def _apply_rule_action(self, action: Dict[str, Any], current_factor: float) -> float:
        """
        Applies the action from a rule's 'then' block.
        This version modifies a single adjustment factor.
        """
        if action.get('action') == 'apply_multiplier':
            multiplier = action.get('multiplier', 1.0)
            weight = action.get('weight', 1.0)
            # The formula blends the multiplier effect based on the weight
            # This can be changed to a simple multiplication if desired (factor * multiplier)
            current_factor *= (1 + (multiplier - 1) * weight)
        
        return current_factor
