from typing import Dict, Any, List, Optional

from ..interfaces import IAnalyzer
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

class AdaptiveConfidenceAnalyzer(IAnalyzer):
    """
    Calculates an adaptive confidence threshold based on declarative rules
    that evaluate market context provided by other analyzers.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the analyzer with a base confidence and a list of adjustment rules.

        Args:
            config (Dict[str, Any]): A dictionary containing:
                - 'base_confidence' (float): The starting confidence threshold.
                - 'rules' (List[Dict]): A list of rules for adjustments.
        """
        self.config = config or {}
        self.base_confidence = self.config.get('base_confidence', 0.55)
        self.rules = self.config.get('rules', [])
        logger.info(f"AdaptiveConfidenceAnalyzer initialized with base confidence: {self.base_confidence}")

    def analyze(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Calculates the adaptive confidence by applying rules to the input context.

        Args:
            data (Dict[str, Any]): A dictionary of context features from other analyzers,
                                     e.g., {'market_regime': 'Volatile', 'sentiment_std': 0.6}.

        Returns:
            Dict[str, Any]: A dictionary containing the 'adaptive_confidence_threshold'.
        """
        confidence_threshold = self.base_confidence

        for rule in self.rules:
            try:
                if self._evaluate_rule_conditions(rule.get('if', {}), data):
                    confidence_threshold = self._apply_rule_action(rule.get('then', {}), confidence_threshold)
            except Exception as e:
                logger.error(f"Error processing rule '{rule.get('name', 'Unnamed')}': {e}", exc_info=True)
        
        # Cap the confidence threshold to a reasonable maximum
        final_threshold = min(confidence_threshold, self.config.get('max_confidence', 0.85))

        logger.info(f"Calculated adaptive confidence threshold: {final_threshold:.4f}")
        return {'adaptive_confidence_threshold': final_threshold}

    def _evaluate_rule_conditions(self, conditions: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """
        Evaluates the 'if' block of a rule, supporting 'all' (AND) and 'any' (OR) logic.
        """
        if 'all' in conditions:
            return all(self._check_condition(cond, context) for cond in conditions['all'])
        if 'any' in conditions:
            return any(self._check_condition(cond, context) for cond in conditions['any'])
        return False

    def _check_condition(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """
        Checks a single condition from a rule.
        """
        feature = condition.get('context_feature')
        if feature not in context:
            return False

        context_value = context[feature]

        if 'is' in condition and context_value == condition['is']:
            return True
        if 'is_not' in condition and context_value != condition['is_not']:
            return True
        if 'greater_than' in condition and context_value > condition['greater_than']:
            return True
        if 'less_than' in condition and context_value < condition['less_than']:
            return True
        
        return False

    def _apply_rule_action(self, action: Dict[str, Any], current_threshold: float) -> float:
        """
        Applies the action from a rule's 'then' block, modifying the threshold.
        """
        if action.get('action') == 'increase_threshold':
            return current_threshold + action.get('value', 0.0)
        if action.get('action') == 'decrease_threshold':
            return current_threshold - action.get('value', 0.0)
        if action.get('action') == 'set_threshold':
            return action.get('value', current_threshold)
        
        return current_threshold
